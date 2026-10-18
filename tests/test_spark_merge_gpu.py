"""
Targeted CUDA checks for the upstream-1.6 / GB10 fork integration.

Run only while the inference server is stopped:
    PYTHONPATH=. python -m pytest -q tests/test_spark_merge_gpu.py
    EXL3_GR_INT8=0 PYTHONPATH=. python -m pytest -q \
        tests/test_gr_mix_tiled.py -k 'fused_decode or weighted_copy or tiled_tables'

These tests use synthetic weights, not a model download. They check folded-int8 GR
table parity, quantized-kernel arithmetic, retained upstream prefill tables and
runtime per-expert mixed-K decode against separately reconstructed dense weights.
Model-level teacher-forced logits and MTP acceptance remain separate release gates.
"""
import pytest
import torch
import torch.nn.functional as F

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import hyperconnections as hc

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DEVICE = torch.device("cuda:0")



@pytest.fixture(autouse=True)
def precise_torch_reference():
    # Do not let a deployment's TF32 preference weaken the float32 ground truth.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous

def relative_error(actual, expected):
    scale = expected.float().abs().max().clamp_min(1e-12)
    return ((actual.float() - expected.float()).abs().max() / scale).item()


def make_gr(monkeypatch, *, int8, combine, seed=13):
    monkeypatch.setattr(hc, "_GR_INT8", int8)
    torch.manual_seed(seed)
    H, D, rank = 4, 2560, 320
    site = hc.GatedResidual(None, "spark_test", H, D, 1e-6, combine)
    site.device = DEVICE
    site.norm_w_raw = torch.randn(H * D, device=DEVICE) * 0.1
    down = torch.randn(rank, H * D, device=DEVICE) / (H * D) ** 0.5
    up = torch.randn(H * D, rank, device=DEVICE) / rank ** 0.5
    inject = torch.randn(H, H * D, device=DEVICE) / (H * D) ** 0.5 if combine else None
    site._prepare(down, up, inject, keep_source_weights=True)
    return site, down, up, inject


def quantized_gr_reference(site, streams):
    # Independent torch implementation of the historical folded-int8 kernel's
    # arithmetic. Quantization error is separated from CUDA arithmetic error.
    H, D = site.hc_mult, site.hidden_size
    raw = streams.reshape(-1, H, D).float()
    unit = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + site.rms_eps)
    folded = site.fn_q.float() * site.fn_s[:, None]
    projections = F.linear(unit.flatten(1), folded) / H
    hidden = F.silu(projections[:, :site.rank])
    up = (site.upx_q.float()
          * site.upx_s.reshape(H, D // 4, 4)[:, :, None, :])
    up = up.permute(0, 1, 3, 2).reshape(H * D, site.rank)
    gates = torch.sigmoid(F.linear(hidden, up)).reshape(-1, H, D)
    mixed = (gates * unit * site.w_h.reshape(H, D).float()).mean(1)
    post = 2 * torch.sigmoid(projections[:, site.rank:]) if site.use_combine else None
    return post, mixed


@pytest.mark.parametrize("combine", [False, True])
def test_gr_int8_preserves_folded_tables_and_kernel_math(monkeypatch, combine):
    site, down, up, inject = make_gr(monkeypatch, int8=True, combine=combine)
    assert site.fn_q is not None
    assert site.proj_h is not None and site.upx_h is not None

    original_projection = torch.cat([down.half()] + ([] if inject is None else [inject.half()]))
    folded = (original_projection.float()
              * site.w_h.float()).half().float()
    scale = folded.abs().amax(1).clamp_min(1e-8) / 127
    quant = (folded / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    assert torch.equal(site.fn_q, quant)
    assert torch.equal(site.fn_s, scale)
    # The persistent fp16 projection must stay UNFOLDED for upstream tiled prefill.
    assert torch.equal(site.proj_h[:site.proj_m], original_projection)

    for rows in (1, 2, 5, 8):
        torch.manual_seed(rows)
        streams = torch.randn(1, rows, 4, 2560, device=DEVICE) * 3
        post_ref, mixed_ref = quantized_gr_reference(site, streams)
        post, mixed = site._mix(streams, cached=False)
        err = relative_error(mixed, mixed_ref)
        assert err < 3e-3, ("int8 mixed", rows, combine, err)
        if combine:
            err = relative_error(post, post_ref)
            assert err < 1e-3, ("int8 post", rows, combine, err)
        else:
            assert post is None
        post2, mixed2 = site._mix(streams, cached=False)
        assert torch.equal(mixed, mixed2)
        assert not combine or torch.equal(post, post2)


@pytest.mark.parametrize("combine", [False, True])
def test_gr_fp16_decode_and_tiled_prefill_share_unfolded_tables(monkeypatch, combine):
    site, _, _, _ = make_gr(monkeypatch, int8=False, combine=combine)
    assert site.fn_q is None
    assert site.tiled, "CUDA extension must include upstream deterministic GR prefill"
    for rows in (1, 8, 33, 257):
        torch.manual_seed(rows)
        streams = torch.randn(1, rows, 4, 2560, device=DEVICE) * 3
        post_ref, mixed_ref = site._mix_ref(streams)
        post, mixed = site._mix(streams, cached=False)
        assert relative_error(mixed, mixed_ref.reshape(rows, 2560)) < 3e-3
        if combine:
            assert relative_error(post, post_ref.reshape(rows, 4)) < 1e-3
        if rows > site.FUSED_MAX_R:
            # Adding int8 decode tables must not change the prefill result or
            # make it depend on the released checkpoint-oriented up table.
            saved_up = site.up_h
            site._quantize_int8(4, 2560)
            site.up_h = None
            post2, mixed2 = site._mix(streams, cached=False)
            assert torch.equal(mixed, mixed2)
            assert not combine or torch.equal(post, post2)
            site.fn_q = site.fn_s = site.upx_q = site.upx_s = None
            site.up_h = saved_up


class MixedProjection:
    def __init__(self, in_features, out_features, bitrates, gen):
        self.bitrates = torch.tensor(bitrates, dtype=torch.int32)
        self.trellis, self.suh, self.svh, self.weights = [], [], [], []
        for bits in bitrates:
            shape = (in_features // 16, out_features // 16, 16 * bits)
            trellis = torch.randint(0, 65536, shape, dtype=torch.int32,
                                    generator=gen).to(torch.int16).to(DEVICE)
            def scale(size):
                mag = torch.rand(size, generator=gen) * 0.2 + 0.9
                sign = torch.where(torch.rand(size, generator=gen) < 0.5, -1.0, 1.0)
                return (mag * sign).half().to(DEVICE)
            self.trellis.append(trellis)
            self.suh.append(scale(in_features))
            self.svh.append(scale(out_features))
            weight = torch.empty((in_features, out_features), dtype=torch.half, device=DEVICE)
            ext.reconstruct(weight, trellis, bits, False, True)
            self.weights.append(weight)
        self.tabs = tuple(torch.tensor([t.data_ptr() for t in tensors],
                                       dtype=torch.long, device=DEVICE)
                          for tensors in (self.trellis, self.suh, self.svh))

    def dense(self, x, expert):
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, self.suh[expert], None, 1.0)
        y = torch.empty((x.shape[0], self.weights[expert].shape[1]),
                        dtype=torch.half, device=DEVICE)
        ext.hgemm(xh, self.weights[expert], y)
        ext.had_r_128(y, y, None, self.svh[expert], 1.0)
        return y.float()


@pytest.mark.parametrize("hidden,intermediate", [(256, 256), (2560, 768)])
@pytest.mark.parametrize("activation,gu_dtype", [("silu", torch.half), ("gelu", torch.float)])
def test_runtime_mixedk_coop_matches_dense_and_reuses_scratch(hidden, intermediate, activation, gu_dtype):
    assert hasattr(ext, "CoopMK"), "GB10 build must include the fork's CoopMK extension"
    gen = torch.Generator().manual_seed(47)
    E, topk, max_rows = 8, 4, 8
    # Every integer bitrate 1..8 appears WITHIN each projection, unlike tests
    # that change a single uniform bitrate between gate/up/down launches.
    gate = MixedProjection(hidden, intermediate, list(range(1, 9)), gen)
    up = MixedProjection(hidden, intermediate, list(range(1, 9)), gen)
    down = MixedProjection(intermediate, hidden, list(range(8, 0, -1)), gen)
    slots = max_rows * topk
    empty = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=DEVICE)
    had_g, had_u = empty((slots, hidden), torch.half), empty((slots, hidden), torch.half)
    gu_g = empty((slots, intermediate), gu_dtype)
    gu_u = empty((slots, intermediate), gu_dtype)
    act = empty((slots, intermediate), torch.half)
    dout = empty((slots, hidden), torch.float)
    ctr = torch.zeros(slots * (intermediate // 128) + max_rows * (hidden // 128)
                      + 2 + slots + 1 + slots, dtype=torch.int, device=DEVICE)
    out = empty((max_rows, hidden), torch.float)
    coop = ext.CoopMK(
        hidden, *gate.tabs, *up.tabs, *down.tabs, None, None, None,
        gate.bitrates, up.bitrates, down.bitrates, False, True,
        0 if activation == "silu" else 1, 0.0, True,
        had_g, had_u, gu_g, gu_u, act, dout, ctr, out, None, -1, -1, 3)
    for rows in (1, 4, 8, 1):
        x = (torch.randn((rows, hidden), generator=gen) * 0.05).half().to(DEVICE)
        selected = torch.stack([torch.randperm(E, generator=gen)[:topk] for _ in range(rows)]).to(DEVICE)
        # Guarantee every width is selected across the multi-row calls.
        if rows >= 4:
            selected[0] = torch.tensor([0, 1, 2, 3], device=DEVICE)
            selected[1] = torch.tensor([4, 5, 6, 7], device=DEVICE)
        routing = (torch.rand((rows, topk), generator=gen) * 0.9 + 0.1).half().to(DEVICE)
        routing[0, 0] = 0
        ref = torch.zeros((rows, hidden), device=DEVICE)
        for row in range(rows):
            for pick in range(topk):
                expert = selected[row, pick].item()
                g = gate.dense(x[row:row + 1], expert)
                u = up.dense(x[row:row + 1], expert)
                a = F.silu(g) if activation == "silu" else F.gelu(g, approximate="tanh")
                ref[row] += routing[row, pick].float() * down.dense((a * u).half(), expert)[0]
        for plan in (3, 4):
            out.fill_(float("nan"))
            coop.run(x, selected, routing, None, -1, -1, plan)
            first = out[:rows].clone()
            coop.run(x, selected, routing, None, -1, -1, plan)
            torch.cuda.synchronize()
            assert torch.isfinite(first).all()
            assert torch.equal(first, out[:rows]), ("nondeterministic", rows, plan)
            err = relative_error(first, ref)
            assert err < 3e-3, (hidden, intermediate, activation, rows, plan, err)


def test_device_mtp_keeps_pinned_positions_stable_during_async_uploads():
    """A queued GPU delay exposes host-side staging overwrites deterministically."""
    import importlib.util
    from pathlib import Path

    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("This torch build has no CUDA delay primitive for the race test")
    path = Path(__file__).with_name("test_mtp_row_budget_cpu.py")
    spec = importlib.util.spec_from_file_location("mtp_source_probe", path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    probe = harness._probe(2)
    probe.iterate_draftmodel_mtp_gen.__func__.__globals__["_MTP_DEVICE_DRAFT"] = True
    for job in probe.active_jobs:
        job.mtp_last_hidden = job.mtp_last_hidden.to(DEVICE)
    probe.draft_ids_pinned = torch.zeros((2, 5), dtype=torch.long, pin_memory=True)
    staging = {}

    def stage(name, *shape):
        key = (name, shape)
        if key not in staging:
            staging[key] = torch.zeros(shape, dtype=torch.int32, pin_memory=True)
        return staging[key]
    probe._staging = stage

    class AsyncDraft:
        def __init__(self):
            self.positions = []
            self.last_ids = None
            self.delay = False

        def forward(self, ids, params):
            # Queue delay after the initial pageable input-ID upload, which may
            # synchronize host staging and otherwise accidentally hide the race.
            if self.delay:
                torch.cuda._sleep(10_000_000)
            self.positions.append(params["cache_seqlens"].to(DEVICE, non_blocking=True))
            self.last_ids = ids
            return params["target_hidden"] + 0.125

        def sample_from_state(self, state, params):
            return self.last_ids

    probe.draft_model = AsyncDraft()
    # Warm allocations first: pinning memory for the first time can itself
    # synchronize CUDA and hide the race this test is intended to expose.
    probe.iterate_draftmodel_mtp_gen([])
    torch.cuda.synchronize()
    probe.draft_model.positions.clear()
    probe.draft_model.delay = True
    stream = torch.cuda.Stream(device=DEVICE)
    with torch.cuda.stream(stream):
        result = probe.iterate_draftmodel_mtp_gen([])
    stream.synchronize()
    assert result.shape[-1] == 3
    assert [p.cpu().tolist() for p in probe.draft_model.positions] == [[4, 4], [5, 5], [6, 6]]
