"""Real-pack unified mixed-K NOSYNC equivalence and asynchronous buffer tests.

GPU owner only:
  EXL3_MOE_COOP_MIXEDK=0 \
  SPARK_MIXEDK_MODEL=/home/cruzspark/models/flashnext-exl3-sage-4.15bpw \
  python -m pytest tests/test_mixedk_nosync_pack_gpu.py -q -s

Only one actual MoE layer is loaded with the production single-device loader;
all 512 experts, shared expert, and router weights are real. Router selections
are controlled to exercise changing counts, concentrated rows, and capacity
boundaries. This tests dispatch/scratch correctness, not the router algorithm.
No model files or installed runtime configuration are changed.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
@torch.inference_mode()
def actual_layer():
    path = os.environ.get("SPARK_MIXEDK_MODEL")
    if not path:
        pytest.skip("Set SPARK_MIXEDK_MODEL to the actual mixed-K Qwen pack")
    from exllamav3 import Config, Model
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.util.tensor import g_tensor_cache
    moe = importlib.import_module("exllamav3.modules.block_sparse_mlp")
    torch.cuda.set_device(0)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(moe, "_COOP_MIXEDK_ENV", "0")
        patch.setattr(moe, "COOP_MIXEDK", False)
        patch.setattr(moe, "MIXEDK_MIN_ROWS", 0)
        patch.setattr(moe, "FUSED_DET", True)
        patch.setenv("EXL3_MOE_MIXEDK_GROUPS", "0")
        patch.setenv("EXL3_MOE_MIXEDK_CMAX", "0")
        config = Config.from_directory(path)
        model = Model.from_config(config, component="text")
        key = os.environ.get("SPARK_MIXEDK_LAYER_KEY", "model.language_model.layers.0.mlp")
        layer = model.find_module(key)
        assert isinstance(layer, BlockSparseMLP), f"{key} is not a MoE layer"
        # Use exactly the deferred-load lifecycle used by Model.load(), restricted
        # to this module; no target KV cache, PLE table, or whole-model allocation.
        model._load_single(False, torch.device("cuda:0"), config, [layer], False)
        g_tensor_cache.drop_all()
        assert layer.hidden_size == layer.expert_size == 2560
        assert layer.intermediate_size == 640
        assert layer.num_experts == layer.num_local_experts == 512
        assert layer.num_experts_per_tok == 10
        assert layer.mixedk_unified and not layer.uniform_expert_q
        assert layer.coopmk is None and not layer.cpu_offload and layer.cpu_split_first is None
        assert layer.routing_gate is not None
        patch.setattr(layer, "routing_fn",
                      lambda rows, cfg, y, params: (params["_test_selected"], params["_test_weights"]))
        print(json.dumps({
            "model": str(Path(path).resolve()), "layer": layer.key,
            "config_sha256": hashlib.sha256((Path(path) / "config.json").read_bytes()).hexdigest(),
            "engine_module": moe.__file__,
            "engine_module_sha256": hashlib.sha256(Path(moe.__file__).read_bytes()).hexdigest(),
            "K_histograms": {name: dict(Counter(float(p.inner.K) for p in projections))
                             for name, projections in (("gate", layer.gates), ("up", layer.ups),
                                                       ("down", layer.downs))},
        }), flush=True)
        try:
            yield SimpleNamespace(layer=layer, moe=moe, ext=moe.ext)
        finally:
            torch.cuda.synchronize()
            layer.unload()
            g_tensor_cache.drop_all()


def make_case(batch, query, concentrated, seed, topk=10):
    rows, experts, hidden = batch * query, 512, 2560
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(batch, query, hidden, generator=gen) * .25).half().pin_memory()
    if concentrated:
        selected = torch.randperm(experts, generator=gen)[:topk].expand(rows, topk).clone()
    else:
        selected = torch.stack([torch.randperm(experts, generator=gen)[:topk] for _ in range(rows)])
    weights = torch.rand(rows, topk, generator=gen) + .05
    weights /= weights.sum(dim=-1, keepdim=True)
    weights = weights.half()
    weights[0, 0] = 0.  # A zero-weight route must still have a count/slot.
    expected = torch.zeros(experts + 1, dtype=torch.long)
    for expert, count in Counter(selected.flatten().tolist()).items():
        expected[expert] = count
    return SimpleNamespace(
        x=x, selected=selected.pin_memory(), weights=weights.pin_memory(),
        expected_counts=expected, rows=rows, topk=topk, batch=batch, query=query,
        concentrated=concentrated, seed=seed,
    )


def to_device(case, buffers=None):
    if buffers is None:
        return (case.x.cuda(), case.selected.cuda(), case.weights.cuda())
    key = (case.batch, case.query, case.topk)
    if key not in buffers:
        buffers[key] = (torch.empty_like(case.x, device="cuda"),
                        torch.empty_like(case.selected, device="cuda"),
                        torch.empty_like(case.weights, device="cuda"))
    x, selected, weights = buffers[key]
    # Immutable pinned source tensors are retained by the caller until the stream
    # finishes. Reusing the GPU destinations is ordered on that same stream.
    x.copy_(case.x, non_blocking=True)
    selected.copy_(case.selected, non_blocking=True)
    weights.copy_(case.weights, non_blocking=True)
    return x, selected, weights


@contextmanager
def tracking(state, enabled, forbid_host_reads=False):
    records = []
    bincount_calls = []
    original_mixed = state.ext.exl3_moe_mixedk
    original_gather = state.ext.exl3_moe_gather
    original_bincount = torch.bincount
    current = {"label": None}

    def mixed(*args, **kwargs):
        assert not kwargs and len(args) == 35, "Review tracker against changed native call ABI"
        result = original_mixed(*args)
        records.append({
            "label": current["label"], "kind": "mixed",
            "counts": args[2].clone(), "tokens": args[3].clone(), "weights": args[4].clone(),
            "base": args[31].clone() if args[31] is not None else None,
            "num_active": args[29], "concurrency": args[5].shape[0],
            "range_tile": tuple(args[32:35]),
        })
        return result

    def gather(*args, **kwargs):
        assert not kwargs and len(args) == 8
        result = original_gather(*args)
        records.append({
            "label": current["label"], "kind": "gather",
            "selected": args[2].clone(), "inverse_order": args[3].clone(),
            "starts": args[4].clone(), "base": args[5].clone(), "kinds": args[6].clone(),
            "weights": args[7].clone(),
        })
        return result

    def bincount(*args, **kwargs):
        bincount_calls.append(current["label"])
        if forbid_host_reads:
            raise AssertionError("Eligible NOSYNC path called torch.bincount")
        return original_bincount(*args, **kwargs)

    def blocked(*args, **kwargs):
        raise AssertionError("Eligible NOSYNC path performed Tensor.item/tolist")

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(state.moe, "MIXEDK_NOSYNC", enabled))
        stack.enter_context(mock.patch.object(state.ext, "exl3_moe_mixedk", mixed))
        stack.enter_context(mock.patch.object(state.ext, "exl3_moe_gather", gather))
        stack.enter_context(mock.patch.object(torch, "bincount", bincount))
        if forbid_host_reads:
            stack.enter_context(mock.patch.object(torch.Tensor, "item", blocked))
            stack.enter_context(mock.patch.object(torch.Tensor, "tolist", blocked))
        yield SimpleNamespace(records=records, bincount_calls=bincount_calls, current=current)


def forward(state, case, device_inputs):
    x, selected, weights = device_inputs
    out = state.layer.forward(x, {"_test_selected": selected, "_test_weights": weights},
                              out_dtype=torch.float32)
    # Outputs and routing buffers can be reused by later module calls. Snapshot
    # on the producer stream before enqueueing the next input-buffer update.
    return out.clone()


def compare_records(reference, candidate, case):
    assert len(reference) == len(candidate)
    mixed_count = 0
    for old, new in zip(reference, candidate):
        assert old["kind"] == new["kind"]
        for key in old:
            if isinstance(old[key], torch.Tensor):
                assert torch.equal(old[key], new[key]), f"{old['kind']} {key} changed"
            elif key not in ("num_active", "label"):
                assert old[key] == new[key], f"{old['kind']} {key} changed"
        if old["kind"] == "mixed":
            mixed_count += 1
            assert torch.equal(new["counts"].cpu(), case.expected_counts), "Expert counts are stale/wrong"
            assert min(old["num_active"], old["concurrency"]) == min(new["num_active"], new["concurrency"])
    if case.rows <= 256:
        assert mixed_count >= 1, "Test did not exercise the unified mixed-K kernel"


def check_outputs(reference, candidate):
    assert torch.isfinite(reference).all() and torch.isfinite(candidate).all()
    assert torch.equal(reference, candidate), (
        f"NOSYNC changed output: maxabs={(reference-candidate).abs().max().item()}, "
        f"changed={(reference != candidate).sum().item()}")


@torch.inference_mode()
def test_cold_buffer_guard(actual_layer):
    state = actual_layer
    case = make_case(1, 6, False, 20261008)
    inputs = to_device(case)
    # Explicitly exercise the lazy-buffer predicate even if this test is selected
    # after another test: no queued GPU work remains at this boundary.
    torch.cuda.synchronize()
    state.layer._mkd_bufs = None
    with tracking(state, True) as cold:
        candidate = forward(state, case, inputs)
    assert cold.bincount_calls, "Cold path must keep the original readback until buffers exist"
    assert state.layer._mkd_bufs is not None
    assert state.layer._mkd_fused_rows == 256
    assert state.layer._mkd_bufs.temp_state_g.shape[0] <= 10
    with tracking(state, False) as old:
        reference = forward(state, case, inputs)
    check_outputs(reference, candidate)
    compare_records(old.records, cold.records, case)


@pytest.mark.parametrize("batch,query,concentrated", [
    (1, 1, False), (1, 6, True), (1, 6, False), (1, 16, True),
    (4, 6, True), (1, 26, True), (1, 257, True),
], ids=["q1", "q6_concentrated", "q6_spread", "q16_tier_limit",
        "batch4_q6_fallthrough", "assignments_over256", "expert_count_over256"])
@torch.inference_mode()
def test_warm_counts_slots_and_capacity_guards(actual_layer, batch, query, concentrated):
    state = actual_layer
    case = make_case(batch, query, concentrated, 411 + batch + query)
    inputs = to_device(case)
    # Warm actual buffers/graphs using the reference path.
    with tracking(state, False) as old:
        reference = forward(state, case, inputs)
    assert old.bincount_calls
    eligible = case.rows <= 16 and case.rows * case.topk <= state.layer._mkd_fused_rows
    with tracking(state, True, forbid_host_reads=eligible) as new:
        candidate = forward(state, case, inputs)
    assert bool(new.bincount_calls) == (not eligible)
    check_outputs(reference, candidate)
    compare_records(old.records, new.records, case)
    if case.rows == 24:
        assert any(r["kind"] == "mixed" and r["range_tile"][2] == 32 for r in new.records)
    print(f"rows={case.rows} topk={case.topk} concentrated={concentrated} "
          f"nosync_eligible={eligible}: exact output/count/slot match", flush=True)


@torch.inference_mode()
def test_topk_below_concurrency_keeps_readback(actual_layer):
    state = actual_layer
    # This is an explicit dispatch guard control, not a proposed model setting.
    warm = make_case(1, 6, False, 620)
    with tracking(state, False):
        forward(state, warm, to_device(warm))
    concurrency = int(state.layer._mkd_bufs.temp_state_g.shape[0])
    assert concurrency >= 2
    topk = concurrency - 1
    case = make_case(1, 6, True, 621, topk=topk)
    inputs = to_device(case)
    with mock.patch.object(state.layer, "num_experts_per_tok", topk):
        with tracking(state, False) as old:
            reference = forward(state, case, inputs)
        with tracking(state, True) as new:
            candidate = forward(state, case, inputs)
    assert new.bincount_calls
    check_outputs(reference, candidate)
    compare_records(old.records, new.records, case)


@torch.inference_mode()
def test_delayed_stream_reused_inputs_and_changing_routes(actual_layer):
    state = actual_layer
    cases = [make_case(1, q, bool(i % 2), 9500 + i) for i, q in
             enumerate((1, 6, 16, 6, 1, 6, 16, 1, 6))]
    references, reference_records = [], []
    for case in cases:
        with tracking(state, False) as old:
            references.append(forward(state, case, to_device(case)))
        reference_records.append(old.records)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    buffers, actual = {}, []
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        with tracking(state, True, forbid_host_reads=True) as new:
            for index, case in enumerate(cases):
                new.current["label"] = index
                actual.append(forward(state, case, to_device(case, buffers)))
    stream.synchronize()
    for index, (case, expected, got) in enumerate(zip(cases, references, actual)):
        check_outputs(expected, got)
        records = [r for r in new.records if r["label"] == index]
        compare_records(reference_records[index], records, case)
    assert not new.bincount_calls
    print("Delayed nondefault stream: nine changing routes with reused GPU buffers "
          "matched reference outputs/counts/slots bitwise; no item/tolist/bincount", flush=True)
