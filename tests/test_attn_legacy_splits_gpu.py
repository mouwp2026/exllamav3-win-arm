"""Compare the split-plan option with byte-checked original attention kernels.

Only the GPU owner should run this:
  SPARK_ATTN_LEGACY_SOURCE=/original94/.../attention_fn/triton_paged.py \
  SPARK_ATTN_BASELINE_SOURCE=/frozen4b/triton_paged.py \
  python -m pytest tests/test_attn_legacy_splits_gpu.py -q -s

Uses the original Triton bodies, AOT signatures/alignment attributes used by BC,
the actual GB10 split-cap formula, shuffled physical pages and FP16 KV caches.
Candidate can be supplied with SPARK_ATTN_CANDIDATE_SOURCE without installing it.
This does not load a model or modify any runtime checkout.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import sys
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

GOLDEN_SHA256 = {
    "legacy": "b7235157f6efa0d2ffcf2bf37586cccdb36be3ac23cb7e10ff704334aeb28000",
    "baseline": "0315b952b17d079fca1cdb481ee9ca814cf8707a895b6f75db3ac8518c870fd1",
}
NAMES = {
    "_get_h32", "_rot_h32", "_qc_plane_kt", "_qc_plane_v", "_qc_load_kt", "_qc_load_v",
    "_paged_attn_decode_split_kernel", "_paged_attn_decode_combine_kernel",
    "decode_row_layout", "decode_split_programs", "combine_subtiles",
}


def load_source(path: Path, label: str):
    source = path.read_bytes()
    if label in GOLDEN_SHA256:
        assert hashlib.sha256(source).hexdigest() == GOLDEN_SHA256[label], (
            f"{label} source differs from the declared immutable reference")
    tree = ast.parse(source, filename=str(path))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    found = {n.name for n in functions}
    assert {"_paged_attn_decode_split_kernel", "_paged_attn_decode_combine_kernel"} <= found
    tree.body = functions
    name = f"spark_attn_compare_{label}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    import triton
    import triton.language as tl
    module.__dict__.update(torch=torch, triton=triton, tl=tl, _decode_legacy_splits=False, _h32_cache={})
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def sources():
    paths = {}
    for label in ("legacy", "baseline"):
        value = os.environ.get(f"SPARK_ATTN_{label.upper()}_SOURCE")
        if not value:
            pytest.skip(f"Set SPARK_ATTN_{label.upper()}_SOURCE to the frozen source")
        paths[label] = Path(value).resolve(strict=True)
    candidate = os.environ.get("SPARK_ATTN_CANDIDATE_SOURCE")
    paths["candidate"] = (Path(candidate) if candidate else Path(__file__).resolve().parents[1] /
                          "exllamav3/modules/attention_fn/triton_paged.py").resolve(strict=True)
    return {name: load_source(path, name) for name, path in paths.items()}


def aot_launch(fn, runtime, constants, signature, grid, warps, stages, aligned):
    """AOT compile using the BC pointer contracts, not the JIT launch inference."""
    import triton
    from triton.compiler import ASTSource
    attrs = {}
    if aligned:
        names = {"q", "k_cache", "v_cache", "block_table", "out", "partial_o", "partial_ml",
                 "k_scales", "v_scales", "h32"}
        for name in names.intersection(signature):
            assert runtime[name].data_ptr() % 16 == 0
            attrs[(fn.arg_names.index(name),)] = [["tt.divisibility", 16]]
    sig = signature | {n: "constexpr" for n in constants}
    assert set(sig) == set(fn.arg_names)
    compiled = triton.compile(
        ASTSource(fn=fn, signature=sig, constexprs=constants, attrs=attrs),
        options={"num_warps": warps, "num_stages": stages})
    # The generated Python launcher keeps constexpr positions in its argument
    # list even though they are omitted from the device ABI.
    bound = runtime | constants
    compiled[grid](*[bound[name] for name in fn.arg_names])


def run(module, label, enabled, q, kc, vc, block, seqlens, qc=None, kv_heads_override=None):
    import triton
    batch, q_len, q_heads, hd = q.shape
    kv_heads = kc.shape[2] if qc is None else kv_heads_override
    assert kv_heads is not None
    group = q_heads // kv_heads
    hp = triton.next_power_of_2(hd)
    if label == "legacy":
        block_m = triton.next_power_of_2(q_len)
        block_h = max(16 // block_m, 1)
        rows, blocks = block_m * block_h, triton.cdiv(group, block_h)
    else:
        rows, blocks = module.decode_row_layout(q_len, group, hp)
    programs = batch * kv_heads * blocks
    occupancy = programs
    if label == "candidate":
        module._decode_legacy_splits = enabled
        occupancy = module.decode_split_programs(batch, q_len, kv_heads, group, blocks)
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    cap = max(1, min(2 * sms // occupancy, 128))
    bn = max(16, 8192 // hp)
    bound = block.shape[1] * 256 + q_len
    splits = max(1, min(cap, triton.cdiv(bound, bn)))
    span = triton.cdiv(triton.cdiv(bound, splits), bn) * bn
    out = torch.full_like(q, float("nan"))
    po = torch.empty(programs * cap * rows * hp, device=q.device, dtype=torch.float32)
    ml = torch.empty(programs * cap * rows * 2, device=q.device, dtype=torch.float32)
    if qc is None:
        sk, sv, qck, qcv, h32 = q, q, 0, 0, q
    else:
        sk, sv, qck, qcv = qc
        h32 = module._get_h32(q.device)
    runtime = dict(q=q, k_cache=kc, v_cache=vc, block_table=block, cache_seqlens=seqlens,
                   out=out, partial_o=po, partial_ml=ml, k_scales=sk, v_scales=sv, h32=h32,
                   split_len=span, num_pages_per_seq=block.shape[1], num_splits=splits, sinks=q)
    signature = {
        "q": "*fp16", "k_cache": "*fp16", "v_cache": "*fp16",
        "block_table": "*i32", "cache_seqlens": "*i32", "out": "*fp16",
        "partial_o": "*fp32", "partial_ml": "*fp32", "k_scales": "*fp16",
        "v_scales": "*fp16", "h32": "*fp16", "split_len": "i32",
        "num_pages_per_seq": "i32", "num_splits": "i32", "sinks": "*fp32",
    }
    if qc is not None:
        signature.update(k_cache="*i32", v_cache="*i32")
    constants = dict(
        QCK=qck, QCV=qcv, q_len=q_len, kv_append_len=q_len, n_q_heads=q_heads,
        n_kv_heads=kv_heads, page_size=256, head_dim=hd, HD_PAD=hp,
        scale=hd ** -.5, CAUSAL=True, WINDOW_LEFT=-1, WINDOW_RIGHT=-1,
        SOFTCAP=0., FINAL=False, HAS_SINKS=False, BLOCK_ROWS=rows, BLOCK_N=bn,
    )
    if label == "legacy":
        constants.update(BLOCK_M=block_m, BLOCK_H=block_h)
    aot_launch(module._paged_attn_decode_split_kernel, runtime, constants, signature,
               (programs, cap, 1), 4, 2, aligned=label != "legacy")
    rs, ds = module.combine_subtiles(rows, hp)
    c_runtime = {k: runtime[k] for k in ("partial_o", "partial_ml", "out", "h32", "num_splits", "sinks")}
    c_signature = {k: signature[k] for k in c_runtime}
    c_constants = dict(QCV=qcv, HAS_SINKS=False, q_len=q_len, n_q_heads=q_heads,
                       n_kv_heads=kv_heads, head_dim=hd, HD_PAD=hp,
                       BLOCK_ROWS=rows, ROWS_SUB=rs, D_SUB=ds)
    if label == "legacy":
        c_constants.update(BLOCK_M=block_m, BLOCK_H=block_h)
    else:
        c_constants["V_DIM"] = hd
    aot_launch(module._paged_attn_decode_combine_kernel, c_runtime, c_constants, c_signature,
               (programs, (rows // rs) * (hp // ds), 1), 4, 1,
               aligned=label != "legacy" and not (label == "candidate" and enabled))
    assert torch.isfinite(out).all()
    return out, {"programs": programs, "cap": cap, "splits": splits, "span": span,
                 "rows": rows, "sms": sms}


@pytest.mark.parametrize("batch,q_len,prefix,group,hd", [
    (1, 6, 1023, 12, 256),
    (1, 1, 1023, 12, 256),
    (1, 6, 255, 12, 256),
    (1, 4, 1023, 12, 256),
    (1, 8, 1023, 12, 256),
    (1, 5, 255, 6, 128),
    (2, 6, 1023, 12, 256),
    (1, 16, 4095, 12, 256),
], ids=["spark_q6_p1023", "spark_q1_p1023", "spark_q6_p255", "power2_q4",
        "power2_q8", "gqa6_q5", "batch2_q6", "long_q16"])
def test_actual_old_aot_numerics(sources, batch, q_len, prefix, group, hd):
    generator = torch.Generator().manual_seed(941603 + q_len + prefix + group)
    # Match the quality probe's five-page allocation for its short-prefix cases.
    kv_heads, pages = 2, (max(prefix + q_len, 1072) + 255) // 256
    q = torch.randn(batch, q_len, kv_heads * group, hd, generator=generator).half().cuda()
    kc = torch.randn(batch * pages, 256, kv_heads, hd, generator=generator).half().cuda()
    vc = torch.randn(batch * pages, 256, kv_heads, hd, generator=generator).half().cuda()
    block = torch.randperm(batch * pages, generator=generator).reshape(batch, pages).int().cuda()
    seqlens = torch.full((batch,), prefix, dtype=torch.int32, device="cuda")
    old, old_plan = run(sources["legacy"], "legacy", False, q, kc, vc, block, seqlens)
    baseline, baseline_plan = run(sources["baseline"], "baseline", False, q, kc, vc, block, seqlens)
    off, off_plan = run(sources["candidate"], "candidate", False, q, kc, vc, block, seqlens)
    compatible, compatible_plan = run(sources["candidate"], "candidate", True, q, kc, vc, block, seqlens)
    assert torch.equal(off, baseline), "Disabled option changed frozen4b output"
    assert off_plan == baseline_plan
    assert compatible_plan["cap"] == old_plan["cap"]
    assert compatible_plan["splits"] == old_plan["splits"]
    assert compatible_plan["span"] == old_plan["span"]
    assert compatible_plan["programs"] == baseline_plan["programs"]
    changed = int((compatible != old).sum())
    max_abs = float((compatible.float() - old.float()).abs().max())
    print(f"q{q_len} p{prefix} b{batch} g{group} d{hd}: old={old_plan}, packed={baseline_plan}, "
          f"compat_changed={changed}/{old.numel()}, max_abs={max_abs}, "
          f"default_changed={int((baseline != old).sum())}")
    assert torch.equal(compatible, old), (
        f"Historical split plan did not preserve original AOT output: {changed} changed, maxabs={max_abs}")
    if (batch, q_len, prefix, group, hd) == (1, 6, 1023, 12, 256) and old_plan["sms"] == 48:
        assert not torch.equal(old, baseline), "Fixture must expose the old/new partition-rounding difference"


@pytest.mark.parametrize("batch,q_len,prefix", [
    (1, 1, 1023), (1, 6, 255), (1, 6, 1023), (4, 6, 1023),
], ids=["q8_q1", "q8_q6_page_boundary", "q8_q6", "q8_batch4_q6"])
@torch.inference_mode()
def test_actual_old_aot_q8_cache_numerics(sources, batch, q_len, prefix):
    from exllamav3.ext import exllamav3_ext as ext
    kv_heads, group, hd, bits = 2, 12, 256, 8
    pages = (max(prefix + q_len, 1072) + 255) // 256
    gen = torch.Generator().manual_seed(841603 + batch + q_len + prefix)
    q = torch.randn(batch, q_len, kv_heads * group, hd, generator=gen).half().cuda()
    kc = torch.randn(batch * pages, 256, kv_heads, hd, generator=gen).half().cuda()
    vc = torch.randn(batch * pages, 256, kv_heads, hd, generator=gen).half().cuda()
    block = torch.randperm(batch * pages, generator=gen).reshape(batch, pages).int().cuda()
    seqlens = torch.full((batch,), prefix, dtype=torch.int32, device="cuda")
    def quantize(cache):
        rows = batch * pages * 256
        packed = torch.empty(rows, kv_heads * hd // 32 * bits, dtype=torch.int32, device="cuda")
        scales = torch.empty(rows, kv_heads * hd // 32, dtype=torch.float16, device="cuda")
        # Same native quantizer/zero companding as CacheLayer_quant(K8,V8).
        ext.quant_cache_cont(cache.view(rows, kv_heads * hd), packed, scales, 0.)
        return packed.view(batch * pages, 256, -1), scales.view(batch * pages, 256, -1)
    pk, sk = quantize(kc)
    pv, sv = quantize(vc)
    kw = {"qc": (sk, sv, bits, bits), "kv_heads_override": kv_heads}
    old, old_plan = run(sources["legacy"], "legacy", False, q, pk, pv, block, seqlens, **kw)
    baseline, baseline_plan = run(sources["baseline"], "baseline", False, q, pk, pv, block, seqlens, **kw)
    off, off_plan = run(sources["candidate"], "candidate", False, q, pk, pv, block, seqlens, **kw)
    compatible, compatible_plan = run(sources["candidate"], "candidate", True, q, pk, pv, block, seqlens, **kw)
    assert off_plan == baseline_plan and torch.equal(off, baseline), "Disabled option changed frozen4b Q8"
    assert compatible_plan["cap"] == old_plan["cap"]
    assert compatible_plan["splits"] == old_plan["splits"]
    assert compatible_plan["span"] == old_plan["span"]
    assert compatible_plan["programs"] == baseline_plan["programs"]
    changed = int((compatible != old).sum())
    max_abs = float((compatible.float() - old.float()).abs().max())
    print(f"K8/V8 b{batch} q{q_len} p{prefix}: compatible changed={changed}/{old.numel()}, "
          f"maxabs={max_abs}, default changed={int((baseline != old).sum())}", flush=True)
    assert torch.equal(compatible, old), f"Q8 original-output mismatch: {changed} changed, maxabs={max_abs}"
