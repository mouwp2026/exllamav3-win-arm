"""CPU checks for the opt-in dense decode split plan (no model imports/CUDA).

For the independent frozen-source checks:
  SPARK_ATTN_LEGACY_BC_SOURCE=/path/to/94ba01d/bc_attn.py \
  SPARK_ATTN_BASELINE_BC_SOURCE=/path/to/4b08348f/bc_attn.py \
  python -m pytest tests/test_attn_legacy_splits_cpu.py -q

The original configure functions are executed with allocation/compiler stubs,
so the expected cap and kernel constants come from actual immutable sources.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGED = ROOT / "exllamav3/modules/attention_fn/triton_paged.py"
BC = ROOT / "exllamav3/modules/attention_fn/bc_attn.py"
GOLDEN = {
    "legacy": "bbcef6b6548d85666801ca4f529f7cfd276ad8c03d14612d92780ace2bef45e4",
    "baseline": "1326b9bf1d64f57dee24cb5f5cbeb657aa7c855f376cb8c2d0feed914edc5902",
}
TRITON = SimpleNamespace(
    next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
    cdiv=lambda a, b: (a + b - 1) // b,
)


def helpers(enabled):
    tree = ast.parse(PAGED.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in {"decode_row_layout", "decode_split_programs", "combine_subtiles"}]
    ns = {"triton": TRITON, "_decode_legacy_splits": enabled}
    exec(compile(tree, str(PAGED), "exec"), ns)
    return ns


class Buffer:
    def view(self, *shape):
        return self

    def zero_(self):
        return self


def configure(path, bsz, q_len, group, hd, enabled):
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BCAttn")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_configure")
    # Only remove imports; all dispatch arithmetic, kernel constants/signatures,
    # allocation sizes, and configure_slot arguments execute unmodified.
    fn.body = [n for n in fn.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    module = ast.Module(body=[fn], type_ignores=[])
    calls, allocations, compiled = [], [], []

    def get(device, shape, dtype, tag):
        allocations.append((tag, tuple(shape), dtype))
        return Buffer()

    def bucket(device, elements, dtype, tag):
        allocations.append((tag, elements, dtype))
        return Buffer()

    def compile_kernel(device, kernel, sig, consts, warps, stages, **kwargs):
        result = SimpleNamespace(grid_y=None, name=kernel)
        compiled.append((kernel, sig, consts, warps, stages, kwargs, result))
        return result

    ns = helpers(enabled)
    ns.update(
        triton=TRITON,
        torch=SimpleNamespace(half="fp16", float="fp32", float32="fp32"),
        PAGE_SIZE=256,
        _SPLIT_WARPS=4, _SPLIT_STAGES=2,
        attn_decode_config=lambda dev, hp: (max(16, 8192 // hp), 4, 2),
        _get_sm_count=lambda dev: 48,
        _normalize_window=lambda window: (-1, -1),
        _compile_kernel=compile_kernel,
        _paged_attn_decode_split_kernel="split",
        _paged_attn_decode_combine_kernel="combine",
        _paged_kv_update_kernel="update",
        g_tensor_cache=SimpleNamespace(get=get, get_bucketed=bucket),
    )
    exec(compile(module, str(path), "exec"), ns)
    obj = SimpleNamespace(
        device=SimpleNamespace(index=0), head_dim=hd, v_head_dim=hd,
        num_q_heads=2 * group, num_kv_heads=2, quant=False, k_bits=0, v_bits=0,
        window_size=None, sm_scale=hd ** -.5, softcap=None, sinks=None,
        gate_mode=0, qsa=False, hidden_size=2560, hidden_padded=2560,
        g_weight=None, o_dtype="fp32", _pointer_range_32=lambda: False,
        bc=SimpleNamespace(configure_slot=lambda *args: calls.append(args)),
    )
    ns["_configure"](obj, bsz, q_len, True, 0)
    assert len(calls) == 1
    return calls[0], allocations, compiled


@pytest.fixture(scope="module")
def sources():
    paths = {}
    for label, digest in GOLDEN.items():
        value = os.environ.get(f"SPARK_ATTN_{label.upper()}_BC_SOURCE")
        if not value:
            pytest.skip(f"Set SPARK_ATTN_{label.upper()}_BC_SOURCE")
        p = Path(value).resolve(strict=True)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
        paths[label] = p
    return paths


def test_split_occupancy_is_separate_from_packed_grid():
    off, on = helpers(False), helpers(True)
    for q_len in range(1, 17):
        for group in (1, 2, 3, 4, 6, 8, 12, 24):
            for bsz in (1, 2, 4, 8):
                rows, blocks = off["decode_row_layout"](q_len, group, 256)
                assert (rows, blocks) == on["decode_row_layout"](q_len, group, 256)
                assert off["decode_split_programs"](bsz, q_len, 2, group, blocks) == bsz * 2 * blocks
                legacy_heads = max(16 // TRITON.next_power_of_2(q_len), 1)
                assert on["decode_split_programs"](bsz, q_len, 2, group, blocks) == (
                    bsz * 2 * TRITON.cdiv(group, legacy_heads))


@pytest.mark.parametrize("bsz,q_len,group,hd", [
    (1, 1, 12, 256), (1, 6, 12, 256), (4, 6, 12, 256),
    (1, 5, 6, 128), (2, 16, 3, 64),
])
def test_actual_frozen_configurations(sources, bsz, q_len, group, hd):
    old, _, old_compiled = configure(sources["legacy"], bsz, q_len, group, hd, False)
    base, base_alloc, base_compiled = configure(sources["baseline"], bsz, q_len, group, hd, False)
    off, off_alloc, off_compiled = configure(BC, bsz, q_len, group, hd, False)
    on, on_alloc, on_compiled = configure(BC, bsz, q_len, group, hd, True)
    assert off[13:16] == base[13:16], "Disabled option changed live block/cap/grid arguments"
    assert off_alloc == base_alloc, "Disabled option changed static allocation footprint"
    for a, b in zip(off_compiled, base_compiled):
        assert a[:6] == b[:6] and a[6].grid_y == b[6].grid_y
    assert on[13:15] == old[13:15], "Enabled option failed to preserve actual old block/cap"
    assert on[15] == off[15], "Enabled option changed the packed launch grid"
    for a, b in zip(on_compiled, off_compiled):
        assert a[0] == b[0] and a[2:6] == b[2:6] and a[6].grid_y == b[6].grid_y
        if a[0] == "combine":
            assert a[1] == {n: t.removesuffix(":16") for n, t in b[1].items()}
            old_pointers = {n: t for n, t in old_compiled[1][1].items() if t.startswith("*")}
            assert {n: t for n, t in a[1].items() if t.startswith("*")} == old_pointers
        else:
            assert a[1] == b[1], "Only combine loses alignment specialization"
    po = next(n for tag, n, dtype in on_alloc if tag == "bca_po")
    rows = on_compiled[0][2]["BLOCK_ROWS"]
    hp = on_compiled[0][2]["HD_PAD"]
    assert po >= on[15] * on[14] * rows * hp, "Partial allocation is too small for the real grid"


def test_spark_trace_geometry(sources):
    old, _, _ = configure(sources["legacy"], 1, 6, 12, 256, False)
    off, _, _ = configure(BC, 1, 6, 12, 256, False)
    on, _, _ = configure(BC, 1, 6, 12, 256, True)
    assert old[13:15] == (32, 8)
    assert off[13:16] == (32, 16, 6)
    assert on[13:16] == (32, 8, 6)
    bound = 5 * 256 + 6
    span = lambda cap: TRITON.cdiv(TRITON.cdiv(bound, cap), 32) * 32
    assert span(old[14]) == span(on[14]) == 192
    assert span(off[14]) == 96
