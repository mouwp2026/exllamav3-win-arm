"""Guard the Python/CUDA half-bit ABI for the fork's mixed-K MoE path.

The CUDA extension is not imported: loading it would JIT-build and run GPU code. The
serializer and array construction are extracted from their real source and exercised on CPU.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "exllamav3/modules/block_sparse_mlp.py").read_text()


def _serializer():
    tree = ast.parse(SOURCE)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_mixedk_k2")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])), "<mixedk>", "exec"), namespace)
    return namespace["_mixedk_k2"]


def test_mixedk_serializes_integer_and_fractional_bitrates_in_half_bit_units():
    encode = _serializer()
    for bitrate, expected in ((1, 2), (2, 4), (4, 8), (8, 16), (1.5, 3), (2.5, 5), (3.5, 7)):
        assert encode(bitrate) == expected
    for invalid in (0, 4.5, 8.5, 2.25):
        with pytest.raises(ValueError):
            encode(invalid)


@pytest.mark.parametrize("gated", [True, False])
def test_mixedk_device_arrays_keep_half_bit_rates(gated):
    tree = ast.parse(SOURCE)
    gate_branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                       and len(n.body) == 1 and isinstance(n.body[0], ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "_kg" for t in n.body[0].targets))
    assignments = [next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
                   for name in ("_ku", "_kd")]
    code = compile(ast.fix_missing_locations(ast.Module(body=[gate_branch, *assignments], type_ignores=[])),
                   "<mixedk-arrays>", "exec")

    class FakeTorch:
        int32 = "int32"

        @staticmethod
        def tensor(values, *, dtype, device):
            assert (dtype, device) == ("int32", "cuda:0")
            return tuple(int(value) for value in values)  # mirrors torch.int32 truncation

    def linear(rate):
        return SimpleNamespace(inner=SimpleNamespace(K=rate))

    state = SimpleNamespace(gated=gated, device="cuda:0",
                            gates=[linear(1.5), linear(4)],
                            ups=[linear(2.5), linear(6)],
                            downs=[linear(3.5), linear(8)])
    ns = {"self": state, "_ne": 2, "_torch": FakeTorch, "_mixedk_k2": _serializer()}
    exec(code, ns)
    assert ns["_kg"] == ((3, 8) if gated else (5, 12))
    assert ns["_ku"] == (5, 12)
    assert ns["_kd"] == (7, 16)


def test_mixedk_all_three_device_arrays_use_serializer():
    # A correct helper is useless if either gate branch or another projection still truncates K.
    tree = ast.parse(SOURCE)
    for name in ("_kg", "_ku", "_kd"):
        assigned = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert len(assigned) == (2 if name == "_kg" else 1)
        for assignment in assigned:
            assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == "_mixedk_k2" for n in ast.walk(assignment.value)), name
    kernel = (ROOT / "exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh").read_text()
    for projection in ("gate", "up", "down"):
        assert f"const int K_{projection} = K_{projection}_arr[expert_idx];" in kernel
    for k2, bits in ((2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6), (14, 7), (16, 8)):
        assert f"case {k2}:" in kernel and f"exl3_gemm_kernel_inner<{bits}, false, false, cb" in kernel
    for k2, bits in ((3, 1), (5, 2), (7, 3)):
        assert f"case {k2}:  if constexpr (cb == 2) exl3_gemm_kernel_inner<{bits}, true, false, cb" in kernel
