"""CPU checks for the optional legacy convolution arithmetic switch.

Exercises production wrapper dispatch without CUDA imports, including both
launch layouts, dtype scope and strict explicit overrides. Actual arithmetic is
checked against immutable old/new Triton sources in the companion GPU tests.
"""
from __future__ import annotations

import ast
import contextlib
import os
from pathlib import Path
import types

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "exllamav3/modules/gated_delta_net_fn/conv1d.py"


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = tuple(shape), dtype
        self.device, self.is_cuda = "cuda:0", True

    def is_contiguous(self):
        return True

    def dim(self):
        return len(self.shape)

    def size(self, dim):
        return self.shape[dim]


def production_wrapper(monkeypatch, env):
    monkeypatch.setenv("EXL3_GDN_CONV_BF16_PRODUCT", env)
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    records = []
    kernels = {node.name: node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name.startswith("_causal_conv1d_")}

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def call(*args, **kwargs):
                names = [arg.arg for arg in kernels[self.name].args.args]
                records.append((self.name, grid, dict(zip(names, args)) | kwargs))
            return call

    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "causal_conv1d_update_slotted_triton")
    mode = next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_legacy_bf16_product" for t in node.targets))
    tree.body = [mode, wrapper]
    namespace = {
        "os": os, "torch": types.SimpleNamespace(
            Tensor=Tensor, dtype=str, bfloat16="bf16",
            empty=lambda shape, dtype, device: Tensor(shape, dtype),
            cuda=types.SimpleNamespace(device=lambda _: contextlib.nullcontext())),
        "triton": types.SimpleNamespace(
            next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
            cdiv=lambda x, y: (x+y-1)//y),
        **{name: Kernel(name) for name in kernels},
    }
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace["causal_conv1d_update_slotted_triton"], records


def arguments(seq, x_dtype="bf16", state_dtype="bf16", weight_dtype="bf16"):
    return [Tensor((2, 67, seq), x_dtype), Tensor((4, 67, 9), state_dtype),
            Tensor((2,), "int64"), Tensor((67, 4), weight_dtype)]


@pytest.mark.parametrize("seq", [6, 257])
@pytest.mark.parametrize("env,expected", [("0", False), ("1", True)])
def test_option_controls_both_output_launches_only(monkeypatch, seq, env, expected):
    fn, records = production_wrapper(monkeypatch, env)
    out = fn(*arguments(seq), transpose_output=True, history=True)
    assert out.shape == (2, seq, 67) and out.dtype == "bf16"
    outputs = [fields for name, _, fields in records if "state_kernel" not in name]
    assert len(outputs) == 1 and outputs[0]["legacy_bf16_product"] is expected
    state_updates = [fields for name, _, fields in records if "state_kernel" in name]
    assert len(state_updates) == int(seq > 256)
    assert all("legacy_bf16_product" not in fields for fields in state_updates)


@pytest.mark.parametrize("dtypes", [("fp16", "bf16", "bf16"),
                                    ("fp32", "bf16", "bf16"),
                                    ("bf16", "fp32", "bf16"),
                                    ("bf16", "bf16", "fp16")])
def test_environment_option_preserves_other_dtype_paths(monkeypatch, dtypes):
    fn, records = production_wrapper(monkeypatch, "1")
    fn(*arguments(257, *dtypes))
    assert records[0][2]["legacy_bf16_product"] is False
    with pytest.raises(ValueError, match="requires bf16"):
        fn(*arguments(257, *dtypes), legacy_bf16_product=True)


def test_explicit_false_and_true_override_import_setting(monkeypatch):
    for env, explicit in [("1", False), ("0", True)]:
        fn, records = production_wrapper(monkeypatch, env)
        fn(*arguments(33), legacy_bf16_product=explicit)
        assert records[0][2]["legacy_bf16_product"] is explicit


@pytest.mark.parametrize("bad", [1, 0, "true", 0.0])
def test_explicit_switch_rejects_ambiguous_types(monkeypatch, bad):
    fn, _ = production_wrapper(monkeypatch, "0")
    with pytest.raises(TypeError, match="bool or None"):
        fn(*arguments(33), legacy_bf16_product=bad)
