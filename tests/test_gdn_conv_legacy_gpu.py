"""Compare an optional bf16 product path to actual frozen convolution sources.

Run only when the GPU owner schedules it:
  SPARK_CONV_LEGACY_SOURCE=/old94/.../gated_delta_net_fn/conv1d.py \
  SPARK_CONV_BASELINE_SOURCE=/frozen-b578/.../gated_delta_net_fn/conv1d.py \
  python -m pytest tests/test_gdn_conv_legacy_gpu.py -q -s

The two reference files must be byte-for-byte original sources (hash checked).
Loading extracts their original Triton functions and wrapper without importing
either model package. No hand-reimplemented convolution is used as the golden
reference. The candidate defaults to this checkout or SPARK_CONV_CANDIDATE_SOURCE.
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
    "legacy": "a08e99093b0d76e33cf256931ad027b019404caf7b92ffd07b2d84b8187c0452",
    "baseline": "a16982f4d5e00fbd18473c6ada94a5ca5d42d5d482a2686b2f6105f4e627ec66",
}
NAMES = {
    "_causal_conv1d_update_slotted_kernel",
    "_causal_conv1d_update_slotted_output_kernel",
    "_causal_conv1d_update_slotted_state_kernel",
    "causal_conv1d_update_slotted_triton",
}


def load_source(path: Path, label: str):
    source = path.read_bytes()
    if label in GOLDEN_SHA256:
        assert hashlib.sha256(source).hexdigest() == GOLDEN_SHA256[label], (
            f"{label} source differs from the declared immutable reference")
    tree = ast.parse(source, filename=str(path))
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in NAMES]
    assert {node.name for node in functions} == NAMES
    # Preserve each original location/source file so Triton's inspect.getsource
    # sees the exact original JIT body, rather than a rewritten reference.
    tree.body = functions
    name = f"spark_conv_compare_{label}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    import triton
    import triton.language as tl
    module.__dict__.update(torch=torch, triton=triton, tl=tl, _legacy_bf16_product=False)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module.causal_conv1d_update_slotted_triton


@pytest.fixture(scope="module")
def functions():
    paths = {}
    for label in ("legacy", "baseline"):
        value = os.environ.get(f"SPARK_CONV_{label.upper()}_SOURCE")
        if not value:
            pytest.skip(f"Set SPARK_CONV_{label.upper()}_SOURCE to the frozen source")
        paths[label] = Path(value).resolve(strict=True)
    candidate = os.environ.get("SPARK_CONV_CANDIDATE_SOURCE")
    paths["candidate"] = (Path(candidate) if candidate else Path(__file__).resolve().parents[1] /
                          "exllamav3/modules/gated_delta_net_fn/conv1d.py").resolve(strict=True)
    return {name: load_source(path, name) for name, path in paths.items()}


@pytest.mark.parametrize("seq,history,bias,token_major", [
    (1, False, False, False),
    (6, True, True, False),
    (33, False, False, False),
    (255, True, True, False),
    (256, False, True, False),
    (257, True, False, False),
    (1023, False, True, False),
    (1071, True, True, False),
    (6, True, True, True),
    (1023, False, True, True),
])
def test_matches_actual_legacy_and_preserves_default(functions, seq, history, bias, token_major):
    # 67 channels crosses both full and partial 32-channel CTAs; nontrivial slot
    # permutation and unused slots detect accidental state/address changes.
    generator = torch.Generator().manual_seed(71425 + seq)
    batch, dim, slots_count, kernel = 2, 67, 4, 4
    state_size = kernel + (7 if history else 0)
    x = torch.randn(batch, dim, seq, generator=generator).bfloat16().cuda()
    state = torch.randn(slots_count, dim, state_size, generator=generator).bfloat16().cuda()
    weight = (torch.randn(dim, kernel, generator=generator) * .5).bfloat16().cuda()
    bias_tensor = (torch.randn(dim, generator=generator) * .1).bfloat16().cuda() if bias else None
    slots = torch.tensor([2, 0], dtype=torch.long, device="cuda")
    candidate_x = x.transpose(1, 2).contiguous() if token_major else x

    old_state = state.clone()
    expected = functions["legacy"](
        x, old_state, slots, weight, bias_tensor, transpose_output=True, history=history)
    new_state = state.clone()
    actual = functions["candidate"](
        candidate_x, new_state, slots, weight, bias_tensor, transpose_output=True,
        history=history, token_major=token_major, out_dtype=torch.bfloat16,
        legacy_bf16_product=True)
    assert torch.equal(actual, expected), (
        f"Legacy output differs: max abs {(actual.float()-expected.float()).abs().max().item()}")
    assert torch.equal(new_state, old_state), "Legacy mode changed history/cache update"

    baseline_state = state.clone()
    baseline = functions["baseline"](
        candidate_x, baseline_state, slots, weight, bias_tensor, transpose_output=True,
        history=history, token_major=token_major, out_dtype=torch.bfloat16)
    default_state = state.clone()
    default = functions["candidate"](
        candidate_x, default_state, slots, weight, bias_tensor, transpose_output=True,
        history=history, token_major=token_major, out_dtype=torch.bfloat16)
    assert torch.equal(default, baseline), "Disabled option changed frozen b578 output"
    assert torch.equal(default_state, baseline_state), "Disabled option changed frozen b578 state"
    if seq == 257:
        assert not torch.equal(expected, baseline), "Fixture must expose the actual old/new arithmetic difference"
        print(f"seq257 bf16 product differences: {(expected != baseline).sum().item()}/{expected.numel()}")
