"""CPU regression for exact historical int8 GR table preparation.

The deployed 94ba01d _prepare path copied fp16 projection rows to float32
scratch, multiplied by w_h.float(), and stored fn_h as fp16. w_h itself is
the fp16-rounded (hc_norm.weight.float() + 1). Keeping the intermediate
rounding matters before the final per-row int8 quantization.

Extract only the production quantizer AST so this test runs without loading
the CUDA extension or importing the model package.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def production_quantizer():
    path = Path(__file__).resolve().parents[1] / "exllamav3/modules/hyperconnections.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GatedResidual")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_quantize_int8")
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_quantize_int8"]


@pytest.mark.parametrize("combine", [False, True])
def test_int8_projection_matches_94ba01d_fp16_norm_fold(combine):
    gen = torch.Generator().manual_seed(941605 + combine)
    H, D, rank = 4, 128, 64
    M = rank + (H if combine else 0)
    Mpad = (M + 63) // 64 * 64
    norm_raw = torch.randn((H, D), generator=gen) * 0.1
    norm_float = norm_raw.float() + 1.0
    w_h = norm_float.flatten().half()
    projection = torch.randn((Mpad, H * D), generator=gen).half()
    upx = torch.randn((H, D // 4, rank, 4), generator=gen).half()
    site = SimpleNamespace(
        proj_h=projection, proj_m=M, norm_w=norm_float, w_h=w_h, upx_h=upx
    )

    # Keep the legacy source's distinct fp32 scratch -> multiply -> fp16
    # storage stages, rather than expressing the expected result using the
    # candidate's folded-expression implementation.
    tmp = torch.empty((M, H * D), dtype=torch.float32)
    tmp.copy_(projection[:M])
    tmp *= w_h.float()
    legacy_fn_h = tmp.half().contiguous()
    legacy_f = legacy_fn_h.float()
    legacy_scale = legacy_f.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    legacy_quant = torch.round(legacy_f / legacy_scale[:, None]).clamp_(-128, 127).to(torch.int8)

    # Verify this fixture would detect the incorrect unrounded-norm merge.
    unrounded_f = (projection[:M].float() * norm_float.flatten()).half().float()
    wrong_scale = unrounded_f.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    wrong_quant = torch.round(unrounded_f / wrong_scale[:, None]).clamp_(-128, 127).to(torch.int8)
    assert not torch.equal(wrong_quant, legacy_quant), "Fixture must cross int8 rounding thresholds"

    original_projection, original_upx = projection.clone(), upx.clone()
    production_quantizer()(site, H, D)
    assert torch.equal(site.fn_q, legacy_quant)
    assert torch.equal(site.fn_s, legacy_scale)
    # Quantization must leave upstream's unfolded fp16 prefill tables intact.
    assert torch.equal(site.proj_h, original_projection)
    assert torch.equal(site.upx_h, original_upx)
