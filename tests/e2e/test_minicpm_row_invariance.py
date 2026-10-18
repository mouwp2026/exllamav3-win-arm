"""Real-weight GPU checks for MiniCPM-V-4.7 verifier row invariance.

Set MINICPM_TARGET_DIR to an EXL3 target directory to enable. These checks use exact
Tensor equality, not tolerances; run on the same build used for speculative verify.
"""
from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("MINICPM_TARGET_DIR"),
    reason="set MINICPM_TARGET_DIR to a MiniCPM-V-4.7 EXL3 target",
)


def _load_target():
    from exllamav3 import Config, Model

    target = os.environ["MINICPM_TARGET_DIR"]
    config = Config.from_directory(target)
    model = Model.from_config(config)
    model.load(device=torch.device("cuda:0"), progressbar=False)
    return model


def _blocks(model):
    return [m for m in model.modules if type(m).__name__ == "TransformerBlock"]


def test_minicpm_router_selection_and_weights_are_row_invariant():
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    with torch.inference_mode():
        model = _load_target()
        blocks = _blocks(model)
        for layer in (0, 39):
            mlp = blocks[layer].mlp
            width = mlp.hidden_size
            torch.manual_seed(1947 + layer)
            rows = torch.randn((8, width), device=torch.device("cuda:0"), dtype=torch.half)
            for count in (2, 3, 4, 5, 8):
                selected, weights = mlp.routing_fn(
                    count, mlp.routing_cfg, rows[:count].clone(), {}
                )
                serial = []
                for i in range(count):
                    selected_i, weights_i = mlp.routing_fn(
                        1, mlp.routing_cfg, rows[i:i + 1].clone(), {}
                    )
                    serial.append((selected_i.clone(), weights_i.clone()))
                expected_selected = torch.cat([s.reshape(1, -1) for s, _ in serial], dim=0)
                expected_weights = torch.cat([w.reshape(1, -1) for _, w in serial], dim=0)
                assert torch.equal(selected.reshape(count, -1), expected_selected), (
                    f"layer {layer}, M={count}: selected expert ids differ"
                )
                assert torch.equal(weights.reshape(count, -1), expected_weights), (
                    f"layer {layer}, M={count}: routing weights differ"
                )
