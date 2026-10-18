from pathlib import Path
from types import SimpleNamespace
import os

import torch

from exllamav3.architecture.architectures import get_architectures
from exllamav3.architecture.qwen3_dspark import (
    Qwen3DSparkConfig,
    Qwen3DSparkModel,
    dspark_sequential_sample,
    audit_checkpoint_tensors,
)
from exllamav3.architecture.dflash import DFlashModel
from exllamav3.generator.generator import _dflash_layout

import pytest

# Local copy of openbmb/MiniCPM-V-4.7-35B-A3B-DSpark (revision 5facb75c). Tests that read the
# checkpoint skip unless DSPARK_CHECKPOINT points at it.
_CHECKPOINT_ENV = os.environ.get("DSPARK_CHECKPOINT")
CHECKPOINT = Path(_CHECKPOINT_ENV) if _CHECKPOINT_ENV else None
needs_checkpoint = pytest.mark.skipif(
    CHECKPOINT is None or not (CHECKPOINT / "config.json").exists(),
    reason = "set DSPARK_CHECKPOINT to a local openbmb/MiniCPM-V-4.7-35B-A3B-DSpark directory",
)


@needs_checkpoint
def test_official_architecture_constructs_with_raw_taps_and_dspark_heads():
    entry = get_architectures()["Qwen3DSparkModel"]
    assert entry["config_class"] is Qwen3DSparkConfig
    assert entry["model_class"] is Qwen3DSparkModel
    config = Qwen3DSparkConfig(str(CHECKPOINT))
    assert config.target_layer_ids == [3, 11, 19, 27, 35]
    assert config.block_size == 15
    assert config.markov_rank == 256
    assert config.enable_confidence_head is True
    assert config.confidence_head_with_markov is True
    model = Qwen3DSparkModel(config)
    assert model.fwd_end_idx < len(model.modules)
    assert model.modules[model.fwd_end_idx:] == [model.markov_w1, model.markov_w2, model.confidence]
    assert model.markov_w1.caps["prefer_cpu"] is False


@needs_checkpoint
def test_anchor_is_first_of_fixed_fifteen_prediction_rows():
    config = Qwen3DSparkConfig(str(CHECKPOINT))
    assert config.block_size == 15
    assert config.sample_from_anchor is True
    assert config.default_draft_size == 15


def test_forward_runs_backbone_prefix_and_never_auxiliary_heads(monkeypatch):
    assert Qwen3DSparkModel.forward is not DFlashModel.forward

    calls = []

    class AddOne:
        def prepare_for_device(self, x, params):
            return x

        def forward(self, x, params):
            calls.append("backbone")
            return x + 1

    class AuxiliaryHead:
        def prepare_for_device(self, x, params):
            raise AssertionError("auxiliary head entered normal forward path")

        def forward(self, x, params):
            raise AssertionError("auxiliary head entered normal forward path")

    model = object.__new__(Qwen3DSparkModel)
    model.modules = [AddOne(), AddOne(), AuxiliaryHead()]
    model.fwd_end_idx = 2
    model.prepare_inputs = lambda input_ids, params: input_ids.float()
    monkeypatch.setattr(
        "exllamav3.architecture.qwen3_dspark.advance_recurrent_states",
        lambda input_ids, params, active_model: calls.append("advance"),
    )

    result = model.forward(torch.tensor([[1]]), {})
    assert result.tolist() == [[3.0]]
    assert calls == ["backbone", "backbone", "advance"]


def test_sequential_sampler_adds_previous_token_markov_bias_and_accepts_optional_confidence():
    base = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    embeddings = torch.tensor([[0.0], [2.0], [4.0]])
    projection = torch.tensor([[0.0], [0.0], [1.0]])
    seen = []

    def embed(token_ids):
        return embeddings[token_ids]

    def bias(markov):
        return markov @ projection.T

    def confidence(position_hidden, markov_embed):
        seen.append((position_hidden.clone(), markov_embed.clone()))
        return torch.sigmoid(position_hidden[..., 0] + markov_embed[..., 0])

    sampled, confidences = dspark_sequential_sample(
        base, anchor_ids=torch.tensor([0]), markov_embed_fn=embed,
        markov_bias_fn=bias, confidence_fn=confidence,
        position_hidden=torch.tensor([[[0.0], [0.0]]]),
    )
    assert sampled.shape == (1, 2)
    assert sampled.tolist() == [[1, 2]]
    assert len(seen) == 2
    assert torch.allclose(confidences, torch.tensor([[0.5, torch.sigmoid(torch.tensor(2.0))]]))


def test_sequential_sampler_runs_without_confidence_head():
    base = torch.zeros((1, 2, 3))
    sampled, confidence = dspark_sequential_sample(
        base, anchor_ids=torch.tensor([2]),
        markov_embed_fn=lambda token_ids: torch.zeros((token_ids.shape[0], 1)),
        markov_bias_fn=lambda markov: torch.zeros((markov.shape[0], 3)),
        confidence_fn=None, position_hidden=None,
    )
    assert sampled.shape == (1, 2)
    assert confidence is None


def test_generator_layout_keeps_anchor_predictions_and_aligns_confidence():
    tensor = torch.tensor([[10, 11, 12]])
    assert _dflash_layout(tensor, sample_from_anchor=True).tolist() == [[10, 11, 12]]
    assert _dflash_layout(tensor, sample_from_anchor=False).tolist() == [[11, 12]]


def test_confidence_head_returns_sigmoid_probability():
    class FakeConfidence:
        def forward(self, hidden, params):
            assert hidden.shape == (1, 3)
            return torch.tensor([[2.0]])

    model = object.__new__(Qwen3DSparkModel)
    model.confidence = FakeConfidence()
    model.config = SimpleNamespace(confidence_head_with_markov=True)
    confidence = model._confidence_function({})
    result = confidence(torch.zeros((1, 2)), torch.zeros((1, 1)))
    assert torch.allclose(result, torch.sigmoid(torch.tensor([2.0])))


@needs_checkpoint
def test_checkpoint_consumption_audit_accounts_for_every_draft_tensor_and_borrowed_target_parts():
    from safetensors import safe_open

    path = CHECKPOINT / "model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
    report = audit_checkpoint_tensors(keys)
    assert report["unconsumed"] == set()
    assert report["consumed"] == keys
    assert report["borrowed_target"] == {"embed_tokens", "lm_head"}
    assert "markov_head.markov_w1.weight" in report["consumed"]
    assert "markov_head.markov_w2.weight" in report["consumed"]
    assert "confidence_head.proj.weight" in report["consumed"]
    assert "confidence_head.proj.bias" in report["consumed"]
