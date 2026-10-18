from __future__ import annotations

import os
import torch
from typing_extensions import override

from ..cache.recurrent_util import advance_recurrent_states
from ..modules import Embedding, Linear
from .dflash import DFlashConfig, DFlashModel

"""
DSpark draft (Qwen3DSparkModel; openbmb/MiniCPM-V-4.7-35B-A3B-DSpark).

Config validation, the conversion passthrough of the Markov/confidence heads, the attach checks and the
draft-pack tooling are by r0b0tlab (github.com/r0b0tlab/exllamav3, branch minicpm-v4.7, commit 4d083d25),
merged with permission. The runtime drafting path (sequential Markov sampling, confidence head, exact verify)
is this fork's.

Attribution: the tensor layout, the head semantics and the depth convention below were read from vLLM
(vllm/model_executor/models/qwen3_dspark.py and vllm/v1/worker/gpu/spec_decode/dspark/speculator.py, Apache-2.0,
Copyright contributors to the vLLM project). No vLLM code is copied; this file only needs the names and shapes.

What the checkpoint holds (bf16, 62 tensors, 389,223,169 parameters, no embedding and no LM head: both are the target's):

    fc.weight                                  [2048, 10240]    EXL3   concatenated taps -> hidden
    hidden_norm.weight                         [2048]           verbatim (module walk misses it: extras fallback)
    layers.{0-4}.input_layernorm.weight        [2048]           module walk (norm)
    layers.{0-4}.post_attention_layernorm.w    [2048]           module walk (norm)
    layers.{0-4}.self_attn.q_proj / o_proj     [2048, 2048]     EXL3
    layers.{0-4}.self_attn.k_proj / v_proj     [512, 2048]      EXL3
    layers.{0-4}.self_attn.q_norm / k_norm     [256]            module walk (norm)
    layers.{0-4}.mlp.gate_proj / up_proj       [6144, 2048]     EXL3
    layers.{0-4}.mlp.down_proj                 [2048, 6144]     EXL3
    norm.weight                                [2048]           module walk (norm)
    markov_head.markov_w1.weight               [248144, 256]    verbatim (extras)
    markov_head.markov_w2.weight               [248144, 256]    verbatim (extras)
    confidence_head.proj.weight / .bias        [1, 2304] / [1]  verbatim (extras)

Every quantized linear has both dims divisible by 128, so nothing is padded (fc is built with pad_to = 1 anyway).
Converting this checkpoint as a plain DFlashDraftModel would silently DROP markov_head.* and confidence_head.*
(the compile step only carries what a module or get_additional_compiled_tensors names); this shim names them, and
the config reader below refuses a directory (source or pack) that lacks them.

How the open design questions were settled (from config.json and the vLLM source):

- sliding_window: null. The generic reader treats null as missing and DFlashConfig falls back to its 2048 default, but
  every layer_types entry is "full_attention" so Attention receives sliding_window = -1: the default is inert. The
  shim pins config.sliding_window = None and refuses anything but all-full-attention (vLLM refuses too).
- rope_parameters {rope_theta 1e7, rope_type "default", rope_dim 256}: the fork does not read rope_dim. Rotary width
  is head_dim * partial_rotary_factor = 256 = the whole head (vLLM's get_rope uses rope_dim as the rotary width, also
  256), NEOX style, plain 1-D positions. The shim asserts rope_dim is absent or equal to head_dim so a checkpoint that
  rotates a fraction of the head fails loudly instead of being run wrong. The target's M-RoPE does not apply to the
  draft.
- Taps: target_layer_ids [3, 11, 19, 27, 35] are the residual stream AFTER target layer N (vLLM adds 1 to its capture
  index): tap_shift = 0, the exllamav3 export index is used as is.
- Depth: block_size 15 is the draft's query-row count AND the number of drafted tokens (vLLM `sample_from_anchor`
  defaults to True: "anchor + N-1 noise, NOT 1 + N; every query position is a prediction", the anchor row predicts
  the first draft). config.json does not set sample_from_anchor, so the default applies.

Runtime (this fork): the anchor row is draft 1 (15 drafts per round), the Markov bias is applied left to right
in dspark_sequential_sample, and the confidence head truncates the window (EXL3_DSPARK_CONF, default 0.5).

Still TODO(calibrated): a calibrated quantization arm needs Hessians from real target taps; until then draft packs
are uncalibrated-only (caps["uncalibrated_quantize"], H = identity).
"""


class Qwen3DSparkConfig(DFlashConfig):

    arch_string = "Qwen3DSparkModel"

    # Taps are the output of target layer N, exactly the exllamav3 export index (vLLM captures at N + 1)
    tap_shift = 0

    # Tensor namespaces carried through compilation unchanged
    passthrough_prefixes = ("markov_head", "confidence_head")

    def __init__(
        self,
        directory: str,
        model_classes: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes or {"text": Qwen3DSparkModel},
            **kwargs
        )

        # Whole-context, non-causal full attention: no sliding window anywhere. sliding_window is null in the
        # checkpoint, the base reader substitutes 2048 for it, which stays inert only because of the check below
        assert all(t == "full_attention" for t in self.layer_types), \
            "Qwen3DSparkModel: every draft layer must be full_attention"
        assert not self.read_cfg(bool, "use_sliding_window", False), \
            "Qwen3DSparkModel: use_sliding_window is not supported"
        assert self.read_cfg(int, "sliding_window", None) is None, \
            "Qwen3DSparkModel: sliding_window is not supported"
        self.sliding_window = None

        # RoPE: the whole head is rotated with plain 1-D positions at rope_theta (default type)
        rope_dim = self.read_cfg(int, "rope_parameters->rope_dim", None)
        assert rope_dim is None or rope_dim == self.head_dim, \
            f"Qwen3DSparkModel: rope_dim {rope_dim} is not the head width {self.head_dim}, " \
            f"partial rotation is not implemented for this draft"
        assert self.read_cfg(str, "rope_parameters->rope_type", "default") == "default", \
            "Qwen3DSparkModel: only rope_type default is supported"
        assert self.read_cfg(float, "partial_rotary_factor", 1.0) == 1.0, \
            "Qwen3DSparkModel: partial_rotary_factor must be 1.0"

        # DSpark switches. Top-level keys are what vLLM reads, dflash_config repeats them: they must agree
        def both(expected_type, key, default):
            top = self.read_cfg(expected_type, key, None)
            nested = self.read_cfg(expected_type, f"dflash_config->{key}", None)
            assert top is None or nested is None or top == nested, \
                f"Qwen3DSparkModel: {key} is {top} at the top level and {nested} in dflash_config"
            v = top if top is not None else nested
            return default if v is None else v
        self.markov_rank = both(int, "markov_rank", None)
        assert self.markov_rank, "Qwen3DSparkModel: markov_rank is required"
        self.enable_confidence_head = both(bool, "enable_confidence_head", False)
        self.confidence_head_with_markov = both(bool, "confidence_head_with_markov", False)
        self.confidence_in = self.hidden_size + (self.markov_rank if self.confidence_head_with_markov else 0)
        self.assert_cfg(str, "dflash_config->markov_head_type", "vanilla", True)
        self.assert_cfg(str, "dflash_config->projector_type", "dspark", True)
        self.assert_cfg(str, "dflash_config->attention_mode", "gqa", True)
        self.assert_cfg(bool, "sample_from_anchor", True, True)
        self.draft_vocab_size = self.read_cfg(int, "draft_vocab_size", self.vocab_size)
        assert self.vocab_size is not None and self.draft_vocab_size == self.vocab_size, \
            "Qwen3DSparkModel: a reduced draft vocabulary (draft_vocab_size != vocab_size) is not supported"
        self.sample_from_anchor = both(bool, "sample_from_anchor", True)
        self.default_draft_size = self.block_size if self.sample_from_anchor else self.block_size - 1
        if self.confidence_head_with_markov and not self.enable_confidence_head:
            raise ValueError("confidence_head_with_markov requires enable_confidence_head")
        if self.sample_from_anchor and self.block_size != 15:
            # The runtime adapter supports the official fixed-block checkpoint only
            raise ValueError("Qwen3 dSparK anchor-as-first adapter requires block_size=15")

        # The heads ride along as raw tensors: a directory without them would convert to an unusable draft
        self.check_head_tensors()


    def head_tensor_shapes(self) -> dict[str, list[int]]:
        shapes = {
            "markov_head.markov_w1.weight": [self.vocab_size, self.markov_rank],
            "markov_head.markov_w2.weight": [self.draft_vocab_size, self.markov_rank],
        }
        if self.enable_confidence_head:
            shapes["confidence_head.proj.weight"] = [1, self.confidence_in]
            shapes["confidence_head.proj.bias"] = [1]
        return shapes


    def check_head_tensors(self):
        found = {}
        for prefix in self.passthrough_prefixes:
            found.update(self.stc.list_tensors(prefix = prefix))
        want = self.head_tensor_shapes()
        for name, shape in want.items():
            assert name in found, f"Qwen3DSparkModel: {name} is missing from {self.directory}"
            assert list(found[name]["shape"]) == shape, \
                f"Qwen3DSparkModel: {name} has shape {found[name]['shape']}, expected {shape}"
        extra = sorted(set(found) - set(want))
        assert not extra, f"Qwen3DSparkModel: unexpected head tensors {extra}"


class Qwen3DSparkModel(DFlashModel):
    """DFlash Qwen3 draft stack plus its sequential Markov/confidence heads."""

    config_class = Qwen3DSparkConfig

    def __init__(self, config: Qwen3DSparkConfig, **kwargs):
        super().__init__(config, **kwargs)
        # The DFlash backbone ends at its final norm. Markov/confidence heads are
        # sampled explicitly and must never enter the normal module forward chain.
        self.fwd_end_idx = len(self.modules)
        self.markov_w1 = Embedding(
            config=config, key="markov_head.markov_w1", vocab_size=config.vocab_size,
            hidden_size=config.markov_rank, allow_table=False,
        )
        # This table is consulted once per drafted position. Keep it on-device to
        # avoid a host synchronization in the sequential Markov loop.
        self.markov_w1.caps["prefer_cpu"] = False
        self.markov_w2 = Linear(
            config=config, key="markov_head.markov_w2", in_features=config.markov_rank,
            out_features=config.draft_vocab_size, qmap=None, trim_padded_out=True,
        )
        self.modules += [self.markov_w1, self.markov_w2]
        self.confidence = None
        if config.enable_confidence_head:
            input_size = config.hidden_size + (config.markov_rank if config.confidence_head_with_markov else 0)
            self.confidence = Linear(
                config=config, key="confidence_head.proj", in_features=input_size,
                out_features=1, qmap=None, pad_to=1,
            )
            self.modules.append(self.confidence)
        self.draft_conf_threshold = float(os.environ.get("EXL3_DSPARK_CONF", "0.5"))
        # Markov walk over the top-K candidates of each row's base logits (the reference engines walk the top K
        # too); 0 walks the whole vocabulary. Proposals only steer acceptance, never the verified output
        self.markov_top_k = int(os.environ.get("EXL3_DSPARK_TOPK", "64"))
        self._w2t = None
        self.caps.update({
            # H = identity quantization; a calibrated arm is a TODO (see the module docstring)
            "uncalibrated_quantize": True,
            "dflash_sample_from_anchor": config.sample_from_anchor,
            "default_draft_size": config.default_draft_size,
            # One-pass verification is exact on row-invariant targets for verify windows of up to
            # max_verify_rows rows (anchor + drafts): the decode and verify paths share kernel arithmetic
            "max_verify_rows": int(os.environ.get("EXL3_DSPARK_MAX_VERIFY_ROWS", "8")),
        })
        self.caps["exact_serial_verify"] = os.environ.get("EXL3_DSPARK_SERIAL_VERIFY", "0") == "1"

    def enable_exact_verification(self):
        """Verify every proposal through its own one-token target forward (slow reference path,
        also selected with EXL3_DSPARK_SERIAL_VERIFY=1). The default one-pass verification is
        bit-identical to it on row-invariant targets."""
        self.caps["exact_serial_verify"] = True

    @classmethod
    def checkpoint_tensor_consumption(cls, names):
        return audit_checkpoint_tensors(names)

    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if self.config.sample_from_anchor:
            if input_ids.shape[-1] != 1:
                raise ValueError("Qwen3 dSparK requires exactly one anchor token per request")
            params["dspark_anchor_ids"] = input_ids[:, -1]
        return super().prepare_inputs(input_ids, params)

    @override
    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict | None = None) -> torch.Tensor:
        """Run only the DFlash backbone; Markov/confidence heads are sampled separately."""
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        for module in self.modules[:self.fwd_end_idx]:
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        advance_recurrent_states(input_ids, params, self)
        return x

    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        target = self.attached_model()
        lm_head = target.modules[target.logit_layer_idx]
        # Only the rows the generator can verify this round need the head and the Markov walk (the window is capped
        # by num_draft_tokens and max_verify_rows); the anchor-as-first layout keeps row r = draft r
        rows_used = params.get("dspark_rows")
        if rows_used is not None and self.config.sample_from_anchor and 0 < rows_used < state.shape[1]:
            state = state[:, :rows_used].contiguous()
        logits = lm_head.forward(lm_head.prepare_for_device(state, params), params).float()
        if self.config.sample_from_anchor:
            anchor_ids = params["dspark_anchor_ids"]

            def markov_embed(token_ids):
                return self.markov_w1.forward(token_ids, params, out_dtype=state.dtype).to(
                    state.device, dtype=state.dtype
                )

            def markov_bias(markov):
                return self.markov_w2.forward(
                    markov.unsqueeze(1), params, out_dtype=logits.dtype
                ).squeeze(1)

            if self.markov_top_k > 0:
                sampled, confidence = self._sample_top_k(logits, anchor_ids, state, params)
            else:
                sampled, confidence = dspark_sequential_sample(
                    logits,
                    anchor_ids=anchor_ids,
                    markov_embed_fn=markov_embed,
                    markov_bias_fn=markov_bias,
                    confidence_fn=self._confidence_function(params),
                    position_hidden=state,
                )
            if confidence is not None:
                params["draft_conf"] = confidence
                threshold = float(params.get("dspark_conf_threshold", self.draft_conf_threshold))
                keep = confidence >= threshold
                lengths = torch.cumprod(keep.to(torch.int32), dim=-1).sum(dim=-1)
                params["draft_confidence_len"] = int(lengths.max().item())
            return sampled
        raise NotImplementedError("Qwen3 dSparK fill-in mode is not implemented")

    def _sample_top_k(self, logits: torch.Tensor, anchor_ids: torch.Tensor, state: torch.Tensor, params: dict):
        """Left-to-right Markov walk restricted to each row's top-K base candidates: per row, a K x rank gather of
        markov_w2 instead of a full-vocabulary projection, then the confidence head on all rows in one call."""
        vocab = self.config.vocab_size
        base = logits[..., :vocab]
        bsz, rows, _ = base.shape
        k = min(self.markov_top_k, vocab)
        top_v, top_i = torch.topk(base, k, dim=-1)                                  # (b, r, k)
        if self._w2t is None:
            w2 = self.markov_w2.inner.weight                                      # (rank, vocab_padded)
            self._w2t = w2[:, :vocab].t().contiguous()                            # (vocab, rank)
        w1 = self.markov_w1.embedding.weight                                     # (vocab, rank)
        cand_w2 = self._w2t[top_i]                                                # (b, r, k, rank)
        prev = anchor_ids.reshape(bsz).to(base.device)
        out = torch.empty((bsz, rows), dtype=torch.long, device=base.device)
        marks = []
        for r in range(rows):
            m = w1[prev].to(cand_w2.dtype)                                        # (b, rank)
            marks.append(m)
            bias = torch.bmm(cand_w2[:, r], m.unsqueeze(-1)).squeeze(-1)          # (b, k)
            j = torch.argmax(top_v[:, r] + bias.float(), dim=-1, keepdim=True)
            tok = top_i[:, r].gather(1, j).squeeze(1)
            out[:, r] = tok
            prev = tok
        confidence = None
        if self.confidence is not None:
            hidden = state
            if self.config.confidence_head_with_markov:
                hidden = torch.cat((state, torch.stack(marks, dim=1).to(state.dtype)), dim=-1)
            confidence = torch.sigmoid(self.confidence.forward(hidden, params).squeeze(-1).float())
        return out, confidence

    def _confidence_function(self, params):
        if self.confidence is None:
            return None

        def project(hidden, markov_embed):
            if self.config.confidence_head_with_markov:
                hidden = torch.cat((hidden, markov_embed), dim=-1)
            logits = self.confidence.forward(hidden, params).squeeze(-1)
            return torch.sigmoid(logits.float())

        return project

    @override
    def attach_to(self, target):
        if target.loaded_tp:
            raise NotImplementedError("Qwen3DSparkModel does not support tensor-parallel targets")
        if target.config.vocab_size != self.config.vocab_size:
            raise ValueError(
                f"DSpark vocabulary size {self.config.vocab_size} does not match "
                f"target vocabulary size {target.config.vocab_size}"
            )
        if target.config.hidden_size != self.config.hidden_size:
            raise ValueError(
                f"DSpark hidden size {self.config.hidden_size} does not match "
                f"target hidden size {target.config.hidden_size}"
            )
        if not 0 <= self.config.mask_token_id < target.config.vocab_size:
            raise ValueError("DSpark mask_token_id is outside the target vocabulary")
        if any(not 0 <= layer_id < target.config.num_hidden_layers
               for layer_id in self.config.target_layer_ids):
            raise ValueError("DSpark target_layer_ids contains a layer outside the target model")
        super().attach_to(target)

    @classmethod
    @override
    def get_additional_compiled_tensors(cls, config: Qwen3DSparkConfig) -> dict:
        # hidden_norm (stored in DFlashInputLayer under a key outside the fc prefix) plus the Markov tables and the
        # confidence head: raw tensors outside any Linear, copied bit for bit from the source
        tensors = dict(super().get_additional_compiled_tensors(config))
        for prefix in config.passthrough_prefixes:
            tensors.update(config.stc.list_tensors(prefix = prefix))
        return tensors


def dspark_sequential_sample(
    base_logits: torch.Tensor,
    *,
    anchor_ids: torch.Tensor,
    markov_embed_fn,
    markov_bias_fn,
    confidence_fn=None,
    position_hidden: torch.Tensor | None = None,
):
    """Sample query rows left-to-right using the previously selected token.

    The anchor is the predecessor of row zero, not an emitted output column.
    Embedding and projection are callables so production modules own placement and
    dtype while the autoregressive dependency stays independently testable.
    """
    batch, rows, _ = base_logits.shape
    previous = anchor_ids.reshape(batch).to(base_logits.device)
    output = torch.empty((batch, rows), dtype=torch.long, device=base_logits.device)
    confidence = []
    for row in range(rows):
        markov = markov_embed_fn(previous)
        bias = markov_bias_fn(markov)
        token = torch.argmax(base_logits[:, row, :] + bias.to(base_logits.dtype), dim=-1)
        output[:, row] = token
        if confidence_fn is not None:
            if position_hidden is None:
                raise ValueError("position_hidden is required when confidence_fn is supplied")
            confidence.append(confidence_fn(position_hidden[:, row, :], markov))
        previous = token
    return output, (torch.stack(confidence, dim=1) if confidence else None)


def audit_checkpoint_tensors(names):
    """Classify every official checkpoint tensor; target embedding/head are shared.

    The official checkpoint ships neither target embedding nor lm_head. ExLlama
    borrows them from the attached target model; all present checkpoint tensors
    must belong to one of the draft module keyspaces below.
    """
    names = set(names)
    draft_prefixes = (
        "fc.", "hidden_norm.", "layers.", "norm.",
        "markov_head.markov_w1.", "markov_head.markov_w2.", "confidence_head.proj.",
    )
    consumed = {name for name in names if name.startswith(draft_prefixes)}
    return {
        "consumed": consumed,
        "unconsumed": names - consumed,
        "borrowed_target": {"embed_tokens", "lm_head"},
    }
