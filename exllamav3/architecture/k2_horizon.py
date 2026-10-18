"""K2 Horizon dense/MoE decoder with Mixture-of-Values attention."""
from __future__ import annotations

import torch
from typing_extensions import override
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn
from ..modules.k2_horizon import K2GroupedRMSNorm, MoVAValueProjection, K2BlockSparseMLP


class K2HorizonConfig(Config):
    arch_string = "K2HorizonForCausalLM"

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": K2HorizonModel}, **kwargs)
        overlay = kwargs.get("routing_bias_overlay")
        if overlay:
            if not overlay.endswith(".safetensors"):
                raise ValueError("routing_bias_overlay must be a .safetensors file")
            import os
            if not os.path.isfile(overlay):
                raise ValueError(f"routing_bias_overlay does not exist: {overlay}")
            self.stc.add_tensor_files(overlay)
        use_sliding_window = self.read_cfg(bool, "use_sliding_window", False)
        sliding_window = self.read_cfg(int, "sliding_window", None)
        # A finite window is ambiguous even if the flag is false/absent. Do not
        # silently run global attention until K2's mask semantics are supported.
        if use_sliding_window or sliding_window is not None:
            raise ValueError("K2 Horizon sliding-window attention is not supported")
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.head_dim = self.read_cfg(int, "head_dim", self.hidden_size // self.read_cfg(int, "num_attention_heads", no_default))
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", self.intermediate_size)
        self.num_experts = self.read_cfg(int, "num_experts", 0)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", 0)
        self.mova_num_experts = self.read_cfg(int, "mova_num_experts", 0)
        self.mova_num_experts_per_tok = self.read_cfg(int, "mova_num_experts_per_tok", 0)
        self.num_shared_experts = self.read_cfg(int, "num_shared_experts", 0)
        self.mlp_only_layers = self.read_cfg(list, "mlp_only_layers", [])
        self.decoder_sparse_step = self.read_cfg(int, "decoder_sparse_step", 1)
        self.layernorm_num_groups = self.read_cfg(int, "layernorm_num_groups", 1)
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", 1e-6)
        self.router_scaling_factor = self.read_cfg(float, "router_scaling_factor", 1.0)
        self.norm_topk_prob = self.read_cfg(bool, "norm_topk_prob", True)
        self.query_key_norm = self.read_cfg(bool, "query_key_norm", True)
        self.moe_gate_bias = self.read_cfg(bool, "moe_gate_bias", False)
        self.attention_gate_func = self.read_cfg(str, "attention_gate_func", None)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.rope_head_dim = self.read_cfg(int, "rope_head_dim", self.head_dim)
        if self.rope_head_dim != self.head_dim:
            raise ValueError("K2 Horizon rope_head_dim != head_dim (partial RoPE) is not supported")
        self.assert_cfg(str, "hidden_act", "silu", optional=True)
        self.assert_cfg(str, "router_score_func", "sigmoid", optional=True)
        if self.attention_gate_func not in (None, "softplus"):
            raise ValueError("K2 Horizon supports only per-channel softplus attention gate")
        if self.num_experts and not (self.norm_topk_prob and self.moe_gate_bias):
            raise ValueError("K2 Horizon MoE requires norm_topk_prob and moe_gate_bias")
        if self.mova_num_experts and not self.moe_gate_bias:
            raise ValueError("K2 Horizon MoVA requires moe_gate_bias")
        if self.layernorm_num_groups <= 0 or self.hidden_size % self.layernorm_num_groups:
            raise ValueError("layernorm_num_groups must divide hidden_size")
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)


class K2HorizonModel(Model):
    config_class = K2HorizonConfig

    def __init__(self, config: K2HorizonConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Attention TP assumes a splittable linear V projection. MoVA routes whole
        # experts and has no TP import/collective implementation yet.
        self.caps["supports_tp"] = False
        self.modules.append(Embedding(config, "model.embed_tokens", config.vocab_size, config.hidden_size))
        self.first_block_idx = len(self.modules)
        for idx in range(config.num_hidden_layers):
            base = f"model.layers.{idx}"
            attn_key = base + ".self_attn"
            sparse = (config.num_experts > 0 and idx not in config.mlp_only_layers and
                      (idx + 1) % config.decoder_sparse_step == 0)
            mova = sparse and config.mova_num_experts > 0
            if mova:
                v_proj = MoVAValueProjection(config, attn_key + ".v_proj", config.hidden_size,
                                              config.num_kv_heads * config.head_dim,
                                              config.mova_num_experts, config.mova_num_experts_per_tok,
                                              config.router_scaling_factor, qmap="block.attn.input")
                k_proj = Linear(config, attn_key + ".k_proj", config.hidden_size,
                                config.num_kv_heads * config.head_dim, qmap="block.attn.input",
                                trim_padded_out=True)
            else:
                v_proj = None
                k_proj = None
            if sparse:
                route = Linear(config, base + ".mlp.gate", config.hidden_size,
                               config.num_experts, pad_to=1, load_bias=False, out_dtype=torch.half)
                mlp = K2BlockSparseMLP(
                    config=config, key=base + ".mlp", hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size, num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    key_up="experts.{expert_idx}.up_proj",
                    key_gate="experts.{expert_idx}.gate_proj",
                    key_down="experts.{expert_idx}.down_proj", routing_gate=route,
                    key_e_score_bias="gate.bias", router_type="dots",
                    routed_scaling_factor=config.router_scaling_factor,
                    n_group=1, topk_group=1, qmap="block.mlp",
                    interm_dtype=torch.half, out_dtype=torch.float,
                    shared_experts=GatedMLP(
                        config, base + ".mlp.shared_experts", config.hidden_size,
                        config.moe_intermediate_size * config.num_shared_experts,
                        key_up="up_proj", key_gate="gate_proj", key_down="down_proj",
                        qmap="block.mlp", interm_dtype=torch.half, out_dtype=torch.float,
                    ) if config.num_shared_experts else None,
                )
            else:
                mlp = GatedMLP(
                    config, base + ".mlp", config.hidden_size, config.intermediate_size,
                    key_up="up_proj", key_gate="gate_proj", key_down="down_proj",
                    qmap="block.mlp", interm_dtype=torch.half, out_dtype=torch.float,
                )
            def norm(key, groups):
                return K2GroupedRMSNorm(config, key, groups, config.rms_norm_eps)
            attn = Attention(
                config=config, key=attn_key, layer_idx=idx,
                hidden_size=config.hidden_size, head_dim=config.head_dim,
                num_q_heads=config.num_q_heads, num_kv_heads=config.num_kv_heads,
                rope_settings=config.rope_settings,
                key_q="q_proj", key_k=None if mova else "k_proj",
                k_proj=k_proj, key_v=None if mova else "v_proj",
                v_proj=v_proj, key_o="o_proj",
                key_g="gate_proj" if config.attention_gate_func else None,
                full_gate=bool(config.attention_gate_func),
                gate_softplus=config.attention_gate_func == "softplus",
                q_norm=RMSNorm(config, attn_key + ".q_norm", config.rms_norm_eps,
                               groups=config.num_q_heads) if config.query_key_norm else None,
                k_norm=RMSNorm(config, attn_key + ".k_norm", config.rms_norm_eps,
                               groups=config.num_kv_heads) if config.query_key_norm else None,
                qmap="block.attn", out_dtype=torch.float,
            )
            self.modules.append(TransformerBlock(
                config, base, layer_idx=idx,
                attn_norm=norm(base + ".input_layernorm", config.layernorm_num_groups),
                attn=attn,
                mlp_norm=norm(base + ".post_attention_layernorm", config.layernorm_num_groups),
                mlp=mlp,
            ))
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules.extend([
            K2GroupedRMSNorm(config, "model.norm", config.layernorm_num_groups,
                             config.rms_norm_eps, out_dtype=torch.half),
            Linear(config, "lm_head", config.hidden_size, config.vocab_size,
                   alt_key="model.embed_tokens" if config.tie_word_embeddings and
                   not config.stc.has_tensor("lm_head") else None,
                   qmap="block", caps={"logits_output": True}),
        ])
        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)
