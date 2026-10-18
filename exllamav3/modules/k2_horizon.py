"""K2 Horizon's grouped norms and mixture-of-values attention projection."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from . import Module, Linear, RMSNorm, BlockSparseMLP


def require_routing_bias(config, key: str, count: int):
    """A selection-only bias is learned data, never an implicit zero vector."""
    if not config.stc.has_tensor(key):
        raise ValueError(
            f"Required K2 Horizon selection bias {key} is missing. "
            "Supply the original learned biases in a routing_bias_overlay safetensors file; "
            "the quantized checkpoint may omit them."
        )
    stc = config.stc.find_stc(key)
    header = stc.file_headers[stc.tensor_file_map[key]][key]
    if header["shape"] != [count]:
        raise ValueError(f"K2 Horizon selection bias {key} must have shape [{count}], got {header['shape']}")


def mova_routes(logits: torch.Tensor, bias: torch.Tensor | None, top_k: int, scale: float):
    """Bias determines choices, while *unbiased* sigmoid scores determine weights."""
    scores = logits.float().sigmoid()
    selected = (scores + bias.float() if bias is not None else scores).topk(top_k, dim=-1).indices
    weights = scores.gather(-1, selected)
    if top_k > 1:
        weights = weights / weights.sum(-1, keepdim=True)
    if scale is not None:
        weights = weights * scale
    return selected, weights


def combine_mova_values(hidden, selected, weights, expert_forward, out_features):
    """Group active rows by expert and apply SiLU before the weighted reduction."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    picks = selected.reshape(-1, selected.shape[-1])
    routing = weights.reshape_as(picks)
    output = flat.new_zeros((flat.shape[0], out_features))
    for expert_idx in picks.unique().tolist():
        token, slot = torch.where(picks == expert_idx)
        v = F.silu(expert_forward(expert_idx, flat.index_select(0, token)))
        output.index_add_(0, token, (v * routing[token, slot, None].to(v.dtype)).to(output.dtype))
    return output.reshape(*hidden.shape[:-1], out_features)


def k2_attention_gate(output, gate):
    """Per-channel softplus with the source model's log(2) beta."""
    return output * F.softplus(gate.reshape_as(output), beta=math.log(2))


class K2GroupedRMSNorm(Module):
    def __init__(self, config, key: str, groups: int, eps: float, out_dtype: torch.dtype | None = None):
        super().__init__(config, key, None)
        self.groups = groups
        self.norm = RMSNorm(config, key, eps, groups=groups, out_dtype=out_dtype)
        self.register_submodule(self.norm)

    def optimizer_targets(self):
        return []

    def _grouped(self, x):
        if x.shape[-1] % self.groups:
            raise ValueError(f"K2 norm {self.key}: width {x.shape[-1]} not divisible by {self.groups}")
        return x.reshape(*x.shape[:-1], self.groups, x.shape[-1] // self.groups)

    def forward_torch(self, x, params, out_dtype=None):
        return self.norm.forward_torch(self._grouped(x), params, out_dtype).reshape_as(x)

    def forward(self, x, params, out_dtype=None):
        # Do not fuse with the preceding residual: RMSNorm must see *each group* as a row.
        return self.norm.forward(self._grouped(x), params, out_dtype).reshape_as(x)

    def tp_export(self, plan, producer):
        return {"cls": K2GroupedRMSNorm, "groups": self.groups,
                "norm": self.norm.tp_export(plan, producer), "device": self.device}

    @staticmethod
    def tp_import(local_context, exported, plan):
        norm = RMSNorm.tp_import(local_context, exported["norm"], plan)
        module = K2GroupedRMSNorm(None, norm.key, exported["groups"], norm.rms_norm_eps,
                                  out_dtype=norm.out_dtype)
        module.norm = norm
        module.modules = [norm]
        module.device = local_context["device"]
        return module


class MoVAValueProjection(Module):
    def __init__(self, config, key: str, hidden_size: int, out_features: int,
                 num_experts: int, top_k: int, scaling_factor: float, qmap: str | None = None):
        super().__init__(config, key, qmap)
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.scaling_factor = scaling_factor
        # Attention probes V's quant_type when deciding whether to fuse K/V.
        # A routed projection is never an individual fusible Linear.
        self.quant_type = None
        # The bias is stored at router.bias but MUST NOT enter the linear's logits.
        # EXL3 packs 64 logical experts into 128 physical output columns.
        self.router = Linear(config, key.replace(".v_proj", ".v_router"),
                             hidden_size, num_experts, pad_to=128, trim_padded_out=True,
                             load_bias=False, out_dtype=torch.half)
        prefix = key.replace(".v_proj", ".v_experts")
        self.experts = [Linear(config, f"{prefix}.{idx}", hidden_size, out_features,
                               qmap=qmap, trim_padded_out=True, out_dtype=torch.half)
                        for idx in range(num_experts)]
        self.register_submodule(self.router)
        for expert in self.experts:
            self.register_submodule(expert)
        self.bias = None

    def optimizer_targets(self):
        return [[self.router.optimizer_targets(), [e.optimizer_targets() for e in self.experts]]]

    def load(self, device, **kwargs):
        bias_key = self.router.key + ".bias"
        require_routing_bias(self.config, bias_key, self.num_experts)
        super().load(device, **kwargs)
        self.bias = self.config.stc.get_tensor(bias_key, device, allow_bf16=True, no_defer=True)

    def unload(self):
        super().unload()
        self.bias = None

    def get_tensors(self):
        return {self.router.key + ".bias": self.bias.contiguous()} if self.bias is not None else {}

    def forward(self, x, params, out_dtype=None):
        flat = x.reshape(-1, x.shape[-1])
        logits = self.router.forward(flat, params)[..., :self.num_experts]
        selected, weights = mova_routes(logits, self.bias, self.top_k, self.scaling_factor)
        values = combine_mova_values(flat, selected, weights,
                                     lambda idx, rows: self.experts[idx].forward(rows, params),
                                     self.out_features)
        return values.reshape(*x.shape[:-1], self.out_features).to(out_dtype or x.dtype)

    def make_tp_allocation(self, options):
        raise NotImplementedError("K2 MoVA tensor-parallel allocation is not supported; use layer split")

    def storage_size(self):
        # Attention's TP planner queries this on its V projection, not make_tp_allocation.
        raise NotImplementedError("K2 MoVA tensor-parallel attention is not supported; use layer split")

    def recons_size(self):
        raise NotImplementedError("K2 MoVA tensor-parallel attention is not supported; use layer split")

    def tp_export(self, plan, producer):
        raise NotImplementedError("K2 MoVA tensor-parallel export is not supported; use layer split")

    @staticmethod
    def tp_import(local_context, exported, plan):
        raise NotImplementedError("K2 MoVA tensor-parallel import is not yet supported; use layer split")

    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        raise NotImplementedError("K2 MoVA tensor-parallel attention import is not supported; use layer split")


class K2BlockSparseMLP(BlockSparseMLP):
    def load(self, device, **kwargs):
        require_routing_bias(self.config, self.key + ".gate.bias", self.num_experts)
        super().load(device, **kwargs)
