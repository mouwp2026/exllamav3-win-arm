from __future__ import annotations
from typing_extensions import override
import os
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.device_copy import host_to_device
from ..util.tensor import to2
from . import Module, Linear
from .multilinear import MultiLinear
from collections import defaultdict as _defaultdict
from ..ext import exllamav3_ext as ext
from ..util.backend import MOE_FUSED_ROWS, MOE_BATCH_RECON, MOE_MTILE
from dataclasses import dataclass
from .mlp import MLP, GatedMLP, capture_out_sensitivity
from .rmsnorm import RMSNorm
from .layernorm import LayerNorm
from .block_sparse_mlp_cpu import BlockSparseMLP_CPU
from .moe_batch_recon import PAD_MAX, worst_case_bytes
from ..model.model_tp_alloc import TPAllocation
from ..util import profile_opt
from ..util.tensor import g_tensor_cache, buffered_interleaved_arange
from .block_sparse_mlp_routing import (
    RoutingCFG, ROUTING_ACT_SIGMOID, ROUTING_ACT_SQRTSP,
    routing_std, routing_std_bias, routing_ds3, routing_dots, routing_sqrtsp, routing_sqrtsp_hash,
)

# Row capacity of the fused MoE kernel's per-group temp buffers (experts with more assigned
# rows take the reconstruct paths); EXL3_MOE_FUSED_ROWS overrides for tuning sweeps (per-backend
# defaults in util/backend.py)
TEMP_ROWS_FUSED = MOE_FUSED_ROWS
TEMP_ROWS_GRAPH = 32
# Batched reconstruct tier for the experts above the fused kernel's row capacity at prefill
# (moe_batch_recon.py); EXL3_MOE_BATCH_RECON=0 restores the per-expert reconstruct loop
BATCH_RECON = MOE_BATCH_RECON
# Row tiles for the fused kernel: experts with more than MTILE_T1 rows run through a 32-row tile
# instance, more than MTILE_T2 through a 64-row one (32 for the N = 256 shape), each its own
# launch over its expert range (mul1 codebook only). EXL3_MOE_MTILE=0 keeps the single 16-row
# launch
MTILE = MOE_MTILE
MTILE_T1, MTILE_T2 = 16, 32
# Fused-kernel row capacity per expert when the wide tiles apply: with them the fused kernel
# beats the batched reconstruct tier up to 256 rows (Qwen3.8 4k chunk: 128 -> 256 rows +3%)
FUSED_ROWS_WIDE = int(os.environ.get("EXL3_MOE_FUSED_ROWS_WIDE", 256))
# Deterministic (slot + gather) accumulation for the fused kernel's outputs; EXL3_MOE_FUSED_DET=0
# restores the atomic adds
FUSED_DET = os.environ.get("EXL3_MOE_FUSED_DET", "1") != "0"
# Minimum routed row count (tokens * experts-per-token) for the unified mixed-K kernel, layered on
# top of EXL3_MIXEDK_LEGACY. That switch picks one path for the whole run; this lets decode fall to
# the legacy per-K-group coop dispatch while prefill keeps the unified kernel. The unified kernel
# reads K per expert from a device array (exl3_moe_kernel.cuh) whereas the K-specialized exl3_moe
# instances take it as a template parameter ("K = 0 switches Kg/Ku/Kd at runtime, K > 0 =
# compile-time"), so it cannot unroll the trellis decode loop. Row tiles amortize that at prefill
# but not at one token. 0 leaves the EXL3_MIXEDK_LEGACY behaviour unchanged
MIXEDK_MIN_ROWS = int(os.environ.get("EXL3_MOE_MIXEDK_MIN_ROWS", 0))
# Sync-free unified mixed-K decode (default off). When every expert's row count is bounded by the
# m16 tier (num_tokens <= MTILE_T1) and top_k is at least the temp-buffer concurrency, the unified
# mixed-K dispatch needs no host readback: assignments are counted on the device with a scatter-add
# (torch.bincount reads input.max() back to the host), the kernel is launched with the buffer
# concurrency as its active-expert count instead of the exact in-range count (_ec[_m].tolist()),
# and the handled-set readback, which no consumer reads when the counts stay on the device, is
# skipped. The kernel sizes its grid as min(concurrency, MOE_MAX_GROUPS, num_active) groups, so
# the launch is unchanged whenever the true active count is >= the concurrency, which top_k >=
# concurrency guarantees (every token picks top_k distinct experts). Output is bit-identical.
# The uniform fused path's bincount (all-fused decode, run_fused(-1)) gets the same scatter-add.
MIXEDK_NOSYNC = os.environ.get("EXL3_MOE_MIXEDK_NOSYNC", "0") == "1"
_ROWINV_MOE = os.environ.get("EXL3_ROWINV_MOE", "0") == "1"
MAX_BSZN = 8  # must match MAX_BSZN in exllamav3_ext/libtorch/blocksparse_mlp.h
# Cooperative decode kernels for unified mixed-K layers (default off). At decode and verify shapes
# (bsz <= MAX_BSZN) a mixed-K layer runs through the two-stage cooperative kernels the uniform-K
# layers use (exl3_moe_coop), with each expert's gate / up / down bitrate read from a device table
# instead of fixed per launch (ext.CoopMK): no sort, count, slot tables or gather, and no host
# synchronization. Larger batches and layers the kernels do not cover keep the unified mixed-K
# path (and EXL3_MOE_MIXEDK_NOSYNC there). The routed sum is accumulated in a different order
# than the unified kernel's, so outputs are not bit-identical to it. EXL3_COOPMK_PLAN selects the
# kernel variant set (1 all-K, 2 split by decode kind, 3 all-K capped at 64 registers, 4 split
# capped at 64); 3 was fastest per call on the tested pack
#
# Unset, it defaults to on for sm_121 (GB10 / DGX Spark) only. Measured there on GLM-5.3 EXL3 3.38bpw
# (mixed-K mul1 experts), single-token decode: routed MoE 111 -> 59 ms per token summed over 78
# layers, host syncs ~110 -> 3 per step, teacher-forced decode NLL unchanged (1.245 vs 1.257 unified,
# prefill reference 1.248). Plans 1/2 were ~7% slower than 3; 4 tied 3. EXL3_MOE_COOP_MIXEDK=0/1
# overrides the default on any GPU
_COOP_MIXEDK_ENV = os.environ.get("EXL3_MOE_COOP_MIXEDK")
COOP_MIXEDK = _COOP_MIXEDK_ENV == "1"
COOPMK_PLAN = int(os.environ.get("EXL3_COOPMK_PLAN", "3"))


def _coop_mixedk_enabled(device) -> bool:
    """EXL3_MOE_COOP_MIXEDK if set, else on for sm_121 devices"""
    if _COOP_MIXEDK_ENV is not None:
        return _COOP_MIXEDK_ENV == "1"
    try:
        return torch.device(device).type == "cuda" and torch.cuda.get_device_capability(device) == (12, 1)
    except Exception:
        return False
_coopmk_warned = False


def _coopmk_module():
    """The extension module with the CoopMK kernels, or None for a build without them. Only
    consulted when EXL3_MOE_COOP_MIXEDK=1"""
    global _coopmk_warned
    if hasattr(ext, "CoopMK"):
        return ext
    if not _coopmk_warned:
        _coopmk_warned = True
        print(" !! EXL3_MOE_COOP_MIXEDK=1 but this exllamav3_ext build has no CoopMK kernels; "
              "mixed-K layers keep the unified path")
    return None

@dataclass
class FusedBuffers:
    temp_state_g: torch.Tensor
    temp_state_u: torch.Tensor
    temp_intermediate_g: torch.Tensor
    temp_intermediate_u: torch.Tensor



@dataclass
class ExpertsCFG:
    yh: torch.Tensor
    interm_g: torch.Tensor
    interm_u: torch.Tensor
    interm_a: torch.Tensor
    out_d: torch.Tensor
    out_d2: torch.Tensor
    min_expert: int
    max_expert: int
    out_bszn: torch.Tensor | None = None   # (MAX_BSZN, H) fp32 routed sum of the bsz<=MAX_BSZN path


def _mixedk_k2(K):
    """Serialize EXL3 bitrate in half-bit units for the mixed-K CUDA switch."""
    if K not in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8):
        raise ValueError(f"Unsupported EXL3 bitrate {K} (integer 1..8, or 1.5 / 2.5 / 3.5)")
    return int(round(2 * K))


# Debug: replicate the router on every TP rank and compare local vs broadcast selections
_routing_check = os.environ.get("EXL3_TP_ROUTING_CHECK", "0") != "0"
# Shared expert as a one-expert fused decode launch (see BC_BlockSparseMLP::sh_coop); 0 keeps the
# three-launch BC_GatedMLP graph
_moe_shared_coop = os.environ.get("EXL3_MOE_SHARED_COOP", "1") != "0"

# Router types whose selection runs entirely on the deterministic ext paths (routing_gemm.cu +
# fixed-order top-k with FMA-only activations), so under tensor parallelism every rank can route
# for itself on the (rank-identical) residual stream instead of receiving the selection by
# broadcast from the output rank: two collectives per MoE layer per token fewer. Every router in
# use is on those paths; the grouped DS3 router (n_group > 1, not supported by any loadable
# architecture) is still torch-composed and would keep the broadcast
_replicated_router_types = ("std", "std_bias", "dots", "sqrtsp", "sqrtsp_hash")
_routing_check_stats = {}

def _routing_check_compare(key, local_sel, local_w, sel, w):
    import atexit
    # Stats split by row-count class: single-row calls (fixed-order routing GEMV), other
    # decode-sized calls (<= 32 rows: fused mix path upstream, hgemm routing logits) and
    # prefill-sized calls (tiled mix path, hgemm routing logits)
    R = sel.shape[0]
    key = (key, "bsz1" if R == 1 else "decode" if R <= 32 else "prefill")
    st = _routing_check_stats.get(key)
    if st is None:
        st = _routing_check_stats[key] = {"rows": 0, "sel_mismatch": 0, "w_maxdiff": 0.0, "calls": 0}
        if len(_routing_check_stats) == 1:
            def report():
                for cls in ("bsz1", "decode", "prefill"):
                    vs = [v for k, v in _routing_check_stats.items() if k[1] == cls]
                    if not vs: continue
                    rows = sum(v["rows"] for v in vs); mism = sum(v["sel_mismatch"] for v in vs)
                    wmax = max(v["w_maxdiff"] for v in vs)
                    print(f" -- routing-check (pid {os.getpid()}) [{cls}]: {rows} rows over {len(vs)} layers, "
                          f"selection mismatches {mism} ({mism / max(rows, 1) * 100:.4f}%), max weight diff {wmax:.3e}", flush = True)
            atexit.register(report)
    ls = torch.sort(local_sel, dim = 1).values; bs = torch.sort(sel, dim = 1).values
    same = (ls == bs).all(dim = 1)
    st["rows"] += sel.shape[0]; st["calls"] += 1
    st["sel_mismatch"] += int((~same).sum().item())
    if same.any():
        # weights compared in the sorted-expert order on agreeing rows
        lw = torch.gather(local_w, 1, torch.sort(local_sel, dim = 1).indices)[same]
        bw = torch.gather(w, 1, torch.sort(sel, dim = 1).indices)[same]
        st["w_maxdiff"] = max(st["w_maxdiff"], float((lw.float() - bw.float()).abs().max().item()))


class MoETPAllocation(TPAllocation):
    """MoE prefill scratch scales with the routed rows a rank receives, not linearly with its share
    of experts (see BlockSparseMLP.tp_prefill_bound)"""

    def __init__(self, module: BlockSparseMLP, tensor_split: bool, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.tensor_split = tensor_split

    def overhead(self, tokens: int, s: int, channels: int) -> int:
        return super().overhead(tokens, s, channels) + \
            self.module.tp_prefill_bound(tokens, s, channels, self.tensor_split)


class BlockSparseMLP(BlockSparseMLP_CPU, Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        num_local_experts: int | None = None,
        latent_size: int | None = None,
        key_latent_in: str | None = None,
        key_latent_out: str | None = None,
        latent_in: Linear | None = None,
        latent_out: Linear | None = None,
        latent_hq_bits: int = 0,
        key_up: str | None = None,
        key_gate: str | None = None,
        key_down: str | None = None,
        key_gate_split: str | None = None,
        key_up_split: str | None = None,
        key_gate_up_split: str | None = None,
        key_down_split: str | None = None,
        key_routing_gate: str | None = None,
        key_shared_gate: str | None = None,
        key_e_score_bias: str | None = "gate.e_score_correction_bias",
        key_e_score_bias_vl: str | None = None,
        key_tid2eid: str | None = None,
        key_per_expert_scale: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype = None,
        activation_fn: str = "silu",
        act_limit: float = 0.0,
        interm_dtype: torch.dtype = None,
        interm_div: float = 1.0,
        router_type: str = "std",
        routing_gate: Linear | None = None,
        shared_gate: Linear | None = None,
        routed_scaling_factor: float | None = None,
        n_group: int | None = None,
        topk_group: int | None = None,
        shared_experts: MLP | GatedMLP | None = None,
        shared_experts_post_norm: RMSNorm | LayerNorm | None = None,
        router_pre_norm: RMSNorm | LayerNorm | None = None,
        routed_pre_norm: RMSNorm | LayerNorm | None = None,
        routed_post_norm: RMSNorm | LayerNorm | None = None,
        gates: list[Linear | Module] = None,
        ups: list[Linear | Module] = None,
        downs: list[Linear | Module] = None,
        routing_first: int | None = None,
        routing_last: int | None = None,
        routing_device: int | None = None,
        transposed_load: bool = True,
        transpose_fused_weights: bool = True,
        ftranspose_after_load: bool = True,
        frange_dim: int = 0,
        gate_up_interleaved: bool = False,
        alt_residual_channel: bool = False,
        qbits_key: str = "bits"
    ):
        super().__init__(config, key, None)

        self.interm_dtype = interm_dtype
        self.interm_div = interm_div
        self.router_type = router_type
        if interm_div != 1.0:
            assert router_type in ("dots", "ds3", "std"), \
                "interm_div requires a router type that can fold the compensation into the routing weights"
            if router_type != "std":
                routed_scaling_factor = (routed_scaling_factor if routed_scaling_factor is not None else 1.0) * interm_div
        self.activation_fn = activation_fn
        self.intermediate_size = intermediate_size
        self.intermediate_size_padded = (intermediate_size + 127) // 128 * 128
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.f_threshold = min(self.num_experts // self.num_experts_per_tok, 4)
        self.num_local_experts = num_local_experts if num_local_experts is not None else num_experts
        self.hidden_size = hidden_size
        # Latent MoE (Nemotron-3 Super): the routed experts run at latent_size, between a
        # projection of the block input down to that width and a projection of the routed sum
        # back up. The router and the shared experts see the full-width block input
        self.latent_size = latent_size
        self.expert_size = latent_size or hidden_size
        self.router_type = router_type
        self.act_limit = act_limit
        self.alt_residual_channel = alt_residual_channel

        self.routing_first = routing_first
        self.routing_last = routing_last
        self.routing_device = routing_device

        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group

        assert out_dtype in (torch.float, None), \
            f"BlockSparseMLP output dtype must be float"

        assert shared_experts is None or shared_experts.out_dtype in (torch.float, None), \
            f"Shared experts output dtype must be float"

        assert num_experts_per_tok <= TEMP_ROWS_GRAPH, \
            f"Too many experts per token, max supported is {TEMP_ROWS_GRAPH}"

        if routing_gate is None and key_routing_gate is None:
            self.routing_gate = None
        elif routing_gate is None:
            self.routing_gate = Linear(
                config = config,
                key = f"{key}.{key_routing_gate}",
                in_features = hidden_size,
                out_features = num_experts,
                qmap = None,
                out_dtype = torch.half,
                pad_to = 1,
            )
            self.register_submodule(self.routing_gate)
        else:
            self.routing_gate = routing_gate
            self.register_submodule(self.routing_gate)

        if shared_gate is None and key_shared_gate is None:
            self.shared_gate = None
        elif shared_gate is None:
            self.shared_gate = Linear(
                config = config,
                key = f"{key}.{key_shared_gate}",
                in_features = hidden_size,
                out_features = 1,
                qmap = None,
                out_dtype = torch.float,
                pad_to = 1,
            )
            self.register_submodule(self.shared_gate)
        else:
            self.shared_gate = shared_gate
            self.register_submodule(self.shared_gate)

        if latent_in is None and key_latent_in is None:
            assert latent_size is None, "latent_size requires the latent projection keys"
            self.latent_in = None
            self.latent_out = None
        else:
            assert latent_size, "latent projections require latent_size"
            # The down projection shares the block input (and its Hessian) with the shared
            # experts; the up projection's input is the routed sum
            self.latent_in = latent_in if latent_in is not None else Linear(
                config = config,
                key = f"{key}.{key_latent_in}",
                in_features = hidden_size,
                out_features = latent_size,
                qmap = qmap + ".input" if qmap else None,
                out_dtype = torch.half,
                select_hq_bits = latent_hq_bits,
                qgroup = key + ".latent",
                qbits_key = qbits_key,
            )
            self.latent_out = latent_out if latent_out is not None else Linear(
                config = config,
                key = f"{key}.{key_latent_out}",
                in_features = latent_size,
                out_features = hidden_size,
                qmap = qmap + ".latent_out" if qmap else None,
                out_dtype = torch.float,
                select_hq_bits = latent_hq_bits,
                qgroup = key + ".latent",
                qbits_key = qbits_key,
            )
            self.register_submodule(self.latent_in)
            self.register_submodule(self.latent_out)

        # Non-gated experts (NemotronH: up/down with relu2). Gateless relu2 rides every quantized
        # fast path (relu(u) * u = relu2(u): the BC graphs and mgemm loops skip the gate GEMM, the
        # fused kernel synthesizes the gate lane, the batched reconstruct tier and the CPU worker
        # take up/down only); other gateless activations run the dense per-expert path
        self.gated = (
            key_gate is not None or key_gate_split is not None or key_gate_up_split is not None or
            (gates is not None and len(gates) > 0)
        )

        if gates is not None:
            assert ups is not None and (not self.gated or len(ups) == len(gates))
            assert downs is not None and len(downs) == len(ups)
            self.num_slices = len(ups)
            self.gates = gates
            self.ups = ups
            self.downs = downs

        else:
            self.gates = []
            self.ups = []
            self.downs = []

            # The experts' input is the block input, or the latent projection of it (a
            # different activation, so a different Hessian) in a latent MoE
            expert_qmap = (qmap + (".latent" if latent_size else ".input")) if qmap else None
            for idx in range(self.num_local_experts):

                fkey_gate, fkey_up, fkey_down = (
                    f"{key}.{key_gate_up_split}" if key_gate_up_split else
                    f"{key}.{key_gate_split}" if key_gate_split else
                    None,
                    f"{key}.{key_gate_up_split}" if key_gate_up_split else
                    f"{key}.{key_up_split}" if key_up_split else
                    None,
                    f"{key}.{key_down_split}" if key_down_split else
                    None
                )

                gate = None if not self.gated else Linear(
                    config = config,
                    key = f"{key}.{key_gate}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_gate,
                    fidx = idx,
                    frange = (0, intermediate_size) if key_gate_up_split else None,
                    finterleaved = gate_up_interleaved,
                    in_features = self.expert_size,
                    out_features = intermediate_size,
                    qmap = expert_qmap,
                    out_dtype = self.interm_dtype,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    ftranspose_after_load = ftranspose_after_load,
                    frange_dim = frange_dim,
                    qgroup = key + ".block_gud",
                    qbits_key = qbits_key,
                )
                up = Linear(
                    config = config,
                    key = f"{key}.{key_up}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_up,
                    fidx = idx,
                    frange = (intermediate_size, intermediate_size * 2) if key_gate_up_split else None,
                    finterleaved = gate_up_interleaved,
                    in_features = self.expert_size,
                    out_features = intermediate_size,
                    qmap = expert_qmap,
                    out_dtype = self.interm_dtype,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    ftranspose_after_load = ftranspose_after_load,
                    frange_dim = frange_dim,
                    qgroup = key + ".block_gud",
                    qbits_key = qbits_key,
                    weight_scale = 1.0 / interm_div,
                )
                down = Linear(
                    config = config,
                    key = f"{key}.{key_down}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_down,
                    fidx = idx,
                    in_features = intermediate_size,
                    out_features = self.expert_size,
                    qmap = qmap + f".{idx}.down" if qmap else None,
                    out_dtype = torch.float,
                    allow_input_padding = True,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    ftranspose_after_load = ftranspose_after_load,
                    # The input dim pads to match the gate/up padded output width; padded output
                    # columns (zeros, or quantization noise over zero weights) are trimmed
                    trim_padded_out = True,
                    qgroup = key + ".block_gud",
                    qbits_key = qbits_key,
                )

                self.ups.append(up)
                if gate is not None:
                    self.gates.append(gate)
                self.downs.append(down)

                self.register_submodule(up)
                self.register_submodule(gate)
                self.register_submodule(down)

        if self.gated:
            self.gateless_act = None
            match activation_fn:
                case "silu":
                    self.activation_fn_call = ext.silu_mul
                    self.activation_fn_idx = 0
                case "gelu":
                    self.activation_fn_call = ext.gelu_mul
                    self.activation_fn_idx = 1
                case "swiglu_oai":
                    self.activation_fn_call = ext.silu_oai_mul
                    self.activation_fn_idx = 3
                case "relu2":
                    self.activation_fn_call = ext.relu2_mul
                    self.activation_fn_idx = 2
                case _:
                    raise ValueError(f"Unknown activation function {activation_fn}")
        else:
            # Gateless relu2 rides the gated fast paths via relu(u) * u = relu2(u): the act call
            # sites pass (u, u, a) and the fused MoE kernel's MOE_ACT_RELU2_NOGATE synthesizes
            # the gate lane from u
            self.activation_fn_call = ext.relu_mul if activation_fn == "relu2" else None
            self.activation_fn_idx = 2 if activation_fn == "relu2" else -1
            match activation_fn:
                case "silu": self.gateless_act = F.silu
                case "gelu": self.gateless_act = lambda x: F.gelu(x, approximate = "tanh")
                case "relu2": self.gateless_act = lambda x: torch.square(F.relu(x))
                case _:
                    raise ValueError(f"Unknown gateless activation function {activation_fn}")

        self.is_quantized = False
        self.uniform_expert_q = True
        self.support_fused = False
        self.support_quant_paths = False
        self.multi_gate = None
        self.multi_up = None
        self.multi_down = None

        self.routing_cfg = None
        self.experts_cfg = None

        # Persistent broadcast targets for TP ranks without the router (see forward)
        self.bcast_sel_bsz1 = None
        self.bcast_weights_bsz1 = None

        self.e_score_correction_bias = None
        self.e_score_correction_bias_key = key_e_score_bias
        self.e_score_bias_vl = None
        self.e_score_bias_vl_key = key_e_score_bias_vl
        self.tid2eid = None
        self.tid2eid_key = key_tid2eid
        self.per_expert_scale = None
        self.per_expert_scale_key = key_per_expert_scale

        self.shared_experts = shared_experts
        if shared_experts is not None:
            self.register_submodule(shared_experts)

        match router_type:
            case "std": self.routing_fn = routing_std
            case "std_bias": self.routing_fn = routing_std_bias
            case "ds3": self.routing_fn = routing_ds3
            case "dots": self.routing_fn = routing_dots
            case "sqrtsp": self.routing_fn = routing_sqrtsp
            case "sqrtsp_hash": self.routing_fn = routing_sqrtsp_hash
            case _: raise ValueError(f"Unknown router type {router_type}")

        self.tp_reduce = False

        self.shared_experts_post_norm = shared_experts_post_norm
        self.router_pre_norm = router_pre_norm
        self.routed_pre_norm = routed_pre_norm
        self.routed_post_norm = routed_post_norm
        self.register_submodule(self.shared_experts_post_norm)
        self.register_submodule(self.router_pre_norm)
        self.register_submodule(self.routed_pre_norm)
        self.register_submodule(self.routed_post_norm)

        self.bc = None
        self.bc_sh_exp = False
        self.coopmk = None
        self.coopmk_out = None
        self.fused_mode_buffers = None
        self.mtile_ok = False
        self.fused_rows = TEMP_ROWS_FUSED
        self.batch_recon = None
        self._cpu_init_state()

    @override
    def optimizer_targets(self):
        g, u, d = [], [], []
        for m in self.gates: g += m.optimizer_targets()
        for m in self.ups: u += m.optimizer_targets()
        for m in self.downs: d += m.optimizer_targets()
        if self.latent_in is not None:
            u = self.latent_in.optimizer_targets() + u
            d = d + self.latent_out.optimizer_targets()
        if self.shared_experts:
            s = self.shared_experts.optimizer_targets()
            return [s, [g + u, d]]
        else:
            return [[g + u, d]]


    def load_local(self, **kwargs):

        # Test if experts can be fused
        num_exl3_tensors = 0
        num_nonexl3_tensors = 0
        for l in self.gates + self.ups + self.downs:
            if l.quant_type == "exl3":
                num_exl3_tensors += 1
            else:
                num_nonexl3_tensors += 1
        if num_exl3_tensors and num_nonexl3_tensors:
            print(f" !! Warning, partially quantized block-sparse MLP layer: {self.key}")
        self.is_quantized = (num_exl3_tensors > 0 and num_nonexl3_tensors == 0)

        # Mixed-K: skip MultiLinear/fused when experts differ in (K, mcg, mul1).
        def _uniform_q(ls):
            return not ls or len({(l.inner.K, l.inner.mcg, l.inner.mul1) for l in ls}) <= 1
        self.uniform_expert_q = self.is_quantized and all(
            _uniform_q(ls) for ls in ((self.gates if self.gated else []), self.ups, self.downs))
        if self.is_quantized and not self.uniform_expert_q:
            print(f" -- Mixed-K experts in {self.key}: dense per-expert path")

        # The quantized fast paths (mgemm/BC/fused kernels) don't yet support per-expert biases,
        # activations other than silu/gelu (or gateless relu2), or trimmed (padded) down
        # projections; configurations with any of those run every batch size through the dense
        # per-expert path, which handles all of them (gpt-oss)
        self.support_quant_paths = (
            self.is_quantized and self.uniform_expert_q and
            (self.activation_fn in ("silu", "gelu") if self.gated else self.activation_fn == "relu2") and
            all(l.inner.bias is None for l in self.gates + self.ups + self.downs) and
            all(not l.trim_padded_out or l.out_features == l.out_features_unpadded for l in self.downs)
        )

        # The fused bsz<=MAX_BSZN kernels (BC_BlockSparseMLP.run_bszN) additionally support
        # the gpt-oss activation, per-expert biases (all-or-nothing per projection) and padded
        # dims (input zero-padded and output trimmed in the kernel); the dense per-expert path
        # covers those configurations for every other batch shape
        def _uniform_bias(ls):
            has = [l.inner.bias is not None for l in ls]
            return all(has) or not any(has)
        self.support_bc_bszn = (
            self.is_quantized and self.uniform_expert_q and
            (self.activation_fn in ("silu", "gelu", "swiglu_oai") if self.gated else self.activation_fn == "relu2") and
            _uniform_bias(self.gates) and _uniform_bias(self.ups) and _uniform_bias(self.downs) and
            not self.config.infer_params.no_reconstruct
        )

        self.mixedk_k_groups = []
        # Make fused modules (only used by the quantized fast paths). Gateless experts have no
        # gate MultiLinear; the up module doubles as a placeholder wherever the fast paths want
        # gate pointer tables (never dereferenced, the gate GEMMs are skipped)
        if (self.support_quant_paths or self.support_bc_bszn) and not self.config.infer_params.no_reconstruct:
            self.multi_gate = MultiLinear(self.device, self.gates, allow_bias = True) if self.gated else None
            self.multi_up = MultiLinear(self.device, self.ups, allow_bias = True)
            self.multi_down = MultiLinear(self.device, self.downs, allow_bias = True)

        # Mixed-K: build per-K-group MultiLinears for fused dispatch
        if self.is_quantized and not self.uniform_expert_q and not self.config.infer_params.no_reconstruct:
            buckets = _defaultdict(list)
            for i in range(len(self.ups)):
                gk = self.gates[i].inner.K if self.gated else self.ups[i].inner.K
                uk = self.ups[i].inner.K
                dk = self.downs[i].inner.K
                mcg_v = self.ups[i].inner.mcg
                mul1_v = self.ups[i].inner.mul1
                buckets[(gk, uk, dk, mcg_v, mul1_v)].append(i)
            for (gk, uk, dk, mcg_v, mul1_v), experts in sorted(buckets.items(), key=lambda x: -len(x[1])):
                if len(experts) < 4:
                    continue
                try:
                    mg = MultiLinear(self.device, [self.gates[e] for e in experts], allow_bias=True) if self.gated else None
                    mu = MultiLinear(self.device, [self.ups[e] for e in experts], allow_bias=True)
                    md = MultiLinear(self.device, [self.downs[e] for e in experts], allow_bias=True)
                except Exception:
                    continue
                g2l = {e: j for j, e in enumerate(experts)}
                self.mixedk_k_groups.append((experts, mg, mu, md, g2l))
            if self.mixedk_k_groups:
                n_fused = sum(len(g[0]) for g in self.mixedk_k_groups)
                print(f" -- Mixed-K fused groups in {self.key}: {len(self.mixedk_k_groups)} groups, {n_fused}/{len(self.ups)} experts fused")

            # === Mixed-K unified kernel: one launch for ALL experts regardless of K ===
            # Check: all experts share the same codebook type (mcg/mul1), differ only in K
            _all_mcg = set()
            _all_mul1 = set()
            for ls in ([self.gates] if self.gated else []) + [self.ups, self.downs]:
                for l in ls:
                    _all_mcg.add(l.inner.mcg)
                    _all_mul1.add(l.inner.mul1)
            self.mixedk_unified = False
            if (len(_all_mcg) == 1 and len(_all_mul1) == 1 and
                list(_all_mcg)[0] != list(_all_mul1)[0] and
                (self.activation_fn in ("silu", "gelu") if self.gated else self.activation_fn == "relu2") and
                all(l.inner.bias is None for l in self.gates + self.ups + self.downs) and
                all(not l.trim_padded_out or l.out_features == l.out_features_unpadded for l in self.downs) and
                hasattr(ext, "exl3_moe_mixedk")):
                import torch as _torch
                _ne = len(self.ups)
                # CUDA mixed-K switches on half-bit units (2*K), not integer bitrates.
                if self.gated:
                    _kg = _torch.tensor([_mixedk_k2(self.gates[i].inner.K) for i in range(_ne)], dtype=_torch.int32, device=self.device)
                else:
                    _kg = _torch.tensor([_mixedk_k2(self.ups[i].inner.K) for i in range(_ne)], dtype=_torch.int32, device=self.device)
                _ku = _torch.tensor([_mixedk_k2(self.ups[i].inner.K) for i in range(_ne)], dtype=_torch.int32, device=self.device)
                _kd = _torch.tensor([_mixedk_k2(self.downs[i].inner.K) for i in range(_ne)], dtype=_torch.int32, device=self.device)
                self.mixedk_K_gate_arr = _kg
                self.mixedk_K_up_arr = _ku
                self.mixedk_K_down_arr = _kd
                # Build unified pointer tables for ALL experts
                self.mixedk_ptrs_gate_trellis = _torch.tensor(
                    [(self.gates[i].inner.trellis.data_ptr() if self.gated else self.ups[i].inner.trellis.data_ptr()) for i in range(_ne)],
                    dtype=_torch.long, device=self.device)
                self.mixedk_ptrs_gate_suh = _torch.tensor(
                    [(self.gates[i].inner.suh.data_ptr() if self.gated else self.ups[i].inner.suh.data_ptr()) for i in range(_ne)],
                    dtype=_torch.long, device=self.device)
                self.mixedk_ptrs_gate_svh = _torch.tensor(
                    [(self.gates[i].inner.svh.data_ptr() if self.gated else self.ups[i].inner.svh.data_ptr()) for i in range(_ne)],
                    dtype=_torch.long, device=self.device)
                self.mixedk_ptrs_up_trellis = _torch.tensor(
                    [self.ups[i].inner.trellis.data_ptr() for i in range(_ne)], dtype=_torch.long, device=self.device)
                self.mixedk_ptrs_up_suh = _torch.tensor(
                    [self.ups[i].inner.suh.data_ptr() for i in range(_ne)], dtype=_torch.long, device=self.device)
                self.mixedk_ptrs_up_svh = _torch.tensor(
                    [self.ups[i].inner.svh.data_ptr() for i in range(_ne)], dtype=_torch.long, device=self.device)
                self.mixedk_ptrs_down_trellis = _torch.tensor(
                    [self.downs[i].inner.trellis.data_ptr() for i in range(_ne)], dtype=_torch.long, device=self.device)
                self.mixedk_ptrs_down_suh = _torch.tensor(
                    [self.downs[i].inner.suh.data_ptr() for i in range(_ne)], dtype=_torch.long, device=self.device)
                self.mixedk_ptrs_down_svh = _torch.tensor(
                    [self.downs[i].inner.svh.data_ptr() for i in range(_ne)], dtype=_torch.long, device=self.device)
                self.mixedk_mcg = list(_all_mcg)[0]
                self.mixedk_mul1 = list(_all_mul1)[0]
                # EXL3_MIXEDK_LEGACY=1 selects the legacy per-K-group coop dispatch.
                # Measured +5% on GB10 (sm_121): the grouped coop kernel is ~2.9x
                # cheaper per call than the unified mixed-K kernel there.
                _legacy = os.environ.get("EXL3_MIXEDK_LEGACY", "0") != "0"
                self.mixedk_unified = not _legacy
                # keep mixedk_k_groups populated so the legacy path can be selected at runtime
                print(f" -- Mixed-K UNIFIED kernel enabled in {self.key}: {_ne} experts, K_up range [{_ku.min().item()},{_ku.max().item()}]")

            # Enable fully fused kernel if possible (uniform mcg or mul1 codebook across gate/up/down,
            # and an activation the fused kernel implements)
            if self.multi_up is not None:
                cbs = (
                    self.multi_gate.q_cb() if self.gated else self.multi_up.q_cb(),
                    self.multi_up.q_cb(),
                    self.multi_down.q_cb(),
                )
                self.support_fused = (
                    cbs[0] == cbs[1] == cbs[2] and cbs[0] in ((True, False), (False, True)) and
                    self.support_quant_paths
                )

        # Temp buffers for graph, dq and fused-bsz1 paths
        numex = self.num_experts_per_tok
        H = self.expert_size
        # The gate/up input width and the down output width are the (possibly 128-padded)
        # quantized dims; both equal H for aligned models
        Hi = self.ups[0].in_features
        Ho = self.downs[0].out_features
        I = self.intermediate_size_padded
        device = self.device

        # bszn_rows bounds the fused bsz 1..MAX_BSZN path (BC_BlockSparseMLP.run_bszN, one
        # slot per (token, expert) pick); buffers grow to whichever of that or the single-expert
        # graph loop's TEMP_ROWS_GRAPH requirement is larger, sharing the one cache entry per name
        # (g_tensor_cache is exact-shape-keyed and never evicts -- growing a differently-shaped
        # second entry under the same name would silently double memory forever)
        bszn_rows = MAX_BSZN * numex

        temp_hidden = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH * 2, bszn_rows), Hi), torch.half, "moe1_temp_hidden")
        temp_interm = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH * 2, 2 * bszn_rows), I), self.interm_dtype, "moe1_temp_interm")
        temp_activa = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH, bszn_rows), I), torch.half, "moe1_temp_activa")
        temp_output = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH, bszn_rows), Ho), torch.float, "moe1_temp_output")

        yh = temp_hidden[:bszn_rows].view(bszn_rows, 1, Hi)
        interm_g = temp_interm[:bszn_rows].view(bszn_rows, 1, I)
        interm_u = temp_interm[bszn_rows:bszn_rows*2].view(bszn_rows, 1, I)
        interm_a = temp_activa[:bszn_rows].view(bszn_rows, 1, I)
        yh2 = temp_hidden
        interm_gu = temp_interm
        interm_a2 = temp_activa
        out_d = temp_output[:bszn_rows].view(bszn_rows, 1, Ho)
        out_d2 = temp_output

        # Completion counters + expert-run table of the fused decode kernels
        # (BC_BlockSparseMLP.run_bszN): the kernels reset each other's counters for the next
        # call, so they only need zeroing once
        coop_ctr = g_tensor_cache.get(device, (bszn_rows * (I // 128) + MAX_BSZN * (Ho // 128) + 2 * bszn_rows + 3,), torch.int, "moe1_coop_ctr")
        coop_ctr.zero_()
        # Rotated up-projection input per slot for the fused kernels at bsz > 1 (yh holds the gate's)
        had_u = g_tensor_cache.get(device, (bszn_rows, Hi), torch.half, "moe1_temp_hidden_u")

        # Expert interval for split module (-1, -1) indicate no split
        mine, maxe = self.routing_first, self.routing_last
        if mine is None or maxe - mine == self.num_experts:
            mine, maxe = -1, -1

        # Routed sum of the fused decode path, one exact-width row per token (the kernel trims the
        # padded down width on the way out)
        out_bszn = g_tensor_cache.get(device, (MAX_BSZN, H), torch.float, "moe1_out_bszn")

        cfg = ExpertsCFG(
            yh = yh,
            interm_g = interm_g,
            interm_u = interm_u,
            interm_a = interm_a,
            out_d = out_d,
            out_d2 = out_d2,
            min_expert = mine,
            max_expert = maxe,
            out_bszn = out_bszn,
        )
        self.experts_cfg = cfg

        if (self.support_quant_paths or self.support_bc_bszn) \
                and not self.config.infer_params.no_reconstruct:

            # Embed bound classes for shared experts and shared gate
            sh_exp_bc = None
            sh_exp_t = None
            sh_gate_bc = None
            self.bc_sh_exp = False
            if (
                self.shared_experts
                and isinstance(self.shared_experts, GatedMLP)
                and self.shared_experts.bc is not None
                and self.shared_experts_post_norm is None   # TODO: embed post_norm in BC
                and not self.alt_residual_channel  # TODO: allow residual channel switching in BC (Gemma4)
                and self.latent_in is None  # the graph would merge the shared output at the latent width
            ):
                self.bc_sh_exp = True
                sh_exp_bc = self.shared_experts.bc
                sh_exp_t = torch.empty((1, MAX_BSZN, self.hidden_size), dtype = torch.float, device = self.device)
                if self.shared_gate:
                    assert self.shared_gate.quant_type == "fp16"
                    sh_gate_bc = self.shared_gate.inner.bc

            # Pointer lists for fused modes. Gateless experts reuse the up tables as gate
            # placeholders: valid memory for the (uniform) table loads, never dereferenced
            u_trellis_ptr = torch.tensor([l.inner.trellis.data_ptr() for l in self.ups])
            u_suh_ptr = torch.tensor([l.inner.suh.data_ptr() for l in self.ups])
            u_svh_ptr = torch.tensor([l.inner.svh.data_ptr() for l in self.ups])
            if self.gated:
                g_trellis_ptr = torch.tensor([l.inner.trellis.data_ptr() for l in self.gates])
                g_suh_ptr = torch.tensor([l.inner.suh.data_ptr() for l in self.gates])
                g_svh_ptr = torch.tensor([l.inner.svh.data_ptr() for l in self.gates])
            else:
                g_trellis_ptr, g_suh_ptr, g_svh_ptr = u_trellis_ptr, u_suh_ptr, u_svh_ptr
            gu_trellis_ptr = torch.stack((g_trellis_ptr, u_trellis_ptr), dim = 0).T.contiguous().to(self.device)
            gu_suh_ptr = torch.stack((g_suh_ptr, u_suh_ptr), dim = 0).T.contiguous().to(self.device)
            gu_svh_ptr = torch.stack((g_svh_ptr, u_svh_ptr), dim = 0).T.contiguous().to(self.device)

            dq_temp_up = g_tensor_cache.get(device, (Hi, I), torch.half, "dq_temp")
            dq_temp_down = dq_temp_up.view(I, Ho)

            # Per-expert bias pointer tables for the fused decode path
            def _bias_ptrs(ls):
                if ls[0].inner.bias is None:
                    return None
                return torch.tensor([l.inner.bias.data_ptr() for l in ls],
                                    dtype = torch.long, device = device)
            gate_bias_ptrs = _bias_ptrs(self.gates) if self.gated else None
            up_bias_ptrs = _bias_ptrs(self.ups)
            down_bias_ptrs = _bias_ptrs(self.downs)

            # Bound class for the fused-decode, graph and dq paths (gateless: the up module stands in
            # for the unused gate pointer args, and the gates list is empty)
            multi_gate = self.multi_gate if self.gated else self.multi_up
            self.bc = ext.BC_BlockSparseMLP(
                yh2,
                cfg.yh,
                interm_gu,
                cfg.interm_g,
                cfg.interm_u,
                cfg.interm_a,
                interm_a2,
                cfg.out_d,
                cfg.out_d2,
                sh_exp_t,
                coop_ctr,
                had_u,
                dq_temp_up,
                dq_temp_down,
                cfg.min_expert,
                cfg.max_expert,
                multi_gate.ptrs_trellis,
                multi_gate.ptrs_suh,
                multi_gate.ptrs_svh,
                multi_gate.K,
                multi_gate.mcg,
                multi_gate.mul1,
                self.multi_up.ptrs_trellis,
                self.multi_up.ptrs_suh,
                self.multi_up.ptrs_svh,
                self.multi_up.K,
                self.multi_up.mcg,
                self.multi_up.mul1,
                self.multi_down.ptrs_trellis,
                self.multi_down.ptrs_suh,
                self.multi_down.ptrs_svh,
                self.multi_down.K,
                self.multi_down.mcg,
                self.multi_down.mul1,
                self.activation_fn == "silu",
                self.activation_fn == "gelu",
                self.activation_fn == "swiglu_oai",
                sh_exp_bc,
                sh_gate_bc,
                self.act_limit,
                [x.inner.bc for x in self.gates],
                [x.inner.bc for x in self.ups],
                [x.inner.bc for x in self.downs],
                gu_trellis_ptr,
                gu_suh_ptr,
                gu_svh_ptr,
                cfg.out_bszn,
                gate_bias_ptrs,
                up_bias_ptrs,
                down_bias_ptrs,
                act_relu2 = self.activation_fn == "relu2",
                sh_coop = self.shared_coop_ok(),
            )

            # Larger buffers for fused path, if supported. Wide row tiles (32 / 64 rows, separate
            # N = 128 kernel instances) exist for the mul1 codebook; with them the fused tier's
            # row capacity rises to FUSED_ROWS_WIDE. Dims that are multiples of 256 keep the
            # N = 256 instance for the <= 16-row launch only (it is the faster 16-row tiling)
            if self.support_fused:
                self.mtile_ok = MTILE and bool(self.multi_up.mul1)
                self.fused_rows = FUSED_ROWS_WIDE if self.mtile_ok else TEMP_ROWS_FUSED
                R = self.fused_rows
                C = ext.exl3_moe_max_concurrency(torch.device(device).index)
                self.fused_mode_buffers = FusedBuffers(
                    temp_state_g = g_tensor_cache.get(device, (C, R, H), torch.half, "moe2_temp_state_g"),
                    temp_state_u = g_tensor_cache.get(device, (C, R, H), torch.half, "moe2_temp_state_u"),
                    temp_intermediate_g = g_tensor_cache.get(device, (C, R, I), torch.half, "moe2_temp_intermediate_g"),
                    temp_intermediate_u = g_tensor_cache.get(device, (C, R, I), torch.half, "moe2_temp_intermediate_u"),
                )
                self.f_threshold = min(self.num_experts // self.num_experts_per_tok, 4)

        self._coopmk_setup()


    def _coopmk_setup(self):
        """Bind the cooperative runtime-K decode kernels to this unified mixed-K layer
        (EXL3_MOE_COOP_MIXEDK=1, default on sm_121). self.coopmk stays None, and the layer keeps the
        unified mixed-K path, whenever the layer is outside what the kernels implement"""
        self.coopmk = None
        self.coopmk_out = None
        if not getattr(self, "mixedk_unified", False):
            return
        if not _coop_mixedk_enabled(self.device):
            return
        # Gated silu/gelu, mul1 codebook, integer bitrates 1..8, no biases (mixedk_unified already
        # requires the last), whole expert range on this device
        if not self.gated or self.activation_fn not in ("silu", "gelu"):
            return
        if self.mixedk_mcg or not self.mixedk_mul1:
            return
        if self.num_local_experts != self.num_experts or self.cpu_split_first is not None:
            return
        ks = [l.inner.K for l in self.gates + self.ups + self.downs]
        if any(float(k) != int(k) or not 1 <= int(k) <= 8 for k in ks):
            return
        H = self.expert_size
        Hi = self.ups[0].in_features
        I = self.intermediate_size_padded
        Ho = self.downs[0].out_features
        slots = MAX_BSZN * self.num_experts_per_tok
        if Hi % 128 or I % 128 or Ho % 128 or H > Ho or H > Hi or slots > 256:
            return
        m = _coopmk_module()
        if m is None:
            return
        device = self.device
        interm = self.interm_dtype if self.interm_dtype in (torch.half, torch.float) else torch.half
        buf = lambda shape, dtype, name: g_tensor_cache.get(device, shape, dtype, name)
        ctr = buf((slots * (I // 128) + MAX_BSZN * (Ho // 128) + 2 + (slots + 1) + slots,), torch.int, "coopmk_ctr")
        ctr.zero_()
        out = buf((MAX_BSZN, H), torch.float, "coopmk_out")
        k_tab = lambda ls: torch.tensor([int(l.inner.K) for l in ls], dtype = torch.int32)
        self.coopmk = m.CoopMK(
            Hi,
            self.mixedk_ptrs_gate_trellis, self.mixedk_ptrs_gate_suh, self.mixedk_ptrs_gate_svh,
            self.mixedk_ptrs_up_trellis, self.mixedk_ptrs_up_suh, self.mixedk_ptrs_up_svh,
            self.mixedk_ptrs_down_trellis, self.mixedk_ptrs_down_suh, self.mixedk_ptrs_down_svh,
            None, None, None,
            k_tab(self.gates), k_tab(self.ups), k_tab(self.downs),
            bool(self.mixedk_mcg), bool(self.mixedk_mul1),
            0 if self.activation_fn == "silu" else 1,
            float(self.act_limit),
            True,
            buf((slots, Hi), torch.half, "coopmk_had_g"),
            buf((slots, Hi), torch.half, "coopmk_had_u"),
            buf((slots, I), interm, "coopmk_gu_g"),
            buf((slots, I), interm, "coopmk_gu_u"),
            buf((slots, I), torch.half, "coopmk_act"),
            buf((slots, Ho), torch.float, "coopmk_d_out"),
            ctr, out,
            None,
            -1, -1,
            COOPMK_PLAN,
        )
        self.coopmk_out = out
        print(f" -- Mixed-K coop decode kernels in {self.key}: plan {self.coopmk.plan}, "
              f"variants A {self.coopmk.stage_variants(0)} B {self.coopmk.stage_variants(1)}")


    def _coopmk_forward(self, y, selected_experts, routing_weights, eshape):
        """Routed sum of bsz <= MAX_BSZN tokens through the cooperative runtime-K kernels: two
        launches (three at bsz > 1), launch-only, no host synchronization"""
        bsz = y.shape[0]
        self.coopmk.run(y, selected_experts.contiguous(), routing_weights.contiguous(), None, -1, -1, -1)
        return self.coopmk_out[:bsz].view(eshape)


    def load_routing(self, **kwargs):

        per_expert_scale = self.per_expert_scale
        if self.interm_div != 1.0 and self.router_type == "std":
            # std routing has no scaling factor; fold the interm_div compensation into the
            # per-expert scale, which routing_std applies after top-k normalization. Both the
            # GPU and CPU-offload load paths come through here. The folded scale only goes to the
            # routing config: self.per_expert_scale stays the checkpoint tensor, which is what
            # get_tensors and the TP export hand on
            if per_expert_scale is None:
                per_expert_scale = torch.full(
                    (self.num_experts,), self.interm_div, dtype = torch.bfloat16, device = self.device)
            else:
                per_expert_scale = (per_expert_scale.float() * self.interm_div).to(torch.bfloat16)

        router_logits_bsz1 = torch.empty((1, self.num_experts), dtype = torch.half, device = self.device)
        routing_weights_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.half, device = self.device)
        selected_experts_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.long, device = self.device)

        self.routing_cfg = RoutingCFG(
            gate_tensor = self.routing_gate.inner.weight,
            router_bias = getattr(self.routing_gate.inner, "bias", None),
            gate_tensor_t = None,  # created lazily on first bsz-1 call (weights may be deferred here)
            num_experts = self.num_experts,
            num_experts_per_tok = self.num_experts_per_tok,
            router_logits_bsz1 = router_logits_bsz1,
            routing_weights_bsz1 = routing_weights_bsz1,
            selected_experts_bsz1 = selected_experts_bsz1,
            e_score_correction_bias = self.e_score_correction_bias,
            e_score_bias_h = None,
            tid2eid = self.tid2eid,
            e_score_bias_vl = self.e_score_bias_vl,
            routed_scaling_factor = self.routed_scaling_factor,
            n_group = self.n_group,
            topk_group = self.topk_group,
            per_expert_scale = per_expert_scale,
        )


    @override
    def load(self, device: torch.Device, **kwargs):
        # CPU expert offload (see block_sparse_mlp_cpu.py): a whole-layer claim replaces the
        # GPU load entirely; a split registration shrinks the module to its GPU slice first
        if self.cpu_maybe_offload_load(device, **kwargs):
            return
        self.cpu_maybe_split_load(device, **kwargs)
        super().load(device, **kwargs)

        if self.e_score_correction_bias_key:
            for k in [self.e_score_correction_bias_key, "gate.e_score_correction_bias"]:
                esb = self.config.stc.get_tensor(
                    f"{self.key}.{k}",
                    self.device,
                    optional = True,
                    allow_bf16 = True,
                    no_defer = True,
                )
                if esb is not None:
                    self.e_score_correction_bias = esb if esb.dtype == torch.half else esb.float()
                    break
        if self.e_score_bias_vl_key:
            esb_vl = self.config.stc.get_tensor(
                f"{self.key}.{self.e_score_bias_vl_key}",
                self.device,
                optional = True,
                allow_bf16 = True,
                no_defer = True,
            )
            self.e_score_bias_vl = esb_vl.float() if esb_vl is not None else None
        if self.tid2eid_key:
            self.tid2eid = self.config.stc.get_tensor(
                f"{self.key}.{self.tid2eid_key}",
                self.device,
                no_defer = True,
            )
        if self.per_expert_scale_key:
            self.per_expert_scale = self.config.stc.get_tensor(
                f"{self.key}.{self.per_expert_scale_key}",
                self.device,
                optional = True,
                allow_bf16 = True,
            )
        if device is not None and torch.device(device).type == "cuda":
            self.cpu_post_load()
            self.load_local(**kwargs)
            self.load_routing(**kwargs)


    def _batch_recon_layer(self, y):
        """Per-module batched reconstruct state (moe_batch_recon.BatchReconLayer), built on
        first use; None when the module's configuration isn't covered (same conditions as the
        DQ path plus unpadded hidden dims; fp32 intermediates as in the DQ path are supported)"""
        if self.batch_recon is False:
            return None
        if self.batch_recon is None:
            ok = (
                BATCH_RECON and self.bc is not None and self.support_quant_paths and
                self.interm_dtype in (torch.half, None, torch.float) and self.multi_up is not None and
                self.multi_up.in_features == y.shape[1] and
                self.multi_down.out_features == y.shape[1] and
                self.multi_down.in_features == self.multi_up.out_features
            )
            if not ok:
                self.batch_recon = False
                return None
            from .moe_batch_recon import BatchReconLayer
            mg, mu, md = self.multi_gate, self.multi_up, self.multi_down
            scales = {p: ([l.inner.suh for l in m.linears], [l.inner.svh for l in m.linears])
                      for p, m in (("g", mg), ("u", mu), ("d", md)) if m is not None}
            self.batch_recon = BatchReconLayer(
                (mg.in_features, mg.out_features, mg.K) if mg is not None else None,
                (mu.in_features, mu.out_features, mu.K),
                (md.in_features, md.out_features, md.K),
                mg.q_cb() if mg is not None else None, mu.q_cb(), md.q_cb(),
                self.activation_fn, self.act_limit, self.device, scales,
                interm_fp32 = self.interm_dtype != torch.half)
            self.batch_recon.set_static_pointers(
                mg.ptrs_trellis if mg is not None else None, mu.ptrs_trellis, md.ptrs_trellis)
        return self.batch_recon

    def prefill_worst_case_parts(self, rows: int, assignments: int) -> tuple[int, int]:
        """Upper bound on the GPU-expert prefill transients of a `rows`-token chunk with
        `assignments` routed rows landing on GPU-resident experts, as (fixed, variable) bytes:
        the fp32 accumulator, the output state and the deterministic slot scratch (every
        assignment slotted, padded) live for the whole call; the larger of one
        batched-reconstruct group and the per-expert dequant path with every row on one expert
        is the working set on top. The autosplit measuring forward routes a dummy state, so
        what it sees of these depends on that routing; the loader takes this bound instead
        (autosplit_extra_measure)"""
        if self.cpu_offload or self.multi_up is None:
            return 0, 0
        h = self.expert_size
        recon = self._batch_recon_layer(torch.empty((1, h), dtype = torch.half, device = self.device))
        recon_dims = (recon.dims_g, recon.dims_u, recon.dims_d, recon.interm_fp32) if recon is not None else None
        return self._prefill_bound(rows, assignments, self.intermediate_size_padded, recon_dims)

    def _prefill_bound(self, rows: int, assignments: int, interm: int, recon_dims: tuple | None) -> tuple[int, int]:
        """Shape-only core of prefill_worst_case_parts: `interm` is the (padded) intermediate width
        of the local experts and recon_dims the batched tier's (dims_g, dims_u, dims_d, interm_fp32),
        or None if the layer has no batched tier"""
        h = self.expert_size
        fixed = (rows + 1) * h * 4 + rows * h * 2
        if FUSED_DET:
            fixed += (int(assignments * PAD_MAX) + 1) * h * 4
        r = min(rows, assignments)
        isz = (self.interm_dtype or torch.float).itemsize
        per_expert = r * (2 * interm * isz + 2 * h * 2 + h * 4)
        if self.interm_dtype != torch.half:
            per_expert += r * interm * 2
        batched = worst_case_bytes(*recon_dims, assignments, slot_mode = FUSED_DET) if recon_dims is not None else 0
        return fixed, max(per_expert, batched)

    def autosplit_prepare(self, params):
        """Autosplit loader hook, before the measuring window: the worst-case computation below
        creates the CPU-offload host's per-device stream state and this layer's tier tables as
        a side effect, so running it here makes them resident memory rather than transient"""
        if os.environ.get("EXL3_AUTOSPLIT_WORSTCASE", "1") == "0":
            return
        rows = (params.get("batch_shape") or (1, 0))[1]
        if rows and self.device is not None and self.device.type == "cuda":
            self._autosplit_worst_case(rows)

    def autosplit_extra_measure(self, params):
        """Autosplit loader hook: allocate (and drop) the worst-case prefill transient so the
        device keeps headroom for it (the measuring forward skips the CPU path)"""
        if os.environ.get("EXL3_AUTOSPLIT_WORSTCASE", "1") == "0":
            return
        rows = getattr(self, "_measure_rows", 0)
        if not rows or self.device is None or self.device.type != "cuda":
            return
        total = self._autosplit_worst_case(rows)
        if total > 0:
            t = torch.empty((total,), dtype = torch.uint8, device = self.device)
            del t

    def _autosplit_worst_case(self, rows: int) -> int:
        """Upper bound on this layer's prefill transient for a `rows`-token chunk, in bytes"""
        A = rows * self.num_experts_per_tok
        host = getattr(self, "cpu_host", None)
        if host is not None and getattr(self, "cpu_layer_idx", None) is not None:
            # Split layers: the assignments divide between GPU-resident and CPU-resident
            # experts. Both sides' whole-call buffers live at once; the CPU side's group
            # working set is freed before the GPU side allocates its own. Take the worst split
            total = 0
            for a in (A * i // 4 for i in range(5)):
                gf, gv = self.prefill_worst_case_parts(rows, a)
                cf, cv = host.prefill_worst_case_parts(self.cpu_layer_idx, rows, self.device, A - a)
                total = max(total, gf + cf + max(gv, cv))
        else:
            total = sum(self.prefill_worst_case_parts(rows, A))
        return total

    def _run_batch_recon(self, recon, y, fhs_ext, token_sorted, weight_sorted, expert_count_list, groups,
                         scratch = None, tables = None):
        """Run the planned groups through the batched reconstruct tier. With a slot scratch each
        group's down projection lands in its slots (summed later by exl3_moe_gather); otherwise
        each group accumulates into fhs_ext (rows + 1, the last row a padding sink). Returns the
        set of experts handled."""
        rows = y.shape[0]
        y_ext = torch.cat([y, torch.zeros((1, y.shape[1]), dtype = y.dtype, device = y.device)])
        tok_ext = torch.cat([token_sorted, torch.full((1,), rows, dtype = token_sorted.dtype, device = y.device)])
        w_ext = torch.cat([weight_sorted.half(), torch.zeros((1,), dtype = torch.half, device = y.device)])
        starts = [0]
        for c in expert_count_list:
            starts.append(starts[-1] + c)
        handled = set()
        slot = None
        if scratch is not None:
            # groups were laid out after the fused slots, in plan order, len(grp) * cmax each
            slot = sum(c for c in expert_count_list[:self.num_local_experts or self.num_experts]
                       if 0 < c <= self.fused_rows) if self.fused_mode_buffers is not None else 0
        for grp in groups:
            out_slab = None
            if scratch is not None:
                cmax = max(expert_count_list[e] for e in grp)
                out_slab = scratch[slot : slot + len(grp) * cmax]
                slot += len(grp) * cmax
            recon.run_group(
                y_ext, fhs_ext, tok_ext, w_ext,
                grp, [starts[e] for e in grp], [expert_count_list[e] for e in grp],
                out_slab = out_slab)
            handled.update(grp)
        return handled

    @override
    def unload(self):
        self.cpu_unload()
        self.bc = None
        self.coopmk = None
        self.coopmk_out = None
        self.fused_mode_buffers = None
        self.batch_recon = None
        if self.multi_gate is not None:
            self.multi_gate.unload()
            self.multi_gate = None
        if self.multi_up is not None:
            self.multi_up.unload()
            self.multi_up = None
        if self.multi_down is not None:
            self.multi_down.unload()
            self.multi_down = None
        self.routing_cfg = None
        self.experts_cfg = None
        self.e_score_correction_bias = None
        self.e_score_bias_vl = None
        self.tid2eid = None
        self.per_expert_scale = None
        self.bcast_sel_bsz1 = None
        self.bcast_weights_bsz1 = None
        super().unload()


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        # Row-invariant verify: 2..MAX_BSZN rows run as independent bsz-1 calls so a batched
        # speculative verify computes bit-identical MoE outputs to serial decode
        rows = x.numel() // x.shape[-1]
        if _ROWINV_MOE and 1 < rows <= MAX_BSZN and not self.alt_residual_channel                 and not params.get("autosplit_measure") and not params.get("capture_sens"):
            xf = x.reshape(rows, 1, 1, x.shape[-1]) if x.dim() == 3 else x.reshape(rows, 1, x.shape[-1])
            outs = [self._forward_rows(xf[r], params, out_dtype).clone() for r in range(rows)]
            return torch.cat(outs, dim = -2).reshape(x.shape[:-1] + outs[0].shape[-1:])
        return self._forward_rows(x, params, out_dtype)

    def _forward_rows(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.alt_residual_channel:
            y = params["residual"].view(-1, self.hidden_size)
        else:
            y = x.view(-1, self.hidden_size)
        bsz = y.shape[0]
        bc_sh_exp = False
        if params.get("autosplit_measure"):
            self._measure_rows = bsz
        # Shape of the routed sum: the experts' width is the latent width in a latent MoE
        eshape = x.shape[:-1] + (self.expert_size,)

        # Eligibility for the fused decode kernels (bsz 1..MAX_BSZN): computed up front so it
        # can override the f_threshold-based routing below (bsz>=f_threshold would otherwise
        # always fall through to the exl3_moe/dense path first, capping this tier's reach at
        # f_threshold-1 instead of MAX_BSZN). Expert-range shards (CPU split, TP) are masked
        # inside the kernel (out-of-range picks contribute exact zeros). Shared experts run
        # through BC_GatedMLP's own multi-row graph ahead of the kernel (see mlp.py)
        bszn_eligible = self.bc is not None and bsz <= MAX_BSZN

        # One decision for the unified mixed-K path, taken here and applied at every site below
        # (branch guard, readback suppression, tier plan, slot tables, dispatch) so they cannot
        # disagree. Rejecting a call here repopulates expert_count_list at the readback below,
        # which is what makes the legacy per-K-group coop dispatch reachable for that call.
        # selected_experts is (tokens, experts-per-token) on every routing path, so this is the
        # row count the flattened assignments carry, matching the num_tokens * top_k tests below
        mixedk_unified_ok = getattr(self, "mixedk_unified", False) and \
            bsz * self.num_experts_per_tok >= MIXEDK_MIN_ROWS

        # Routing
        if self.router_pre_norm:
            z = self.router_pre_norm.forward(y, params, out_dtype = torch.half)
        else:
            z = y

        if self.routing_gate is not None:
            selected_experts, routing_weights = self.routing_fn(bsz, self.routing_cfg, z, params)
        elif bsz == 1:
            # Stable broadcast targets rather than per-call allocations
            if self.bcast_sel_bsz1 is None:
                self.bcast_sel_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.long, device = self.device)
                self.bcast_weights_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.half, device = self.device)
            selected_experts = self.bcast_sel_bsz1
            routing_weights = self.bcast_weights_bsz1
        else:
            selected_experts = torch.empty((bsz, self.num_experts_per_tok), dtype = torch.long, device = self.device)
            routing_weights = torch.empty((bsz, self.num_experts_per_tok), dtype = torch.half, device = self.device)

        # Extra norm (Gemma4)
        if self.routed_pre_norm:
            y = self.routed_pre_norm.forward(y, params, out_dtype = torch.half)

        # Latent MoE: the experts (GPU and CPU-resident alike) see the projected input
        if self.latent_in is not None:
            y = self.latent_in.forward(y, params)

        # Calibration: the experts' output sensitivity is weighted by the routing the model would
        # actually do, which is not the selection in use when every expert is activated
        sens_weights = None
        if "capture_sens" in params and self.gated and self.routing_gate is not None:
            sel, w = selected_experts, routing_weights
            if params.get("activate_all_experts"):
                sel, w = self.routing_fn(
                    bsz, self.routing_cfg, z, {k: v for k, v in params.items() if k != "activate_all_experts"})
            sens_weights = torch.zeros((bsz, self.num_experts), dtype = torch.float, device = y.device)
            sens_weights.scatter_(1, sel, w.float())

        if params.get("tp_warmup"):
            # Warmup runs without collectives on garbage-but-finite streams: every rank takes a
            # random (valid, spread-out) selection instead of routing, which exercises the
            # expert paths at realistic row counts (and ranks without a gate have no selection)
            selected_experts.random_(0, self.num_experts)
            routing_weights.fill_(1.0 / self.num_experts_per_tok)
        elif self.routing_device is not None:
            # Selection made on the routing rank and broadcast (router types not yet on the
            # deterministic paths, or EXL3_TP_ROUTING_CHECK, which keeps the broadcast so this
            # rank's own selection can be compared against it)
            check = _routing_check and self.routing_device != self.device and self.routing_gate is not None
            if check:
                local_sel, local_w = selected_experts.clone(), routing_weights.clone()
            params["backend"].broadcast(selected_experts, src_device = self.routing_device)
            params["backend"].broadcast(routing_weights, src_device = self.routing_device)
            if check:
                _routing_check_compare(self.key, local_sel, local_w, selected_experts, routing_weights)

        # CPU expert offload (block_sparse_mlp_cpu.py): split layers hand the tail experts'
        # share to the worker now so it computes concurrently with the GPU expert paths below
        # (folded back in by cpu_split_combine); whole-layer offload replaces the routed sum
        cpu_partial = None
        cpu_pending = None
        if self.cpu_split_first is not None and not params.get("autosplit_measure"):
            cpu_partial, cpu_pending = self.cpu_split_submit(y, bsz, selected_experts, routing_weights)

        if self.cpu_offload:
            final_hidden_states, cpu_pending = self.cpu_offload_issue(
                eshape, y, selected_experts, routing_weights, params)

        # Empty slice
        elif self.intermediate_size == 0 or self.num_local_experts == 0:
            final_hidden_states = torch.zeros(eshape, dtype = torch.float, device = y.device)

        # Unified mixed-K layer at a decode / verify shape: cooperative runtime-K kernels
        # (EXL3_MOE_COOP_MIXEDK=1; self.coopmk is only bound when the layer qualifies)
        elif (
            self.coopmk is not None and bsz <= MAX_BSZN and y.dtype == torch.half and
            selected_experts.dtype == torch.long and routing_weights.dtype == torch.half and
            selected_experts.shape[-1] == self.num_experts_per_tok
        ):
            final_hidden_states = self._coopmk_forward(y, selected_experts, routing_weights, eshape)

        # Torch/C++/fused path
        elif (
            (bsz >= self.f_threshold and not bszn_eligible) or not self.is_quantized or
            self.config.infer_params.no_reconstruct or
            mixedk_unified_ok or
            not (self.support_quant_paths or bszn_eligible)
        ):
            # One spare row: the batched reconstruct tier's padding sink (never read back)
            fhs_ext = torch.zeros((y.shape[0] + 1, y.shape[1]), dtype = torch.float, device = y.device)
            final_hidden_states = fhs_ext[:y.shape[0]]

            # if self.routing_device is None or self.num_local_experts == self.num_experts:
            #     expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = self.num_local_experts)
            # else:
            #     selected_experts -= self.routing_first
            #     invalid = (selected_experts < 0) | (selected_experts >= self.num_local_experts)
            #     shifted = torch.where(invalid, torch.zeros_like(selected_experts), selected_experts + 1)
            #     expert_mask = F.one_hot(shifted, num_classes = self.num_local_experts + 1)[..., 1:]

            if self.num_local_experts is None or self.num_local_experts > 0:

                num_ex = self.num_local_experts or self.num_experts

                num_tokens, top_k = selected_experts.shape
                E = self.num_local_experts

                # Flatten assignments
                flat_expert_global = selected_experts.reshape(-1)               # [num_tokens * top_k]
                flat_weight = routing_weights.reshape(-1)                       # [num_tokens * top_k]

                # Token indices corresponding to each flattened assignment
                flat_token = buffered_interleaved_arange(num_tokens, top_k, device = y.device)

                # Map to local expert ids whenever this module holds a slice (TP shard or
                # CPU expert split), not only in the multi-device routing case
                if self.num_local_experts == self.num_experts:
                    flat_expert_local = flat_expert_global
                else:
                    flat_expert_local = flat_expert_global - self.routing_first
                    valid = (flat_expert_local >= 0) & (flat_expert_local < E)
                    flat_expert_local = torch.where(valid, flat_expert_local, torch.full_like(flat_expert_local, E))

                # Group once by local expert id (including sentinel for expert-P mode)
                order = flat_expert_local.argsort()
                token_sorted = flat_token[order]
                weight_sorted = flat_weight[order]

                # Count how many assignments per expert. With few enough total assignments no
                # expert can exceed the fused kernel's row capacity, so the readback (a CPU sync
                # per layer, ~33% idle at MTP verify shapes) is skipped and everything is fused
                _nosync = (
                    MIXEDK_NOSYNC and mixedk_unified_ok and
                    self.num_local_experts == self.num_experts and num_tokens <= MTILE_T1 and
                    getattr(self, "_mkd_bufs", None) is not None and
                    num_tokens * top_k <= getattr(self, "_mkd_fused_rows", TEMP_ROWS_FUSED) and
                    top_k >= int(self._mkd_bufs.temp_state_g.shape[0])
                )
                # Same device-side count for the uniform fused path when it runs all-fused
                # (run_fused(-1): no count readback follows), e.g. a uniform-K MTP layer
                _nosync_fused = (
                    MIXEDK_NOSYNC and not mixedk_unified_ok and
                    self.fused_mode_buffers is not None and
                    self.num_local_experts == self.num_experts and
                    num_tokens * top_k <= self.fused_rows
                )
                if _nosync or _nosync_fused:
                    expert_count = torch.zeros((E + 1,), dtype = torch.long, device = y.device)
                    expert_count.scatter_add_(0, flat_expert_local, torch.ones_like(flat_expert_local))
                else:
                    expert_count = torch.bincount(flat_expert_local, minlength = E + 1)
                if self.fused_mode_buffers is not None and num_tokens * top_k <= self.fused_rows:
                    expert_count_list = None
                elif mixedk_unified_ok and num_tokens * top_k <= getattr(self, "_mkd_fused_rows", TEMP_ROWS_FUSED):
                    expert_count_list = None
                else:
                    expert_count_list = expert_count.tolist()

                # Tier plan: fused kernel for experts up to self.fused_rows rows, batched
                # reconstruct groups above that up to the tile cap, per-expert reconstruct beyond
                recon = None
                groups = []
                min_rows = 0
                if expert_count_list is None:
                    fused_total = num_tokens * top_k
                else:
                    fused_total = 0
                    if self.fused_mode_buffers is not None:
                        min_rows = self.fused_rows
                        fused_total = sum(c for c in expert_count_list[:num_ex] if 0 < c <= self.fused_rows)
                    elif mixedk_unified_ok:
                        _mkd_r = getattr(self, "_mkd_fused_rows", FUSED_ROWS_WIDE if (MTILE and getattr(self, "mixedk_mul1", False)) else TEMP_ROWS_FUSED)
                        min_rows = _mkd_r
                        fused_total = sum(c for c in expert_count_list[:num_ex] if 0 < c <= _mkd_r)
                    recon = self._batch_recon_layer(y)
                    if recon is not None:
                        lim = max(min_rows, TEMP_ROWS_GRAPH)
                        heavy = [e for e in range(num_ex) if lim < expert_count_list[e] <= recon.max_rows]
                        if heavy:
                            from .moe_batch_recon import plan_groups
                            groups = plan_groups(heavy, lambda e: expert_count_list[e], recon.cap)

                # Deterministic accumulation (FUSED_DET): every assignment of the fused and
                # batched tiers gets a slot in one fp32 scratch (fused experts: count rows each,
                # weighted by the kernel; batched experts: their group's padded row count each,
                # unweighted) and one exl3_moe_gather per layer sums each token's slots in k
                # order, so identical inputs give identical outputs. The atomic alternative adds
                # contributions in arrival order. Slot tables come from the host-side counts, the
                # all-fused fast path builds them on the device. Scratch is prefill-shaped and per
                # call. Its traffic (written once by the GEMMs, read once by the gather) matches
                # what the atomic index_add_ path moved
                scratch = tables = inv_order = None
                if FUSED_DET and (fused_total or groups):
                    A = flat_expert_local.shape[0]
                    inv_order = torch.empty_like(order).scatter_(
                        0, order, torch.arange(A, device = order.device))
                    if expert_count_list is None:
                        # Row num_ex of expert_count is the sentinel bucket for picks outside this
                        # module's expert slice (TP shard, CPU split). Its slots are never written,
                        # so the gather below is restricted to the first num_ex table rows
                        expert_start = torch.cumsum(expert_count, 0) - expert_count
                        tables = torch.stack([expert_start, expert_start, (expert_count > 0).long()])
                        n_slots = fused_total
                    else:
                        import numpy as np
                        E1 = len(expert_count_list)
                        base = np.zeros(E1, dtype = np.int64)
                        kind = np.zeros(E1, dtype = np.int64)
                        starts_np = np.cumsum([0] + expert_count_list[:-1]).astype(np.int64)
                        n_slots = 0
                        if self.fused_mode_buffers is not None:
                            for e in range(num_ex):
                                c = expert_count_list[e]
                                if 0 < c <= self.fused_rows:
                                    base[e] = n_slots; kind[e] = 1; n_slots += c
                        elif mixedk_unified_ok:
                            _mkd_r2 = getattr(self, "_mkd_fused_rows", FUSED_ROWS_WIDE if (MTILE and getattr(self, "mixedk_mul1", False)) else TEMP_ROWS_FUSED)
                            for e in range(num_ex):
                                c = expert_count_list[e]
                                if 0 < c <= _mkd_r2:
                                    base[e] = n_slots; kind[e] = 1; n_slots += c
                        for grp in groups:
                            cmax = max(expert_count_list[e] for e in grp)
                            for b, e in enumerate(grp):
                                base[e] = n_slots + b * cmax; kind[e] = 2
                            n_slots += len(grp) * cmax
                        tables = host_to_device(torch.from_numpy(np.stack([base, starts_np, kind])), y.device)
                    scratch = torch.empty((max(n_slots, 1), y.shape[1]), dtype = torch.float, device = y.device)

                def run_fused(num_active, count_lo = 1, count_hi = self.fused_rows, m_tile = 16):
                    # Gateless: the up module stands in for the gate pointer tables (the kernel
                    # skips the gate GEMM when activation_fn_idx is MOE_ACT_RELU2_NOGATE)
                    multi_gate = self.multi_gate if self.gated else self.multi_up
                    ext.exl3_moe(
                        y,
                        final_hidden_states,
                        expert_count,
                        token_sorted,
                        weight_sorted,
                        self.fused_mode_buffers.temp_state_g,
                        self.fused_mode_buffers.temp_state_u,
                        self.fused_mode_buffers.temp_intermediate_g,
                        self.fused_mode_buffers.temp_intermediate_u,
                        self.activation_fn_idx,
                        multi_gate.K,
                        self.multi_up.K,
                        self.multi_down.K,
                        multi_gate.ptrs_trellis,
                        multi_gate.ptrs_suh,
                        multi_gate.ptrs_svh,
                        self.multi_up.ptrs_trellis,
                        self.multi_up.ptrs_suh,
                        self.multi_up.ptrs_svh,
                        self.multi_down.ptrs_trellis,
                        self.multi_down.ptrs_suh,
                        self.multi_down.ptrs_svh,
                        multi_gate.mcg,
                        multi_gate.mul1,
                        self.multi_up.mcg,
                        self.multi_up.mul1,
                        self.multi_down.mcg,
                        self.multi_down.mul1,
                        self.act_limit,
                        num_active,
                        scratch, tables[0] if tables is not None else None,
                        count_lo, count_hi, m_tile
                    )

                # num_active -1 = unknown (all fused), kernel launches at max concurrency
                if self.fused_mode_buffers is not None:
                    if expert_count_list is None:
                        run_fused(-1)
                    else:
                        counts = [c for c in expert_count_list[:num_ex] if 0 < c <= self.fused_rows]
                        t1 = sum(1 for c in counts if MTILE_T1 < c <= MTILE_T2)
                        t2 = sum(1 for c in counts if c > MTILE_T2)
                        if self.mtile_ok and (t1 or t2):
                            # One launch per row tile over its expert range, largest first
                            t0 = len(counts) - t1 - t2
                            if t2:
                                run_fused(t2, MTILE_T2 + 1, self.fused_rows, 64)
                            if t1:
                                run_fused(t1, MTILE_T1 + 1, MTILE_T2, 32)
                            if t0:
                                run_fused(t0, 1, MTILE_T1, 16)
                        else:
                            run_fused(len(counts))

                # Batched reconstruct tier (into slots when deterministic, else accumulating)
                batched = ()
                if groups:
                    batched = self._run_batch_recon(
                        recon, y, fhs_ext, token_sorted, weight_sorted, expert_count_list, groups,
                        scratch, tables)

                # === Mixed-K unified fused dispatch (one launch for ALL experts) ===
                mixedk_handled = set()
                if mixedk_unified_ok:
                    # Allocate fused buffers on first use
                    if not hasattr(self, "_mkd_bufs") or self._mkd_bufs is None:
                        H = self.expert_size
                        I = self.intermediate_size_padded
                        C_max = ext.exl3_moe_max_concurrency(torch.device(self.device).index)
                        # Fewer, wider expert groups: with a 16-row tile holding only 1-2 rows the
                        # gate/up GEMM (I/128 N-tiles) leaves most of an 8-wide group's blocks idle.
                        # Giving each expert more SMs splits the K loop further and fills them up.
                        # EXL3_MOE_MIXEDK_GROUPS caps the group count (0 = device default)
                        _g = int(os.environ.get("EXL3_MOE_MIXEDK_GROUPS", "0"))
                        if _g > 0:
                            C_max = min(C_max, _g)
                        # Buffer count = max concurrent expert groups. Raising it past the
                        # device default (num_sms // MOE_SMS_PER_EXPERT) lets more experts be
                        # in flight at once, which pays off now that the mixedk kernel fits
                        # two blocks per SM; costs one temp-buffer slice per extra group.
                        _gc = int(os.environ.get("EXL3_MOE_MIXEDK_CMAX", "0"))
                        if _gc > 0:
                            C_max = _gc
                        R = FUSED_ROWS_WIDE if (MTILE and self.mixedk_mul1) else TEMP_ROWS_FUSED
                        self._mkd_fused_rows = R
                        self._mkd_mtile_ok = MTILE and self.mixedk_mul1
                        self._mkd_bufs = FusedBuffers(
                            temp_state_g=g_tensor_cache.get(self.device, (C_max, R, H), torch.half, "mkd_state_g"),
                            temp_state_u=g_tensor_cache.get(self.device, (C_max, R, H), torch.half, "mkd_state_u"),
                            temp_intermediate_g=g_tensor_cache.get(self.device, (C_max, R, I), torch.half, "mkd_interm_g"),
                            temp_intermediate_u=g_tensor_cache.get(self.device, (C_max, R, I), torch.half, "mkd_interm_u"),
                        )
                    _mkd_rows = self._mkd_fused_rows
                    # Count active experts within fused row cap
                    if expert_count_list is not None:
                        counts_fused = [c for c in expert_count_list[:num_ex] if 0 < c <= _mkd_rows]
                    elif _nosync:
                        # Every expert is in the m16 tier (count <= num_tokens <= MTILE_T1), so the
                        # list only needs its length: the buffer concurrency, which the true active
                        # count (>= top_k) meets, so the kernel's grid is unchanged
                        counts_fused = [1] * int(self._mkd_bufs.temp_state_g.shape[0])
                    else:
                        # GPU fast path — small readback of just the fused count
                        _ec = expert_count[:num_ex]
                        _m = (_ec > 0) & (_ec <= _mkd_rows)
                        counts_fused = _ec[_m].tolist()
                    if counts_fused:
                        def run_mixedk_fused(num_active, count_lo=1, count_hi=_mkd_rows, m_tile=16):
                            ext.exl3_moe_mixedk(
                                y,
                                final_hidden_states,
                                expert_count,
                                token_sorted,
                                weight_sorted,
                                self._mkd_bufs.temp_state_g,
                                self._mkd_bufs.temp_state_u,
                                self._mkd_bufs.temp_intermediate_g,
                                self._mkd_bufs.temp_intermediate_u,
                                self.activation_fn_idx,
                                self.mixedk_K_gate_arr,
                                self.mixedk_K_up_arr,
                                self.mixedk_K_down_arr,
                                self.mixedk_ptrs_gate_trellis,
                                self.mixedk_ptrs_gate_suh,
                                self.mixedk_ptrs_gate_svh,
                                self.mixedk_ptrs_up_trellis,
                                self.mixedk_ptrs_up_suh,
                                self.mixedk_ptrs_up_svh,
                                self.mixedk_ptrs_down_trellis,
                                self.mixedk_ptrs_down_suh,
                                self.mixedk_ptrs_down_svh,
                                self.mixedk_mcg,
                                self.mixedk_mul1,
                                self.mixedk_mcg,
                                self.mixedk_mul1,
                                self.mixedk_mcg,
                                self.mixedk_mul1,
                                self.act_limit,
                                num_active,
                                scratch, tables[0] if tables is not None else None,
                                count_lo, count_hi, m_tile,
                            )
                        t1 = sum(1 for c in counts_fused if MTILE_T1 < c <= MTILE_T2)
                        t2 = sum(1 for c in counts_fused if c > MTILE_T2)
                        if self._mkd_mtile_ok and (t1 or t2):
                            t0 = len(counts_fused) - t1 - t2
                            if t2:
                                run_mixedk_fused(t2, MTILE_T2 + 1, _mkd_rows, 64)
                            if t1:
                                run_mixedk_fused(t1, MTILE_T1 + 1, MTILE_T2, 32)
                            if t0:
                                run_mixedk_fused(t0, 1, MTILE_T1, 16)
                        else:
                            run_mixedk_fused(len(counts_fused))
                        # Mark ALL experts with tokens as handled (fused kernel processes them)
                        if expert_count_list is not None:
                            for e in range(num_ex):
                                c = expert_count_list[e]
                                if 0 < c <= _mkd_rows:
                                    mixedk_handled.add(e)
                        elif not _nosync:
                            # (with _nosync, expert_count_list is None and the per-expert fallback
                            # loop below runs zero iterations, so this set has no consumer)
                            _ec = expert_count[:num_ex]
                            for e in ((_ec > 0) & (_ec <= _mkd_rows)).nonzero(as_tuple=True)[0].tolist():
                                mixedk_handled.add(e)
                # === Legacy Mixed-K per-K-group fused dispatch (atomic accumulation) ===
                elif hasattr(self, "mixedk_k_groups") and self.mixedk_k_groups and expert_count_list is not None:
                    _starts = [0] * (num_ex + 1)
                    for _e in range(num_ex):
                        _starts[_e + 1] = _starts[_e] + expert_count_list[_e]
                    for grp_experts, mg, mu, md, g2l in self.mixedk_k_groups:
                        grp_count_list = []
                        grp_token_parts = []
                        grp_weight_parts = []
                        for j, e in enumerate(grp_experts):
                            c = expert_count_list[e] if e < len(expert_count_list) else 0
                            grp_count_list.append(c)
                            if c > 0:
                                s = _starts[e]
                                grp_token_parts.append(token_sorted[s:s+c])
                                grp_weight_parts.append(weight_sorted[s:s+c])
                        grp_total = sum(grp_count_list)
                        if grp_total == 0:
                            continue
                        grp_count_t = torch.tensor(grp_count_list + [0], dtype=torch.long, device=y.device)
                        grp_token_t = torch.cat(grp_token_parts)
                        grp_weight_t = torch.cat(grp_weight_parts)
                        multi_gate = mg if self.gated else mu
                        if not hasattr(self, "_mkd_bufs") or self._mkd_bufs is None:
                            H = self.expert_size
                            I = self.intermediate_size_padded
                            C_max = ext.exl3_moe_max_concurrency(torch.device(self.device).index)
                            R = TEMP_ROWS_FUSED
                            self._mkd_bufs = FusedBuffers(
                                temp_state_g=g_tensor_cache.get(self.device, (C_max, R, H), torch.half, "mkd_state_g"),
                                temp_state_u=g_tensor_cache.get(self.device, (C_max, R, H), torch.half, "mkd_state_u"),
                                temp_intermediate_g=g_tensor_cache.get(self.device, (C_max, R, I), torch.half, "mkd_interm_g"),
                                temp_intermediate_u=g_tensor_cache.get(self.device, (C_max, R, I), torch.half, "mkd_interm_u"),
                            )
                        ext.exl3_moe(
                            y,
                            final_hidden_states,
                            grp_count_t,
                            grp_token_t,
                            grp_weight_t,
                            self._mkd_bufs.temp_state_g,
                            self._mkd_bufs.temp_state_u,
                            self._mkd_bufs.temp_intermediate_g,
                            self._mkd_bufs.temp_intermediate_u,
                            self.activation_fn_idx,
                            multi_gate.K,
                            mu.K,
                            md.K,
                            multi_gate.ptrs_trellis,
                            multi_gate.ptrs_suh,
                            multi_gate.ptrs_svh,
                            mu.ptrs_trellis,
                            mu.ptrs_suh,
                            mu.ptrs_svh,
                            md.ptrs_trellis,
                            md.ptrs_suh,
                            md.ptrs_svh,
                            multi_gate.mcg,
                            multi_gate.mul1,
                            mu.mcg,
                            mu.mul1,
                            md.mcg,
                            md.mul1,
                            self.act_limit,
                            -1,
                            None, None,
                            1, TEMP_ROWS_FUSED, 16,
                        )
                        mixedk_handled.update(grp_experts)

                # One fixed-order gather over every slot
                if scratch is not None:
                    ext.exl3_moe_gather(final_hidden_states, scratch, flat_expert_local, inv_order,
                                        tables[1, :num_ex], tables[0, :num_ex], tables[2, :num_ex], weight_sorted)

                out_state = None
                interm = None
                interm_a = None
                max_count = 0
                start = 0

                # expert_count_list None: everything already handled by the fused kernel above
                for expert_idx in range(num_ex if expert_count_list is not None else 0):
                    count = expert_count_list[expert_idx]
                    end = start + count
                    if count <= min_rows or expert_idx in batched or expert_idx in mixedk_handled:
                        start = end
                        continue

                    top_x = token_sorted[start:end]
                    w = weight_sorted[start:end].unsqueeze(1)

                    current_state = y.index_select(0, top_x)

                    if self.bc is not None and self.support_quant_paths:
                        # Graph path
                        if count <= TEMP_ROWS_GRAPH:
                            self.bc.run_single_expert(current_state, expert_idx)
                            current_state = self.experts_cfg.out_d2[:count]

                        # DQ path
                        else:
                            if count > max_count:
                                out_state = torch.empty((count, self.expert_size), dtype = torch.float, device = self.device)
                                interm = torch.empty((count * 2, self.intermediate_size_padded), dtype = self.interm_dtype, device = self.device)
                                interm_a = interm[:count] if self.interm_dtype == torch.half else \
                                    torch.empty_like(interm[:count], dtype = torch.half)
                                out_state_ = out_state
                                interm_ = interm
                                interm_a_ = interm_a
                                max_count = count
                            elif count == max_count:
                                out_state_ = out_state
                                interm_ = interm
                                interm_a_ = interm_a
                            else:
                                out_state_ = out_state[:count]
                                interm_ = interm[:count * 2]
                                interm_a_ = interm_a[:count]

                            yh = torch.empty((count * 2, self.expert_size), dtype = torch.half, device = self.device)
                            self.bc.run_single_expert_dq(current_state, expert_idx, yh, interm_, interm_a_, out_state_)
                            current_state = out_state_
                    else:

                        # Torch path
                        def mlp(exp_i, xc):
                            u = self.ups[exp_i].forward(xc, params)
                            if self.gated:
                                g = self.gates[exp_i].forward(xc, params)
                                if sens_weights is not None:
                                    exp_g = exp_i if self.num_local_experts == self.num_experts \
                                        else exp_i + self.routing_first
                                    sw = sens_weights[top_x, exp_g]
                                    routed = sw.nonzero().flatten()
                                    if routed.numel():
                                        capture_out_sensitivity(
                                            params["capture_sens"], self.gates[exp_i], self.ups[exp_i],
                                            self.downs[exp_i], g[routed], u[routed], self.activation_fn,
                                            self.act_limit, sw[routed]
                                        )
                                a = u if self.interm_dtype == torch.half else torch.empty_like(u, dtype = torch.half)
                                self.activation_fn_call(g, u, a, self.act_limit)
                            else:
                                a = self.gateless_act(u)
                                if a.dtype != torch.half:
                                    a = a.half()
                            return self.downs[exp_i].forward(a, params)

                        current_state = mlp(expert_idx, current_state)

                    current_state.mul_(w)
                    final_hidden_states.index_add_(0, top_x, current_state)
                    start = end

            final_hidden_states = final_hidden_states.reshape(eshape)

        # Fused decode kernels (bsz 1..MAX_BSZN): two launches run every (token, expert) slot
        # through the expert MLP (no sort/dedup -- overlap between tokens this small is rare and
        # not worth the argsort/bincount host-sync cost that the fused/exl3_moe path pays). Shared
        # experts (if present) run through their own multi-row BC_GatedMLP graph and are merged
        # inside the kernel. Expert-range shards (CPU split, TP) produce a partial sum here:
        # out-of-range picks are masked inside the kernel and contribute exact zeros. Every
        # quantized configuration that reaches this point has self.bc (it is built whenever the
        # quantized paths apply), so this is the last tier
        else:
            if not bszn_eligible:
                import sys
                sys.stderr.write("BSZN_FAIL bsz=%d f_thresh=%d bc=%s quant=%s mixedk=%s support_quant=%s\n" % (
                    bsz, self.f_threshold, self.bc is not None, self.is_quantized,
                    getattr(self, "mixedk_unified", False), getattr(self, "support_quant_paths", False)))
                sys.stderr.flush()
            assert bszn_eligible
            self.bc.run_bszN(y, selected_experts, routing_weights)
            final_hidden_states = self.experts_cfg.out_bszn[:bsz].view(eshape)
            bc_sh_exp = self.bc_sh_exp

        # Independent shared-expert work can cover the CPU job (split tail or whole layer)
        # before collect enqueues its stream wait. Fused shared experts have already run
        # inside the routed path. Keep prefill scheduling and the order of post norms / TP
        # collectives unchanged.
        shared_hidden_states = None
        if cpu_pending is not None and bsz <= MAX_BSZN and self.shared_experts and not bc_sh_exp:
            shared_hidden_states = self.shared_experts.forward(x, params)

        # CPU tail partial folds in before the post norms (nonlinear: they must see the
        # complete routed sum)
        final_hidden_states = self.cpu_split_combine(final_hidden_states, cpu_partial, cpu_pending, eshape)

        # Latent MoE: project the routed sum back to the residual width (linear, so under TP
        # each rank's partial sum projects on its own ahead of the reduction)
        if self.latent_out is not None:
            final_hidden_states = self.latent_out.forward(final_hidden_states.to(torch.half), params)

        # The post norms are nonlinear, so under TP their inputs must be complete sums, not
        # per-rank partials: reduce the routed and shared contributions separately before the
        # norms (Gemma4 MoE), after which every rank holds identical complete tensors and the
        # final reduction is skipped
        pre_norm_reduce = self.tp_reduce and (
            self.routed_post_norm is not None or
            (self.shared_experts is not None and self.shared_experts_post_norm is not None and not bc_sh_exp)
        )
        if pre_norm_reduce:
            self.tp_collect(
                params["backend"],
                final_hidden_states,
                self.intermediate_size > 0 and self.num_local_experts > 0
            )

        # Extra norm (Gemma4)
        if self.routed_post_norm:
            final_hidden_states = self.routed_post_norm.forward(final_hidden_states, params)

        # Shared experts
        if self.shared_experts and not bc_sh_exp:
            y = shared_hidden_states
            if y is None:
                y = self.shared_experts.forward(x, params)
            if pre_norm_reduce:
                self.tp_collect(params["backend"], y, True)
            if self.shared_experts_post_norm:
                y = self.shared_experts_post_norm.forward(y, params)
            if self.shared_gate:
                if bsz > 32:
                    z = self.shared_gate.forward(x, params)
                    ext.add_sigmoid_gate(y, z, final_hidden_states)
                else:
                    ext.add_sigmoid_gate_proj(y, x, final_hidden_states, self.shared_gate.inner.weight)
            else:
                final_hidden_states += y

        # Output reduction
        if self.tp_reduce and not pre_norm_reduce:
            self.tp_collect(
                params["backend"],
                final_hidden_states,
                (self.intermediate_size > 0 and self.num_local_experts > 0) or bool(self.shared_experts)
            )

        if out_dtype is not None:
            final_hidden_states = final_hidden_states.to(out_dtype)
        return final_hidden_states


    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.e_score_correction_bias is not None:
            t[f"{self.key}.{self.e_score_correction_bias_key}"] = self.e_score_correction_bias.contiguous()
        if self.e_score_bias_vl is not None:
            t[f"{self.key}.{self.e_score_bias_vl_key}"] = self.e_score_bias_vl.contiguous()
        if self.tid2eid is not None:
            t[f"{self.key}.{self.tid2eid_key}"] = self.tid2eid.contiguous()
        if self.per_expert_scale is not None:
            t[f"{self.key}.{self.per_expert_scale_key}"] = self.per_expert_scale.contiguous()
        return t


    def shared_coop_ok(self) -> bool:
        """The shared expert can run as a one-expert fused decode launch: an EXL3 GatedMLP with a
        single slice, 128-aligned widths and an activation the fused kernels implement, merged
        without a post-norm at the residual width. Under TP such a shared expert is placed whole
        on one rank (the fused kernels tile the intermediate width in 128s and a split slice
        would rarely stay aligned), whose launch is the only one contributing it."""
        se = self.shared_experts
        if not _moe_shared_coop or se is None or not isinstance(se, GatedMLP):
            return False
        if self.shared_experts_post_norm is not None or self.latent_in is not None or self.alt_residual_channel:
            return False
        if len(se.gates) != 1 or len(se.ups) != 1 or len(se.downs) != 1:
            return False
        if se.activation_fn not in ("silu", "gelu", "relu2"):
            return False
        g, u, d = se.gates[0], se.ups[0], se.downs[0]
        def quantized(l):
            if l.quant_type is not None:
                return l.quant_type == "exl3"
            return self.config is not None and self.config.stc.has_tensor(f"{l.key}.trellis")
        if not (quantized(g) and quantized(u) and quantized(d)):
            return False
        return g.in_features % 128 == 0 and g.out_features % 128 == 0 and d.out_features % 128 == 0 \
            and self.hidden_size <= d.out_features

    def tp_prefill_bound(self, tokens: int, s: int, channels: int, tensor_split: bool) -> int:
        """Prefill transients on a TP rank holding s of channels, the same bound the layer-split
        loader reserves (prefill_worst_case_parts) but taken from config shapes, since the TP split
        is planned before load. Expert split: this rank's share of the assignments on full-width
        experts (a hot expert beyond that share is covered only by the reserve). Tensor split: every
        assignment, on experts holding this rank's intermediate slice"""
        h = self.expert_size
        if s == 0:
            return tokens * h * torch.float.itemsize  # empty slice: zeroed accumulator only
        top_k = self.num_experts_per_tok
        if tensor_split:
            a, interm = tokens * top_k, s * 128
        else:
            a, interm = -(-tokens * top_k * s // channels), self.intermediate_size_padded
        # Whether the batched reconstruct tier will be available is assumed from the tensors
        # being EXL3 rather than checked (_batch_recon_layer needs the loaded layer)
        batched = BATCH_RECON and self.config.stc.has_tensor(f"{self.ups[0].key}.trellis")
        recon_dims = (
            (h, interm) if self.gated else None, (h, interm), (interm, h),
            self.interm_dtype == torch.float
        ) if batched else None
        return sum(self._prefill_bound(tokens, a, interm, recon_dims))

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        if self.shared_gate:
            storage += self.shared_gate.storage_size()
        for g in self.gates: storage += g.storage_size()
        for u in self.ups: storage += u.storage_size()
        for d in self.downs: storage += d.storage_size()
        # The latent projections are replicated on every rank, and so is the routing gate (with
        # its transposed and int8 copies, see block_sparse_mlp_routing._gate_t)
        storage_d = 3 * self.routing_gate.storage_size()
        if self.latent_in is not None:
            storage_d += self.latent_in.storage_size() + self.latent_out.storage_size()
        # Routing selection plus the argsort / sorted token, weight / inverse order index tensors
        overhead_d = self.num_experts_per_tok * 48
        recons = max(
            self.gates[0].recons_size() if self.gated else 0,
            self.ups[0].recons_size(),
            self.downs[0].recons_size()
        )
        use_tp_split = options.get("moe_tensor_split", False)
        tpa = MoETPAllocation(
            module = self,
            tensor_split = use_tp_split,
            key = self.key,
            channel_width = 128 if use_tp_split else 1,
            channel_unit = "channels" if use_tp_split else "experts",
            storage_per_device = storage_d,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            recons_temp = recons,
            channels_to_split = self.ups[0].out_features // 128 if use_tp_split else self.num_experts,
            limit_key = "moe"
        )
        tpa_list = [tpa]
        if self.shared_experts:
            sh = self.shared_experts.make_tp_allocation(options)
            if self.shared_coop_ok():
                for t in sh:
                    t.max_devices = 1
            tpa_list += sh
        return tpa_list


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": BlockSparseMLP,
            "kwargs": {
                "key": self.key,
                "hidden_size": self.hidden_size,
                "latent_size": self.latent_size,
                "intermediate_size": self.intermediate_size,
                "activation_fn": self.activation_fn,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "interm_dtype": self.interm_dtype,
                "interm_div": self.interm_div,
                "router_type": self.router_type,
                # The constructor folds interm_div into the routing scale again on import, so
                # send the pre-fold value (the expert tensors arrive already rescaled)
                "routed_scaling_factor": self.routed_scaling_factor / self.interm_div
                    if (self.interm_div != 1.0 and self.router_type != "std") else self.routed_scaling_factor,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "act_limit": self.act_limit,
                "alt_residual_channel": self.alt_residual_channel,
                "key_tid2eid": self.tid2eid_key,
                "key_e_score_bias_vl": self.e_score_bias_vl_key,
            },
            # Hash-MoE bootstrap layers (DeepSeek-V4): frozen token->experts table, needed
            # wherever routing runs (the output device, like the routing gate)
            "tid2eid": producer.send(self.tid2eid) if self.tid2eid is not None else None,
            "routing_gate": _export(self.routing_gate),
            "shared_gate": _export(self.shared_gate),
            "latent_in": _export(self.latent_in),
            "latent_out": _export(self.latent_out),
            "e_score_correction_bias": producer.send(self.e_score_correction_bias),
            "e_score_bias_vl": producer.send(self.e_score_bias_vl) if self.e_score_bias_vl is not None else None,
            "per_expert_scale": producer.send(self.per_expert_scale),
            "gates": [_export(self.gates[i]) for i in range(self.num_experts)] if self.gated else None,
            "ups": [_export(self.ups[i]) for i in range(self.num_experts)],
            "downs": [_export(self.downs[i]) for i in range(self.num_experts)],
            "shared_experts": self.shared_experts.tp_export(plan, producer) \
                if self.shared_experts is not None else None,
            "shared_experts_post_norm": _export(self.shared_experts_post_norm),
            "router_pre_norm": _export(self.router_pre_norm),
            "routed_pre_norm": _export(self.routed_pre_norm),
            "routed_post_norm": _export(self.routed_post_norm),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        consumer = local_context["consumer"]
        key = exported["kwargs"]["key"]
        device = local_context["device"]
        output_device = local_context["output_device"]
        first, last, unit = plan[key]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        def _import_no_reduce(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan, skip_reduction = True) \
                if exported.get(name) else None

        def _import_i(name, i):
            nonlocal exported, plan
            return exported[name][i]["cls"].tp_import(local_context, exported[name][i], plan) \
                if exported.get(name) else None

        def _import_i_split(name, i, split):
            nonlocal exported, plan
            return exported[name][i]["cls"].tp_import_split(local_context, exported[name][i], plan, split) \
                if exported.get(name) else None

        # Gateless experts (NemotronH) export gates as None; the local module gets an empty
        # gates list so the ctor derives gated = False
        gated = exported.get("gates") is not None

        # Replicated routing: every rank holds the gate and routes for itself (no selection
        # broadcast); the residual stream is rank-identical by construction of the deterministic
        # paths, so the selections agree. The check tool keeps the broadcast for comparison
        replicate = exported["kwargs"]["router_type"] in _replicated_router_types and not _routing_check

        # Tensor parallel
        if unit == "channels":
            num_local_experts = exported["kwargs"]["num_experts"]
            gu_split = (True, first, last)
            d_split = (False, first, last)
            exported["kwargs"]["intermediate_size"] = last - first
            gates = [_import_i_split("gates", i, gu_split) for i in range(num_local_experts)] if gated else []
            ups = [_import_i_split("ups", i, gu_split) for i in range(num_local_experts)]
            downs = [_import_i_split("downs", i, d_split) for i in range(num_local_experts)]
            routing_first = 0
            routing_last = num_local_experts

        # Expert parallel
        elif unit == "experts":
            num_local_experts = last - first
            gates = [_import_i("gates", i) for i in range(first, last)] if gated else []
            ups = [_import_i("ups", i) for i in range(first, last)]
            downs = [_import_i("downs", i) for i in range(first, last)]
            routing_first = first
            routing_last = last

        else:
            assert False

        # A shared expert placed whole on one rank (allocation max_devices = 1, see
        # shared_coop_ok) is absent on the others: its contribution enters the reduction from
        # the owner alone
        sh_present = True
        if exported.get("shared_experts") is not None:
            sh_key = exported["shared_experts"]["kwargs"]["key"]
            if sh_key in plan:
                sh_first, sh_last, _ = plan[sh_key]
                sh_present = sh_last > sh_first
        module = BlockSparseMLP(
            config = None,
            **exported["kwargs"],
            num_local_experts = num_local_experts,
            gates = gates,
            ups = ups,
            downs = downs,
            shared_experts = _import_no_reduce("shared_experts") if sh_present else None,
            shared_gate = _import("shared_gate") if sh_present else None,
            latent_in = _import("latent_in"),
            latent_out = _import("latent_out"),
            routing_gate = _import("routing_gate") if (replicate or device == output_device or _routing_check) else None,
            routing_first = routing_first,
            routing_last = routing_last,
            routing_device = None if replicate else output_device,
            shared_experts_post_norm = _import("shared_experts_post_norm"),
            router_pre_norm = _import("router_pre_norm"),
            routed_pre_norm = _import("routed_pre_norm"),
            routed_post_norm = _import("routed_post_norm"),
        )

        module.device = device
        module.e_score_correction_bias = consumer.recv(exported["e_score_correction_bias"], cuda = True)
        if exported.get("e_score_bias_vl") is not None:
            module.e_score_bias_vl = consumer.recv(exported["e_score_bias_vl"], cuda = True)
        module.per_expert_scale = consumer.recv(exported["per_expert_scale"], cuda = True)
        if exported.get("tid2eid") is not None and (replicate or device == output_device):
            module.tid2eid = consumer.recv(exported["tid2eid"], cuda = True)
        if unit == "channels" or num_local_experts > 0:
            module.load_local()
        if module.routing_gate is not None:
            module.load_routing()
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
            # Routed and shared experts are allocated separately; only when one rank owns both
            # is the trailing collective a plain broadcast
            owner_keys = [key]
            if exported.get("shared_experts") is not None:
                owner_keys.append(exported["shared_experts"]["kwargs"]["key"])
            module.tp_owner = module.tp_single_owner(local_context, *owner_keys)
        return module
