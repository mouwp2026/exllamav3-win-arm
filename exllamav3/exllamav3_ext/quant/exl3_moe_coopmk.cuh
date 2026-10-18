#pragma once

// Per-expert runtime-K fused decode MoE (mixed-K layers): host-side declarations.
// Kernels: exl3_moe_coopmk_kernel.cuh. Launcher and CoopMK class: exl3_moe_coopmk.cu.
// Instances: comp_units/exl3_moe_coopmk_inst_*.cu.

#include <ATen/Tensor.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include "exl3_moe_coop.cuh"

struct CoopMKKernel
{
    void* kernel;
    int smem;
};

// Kernel variants. Each is a pair of stage kernels (A: gate/up, B: down) compiled for a set of
// bitrates (mask); a block whose expert bitrate is outside its variant's mask exits at once
enum CoopMKVariant
{
    COOPMK_V_ALL = 0,       // K 1..8, one launch per stage
    COOPMK_V_REG = 1,       // K 2..4 (register-decoded widths)
    COOPMK_V_STG = 2,       // K 1, 5..8 (shared-memory staged widths)
    COOPMK_V_ALLMB2 = 3,    // K 1..8, __launch_bounds__(512, 2): at most 64 registers
    COOPMK_V_REGMB2 = 4,    // K 2..4, at most 64 registers
    COOPMK_V_COUNT = 5
};

#define COOPMK_VARIANTS(X) X(all) X(reg) X(stg) X(allmb2) X(regmb2)
#define COOPMK_DECL(V) \
    CoopMKKernel coopmk_kernel_a_##V(int Hi, bool wide); \
    CoopMKKernel coopmk_kernel_b_##V(bool wide);
COOPMK_VARIANTS(COOPMK_DECL)
#undef COOPMK_DECL

// Launch plans (per layer, per stage): which variants run, in order
//   1: one all-K launch                     (V_ALL)
//   2: split by decode kind                 (V_REG for K 2..4, V_STG for the rest; one launch if
//                                            the stage's bitrates are all of one kind)
//   3: one all-K launch capped at 64 regs   (V_ALLMB2)
//   4: split, register widths capped at 64  (V_REGMB2 + V_STG)
//   0: default (EXL3_COOPMK_PLAN, else 2)
#define COOPMK_PLAN_DEFAULT 2

class CoopMK
{
public:
    MoeCoopParams p;
    at::Tensor k_tab;                       // int32 [3 * n_local]: gate, up, down bitrates (device)
    std::vector<at::Tensor> keep;           // tables and scratch kept alive for the raw pointers
    uint32_t kset_a;                        // bitrates present in the gate/up stage (bit K)
    uint32_t kset_b;                        // bitrates present in the down stage
    int plan;

    CoopMK
    (
        int Hi,
        at::Tensor g_trellis, at::Tensor g_suh, at::Tensor g_svh,
        at::Tensor u_trellis, at::Tensor u_suh, at::Tensor u_svh,
        at::Tensor d_trellis, at::Tensor d_suh, at::Tensor d_svh,
        c10::optional<at::Tensor> g_bias,
        c10::optional<at::Tensor> u_bias,
        c10::optional<at::Tensor> d_bias,
        at::Tensor k_gate, at::Tensor k_up, at::Tensor k_down,
        bool mcg, bool mul1,
        int act,
        float act_limit,
        bool gated,
        at::Tensor had_g, at::Tensor had_u,
        at::Tensor gu_g, at::Tensor gu_u,
        at::Tensor act_out, at::Tensor d_out,
        at::Tensor ctr, at::Tensor out,
        c10::optional<at::Tensor> sh_gate_w,
        int min_expert, int max_expert,
        int plan
    );

    // Routed sum of bsz <= rows_max tokens into `out` rows 0..bsz-1 (plus sh_out * gate when
    // given). Launch-only: no host synchronization. wide_a / wide_b: -1 auto (the uniform
    // kernel's rule, EXL3_MOE_COOP_WIDE), 0 / 1 force. plan: -1 the object's plan
    void run
    (
        const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
        const c10::optional<at::Tensor>& sh_out,
        int wide_a, int wide_b, int plan_override
    );

    // Variants a stage would launch under a plan (for logging)
    std::vector<int> stage_variants(int stage, int plan_override) const;
};

// Per-variant kernel attributes: (variant, stage, wide, num_regs, local_bytes, static_smem,
// dynamic_smem(Hi), max_blocks_per_sm)
std::vector<std::tuple<int, int, int, int, int, int, int, int>> coopmk_kernel_info(int Hi);
