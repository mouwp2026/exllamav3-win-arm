// Per-expert runtime-K fused decode MoE for mixed-K layers: parameter setup, launcher, kernel
// attribute query. The kernels (exl3_moe_coopmk_kernel.cuh) reuse the uniform coop kernel's code; this
// file mirrors exl3_moe_coop_prepare / exl3_moe_coop_launch (quant/exl3_moe_coop.cu) with the
// bitrates moved from the launch into per-expert device tables.

#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <cstdlib>
#include <algorithm>

#include "../util.h"
#include "../util.cuh"
#include "exl3_moe_coopmk.cuh"
#define COOPMK_DEFINE_ROT
#include "exl3_moe_coopmk_kernel.cuh"

namespace {

typedef CoopMKKernel (*fp_kernel_a)(int, bool);
typedef CoopMKKernel (*fp_kernel_b)(bool);

const fp_kernel_a kernels_a[COOPMK_V_COUNT] =
{
    coopmk_kernel_a_all, coopmk_kernel_a_reg, coopmk_kernel_a_stg, coopmk_kernel_a_allmb2, coopmk_kernel_a_regmb2
};
const fp_kernel_b kernels_b[COOPMK_V_COUNT] =
{
    coopmk_kernel_b_all, coopmk_kernel_b_reg, coopmk_kernel_b_stg, coopmk_kernel_b_allmb2, coopmk_kernel_b_regmb2
};
const uint32_t variant_mask[COOPMK_V_COUNT] =
{
    exl3_coopmk_ns::KM_ALL, exl3_coopmk_ns::KM_REG, exl3_coopmk_ns::KM_STG, exl3_coopmk_ns::KM_ALL, exl3_coopmk_ns::KM_REG
};

int env_int(const char* name, int def)
{
    const char* v = std::getenv(name);
    return (v && *v) ? atoi(v) : def;
}

// Same knobs and rules as the uniform launcher (exl3_moe_coop.cu)
int pick_ksplit(int max_split)
{
    static int forced = -2;
    if (forced == -2) forced = env_int("EXL3_MOE_COOP_KSPLIT", 0);
    return forced > 0 ? std::min(forced, max_split) : 1;
}

int wide_mode()
{
    static int mode = -2;
    if (mode == -2) mode = env_int("EXL3_MOE_COOP_WIDE", -1);
    return mode;
}

bool is_blackwell(int device)
{
    static std::map<int, bool> cache;
    auto it = cache.find(device);
    if (it != cache.end()) return it->second;
    int major = 0;
    cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    bool bw = major >= 10;
    cache[device] = bw;
    return bw;
}

bool pick_wide(int force, int kslices, int slots, int device)
{
    if (force >= 0) return force != 0;
    const int mode = wide_mode();
    if (mode >= 0) return mode != 0;
    if (!is_blackwell(device)) return true;
    return kslices >= 256 || (kslices >= 128 && slots >= 32);
}

void smem_optin(void* kernel, int smem)
{
    if (smem <= 48 * 1024) return;
    static std::map<void*, int> done;
    auto it = done.find(kernel);
    if (it != done.end() && it->second >= smem) return;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    done[kernel] = smem;
}

uint32_t kset_of(const at::Tensor& k)
{
    at::Tensor kc = k.to(at::kCPU).to(at::kInt).contiguous();
    const int* d = kc.data_ptr<int>();
    uint32_t s = 0;
    for (int64_t i = 0; i < kc.numel(); ++i)
    {
        TORCH_CHECK(d[i] >= 1 && d[i] <= 8, "coopmk: bitrate ", d[i], " outside the integer range 1..8");
        s |= 1u << d[i];
    }
    return s;
}

}  // namespace

CoopMK::CoopMK
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
    int plan_
)
{
    TORCH_CHECK(mul1 && !mcg, "coopmk: only the mul1 codebook is instantiated");
    TORCH_CHECK_DTYPE(ctr, kInt);
    TORCH_CHECK(ctr.is_contiguous(), "coopmk: ctr must be contiguous");
    for (const at::Tensor* t : { &g_trellis, &g_suh, &g_svh, &u_trellis, &u_suh, &u_svh, &d_trellis, &d_suh, &d_svh })
        TORCH_CHECK(t->scalar_type() == at::kLong && t->is_contiguous() && t->dim() == 1, "coopmk: pointer tables must be contiguous 1-D int64");
    TORCH_CHECK(act >= 0 && act <= 3, "coopmk: unknown activation");
    const int64_t n = u_trellis.size(0);
    TORCH_CHECK(g_trellis.size(0) == n && d_trellis.size(0) == n, "coopmk: pointer table lengths differ");
    TORCH_CHECK(k_gate.numel() == n && k_up.numel() == n && k_down.numel() == n, "coopmk: K table lengths differ");

    p = {};
    p.Hi = Hi;
    p.I = (int) gu_u.size(-1);
    p.Ho = (int) d_out.size(-1);
    p.H_out = (int) out.size(-1);
    TORCH_CHECK(p.Hi % 128 == 0 && p.I % 128 == 0 && p.Ho % 128 == 0 && p.H_out <= p.Ho, "coopmk: Hi/I/Ho shape");

    TORCH_CHECK_DTYPE(had_g, kHalf);
    TORCH_CHECK_DTYPE(had_u, kHalf);
    TORCH_CHECK_DTYPE(act_out, kHalf);
    TORCH_CHECK_DTYPE(d_out, kFloat);
    TORCH_CHECK_DTYPE(out, kFloat);
    p.gu_f32 = gu_u.scalar_type() == at::kFloat;
    if (!p.gu_f32) TORCH_CHECK_DTYPE(gu_u, kHalf);
    TORCH_CHECK(gu_g.scalar_type() == gu_u.scalar_type(), "coopmk: gate/up scratch dtype mismatch");
    const int slots_max = (int) std::min({ had_g.numel() / p.Hi, had_u.numel() / p.Hi, gu_g.numel() / p.I,
                                           gu_u.numel() / p.I, act_out.numel() / p.I, d_out.numel() / p.Ho });
    auto check_scratch = [&] (const at::Tensor& t, int width)
    {
        TORCH_CHECK(t.is_contiguous() && t.size(-1) == width, "coopmk: scratch shape");
    };
    check_scratch(had_g, p.Hi);
    check_scratch(had_u, p.Hi);
    check_scratch(gu_g, p.I);
    check_scratch(gu_u, p.I);
    check_scratch(act_out, p.I);
    check_scratch(d_out, p.Ho);
    TORCH_CHECK(out.dim() >= 2 && out.stride(-1) == 1, "coopmk: out shape");
    TORCH_CHECK(slots_max >= 1 && slots_max <= 256, "coopmk: scratch must hold 1..256 slots");

    const int rows_max = (int) out.size(-2);
    p.slots_max = slots_max;
    p.ctr_a_len = slots_max * (p.I / 128);
    p.ctr_b_len = rows_max * (p.Ho / 128);
    TORCH_CHECK(ctr.numel() >= exl3_moe_coop_ctr_len(slots_max, rows_max, p.I, p.Ho), "coopmk: counter scratch too small");
    p.ctr_a = (int*) ctr.data_ptr();
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.runs = p.ctr_b + p.ctr_b_len;
    p.rows_max = rows_max;

    p.g_trellis = (const int64_t*) g_trellis.data_ptr();
    p.g_suh = (const int64_t*) g_suh.data_ptr();
    p.g_svh = (const int64_t*) g_svh.data_ptr();
    p.u_trellis = (const int64_t*) u_trellis.data_ptr();
    p.u_suh = (const int64_t*) u_suh.data_ptr();
    p.u_svh = (const int64_t*) u_svh.data_ptr();
    p.d_trellis = (const int64_t*) d_trellis.data_ptr();
    p.d_suh = (const int64_t*) d_suh.data_ptr();
    p.d_svh = (const int64_t*) d_svh.data_ptr();
    p.g_bias = g_bias ? (const int64_t*) g_bias->data_ptr() : nullptr;
    p.u_bias = u_bias ? (const int64_t*) u_bias->data_ptr() : nullptr;
    p.d_bias = d_bias ? (const int64_t*) d_bias->data_ptr() : nullptr;
    p.n_local = (int) n;
    p.act = act;
    p.act_limit = act_limit;
    p.gated = gated;

    p.had_g = (half*) had_g.data_ptr();
    p.had_u = (half*) had_u.data_ptr();
    p.gu_g = gu_g.data_ptr();
    p.gu_u = gu_u.data_ptr();
    p.act_out = (half*) act_out.data_ptr();
    p.d_out = (float*) d_out.data_ptr();
    p.out = (float*) out.data_ptr();
    p.out_stride = (int) out.stride(-2);

    if (sh_gate_w)
    {
        TORCH_CHECK_DTYPE(sh_gate_w.value(), kHalf);
        TORCH_CHECK(sh_gate_w->is_contiguous(), "coopmk: shared gate weight must be contiguous");
        p.sh_gate_w = (const half*) sh_gate_w->data_ptr();
        p.sh_gate_n = (int) sh_gate_w->numel();
    }
    p.min_expert = min_expert;
    p.max_expert = max_expert;

    // Bitrate tables: [gate; up; down] as one int32 device array (gateless: gate row = up row)
    kset_a = (gated ? kset_of(k_gate) : 0u) | kset_of(k_up);
    kset_b = kset_of(k_down);
    auto dev = u_trellis.device();
    k_tab = at::cat({ (gated ? k_gate : k_up).to(dev).to(at::kInt).reshape({-1}),
                      k_up.to(dev).to(at::kInt).reshape({-1}),
                      k_down.to(dev).to(at::kInt).reshape({-1}) }).contiguous();

    plan = plan_ > 0 ? plan_ : env_int("EXL3_COOPMK_PLAN", COOPMK_PLAN_DEFAULT);

    keep = { g_trellis, g_suh, g_svh, u_trellis, u_suh, u_svh, d_trellis, d_suh, d_svh,
             had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out };
    if (g_bias) keep.push_back(*g_bias);
    if (u_bias) keep.push_back(*u_bias);
    if (d_bias) keep.push_back(*d_bias);
    if (sh_gate_w) keep.push_back(*sh_gate_w);
}

std::vector<int> CoopMK::stage_variants(int stage, int plan_override) const
{
    const int pl = plan_override > 0 ? plan_override : plan;
    const uint32_t ks = stage == 0 ? kset_a : kset_b;
    const bool has_reg = (ks & exl3_coopmk_ns::KM_REG) != 0;
    const bool has_stg = (ks & exl3_coopmk_ns::KM_STG) != 0;
    std::vector<int> v;
    switch (pl)
    {
        case 1: v.push_back(COOPMK_V_ALL); break;
        case 3: v.push_back(COOPMK_V_ALLMB2); break;
        case 4:
            if (has_reg) v.push_back(COOPMK_V_REGMB2);
            if (has_stg) v.push_back(COOPMK_V_STG);
            break;
        default:
            if (has_reg && has_stg) { v.push_back(COOPMK_V_REG); v.push_back(COOPMK_V_STG); }
            else if (has_reg) v.push_back(COOPMK_V_REG);
            else v.push_back(COOPMK_V_STG);
            break;
    }
    return v;
}

void CoopMK::run
(
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out,
    int wide_a_force, int wide_b_force, int plan_override
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = x.device().index();

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(rw, kHalf);
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "coopmk: x must be (bsz, H) with unit column stride");
    TORCH_CHECK(sel.is_contiguous() && rw.is_contiguous(), "coopmk: sel/rw must be contiguous");
    TORCH_CHECK(sel.sizes() == rw.sizes() && sel.dim() == 2 && sel.size(0) == x.size(0), "coopmk: sel/rw must be (bsz, topk)");

    MoeCoopParams q = p;
    { static int dbg = env_int("EXL3_MOE_COOP_DBG", 0); q.dbg = dbg; }
    q.x = (const half*) x.data_ptr();
    q.x_stride = (int) x.stride(0);
    q.bsz = (int) x.size(0);
    q.H = (int) x.size(1);
    q.topk = (int) sel.size(-1);
    q.sel = (const int64_t*) sel.data_ptr();
    q.rw = (const half*) rw.data_ptr();
    const int slots = q.bsz * q.topk;
    TORCH_CHECK(q.H % 4 == 0 && q.H <= q.Hi, "coopmk: H/Hi shape");
    TORCH_CHECK(slots <= q.slots_max && q.bsz <= q.rows_max, "coopmk: batch exceeds the scratch");
    TORCH_CHECK(q.min_expert < 0 || q.max_expert - q.min_expert <= q.n_local, "coopmk: expert range exceeds tables");
    if (sh_out)
    {
        TORCH_CHECK_DTYPE(sh_out.value(), kFloat);
        TORCH_CHECK(q.H_out == q.H, "coopmk: shared expert output width must match the routed output");
        TORCH_CHECK(sh_out->is_contiguous() && sh_out->size(-1) == x.size(1) && sh_out->numel() >= x.size(0) * x.size(1),
                    "coopmk: shared expert output shape");
        q.sh_out = (const float*) sh_out->data_ptr();
        if (q.sh_gate_w) TORCH_CHECK(q.sh_gate_n == q.H && q.H % 2 == 0, "coopmk: shared gate width");
    }
    else
        q.sh_gate_w = nullptr;

    const int nproj = q.gated ? 2 : 1;
    q.a_global = q.bsz > 1;
    const bool wide_a = pick_wide(wide_a_force, q.Hi / 16, slots, device);
    const bool wide_b = pick_wide(wide_b_force, q.I / 16, slots, device);

    const int base_a = slots * nproj * (q.I / (wide_a ? 128 : MOE_COOP_COLS));
    const int base_b = slots * (q.Ho / (wide_b ? 128 : MOE_COOP_COLS));
    const int max_split = std::max(1, q.slots_max / slots);
    q.ksplit_a = pick_ksplit(max_split);
    q.ksplit_b = pick_ksplit(max_split);
    const int grid_a = base_a * q.ksplit_a;
    const int grid_b = base_b * q.ksplit_b;

    const int* kp = k_tab.data_ptr<int>();
    void* args[] = { (void*) &q, (void*) &kp };

    if (q.a_global)
    {
        const int rot_items = slots * (q.Hi / 128) * nproj;
        const int rot_grid = CEIL_DIVIDE(rot_items, MOE_COOP_THREADS / 32);
        exl3_coopmk_ns::coopmk_rot_kernel<<<rot_grid, MOE_COOP_THREADS, 0, stream>>>(q);
    }
    for (int v : stage_variants(0, plan_override))
    {
        CoopMKKernel ka = kernels_a[v](q.Hi, wide_a);
        smem_optin(ka.kernel, ka.smem);
        cuda_check(cudaLaunchKernel(ka.kernel, dim3(grid_a), dim3(MOE_COOP_THREADS), args, ka.smem, stream));
    }
    for (int v : stage_variants(1, plan_override))
    {
        CoopMKKernel kb = kernels_b[v](wide_b);
        smem_optin(kb.kernel, kb.smem);
        cuda_check(cudaLaunchKernel(kb.kernel, dim3(grid_b), dim3(MOE_COOP_THREADS), args, kb.smem, stream));
    }
    cuda_check(cudaPeekAtLastError());
}

std::vector<std::tuple<int, int, int, int, int, int, int, int>> coopmk_kernel_info(int Hi)
{
    std::vector<std::tuple<int, int, int, int, int, int, int, int>> r;
    for (int v = 0; v < COOPMK_V_COUNT; ++v)
    for (int stage = 0; stage < 2; ++stage)
    for (int w = 0; w < 2; ++w)
    {
        CoopMKKernel k = stage == 0 ? kernels_a[v](Hi, w != 0) : kernels_b[v](w != 0);
        cudaFuncAttributes a;
        cuda_check(cudaFuncGetAttributes(&a, k.kernel));
        int nb = 0;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k.kernel, MOE_COOP_THREADS, k.smem));
        r.emplace_back(v, stage, w, a.numRegs, (int) a.localSizeBytes, (int) a.sharedSizeBytes, k.smem, nb);
    }
    return r;
}
