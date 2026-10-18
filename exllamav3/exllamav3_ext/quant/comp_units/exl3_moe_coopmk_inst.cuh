#pragma once

// Body of an instance file: the A and B kernels of one variant, both tile geometries, mul1
// codebook (cb = 2), which is what the mixed-K packs use

#include "../exl3_moe_coopmk.cuh"
#include "../exl3_moe_coopmk_kernel.cuh"

#define COOPMK_INST_A(V, MASK, MINB) \
    CoopMKKernel coopmk_kernel_a_##V(int Hi, bool wide) \
    { \
        using namespace exl3_coopmk_ns; \
        const int smem = smem_a_bytes_mk(Hi); \
        if (wide) return { (void*) coopmk_a_kernel<2, true, MASK, MINB>, smem }; \
        return { (void*) coopmk_a_kernel<2, false, MASK, MINB>, smem }; \
    }

#define COOPMK_INST_B(V, MASK, MINB) \
    CoopMKKernel coopmk_kernel_b_##V(bool wide) \
    { \
        using namespace exl3_coopmk_ns; \
        const int smem = smem_b_bytes_mk(); \
        if (wide) return { (void*) coopmk_b_kernel<2, true, MASK, MINB>, smem }; \
        return { (void*) coopmk_b_kernel<2, false, MASK, MINB>, smem }; \
    }
