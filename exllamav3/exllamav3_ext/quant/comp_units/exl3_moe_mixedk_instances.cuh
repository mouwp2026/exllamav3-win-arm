#pragma once

#include "../exl3_moe_common.cuh"

typedef void (*fp_exl3_moe_mixedk_kernel) (EXL3_MOE_MIXEDK_KERNEL_ARGS);

// Mixed-K instances: cb1/cb2 x n128/n256 for m16 (default), plus cb2 x n128 for m32/m64
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb1();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n256_cb1();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n256_cb2();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_m32();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_m64();
// Deeper-pipeline m16 / n128 / mul1 variants (see exl3_moe.cu: EXL3_MK_SHPIPE)
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_sh4fs3();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_sh6fs5();

extern fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_instances[];      // [cb_idx * 2 + N_off]
extern fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_instances_m32[];  // [0]
extern fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_instances_m64[];  // [0]
