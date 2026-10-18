#include "exl3_moe_mixedk_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_m64() { return exl3_moe_mixedk_kernel<128, 2, 64>; }
