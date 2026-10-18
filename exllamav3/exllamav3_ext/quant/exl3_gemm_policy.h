#pragma once

#include <cstdlib>
#include <cstring>

// Opt-in numerical compatibility for deployments qualified with the original
// four CUDA GEMM shapes. Wider row tiles and their separate M buckets remain
// the default. Explicitly forced kernel shapes are not changed by this policy.
inline bool exl3_gemm_legacy_tiles_enabled()
{
#if defined(USE_ROCM)
    return false;
#else
    static const bool enabled = []
    {
        const char* value = std::getenv("EXL3_GEMM_LEGACY_TILES");
        return value && value[0] && std::strcmp(value, "0") != 0;
    }();
    return enabled;
#endif
}

inline int exl3_gemm_autotune_shape_count(int available)
{
    return exl3_gemm_legacy_tiles_enabled() && available > 4 ? 4 : available;
}
