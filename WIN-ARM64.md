# Windows ARM64 branch (RTX Spark N1X / GB10)

This repo is a fork chain, not a fork of upstream alone:

```
turboderp-org/exllamav3
    -> vcruz305/exllamav3          (Linux DGX Spark aarch64 port, snapshot 07030ce)
        -> this repo, branch backup-20261026 (MTP head trim,
           PLE PQ vram, aarch64 port refinements, DFlash2 rollback state)
            -> branch win-arm64    (Windows-on-ARM rebuild)
```

`win-arm64` adds exactly three changes over `backup-20261026`, all needed to
build natively on Windows ARM64 (MSVC + CUDA 13.x arm64):

1. `exllamav3/exllamav3_ext/cpu/moe_handoff.cu`,
   `exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu` - MSVC does not define
   `__aarch64__` (it defines `_M_ARM64`), so the cpu-pause helper fell through
   to `_mm_pause()` and failed to compile. Now uses the MSVC `__yield()`
   intrinsic under `_MSC_VER`.
2. `setup.py` - link `cublas/cusparse/cudart` via absolute
   `%CUDA_PATH%\lib\arm64\*.lib`. torch's build puts the CUDA `lib/x64`
   directory first in the linker search path, which pulls x64 import libs and
   produces a wall of `LNK2001` on ARM64.

Everything else (PQ n-gram VRAM path, MTP head trim, Win32 overlapped disk
gather, etc.) is already in `backup-20261026`.

Build (see main repo docs for the full toolchain):

```powershell
$env:TORCH_CUDA_ARCH_LIST = "12.1"   # GB10 / RTX Spark N1X
pip install -e . --no-build-isolation
```

Weights are NOT part of this repo. This fork is designed to serve
[turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
branch `3.05bpw_h5_ng5` with a PQ-compressed n-gram table
(`EXL3_NGRAM_PQ_META`, see `exllamav3/modules/ngram_embedding.py`).
