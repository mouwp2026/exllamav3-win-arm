# Multi-node layer-split inference

## Scope: PP, not cross-host TP

[`examples/multinode_pipeline.py`](../examples/multinode_pipeline.py) is a
single-request, sequential **pipeline-parallel (PP)** example. Rank 0 embeds the
input, each rank runs a contiguous decoder-layer slice, and the last rank returns
a greedy token or teacher-forced NLL to rank 0. It does not overlap microbatches.
Only one stage computes at a time during single-token decode.

The library's existing TP/EP worker system is single-host. Later four-Spark
experiments used separate cross-host tensor-parallel (TP), layer-owner and
context-parallel (CP) harnesses. Those loaders, head-split MLA imports, RDMA
collectives and speculative servers are **not packaged by this example**.
Do not infer TP speed, long-context quality, or an OpenAI endpoint from PP support.

## Recovered reusable improvements

The tested four-host pipeline changes are included in PR #17:

- `NetEndpoint`: framed TCP tensors and JSON controls, contiguous send payloads,
  byte-oriented `recv_into`, reused pinned CUDA staging, and support for fp16,
  bf16, fp32, int32/int64, uint8, bool and, when Torch supports it, fp8 e4m3fn.
- TCP socket buffer requests of 4 MiB, `TCP_NODELAY` and best-effort Linux
  `TCP_QUICKACK`. Kernel limits can clamp buffer sizes; this does not guarantee
  line-rate throughput or eliminate TCP copies.
- Async H2D staging lifetime: an event is recorded after `recv_tensor(stream=...)`
  and waited on before the next staging-buffer reuse. The caller still needs to
  order consumption of the returned tensor on another CUDA stream. A streamed
  send synchronizes its D2H copy before writing the socket; it is not an async
  network-send API. Endpoints are not thread-safe.
- `NcclEndpoint`: the same tensor/JSON interface using `torch.distributed` p2p,
  byte-preserving payloads, NCCL on CUDA and Gloo for CPU tests. It follows the
  caller's current-stream contract; an explicit `stream` argument is rejected
  **before wire activity**, not silently ignored or scheduled on another stream.
- A slice-only loader and a check that GLM/DeepSeek DSA split boundaries start on
  a `full` indexer layer. Shared indexer selections stay local to the owning stage.
- Cooperative mixed-K MoE decode enabled by default on sm_121. The unified path
  remains for prefill. `EXL3_MOE_COOP_MIXEDK=0` disables that decode default for
  controlled comparisons. Do not replace a mixed-K checkpoint with uniform-K
  weights, or enable a different precision/legacy path as a silent workaround.

Transport support for fp8 does **not** convert the pipeline's hidden states to
fp8; tensor dtype is preserved. NCCL p2p is not a TP all-reduce backend.
These transports have no authentication or encryption: use a trusted, isolated
fabric, firewall job ports, and do not expose them to the public Internet.
See [transport and driver limits](multinode_pipeline_limits.md) for finite frame
limits, TCP I/O deadlines, endpoint poisoning and receive-layout handling.
The manual loop automatically supports recurrent state for
`Glm5NextForConditionalGeneration` (GLM-5.3-Flash). Other recurrent/hybrid
families are rejected before allocation or network setup; general library model
support is not qualification for this PP example.

## Build and runtime identity

Use the same commit, Python/Torch ABI and compatible CUDA driver on every rank.
On GB10 use a toolkit and Torch build supporting sm_121. For a source build in a
prepared virtualenv with Torch, setuptools and Ninja installed:

```bash
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=2
python setup.py build_ext --inplace > build.log 2>&1
```

`MAX_JOBS=2` bounds Ninja concurrency; ensure Ninja is on PATH. Keep the log and
record compiler/toolkit versions, Torch version/CUDA/CXX11 ABI, architecture list,
source tree and produced extension SHA-256. Verify the return code, not just that
an older `.so` exists. Use private build/cache directories, not a live serving
tree. Other GPU architectures need their own target/build.

For a homogeneous four-GB10 cluster, build once on rank 0 and distribute that
exact source-anchored extension to peers, then verify its hash and dependency/
driver compatibility before loading weights. Matching git commits alone do not
prove matching native code. Check the package and extension actually imported:

```bash
git rev-parse HEAD HEAD^{tree}
git status --short
python -c 'import torch, exllamav3, exllamav3_ext; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI); print(exllamav3.__file__); print(exllamav3_ext.__file__)'
sha256sum exllamav3_ext*.so
```

The import can trigger a JIT build if no compatible extension is found; do this
only in the isolated validation environment. Pin and record Triton/JIT caches as
well when comparing performance. Do not benchmark against an unrelated installed
wheel while claiming the checkout's source identity.

## Files and fabric discovery

Each rank needs the **same model configuration and tokenizer**, plus the physical
safetensors shards covering its assigned modules. Rank 0 also needs embedding
weights; the last rank needs final norm/head weights. Full shards on every rank
are acceptable. Check model/config/tokenizer hashes; never modify the original
checkpoint or mix revisions to make a partial shard set load.

Select reachable fabric addresses or DNS names in rank order. Use **numeric
fabric IPs** when requiring a hard TCP connect budget: OS hostname resolution
is outside Python's socket timeout. Inspect each host:

```bash
ip -br addr
rdma link show
ibdev2netdev
ss -ltnp
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

Map the intended network interface and RDMA HCA to `FABRIC_IFACE` and `RDMA_HCA`
on **each host**. Do not copy interface names, GID assumptions or IPs from another
cluster. For NCCL:

```bash
export NCCL_SOCKET_IFNAME="$FABRIC_IFACE" NCCL_IB_HCA="$RDMA_HCA"
export NCCL_IB_DISABLE=0 NCCL_DEBUG=INFO
```

Inspect initialization logs for the intended `NET/IB` path. A successful bootstrap
or a link marked UP does not prove RDMA is being used. TCP needs no NCCL settings.

## Four-rank launch

For a 78-layer GLM-5.3 checkpoint, this tested split is valid:
`0:26,26:46,46:62,62:78`. It is not a generic split for every model. All ranges must
be contiguous, cover all decoder layers, and obey state-sharing boundaries.

Set these on every host, replacing names with your own reachable fabric hosts:

```bash
MODEL=/models/GLM-5.3-EXL3
ADDRS=rank0,rank1,rank2,rank3
SPLITS=0:26,26:46,46:62,62:78
```

Start rank 3, then rank 2, then rank 1 in separate host shells. For each, set
`RANK` accordingly and run:

```bash
python examples/multinode_pipeline.py -m "$MODEL" --rank "$RANK" \
  --addrs "$ADDRS" --splits "$SPLITS" --transport nccl \
  --port 29650 --ctx 8192 --chunk 512
```

Start rank 0 last:

```bash
python examples/multinode_pipeline.py -m "$MODEL" --rank 0 \
  --addrs "$ADDRS" --splits "$SPLITS" --transport nccl \
  --port 29650 --ctx 8192 --chunk 512 \
  --prompt "Explain pipeline parallelism in three sentences." --max_new 200 \
  --nll_file /data/validation.txt
```

`--nll_file` is optional and read only by rank 0. Use a fixed text and save its
hash/token count when comparing configurations. To test TCP, change
`--transport tcp` on **all** ranks. TCP uses `port` on nonzero ranks and `port+1`
on rank 0 for return traffic; NCCL rendezvous uses `port+7`. Reserve those ports
before launch and check every rank's log/exit code. The example has long link
timeouts and no cohort supervisor: use an external bounded launcher/watchdog,
and stop all ranks belonging to your job if a peer fails. Never stop another
operator's job to free a port/GPU.

## Precision, context and memory safety

The default is `Cache(model, max_num_tokens=ctx)` with fp16 cache (MLA stores
fp16 latent/RoPE/indexer planes). The explicit `--cache_quant 4` / `-cq 4` option
selects the canonical quantized cache; `--cache_quant 4,6` sets separate K/V
bitrates. It does not change weight precision or transport dtype. The stage loop
keeps and advances recurrent state automatically for
`Glm5NextForConditionalGeneration`. **Q4 PP and GLM-5.3-Flash recurrent GPU
quality are not qualified by the historical fp16 GLM-5.3 PP measurements.** The
state lifecycle has CPU regression coverage; this is not an exact-checkpoint
hardware qualification. Other recurrent/hybrid families still fail early at the
CLI.
There are no TP, CP, MTP or DFlash switches. `--ctx` is allocated capacity, not
prompt tokens read. Use a multiple of 256, keep prompt plus decode within capacity,
and start with short prompts. Increasing capacity is not a long-context
correctness or throughput result.

The separate Q4 MLA experiments used packed 4-bit latent values plus scale
metadata, fp16 RoPE and separate indexer planes. In replicated TP every rank
holds that cache; owner/CP layouts instead change storage ownership and
collectives. They are not interchangeable with this stage-owned fp16 PP cache,
and their quality/speed receipts do not apply to this example.

On UMA hosts, preflight active process owners and Linux `/proc/meminfo`, then
monitor throughout load, prefill and decode. Require **MemAvailable >= 2 GiB on
every rank and zero positive swap growth** relative to preflight. A cohort
watchdog must stop only its own ranks on a breach. Leave additional practical
headroom (GB10 can start swapping above that floor), price loader/packed-tensor
transients and scratch, and use a staged capacity ladder rather than immediately
allocating a maximum cache. CUDA free-memory reports alone exclude reclaimable
page cache and are not a sufficient admission rule. The example itself does
**not** enforce this watchdog.

Any future speculative path must use a drafter trained for the **exact target**
(hidden width, layer count and target feature-layer IDs). A GLM-5.3-Flash drafter
is not a GLM-5.3 drafter. No drafter is loaded by the packaged PP example.

## Validation and measured scope

The original runtime source at commit
`fa69bb37c7f74e303f501b0adbcd3bff36013c40` was tested on four GB10s with GLM-5.3
EXL3 3.38bpw, the split above, fp16 cache, 8,192-token capacity, 512-token NLL
chunks and 200 generated tokens. The short Wikipedia text scored **1,934
next-token predictions**:

| Transport | NLL | Top-1 | Greedy decode | Median ms/token |
| --- | ---: | ---: | ---: | ---: |
| NCCL/RoCE | 0.893886 | 0.8025 | 8.75 tok/s | 113.7 |
| TCP | 0.893886 | 0.8025 | 8.61 tok/s | 115.3 |

These are historical short-context, single-stream measurements, not a guaranteed
speed or a 1M-token read-in. Matching NLL was also observed with another valid
split. Do not transfer TP/DFlash, CP read-in, or another engine's performance to
this PP implementation. The later TP/CP experiment code and optional native
projection/router/k-split changes remain outside this PR because they need a
complete reusable runtime and fresh exact-source four-rank quality gates;
rejected approximations and overlapping shared-scratch kernels are not enabled.

Run the bounded transport regression suite (no extension build needed):

```bash
python -m pytest tests/model -q -rA
```

The TCP async-staging regression includes a CUDA-only test. CPU/Gloo success is
not a GPU/NCCL quality gate. Before promoting any new runtime change, exercise
both transports on the same checkpoint, fixed teacher-forced text and greedy
prompt; save all rank logs, runtime/native identities, generated IDs/text, scored
count and memory/swap peaks. Report actual read-in length separately from cache
capacity, and retain failed/rejected arms rather than relabeling them as gains.
