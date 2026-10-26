#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "ngram.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <algorithm>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "util.h"
#include "util.cuh"

namespace py = pybind11;

/*

Embedding table fast path (Qwen3.8-Flash-Next PLE table, quantized token embeddings): the hot
loop stays on the CPU where the token ids already live (no device round trips), the scattered
table rows are gathered with threaded preads into a pinned staging buffer, and the trellis rows
are decoded by a memory-bound GPU kernel after one non-blocking H2D copy.

  ngram_hash_cpu:   eos-segmented n-gram hashing + global dedup: emits sorted unique table row
                    ids, the inverse map for the output gather, and each unique row's hash head.
                    Exact port of the torch reference (NGramEmbedding.compute_ngram_ids +
                    torch.unique), which the HF parity test pins down bitwise.
  ngram_gather_cpu: threaded pread of the unique rows (run-coalesced: unique ids are sorted, so
                    adjacent rows merge into single reads) into a caller buffer. This file holds
                    the Linux version; ngram_gather_win.cpp has the overlapped-ReadFile port.
  ngram_dequant:    one block per ring (an n-gram row, or one group of a token embedding row):
                    unpack the tail-biting ring bitstream, decode the mul1 codebook value per
                    element and apply the fp16 scale, then either add the per-head bias or
                    rotate the ring back and apply its sign vector. Matches dequant_rows() in
                    ngram_codec.py (fp16 codebook rounding included).

*/

#define MAX_RING_DIM 256
#define MUL1 0x83DCD12Du

int64_t ngram_hash_cpu
(
    const at::Tensor& ids,           // (bsz, ctx + seq) int64 CPU
    int64_t seq_len,
    const at::Tensor& multipliers,   // (ngram_size) int64 CPU
    const at::Tensor& offsets,       // (num_heads) int64 CPU
    const at::Tensor& sizes,         // (num_heads) int64 CPU
    int64_t heads_per_ngram,
    int64_t eos_token,
    at::Tensor uids,                 // (cap) int64 CPU out, cap >= bsz * seq * num_heads
    at::Tensor inverse,              // (bsz * seq * num_heads) int64 CPU out
    at::Tensor heads                 // (cap) int32 CPU out
)
{
    TORCH_CHECK(ids.device().is_cpu() && ids.dtype() == at::kLong && ids.is_contiguous(),
                "ngram_hash_cpu: ids must be contiguous int64 CPU");
    TORCH_CHECK(ids.dim() == 2, "ngram_hash_cpu: ids must be (bsz, ctx + seq)");
    // Host code reads and writes these through raw pointers
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*> { &multipliers, &offsets, &sizes, &uids, &inverse })
        TORCH_CHECK(t->device().is_cpu() && t->dtype() == at::kLong && t->is_contiguous(),
                    "ngram_hash_cpu: multipliers, offsets, sizes, uids and inverse must be contiguous int64 CPU");
    TORCH_CHECK(heads.device().is_cpu() && heads.dtype() == at::kInt && heads.is_contiguous(),
                "ngram_hash_cpu: heads must be contiguous int32 CPU");
    int64_t bsz = ids.size(0);
    int64_t T = ids.size(1);
    int64_t ctx = T - seq_len;
    int64_t ngram_size = multipliers.numel();
    int64_t H = offsets.numel();
    TORCH_CHECK(seq_len >= 0 && ctx >= 0 && H == (ngram_size - 1) * heads_per_ngram, "ngram_hash_cpu: dims");
    TORCH_CHECK(sizes.numel() == H, "ngram_hash_cpu: sizes must have one entry per head");
    int64_t n = bsz * seq_len * H;
    TORCH_CHECK(uids.numel() >= n && inverse.numel() >= n && heads.numel() >= n,
                "ngram_hash_cpu: output buffers too small");

    const int64_t* idp = (const int64_t*) ids.data_ptr();
    const int64_t* mult = (const int64_t*) multipliers.data_ptr();
    const int64_t* offs = (const int64_t*) offsets.data_ptr();
    const int64_t* szs = (const int64_t*) sizes.data_ptr();
    int64_t* uids_p = (int64_t*) uids.data_ptr();
    int64_t* inv_p = (int64_t*) inverse.data_ptr();
    int32_t* heads_p = (int32_t*) heads.data_ptr();
    for (int64_t h = 0; h < H; ++h)
        TORCH_CHECK(szs[h] > 0, "ngram_hash_cpu: empty hash table for head ", h);
    if (!n) return 0;

    py::gil_scoped_release release;

    struct HK { int64_t hash; int64_t idx; };
    std::vector<HK> hk((size_t) n);

    for (int64_t b = 0; b < bsz; ++b)
    {
        const int64_t* row = idp + b * T;
        int64_t prev_eos = -1;   // last eos position strictly before p
        for (int64_t p = 0; p < T; ++p)
        {
            if (p > 0 && row[p - 1] == eos_token) prev_eos = p - 1;
            if (p >= ctx)
            {
                int64_t seg_start = prev_eos + 1;
                uint64_t mixed = (uint64_t) row[p] * (uint64_t) mult[0];
                int64_t* out = inv_p + (b * seq_len + (p - ctx)) * H;
                for (int64_t s = 1; s < ngram_size; ++s)
                {
                    int64_t src = (p - s >= seg_start && p - s >= 0) ? row[p - s] : eos_token;
                    mixed ^= (uint64_t) src * (uint64_t) mult[s];
                    int64_t lo = (s - 1) * heads_per_ngram;
                    for (int64_t h = lo; h < lo + heads_per_ngram; ++h)
                    {
                        int64_t m = (int64_t) mixed % szs[h];
                        if (m < 0) m += szs[h];
                        int64_t idx = (b * seq_len + (p - ctx)) * H + h;
                        hk[(size_t) idx] = { m + offs[h], idx };
                        (void) out;
                    }
                }
            }
        }
    }

    std::sort(hk.begin(), hk.end(), [](const HK& a, const HK& b) { return a.hash < b.hash; });

    int64_t u = -1;
    int64_t last = -1;
    for (int64_t i = 0; i < n; ++i)
    {
        if (u < 0 || hk[(size_t) i].hash != last)
        {
            last = hk[(size_t) i].hash;
            uids_p[++u] = last;
            // hash head: the offsets are ascending starts of disjoint per-head ranges
            int64_t h = (int64_t) (std::upper_bound(offs, offs + H, last) - offs) - 1;
            heads_p[u] = (int32_t) h;
        }
        inv_p[hk[(size_t) i].idx] = u;
    }
    return u + 1;
}

// Persistent gather pool. Decode-time gathers are a handful of scattered rows whose COLD
// latency dominates: they must all be in flight concurrently (a sequential loop pays one full
// disk latency per row -- measured 16 x ~78us = 1.25ms/token cold). Spawning threads per call
// would cost more than the reads, so lazily started workers pull (offset, dst, bytes) tasks
// from a queue: one task per coalesced run for small gathers, chunked run groups for prefill-
// sized ones. The pool object is intentionally leaked (never destructed) -- joining or
// destroying a static condvar at process exit deadlocks (see the stloader shutdown notes).

#ifndef _WIN32

namespace
{

struct GatherCtx
{
    int fd;
    int64_t base_offset;
    int64_t row_bytes;
    int64_t uid_base;
    const int64_t* up;
    uint8_t* op;
    std::atomic<bool> failed { false };
};

struct GatherPool
{
    std::mutex mx;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    GatherCtx* ctx = nullptr;
    std::vector<std::pair<int64_t, int64_t>> tasks;   // uid-index spans [i0, i1)
    size_t next = 0;
    size_t pending = 0;
    bool started = false;
    std::mutex call_mx;

    static void read_span(GatherCtx* c, int64_t i0, int64_t i1)
    {
        int64_t i = i0;
        while (i < i1)
        {
            // coalesce a run of consecutive rows into one read
            int64_t j = i + 1;
            while (j < i1 && c->up[j] == c->up[j - 1] + 1) ++j;
            int64_t off = c->base_offset + (c->up[i] - c->uid_base) * c->row_bytes;
            int64_t nbytes = (j - i) * c->row_bytes;
            int64_t pos = 0;
            while (pos < nbytes)
            {
                ssize_t got = pread(c->fd, c->op + i * c->row_bytes + pos,
                                    (size_t) (nbytes - pos), (off_t) (off + pos));
                if (got <= 0) { c->failed.store(true); return; }
                pos += got;
            }
            i = j;
        }
    }

    void worker()
    {
        std::unique_lock<std::mutex> lk(mx);
        while (true)
        {
            cv_work.wait(lk, [&] { return next < tasks.size(); });
            auto span = tasks[next++];
            GatherCtx* c = ctx;
            lk.unlock();
            read_span(c, span.first, span.second);
            lk.lock();
            if (--pending == 0)
                cv_done.notify_all();
        }
    }

    void run(GatherCtx* c, std::vector<std::pair<int64_t, int64_t>>& new_tasks)
    {
        std::lock_guard<std::mutex> call_lock(call_mx);
        std::unique_lock<std::mutex> lk(mx);
        if (!started)
        {
            started = true;
            int n = (int) std::min(32u, std::max(4u, std::thread::hardware_concurrency()));
            for (int i = 0; i < n; ++i)
                std::thread(&GatherPool::worker, this).detach();
        }
        ctx = c;
        tasks.swap(new_tasks);
        next = 0;
        pending = tasks.size();
        cv_work.notify_all();
        // The caller drains the queue too: page-cache-warm gathers (~us per read) then finish
        // in-line instead of stalling ~100us on worker wake/sleep latency, while cold gathers
        // still fan out across whichever workers wake in time
        while (next < tasks.size())
        {
            auto span = tasks[next++];
            lk.unlock();
            read_span(c, span.first, span.second);
            lk.lock();
            if (--pending == 0)
                cv_done.notify_all();
        }
        cv_done.wait(lk, [&] { return pending == 0; });
        tasks.clear();
        ctx = nullptr;
    }
};

GatherPool* gather_pool()
{
    static GatherPool* pool = new GatherPool();   // leaked: see note above
    return pool;
}

}  // namespace

void ngram_gather_cpu
(
    int64_t fd,
    int64_t base_offset,
    int64_t row_bytes,
    const at::Tensor& uids,          // (U) int64 CPU, sorted ascending
    int64_t uid_base,                // subtracted from uids (shard-local row index)
    at::Tensor out                   // (U, row_bytes / itemsize) CPU, contiguous
)
{
    TORCH_CHECK(uids.device().is_cpu() && uids.dtype() == at::kLong && uids.is_contiguous(),
                "ngram_gather_cpu: uids must be contiguous int64 CPU");
    TORCH_CHECK(out.device().is_cpu() && out.is_contiguous(), "ngram_gather_cpu: bad out");
    int64_t U = uids.numel();
    TORCH_CHECK(out.size(0) >= U && out.size(1) * out.element_size() == row_bytes,
                "ngram_gather_cpu: out shape");
    if (!U || !row_bytes) return;
    const int64_t* up = (const int64_t*) uids.data_ptr();
    uint8_t* op = (uint8_t*) out.data_ptr();

    py::gil_scoped_release release;

    GatherCtx ctx { (int) fd, base_offset, row_bytes, uid_base, up, op };

    // Coalesce consecutive rows into runs
    std::vector<std::pair<int64_t, int64_t>> runs;   // (first index, count)
    runs.reserve(64);
    for (int64_t i = 0; i < U; )
    {
        int64_t j = i + 1;
        while (j < U && up[j] == up[j - 1] + 1) ++j;
        runs.emplace_back(i, j - i);
        i = j;
    }
    size_t n_runs = runs.size();

    if (n_runs == 1)
    {
        // single contiguous read: no pool round trip
        GatherPool::read_span(&ctx, 0, U);
        TORCH_CHECK(!ctx.failed.load(), "ngram_gather_cpu: short read");
        return;
    }

    // One span task per run keeps every cold read in flight at once; past ~64 runs, group
    // whole runs into span tasks so the queue overhead stays negligible (a worker issues its
    // group's runs sequentially -- pool-width reads still in flight)
    std::vector<std::pair<int64_t, int64_t>> tasks;
    size_t group = (n_runs + 63) / 64;
    tasks.reserve((n_runs + group - 1) / group);
    for (size_t r = 0; r < n_runs; r += group)
    {
        size_t r1 = std::min(n_runs, r + group);
        tasks.emplace_back(runs[r].first, runs[r1 - 1].first + runs[r1 - 1].second);
    }
    gather_pool()->run(&ctx, tasks);
    TORCH_CHECK(!ctx.failed.load(), "ngram_gather_cpu: short read");
}

#endif  // ^ Linux (pread); the Windows version (overlapped ReadFile) is in ngram_gather_win.cpp

__global__ __launch_bounds__(MAX_RING_DIM)
void ngram_dequant_kernel
(
    const int16_t* __restrict__ packed,   // (U, 1 + dim * K / 16)
    const int32_t* __restrict__ heads,    // (U) row of aux per ring, or null for ring index mod num_heads
    const half* __restrict__ aux,         // (num_heads, dim) bias, or signs when rotating; may be null
    half* __restrict__ out,               // (U, dim)
    const int K,
    const int words,                      // 1 + dim * K / 16
    const int num_heads,
    const bool rotate
)
{
    const int r = blockIdx.x;
    const int i = threadIdx.x;
    const int dim = blockDim.x;

    extern __shared__ float sf[];
    uint16_t* sw = (uint16_t*) (sf + dim);
    if (i < words) sw[i] = (uint16_t) packed[(size_t) r * words + i];
    __syncthreads();

    float scale = __half2float(__ushort_as_half(sw[0]));

    // stream bit m of element i lives at ring position ((i - m / K) mod dim) * K + m % K
    uint32_t state = 0;
    #pragma unroll 4
    for (int m = 0; m < 16; ++m)
    {
        int pos = i - m / K;
        if (pos < 0) pos += dim;
        int sb = pos * K + m % K;
        uint32_t bit = (sw[1 + (sb >> 4)] >> (sb & 15)) & 1;
        state |= bit << m;
    }

    // mul1 codebook value, rounded to fp16 like the cached table
    uint32_t prod = state * MUL1;
    float h = 1024.0f + (float) ((prod & 0xff) + ((prod >> 8) & 0xff) +
                                 ((prod >> 16) & 0xff) + ((prod >> 24) & 0xff));
    float k_inv = __half2float(__ushort_as_half(0x1eee));
    float k_bias = __half2float(__ushort_as_half(0xc931));
    float v = __half2float(__float2half_rn(h * k_inv + k_bias)) * scale;

    float a = 0.0f;
    if (aux)
    {
        // The loader validates the bias row count against the head count; the bound here keeps a
        // bad head id from reading outside aux regardless (issue #525)
        int head = heads ? heads[r] : r % num_heads;
        head = max(0, min(head, num_heads - 1));
        a = __half2float(aux[(size_t) head * dim + i]);
    }

    if (rotate)
    {
        // Walsh-Hadamard transform across the block, one butterfly stage per bit of the index
        sf[i] = v;
        __syncthreads();
        for (int s = 1; s < dim; s <<= 1)
        {
            float lo = sf[i & ~s];
            float hi = sf[i | s];
            __syncthreads();
            sf[i] = (i & s) ? lo - hi : lo + hi;
            __syncthreads();
        }
        v = sf[i] * (1.0f / sqrtf((float) dim));
        if (aux) v *= a;
    }
    else v += a;

    out[(size_t) r * dim + i] = __float2half_rn(v);
}

void ngram_dequant
(
    const at::Tensor& packed,                   // (U, 1 + dim * K / 16) int16 CUDA
    int64_t K,
    const c10::optional<at::Tensor>& heads,     // (U) int32 CUDA
    const c10::optional<at::Tensor>& aux,       // (num_heads, dim) half CUDA
    at::Tensor out,                             // (U, dim) half CUDA
    bool rotate
)
{
    const at::cuda::OptionalCUDAGuard device_guard(packed.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(packed, kShort);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(packed.is_contiguous() && out.is_contiguous(), "ngram_dequant: contiguous inputs required");
    int64_t U = packed.size(0);
    int words = (int) packed.size(1);
    int dim = (int) out.size(1);
    TORCH_CHECK(1 <= K && K <= 8 && words == 1 + dim * (int) K / 16, "ngram_dequant: bad K/words");
    TORCH_CHECK(dim <= MAX_RING_DIM && dim % 16 == 0 && out.size(0) >= U, "ngram_dequant: shapes");
    TORCH_CHECK(!rotate || (dim & (dim - 1)) == 0, "ngram_dequant: rotation needs a power of two");
    if (heads)
    {
        TORCH_CHECK_DTYPE((*heads), kInt);
        TORCH_CHECK(aux && heads->is_contiguous() && heads->numel() >= U, "ngram_dequant: heads");
    }
    if (aux)
    {
        TORCH_CHECK_DTYPE((*aux), kHalf);
        TORCH_CHECK(aux->is_contiguous() && aux->dim() == 2 && aux->size(1) == dim, "ngram_dequant: aux shape");
        TORCH_CHECK(!U || aux->size(0) > 0, "ngram_dequant: aux is empty");
    }
    if (!U || !dim) return;

    size_t smem = dim * sizeof(float) + ((words + 1) & ~1) * sizeof(uint16_t);
    ngram_dequant_kernel<<<(unsigned int) U, dim, smem, stream>>>
    (
        (const int16_t*) packed.data_ptr(),
        heads ? (const int32_t*) heads->data_ptr() : nullptr,
        aux ? (const half*) aux->data_ptr() : nullptr,
        (half*) out.data_ptr(),
        (int) K,
        words,
        aux ? (int) aux->size(0) : 1,
        rotate
    );
    cuda_check(cudaPeekAtLastError());
}

// ---- PQ (product-quantized n-gram table) gather + decode, GX10 patch -------------------------
//
// A PQ table (util/_pq_to_pack.py) serves rows as M uint8 segment codes; a row decodes as
// M per-segment 8-bit centroid lookups (fp16 or bf16 values) scaled by the PLE fp8
// weight_scale. The reference form (ngram_embedding._pq_decode_rows) runs M index_selects
// into an fp32 buffer plus a scale and a cast -- 2M+2 launches and a U x (M * seg) fp32
// intermediate per chunk. These kernels fold gather + decode into a single launch that
// writes the final dtype:
//
//   pq_decode_kernel (packed):  row codes arrive in the staging buffer (streamed/disk or RAM
//                               tables) after the H2D copy
//   pq_decode_kernel (table):   row codes are read straight from the device-resident table via
//                               uids -- no intermediate row buffer exists at all
//
// One block per unique row, M * seg threads (160 for M=5/seg=32): thread i = m * seg + s
// decodes codebook[(head * M + m) * K + code_m][s] * scale. The codebooks (heads * M * K * seg
// values, 1.3 MiB for Qwen3.8-Flash-Next) stay L2-resident; every thread of one m reads the
// same code byte, and consecutive s threads read one 64 B codebook span, so both gathers are
// broadcast/coalesced. The codebook element type is templated: the pack ships bf16 centroids,
// but conversions that emit fp16 or fp32 codebooks must decode bit-identically too.

#define MAX_PQ_DIM 1024

template <typename T>
__device__ __forceinline__ float pq_cb_to_float(T v) { return (float) v; }
template <>
__device__ __forceinline__ float pq_cb_to_float(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float pq_cb_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__global__ __launch_bounds__(MAX_PQ_DIM)
void pq_decode_kernel
(
    const uint8_t* __restrict__ packed,     // (U, M) uint8, per-segment codes (packed path)
    const uint8_t* __restrict__ table,      // (rows, M) uint8, device-resident table (gather path)
    const int64_t* __restrict__ uids,       // (U) int64, gather path; unused when table == null
    const int32_t* __restrict__ heads,      // (U) int32, table head of each row
    const T* __restrict__ codebooks,        // (heads * M * K, seg), half/bf16/float
    void* __restrict__ out,                 // (U, M * seg) half or bf16
    const int out_bf16,
    const int M,
    const int K,
    const int seg,
    const float scale
)
{
    const int r = blockIdx.x;
    const int i = threadIdx.x;
    const int m = i / seg;
    const int s = i - m * seg;
    const int head = heads[r];
    const uint8_t code = (table != nullptr)
        ? table[(size_t) uids[r] * (size_t) M + (size_t) m]
        : packed[(size_t) r * (size_t) M + (size_t) m];
    const float v = pq_cb_to_float(codebooks[((size_t) (head * M + m) * (size_t) K + code) * (size_t) seg + s]) * scale;
    if (out_bf16)
        ((__nv_bfloat16*) out)[(size_t) r * (size_t) (M * seg) + (size_t) i] = __float2bfloat16(v);
    else
        ((half*) out)[(size_t) r * (size_t) (M * seg) + (size_t) i] = __float2half_rn(v);
}

static void pq_decode_launch
(
    const void* packed,
    const void* table,
    const int64_t* uids,
    const int32_t* heads,
    const void* codebooks,
    c10::ScalarType cb_dtype,
    void* out,
    bool out_bf16,
    int M,
    int K,
    int seg,
    float scale,
    int64_t U,
    cudaStream_t stream
)
{
    const unsigned int grid = (unsigned int) U;
    const unsigned int dim = (unsigned int) (M * seg);
    switch (cb_dtype)
    {
        case at::kHalf:
            pq_decode_kernel<__half><<<grid, dim, 0, stream>>>
            (
                (const uint8_t*) packed, (const uint8_t*) table, uids, heads,
                (const __half*) codebooks, out, out_bf16 ? 1 : 0, M, K, seg, scale
            );
            break;
        case at::kBFloat16:
            pq_decode_kernel<__nv_bfloat16><<<grid, dim, 0, stream>>>
            (
                (const uint8_t*) packed, (const uint8_t*) table, uids, heads,
                (const __nv_bfloat16*) codebooks, out, out_bf16 ? 1 : 0, M, K, seg, scale
            );
            break;
        case at::kFloat:
            pq_decode_kernel<float><<<grid, dim, 0, stream>>>
            (
                (const uint8_t*) packed, (const uint8_t*) table, uids, heads,
                (const float*) codebooks, out, out_bf16 ? 1 : 0, M, K, seg, scale
            );
            break;
        default:
            TORCH_CHECK(false, "pq_decode: codebooks must be half, bfloat16 or float");
    }
}

void ngram_pq_decode
(
    const at::Tensor& packed,               // (U, M) uint8 CUDA, contiguous
    const at::Tensor& heads,                // (U) int32 CUDA, contiguous
    const at::Tensor& codebooks,            // (heads * M * K, seg) half/bf16/float CUDA
    at::Tensor out,                         // (U, M * seg) half or bf16 CUDA
    int64_t K,
    double scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(packed.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(packed.device().is_cuda() && heads.device().is_cuda() && codebooks.device().is_cuda()
                && out.device().is_cuda(), "ngram_pq_decode: CUDA tensors required");
    TORCH_CHECK(packed.dtype() == at::kByte && heads.dtype() == at::kInt
                && (codebooks.dtype() == at::kHalf || codebooks.dtype() == at::kBFloat16 || codebooks.dtype() == at::kFloat)
                && (out.dtype() == at::kHalf || out.dtype() == at::kBFloat16), "ngram_pq_decode: dtypes");
    TORCH_CHECK(packed.is_contiguous() && heads.is_contiguous() && codebooks.is_contiguous()
                && out.is_contiguous(), "ngram_pq_decode: contiguous inputs required");
    TORCH_CHECK(packed.dim() == 2 && codebooks.dim() == 2 && out.dim() == 2, "ngram_pq_decode: shapes");
    const int64_t U = packed.size(0);
    const int M = (int) packed.size(1);
    const int seg = (int) codebooks.size(1);
    const int dim = M * seg;
    TORCH_CHECK(K > 0 && codebooks.size(0) % ((int64_t) M * K) == 0, "ngram_pq_decode: codebook shape");
    TORCH_CHECK(heads.numel() >= U && out.size(0) >= U && (int) out.size(1) == dim, "ngram_pq_decode: out shape");
    TORCH_CHECK(dim > 0 && dim <= MAX_PQ_DIM, "ngram_pq_decode: dim");
    if (!U) return;

    pq_decode_launch(packed.data_ptr(), nullptr, nullptr, (const int32_t*) heads.data_ptr(),
                     codebooks.data_ptr(), codebooks.scalar_type(), out.data_ptr(),
                     out.dtype() == at::kBFloat16, M, (int) K, seg, (float) scale, U, stream);
    cuda_check(cudaPeekAtLastError());
}

void ngram_pq_gather_decode
(
    const at::Tensor& table,                // (rows, M) uint8 CUDA, contiguous; on-device table
    const at::Tensor& uids,                 // (U) int64 CUDA, contiguous
    const at::Tensor& heads,                // (U) int32 CUDA, contiguous
    const at::Tensor& codebooks,            // (heads * M * K, seg) half/bf16/float CUDA
    at::Tensor out,                         // (U, M * seg) half or bf16 CUDA
    int64_t K,
    double scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(table.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(table.device().is_cuda() && uids.device().is_cuda() && heads.device().is_cuda()
                && codebooks.device().is_cuda() && out.device().is_cuda(), "ngram_pq_gather_decode: CUDA tensors required");
    TORCH_CHECK(table.dtype() == at::kByte && uids.dtype() == at::kLong && heads.dtype() == at::kInt
                && (codebooks.dtype() == at::kHalf || codebooks.dtype() == at::kBFloat16 || codebooks.dtype() == at::kFloat)
                && (out.dtype() == at::kHalf || out.dtype() == at::kBFloat16), "ngram_pq_gather_decode: dtypes");
    TORCH_CHECK(table.is_contiguous() && uids.is_contiguous() && heads.is_contiguous()
                && codebooks.is_contiguous() && out.is_contiguous(), "ngram_pq_gather_decode: contiguous inputs required");
    TORCH_CHECK(table.dim() == 2 && codebooks.dim() == 2 && out.dim() == 2, "ngram_pq_gather_decode: shapes");
    const int64_t U = uids.numel();
    const int M = (int) table.size(1);
    const int seg = (int) codebooks.size(1);
    const int dim = M * seg;
    TORCH_CHECK(K > 0 && codebooks.size(0) % ((int64_t) M * K) == 0, "ngram_pq_gather_decode: codebook shape");
    TORCH_CHECK(heads.numel() >= U && out.size(0) >= U && (int) out.size(1) == dim, "ngram_pq_gather_decode: out shape");
    TORCH_CHECK(dim > 0 && dim <= MAX_PQ_DIM, "ngram_pq_gather_decode: dim");
    if (!U) return;

    pq_decode_launch(nullptr, table.data_ptr(), (const int64_t*) uids.data_ptr(),
                     (const int32_t*) heads.data_ptr(), codebooks.data_ptr(), codebooks.scalar_type(),
                     out.data_ptr(), out.dtype() == at::kBFloat16, M, (int) K, seg, (float) scale, U, stream);
    cuda_check(cudaPeekAtLastError());
}
