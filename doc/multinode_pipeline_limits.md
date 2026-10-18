# Multi-node pipeline transport limits

## Trusted-fabric boundary

TCP framing is **not authentication or encryption**. Use `NetEndpoint` and the
example only on a trusted, access-controlled fabric. The example listens on all
interfaces; firewall its ports and do not expose them to the Internet. Distributed
process-group membership is not a replacement for network access control.

## Finite frame and allocation limits

Both endpoint constructors accept keyword-only `max_tensor_bytes` (default
1 GiB) and `max_object_bytes` (default 16 MiB). `NetEndpoint.listen` and
`NetEndpoint.connect` accept and propagate the same limits. Values must be
positive integers at most the signed int64 maximum; set identical limits at both
ends, and lower them to your deployment's expected frames and memory budget.
The defaults accommodate ordinary chunked PP hidden-state transfers, not an
unbounded full-context transfer or a guarantee of host/GPU memory availability.

Tensor metadata is checked before receive allocation or payload I/O: 0..5
dimensions, nonnegative signed-int64 dimensions, bounded product arithmetic, and
an exact shape × dtype-size payload byte count, including empty tensors and
scalars. Even empty tensors bound their nonzero geometry (treating each zero
as one for the geometry limit), to avoid torch stride/product overflow. Sends
validate before making contiguous/staging/device copies. JSON output is encoded
incrementally and capped before any wire send; peer object lengths are checked
before receive allocation. Limits are per frame, not a total process-memory cap:
output tensors, staging, JSON parsing and stride-copy temporaries can coexist.

Matching strided/transposed receive outputs are supported with a contiguous
temporary and stride-aware copy. CPU scalars round-trip across the supported
dtypes. Bad framing poisons the endpoint rather than allowing the next frame to
be misinterpreted. TCP closes its socket; distributed callers must terminate or
destroy the process group after an error (there is no payload-drain/recovery).

## Deadlines and cancellation

`NetEndpoint(..., io_timeout=60.0)` configures a finite positive socket I/O
budget. `listen` and `connect` also accept `io_timeout`; their `timeout` remains
the accept/connect budget, independently of established I/O. Every connect
attempt uses the remaining monotonic deadline, failed attempts use fresh sockets,
and late connection success is rejected. Use numeric IP addresses when a hard
connect budget is required: OS hostname resolution is not bounded by Python's
socket timeout.

Each header or payload send/receive loop has its own absolute `io_timeout`
deadline, so trickled bytes cannot keep that loop alive indefinitely. Socket EOF,
timeout, or send/receive failure closes and poisons the endpoint. Another thread
may call `close()` to cancel blocked socket I/O; endpoints do not support
concurrent framing operations. Choose a larger **finite** I/O timeout explicitly
for stages that legitimately take more than 60 seconds before producing data.
This timeout does not bound model compute, CUDA copies/events, or JSON parsing.
Distributed I/O uses the process group's configured timeout, not TCP's setting.

`NcclEndpoint.send_tensor/recv_tensor` use the caller's current-stream contract.
Explicit `stream` arguments are rejected before sending or consuming a header.
Make producer data ready on the current stream; the endpoint does not infer
ordering from a tensor produced on an unrelated stream.

## Driver family and context limits

The stage helper retains persistent recurrent state, advances it once per slice
forward using the canonical library utility, and frees it on reset/shutdown; its
lifecycle is covered on CPU. The driver admits
`Glm5NextForConditionalGeneration` automatically because that architecture uses
this lifecycle. Its exact-checkpoint GPU quality is still unqualified until a
matching hardware run compares multi-token prefill, decode and NLL against a
trusted single-model baseline. Every other model advertising
`caps["recurrent_states"]` is rejected before cache allocation, weights, CUDA
setup or network links, including Qwen3.5 and unknown recurrent families.
Nonrecurrent models must also obey the existing DSA full-indexer split
restrictions; this change is not a general recurrent-family qualification.

`ctx` and `chunk` must be positive and `max_new` nonnegative. Empty or over-context
prompts are rejected. NLL inputs require 2..ctx tokens and are rejected rather
than silently truncated. Every stage checks `past_len + input_length <= ctx`
before preparing inputs or executing its slice. Generation stops before a next
forward would exceed context and logs `finish=context_limit`, `eos`, or `max_new`.
A token predicted by the final capacity-fitting forward may be returned without
being fed back into the cache. `max_new=0` performs no generation forwards and
still sends the stop control message.

## CPU regression gate

```bash
python -m pytest tests/model/test_net_transport.py tests/model/test_nccl_transport.py \
    tests/model/test_transport_bounds.py tests/model/test_multinode_driver_bounds.py \
    tests/model/test_multinode_pipeline.py -q
```

These tests exercise real localhost TCP and two-process CPU Gloo round-trips,
invalid-allocation interception, and production driver AST with CPU model leaves.
They do not establish CUDA/NCCL ordering, pinned-memory correctness, full-model
numerical parity, throughput, or four-host readiness. The existing CUDA staging
test returns without execution on CPU-only builds; run the separate GPU gate
before treating that path as qualified.
