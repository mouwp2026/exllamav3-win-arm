"""
Point-to-point tensor transport over torch.distributed (NCCL on GPUs, gloo on CPU), with the same
send_tensor / recv_tensor / send_obj / recv_obj interface as NetEndpoint.

On hosts linked by RDMA-capable NICs (e.g. DGX Spark ConnectX-7 over QSFP, RoCE), NCCL moves the
tensor GPU to GPU without host staging or Python socket loops. Measured on four GB10s on one QSFP
switch (MTU 1500): a 24 KB fp32 hidden state crosses one hop in ~21 us (4-hop ring 83 us), against
~1.1 ms per hop for NetEndpoint over TCP on the same link. Pin the link with NCCL_SOCKET_IFNAME (the
bootstrap interface) and NCCL_IB_HCA (the RoCE device, e.g. rocep1s0f0); NCCL_DEBUG=INFO should
report "via NET/IB". NCCL_IB_DISABLE=1 (socket transport) measured ~1.4 ms per small all-reduce.

Every rank of the job joins one process group (init_group); an endpoint is a (peer rank, device)
pair. Messages are framed by a fixed int64 header, so tensors of any shape and dtype in
_DTYPE_TO_CODE round-trip bit-exactly.
"""

from __future__ import annotations

import json
from typing import Any

import torch
import torch.distributed as dist


class NcclTransportError(Exception):
    pass


class NcclEndpoint:

    _DTYPE_TO_CODE = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
        torch.int32: 3,
        torch.int64: 4,
        torch.uint8: 5,
        torch.bool: 6,
    }
    if hasattr(torch, "float8_e4m3fn"):
        _DTYPE_TO_CODE[torch.float8_e4m3fn] = 7
    _CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}

    _MAGIC = 0x4E43434C  # "NCCL"
    _MSG_TENSOR = 0
    _MSG_OBJECT = 1
    # Finite per-frame limits, configurable per endpoint. Empty shapes also bound
    # their nonzero geometry to prevent torch stride/product overflow.
    max_tensor_bytes = 1024 * 1024 * 1024
    max_object_bytes = 16 * 1024 * 1024
    # header: magic, msg_type, dtype_code, ndim, d0..d4, payload_len
    _HEADER_LEN = 10

    def __init__(self, peer: int, device: torch.device | str | int | None = None, group = None,
                 *, max_tensor_bytes: int = 1024 * 1024 * 1024, max_object_bytes: int = 16 * 1024 * 1024):
        """
        :param peer:
            Global rank of the other end.

        :param device:
            Device the communication buffers live on: the rank's CUDA device under NCCL, ignored
            (CPU) under gloo.
        """
        if not dist.is_initialized():
            raise NcclTransportError("torch.distributed is not initialized; call NcclEndpoint.init_group first")
        self.max_tensor_bytes = self._validate_limit(max_tensor_bytes)
        self.max_object_bytes = self._validate_limit(max_object_bytes)
        self._closed = False
        self.peer = peer
        self.group = group
        backend = dist.get_backend(group)
        self.is_nccl = backend == "nccl"
        if self.is_nccl:
            if device is None:
                device = torch.device("cuda", torch.cuda.current_device())
            self.comm_device = torch.device(device)
        else:
            self.comm_device = torch.device("cpu")
        self._hdr_send = torch.zeros(self._HEADER_LEN, dtype = torch.long, device = self.comm_device)
        self._hdr_recv = torch.zeros(self._HEADER_LEN, dtype = torch.long, device = self.comm_device)

    @staticmethod
    def init_group(
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        device: torch.device | str | int | None = None,
        backend: str | None = None,
        timeout_s: float = 3600,
    ):
        """
        Join (or create, on rank 0's host) the job-wide process group. Every rank calls this once
        before creating endpoints.
        """
        import datetime
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        kwargs = {}
        if backend == "nccl" and device is not None:
            kwargs["device_id"] = torch.device(device)
        dist.init_process_group(
            backend,
            rank = rank,
            world_size = world_size,
            init_method = f"tcp://{master_addr}:{master_port}",
            timeout = datetime.timedelta(seconds = timeout_s),
            **kwargs,
        )

    def _prepare_header(self, msg_type: int, dtype_code: int, shape: tuple, payload_len: int):
        if len(shape) > 5:
            raise NcclTransportError(f"Tensor has {len(shape)} dimensions; max 5 supported")
        dims = list(shape) + [0] * (5 - len(shape))
        h = [self._MAGIC, msg_type, dtype_code, len(shape), *dims, payload_len]
        self._hdr_send.copy_(torch.tensor(h, dtype = torch.long))

    def _recv_header(self, expect_type: int):
        self._ensure_open()
        dist.recv(self._hdr_recv, self.peer, group = self.group)
        h = self._hdr_recv.tolist()
        if h[0] != self._MAGIC:
            self._fail(f"Bad magic: {hex(h[0])}")
        if h[1] != expect_type:
            self._fail(f"Expected message type {expect_type}, got {h[1]}")
        return h

    @staticmethod
    def _validate_limit(n):
        if type(n) is not int or not 0 < n <= (1 << 63) - 1:
            raise NcclTransportError("Byte limits must be positive integers no greater than int64 max")
        return n

    def _ensure_open(self):
        if getattr(self, "_closed", False):
            raise NcclTransportError("Endpoint is closed or poisoned")

    def _fail(self, message):
        self.close()
        raise NcclTransportError(message)

    def _tensor_nbytes(self, shape, dtype_code):
        if not 0 <= len(shape) <= 5:
            raise NcclTransportError(f"Invalid tensor ndim: {len(shape)}")
        if dtype_code not in self._CODE_TO_DTYPE:
            raise NcclTransportError(f"Unknown dtype code: {dtype_code}")
        itemsize = (2, 2, 4, 4, 8, 1, 1, 1)[dtype_code]
        product = 1
        for d in shape:
            if d < 0 or d > (1 << 63) - 1 or product > (self.max_tensor_bytes // itemsize) // max(1, d):
                raise NcclTransportError(f"Tensor shape exceeds byte limit or overflows: {shape}")
            product *= max(1, d)
        n = 0 if 0 in shape else product * itemsize
        if n > self.max_tensor_bytes:
            raise NcclTransportError(f"Tensor exceeds byte limit: {n}")
        return n

    def _validate_tensor_header(self, ndim, dims, dtype_code, payload_len):
        self._ensure_open()
        try:
            if not 0 <= ndim <= 5:
                raise NcclTransportError(f"Invalid tensor ndim: {ndim}")
            shape = tuple(dims[:ndim])
            expected = self._tensor_nbytes(shape, dtype_code)
            if payload_len != expected:
                raise NcclTransportError(f"Payload length {payload_len} does not match shape {shape}")
        except NcclTransportError:
            self.close()
            raise
        return shape, self._CODE_TO_DTYPE[dtype_code]

    def _validate_object_length(self, n):
        self._ensure_open()
        if not 0 < n <= self.max_object_bytes:
            self._fail(f"Object payload length exceeds byte limit or is invalid: {n}")

    def _encode_obj(self, obj):
        self._ensure_open()
        payload = bytearray()
        try:
            for chunk in json.JSONEncoder().iterencode(obj):
                chunk = chunk.encode("utf-8")
                if len(payload) + len(chunk) > self.max_object_bytes:
                    raise NcclTransportError("Object exceeds byte limit")
                payload.extend(chunk)
        except (TypeError, ValueError) as e:
            raise NcclTransportError(f"Object not JSON-serializable: {e}") from e
        return payload

    def send_tensor(self, t: torch.Tensor, stream = None) -> None:
        """Send on the caller's current stream (any device; moved if needed).
        Explicit stream arguments are unsupported; make producer data ready on
        the current stream before calling. No cross-stream ordering is inferred."""
        if stream is not None:
            raise NcclTransportError("Explicit stream is unsupported; use the current stream and stream=None")
        if t.dtype not in self._DTYPE_TO_CODE:
            raise NcclTransportError(f"Unsupported dtype: {t.dtype}")
        self._ensure_open()
        payload_len = self._tensor_nbytes(tuple(t.shape), self._DTYPE_TO_CODE[t.dtype])
        try:
            payload = t.resolve_conj().resolve_neg().contiguous()
            if payload.device != self.comm_device:
                payload = payload.to(self.comm_device)
            # Complete the byte view and header buffer before publishing anything.
            payload = payload.view(-1).view(torch.uint8)
            self._prepare_header(self._MSG_TENSOR, self._DTYPE_TO_CODE[t.dtype], tuple(t.shape), payload_len)
        except Exception as e:
            raise NcclTransportError(f"Tensor preparation failed: {e}") from e
        try:
            dist.send(self._hdr_send, self.peer, group = self.group)
            if payload_len > 0:
                # NCCL has no fp8/bool reductions but p2p is a byte copy: send as uint8
                dist.send(payload, self.peer, group = self.group)
        except Exception as e:
            self.close()
            raise NcclTransportError(f"Tensor send failed: {e}") from e

    def recv_tensor(self, out: torch.Tensor | None = None, device = None, stream = None) -> torch.Tensor:
        """Receive on the caller's current stream; matching strided outputs are copied."""
        if stream is not None:
            raise NcclTransportError("Explicit stream is unsupported; use the current stream and stream=None")
        h = self._recv_header(self._MSG_TENSOR)
        payload_len = h[9]
        shape, dtype = self._validate_tensor_header(h[3], h[4:9], h[2], payload_len)
        target = torch.device(device) if device is not None else (out.device if out is not None else self.comm_device)
        if out is None or tuple(out.shape) != shape or out.dtype != dtype or out.device != self.comm_device or not out.is_contiguous():
            buf = torch.empty(shape, dtype = dtype, device = self.comm_device)
        else:
            buf = out
        if buf.numel() * buf.itemsize != payload_len:
            raise NcclTransportError(f"Payload length {payload_len} does not match shape {shape} {dtype}")
        if payload_len > 0:
            dist.recv(buf.view(-1).view(torch.uint8), self.peer, group = self.group)
        if buf.device != target:
            buf = buf.to(target)
        if out is not None and buf is not out and tuple(out.shape) == shape and out.dtype == dtype:
            out.copy_(buf)
            return out
        return buf

    def send_obj(self, obj: Any) -> None:
        self._ensure_open()
        try:
            payload = self._encode_obj(obj)
            t = torch.frombuffer(bytearray(payload), dtype = torch.uint8).to(self.comm_device)
            self._prepare_header(self._MSG_OBJECT, 0, (), len(payload))
        except Exception as e:
            raise NcclTransportError(f"Object preparation failed: {e}") from e
        try:
            dist.send(self._hdr_send, self.peer, group = self.group)
            if payload:
                dist.send(t, self.peer, group = self.group)
        except Exception as e:
            self.close()
            raise NcclTransportError(f"Object send failed: {e}") from e

    def recv_obj(self) -> Any:
        h = self._recv_header(self._MSG_OBJECT)
        n = h[9]
        self._validate_object_length(n)
        t = torch.empty(n, dtype = torch.uint8, device = self.comm_device)
        dist.recv(t, self.peer, group = self.group)
        try:
            return json.loads(bytes(t.cpu().numpy()).decode("utf-8"))
        except (ValueError, RecursionError, MemoryError) as e:
            # ValueError includes JSON/UTF-8 errors and integer conversion limits.
            self._fail(f"Object parse failed: {e}")

    def close(self) -> None:
        """Poison this endpoint. The caller must tear down the group after a framing error;
        this endpoint cannot drain or recover distributed payloads."""
        self._closed = True
