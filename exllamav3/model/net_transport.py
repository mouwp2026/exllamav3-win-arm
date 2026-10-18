"""
Standalone tensor transport module for cross-host pipeline mode in exllamav3.

Moves torch tensors and small Python control messages between two processes on
DIFFERENT machines over TCP, as the payload path for a pipeline split where one
node runs layers 0..k and the other k+1..N.

The transport layer handles:
- Fixed-size framing with magic, msg type, dtype code, shape, payload length
- Reliable recv loops that handle partial reads and detect EOF
- Fast CUDA tensor staging via reused pinned host memory
- Direct zero-copy recv_into from socket into pinned buffers for GPU transfers
- Support for FP8 (float8_e4m3fn) alongside float16, bfloat16, float32, int32, int64, uint8, bool
- Non-contiguous tensor handling (make contiguous before send)
- Socket buffer tuning for high-throughput networks (QSFP/100GbE)
- Clear error handling with NetTransportError
"""

from __future__ import annotations

import socket
import time
import math
import struct
import json
import io
import sys
from typing import Any

import torch


class NetTransportError(Exception):
    """Raised when framing, EOF, or timeout issues occur during transport."""
    pass


class NetEndpoint:
    """
    Manages TCP socket communication for tensor and object transport.

    Supports bidirectional send/recv of torch tensors and JSON-serializable
    control messages. CUDA tensors are staged through pinned host memory.
    Framing is deterministic; partial reads and EOF are treated as errors.
    """

    # Dtype -> code mapping (1 byte)
    _DTYPE_TO_CODE = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
        torch.int32: 3,
        torch.int64: 4,
        torch.uint8: 5,
        torch.bool: 6,
    }

    # Register float8_e4m3fn if supported in this torch build
    if hasattr(torch, "float8_e4m3fn"):
        _DTYPE_TO_CODE[torch.float8_e4m3fn] = 7

    _CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}

    # Message types
    _MSG_TENSOR = 0
    _MSG_OBJECT = 1
    # Finite per-frame limits, configurable per endpoint. Empty shapes also bound
    # their nonzero geometry to prevent torch stride/product overflow.
    max_tensor_bytes = 1024 * 1024 * 1024
    max_object_bytes = 16 * 1024 * 1024

    # Fixed header layout:
    # magic (4 bytes): 0xDEADBEEF
    # msg_type (1 byte): 0=tensor, 1=object
    # dtype_code (1 byte): 0-7 for tensor types
    # ndim (1 byte): number of dimensions
    # shape (40 bytes): 5 uint64 slots for dims (up to 5D)
    # payload_len (8 bytes): byte count of payload
    # Total: 4 + 1 + 1 + 1 + 40 + 8 = 55 bytes
    _HEADER_FMT = "!I B B B 5Q Q"
    _HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _MAGIC = 0xDEADBEEF

    def __init__(self, sock: socket.socket, is_server: bool = False, buffer_size: int = 4 * 1024 * 1024,
                 *, max_tensor_bytes: int = 1024 * 1024 * 1024, max_object_bytes: int = 16 * 1024 * 1024,
                 io_timeout: float = 60.0):
        """
        Initialize endpoint with an existing socket.

        Args:
            sock: Connected socket.socket instance.
            is_server: Whether this endpoint is the listening side (for logging/clarity).
            buffer_size: TCP send/recv buffer size in bytes (default 4MB for high-speed interconnects).
            max_tensor_bytes: Finite tensor payload/geometry cap (default 1 GiB).
            max_object_bytes: Finite JSON payload cap (default 16 MiB).
            io_timeout: Finite deadline per header/payload I/O loop (default 60s).

        TCP is unauthenticated: use only on an access-controlled trusted fabric.
        close() can cancel blocked I/O. See doc/multinode_pipeline_limits.md.
        """
        self.max_tensor_bytes = self._validate_limit(max_tensor_bytes)
        self.max_object_bytes = self._validate_limit(max_object_bytes)
        self._closed = False
        self.io_timeout = self._validate_timeout(io_timeout)
        self.sock = sock
        self.sock.settimeout(self.io_timeout)
        self.is_server = is_server

        # Low latency socket tuning
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

        # Socket buffer sizing for large prefill chunks (e.g. 100GbE / QSFP)
        if buffer_size > 0:
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
            except OSError:
                pass

        # Quick-ACK on Linux if available to minimize handshake delays
        if hasattr(socket, "TCP_QUICKACK"):
            try:
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
            except OSError:
                pass

        # Pinned staging buffer for CUDA <-> host transfers (reused across calls)
        self._pinned_buffer = None
        self._pinned_size = 0
        # Event recorded after an asynchronous host-to-device copy out of the staging buffer
        # (recv_tensor with a stream). The buffer must not be overwritten until it completes
        self._staging_event = None

    @classmethod
    def listen(
        cls,
        host: str,
        port: int,
        timeout: float = 60.0,
        buffer_size: int = 4 * 1024 * 1024,
        *, io_timeout: float = 60.0,
        max_tensor_bytes: int = 1024 * 1024 * 1024,
        max_object_bytes: int = 16 * 1024 * 1024,
    ) -> NetEndpoint:
        """
        Create a server endpoint that listens for and accepts one peer connection.

        Args:
            host: Interface to bind to (e.g., "0.0.0.0").
            port: Port to listen on.
            timeout: Accept timeout in seconds.
            buffer_size: TCP socket buffer size.

        Returns:
            NetEndpoint connected to the peer.

        Raises:
            NetTransportError: On socket/accept errors or timeout.
        """
        cls._validate_timeout(timeout)
        cls._validate_timeout(io_timeout)
        cls._validate_limit(max_tensor_bytes)
        cls._validate_limit(max_object_bytes)
        server_sock = None
        peer_sock = None
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((host, port))
            server_sock.listen(1)
            server_sock.settimeout(timeout)

            try:
                peer_sock, peer_addr = server_sock.accept()
            finally:
                server_sock.close()

            return cls(peer_sock, is_server = True, buffer_size = buffer_size, io_timeout = io_timeout,
                       max_tensor_bytes = max_tensor_bytes, max_object_bytes = max_object_bytes)
        except Exception as e:
            if peer_sock is not None:
                peer_sock.close()
            raise NetTransportError(f"listen on {host}:{port} failed: {e}") from e
        finally:
            if server_sock is not None:
                server_sock.close()

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        timeout: float = 60.0,
        buffer_size: int = 4 * 1024 * 1024,
        *, io_timeout: float = 60.0,
        max_tensor_bytes: int = 1024 * 1024 * 1024,
        max_object_bytes: int = 16 * 1024 * 1024,
    ) -> NetEndpoint:
        """
        Create a client endpoint that connects to a listening peer.

        Retries with exponential backoff until timeout expires.

        Args:
            host: Remote host to connect to.
            port: Remote port to connect to.
            timeout: Total time to retry before giving up (seconds).
            buffer_size: TCP socket buffer size.

        Returns:
            NetEndpoint connected to the peer.

        Raises:
            NetTransportError: If connection fails after timeout or on other errors.
        """
        cls._validate_timeout(timeout)
        cls._validate_timeout(io_timeout)
        cls._validate_limit(max_tensor_bytes)
        cls._validate_limit(max_object_bytes)
        deadline = time.monotonic() + timeout
        backoff = 0.1
        last_error = None
        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                remaining = min(timeout, deadline - time.monotonic())
                if remaining <= 0:
                    raise socket.timeout("Connection deadline expired")
                sock.settimeout(remaining)
                sock.connect((host, port))
                if time.monotonic() >= deadline:
                    raise socket.timeout("Connection completed after deadline")
                return cls(sock, is_server = False, buffer_size = buffer_size, io_timeout = io_timeout,
                           max_tensor_bytes = max_tensor_bytes, max_object_bytes = max_object_bytes)
            except (socket.timeout, OSError) as e:
                sock.close()
                last_error = e
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(backoff, remaining))
                backoff = min(backoff * 2, 1.0)
            except Exception:
                sock.close()
                raise
        raise NetTransportError(f"connect to {host}:{port} failed after {timeout}s: {last_error}")

    @staticmethod
    def _validate_timeout(value):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise NetTransportError("Timeouts must be finite positive seconds")
        return value

    def _set_io_deadline(self, deadline):
        self._ensure_open()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._fail("Socket I/O deadline expired")
        self.sock.settimeout(remaining)

    @staticmethod
    def _validate_limit(n):
        if type(n) is not int or not 0 < n <= (1 << 63) - 1:
            raise NetTransportError("Byte limits must be positive integers no greater than int64 max")
        return n

    def _ensure_open(self):
        if getattr(self, "_closed", False):
            raise NetTransportError("Endpoint is closed or poisoned")

    def _fail(self, message):
        self.close()
        raise NetTransportError(message)

    def _tensor_nbytes(self, shape, dtype_code):
        if not 0 <= len(shape) <= 5:
            raise NetTransportError(f"Invalid tensor ndim: {len(shape)}")
        if dtype_code not in self._CODE_TO_DTYPE:
            raise NetTransportError(f"Unknown dtype code: {dtype_code}")
        itemsize = (2, 2, 4, 4, 8, 1, 1, 1)[dtype_code]
        product = 1
        for d in shape:
            if d < 0 or d > (1 << 63) - 1 or product > (self.max_tensor_bytes // itemsize) // max(1, d):
                raise NetTransportError(f"Tensor shape exceeds byte limit or overflows: {shape}")
            product *= max(1, d)
        n = 0 if 0 in shape else product * itemsize
        if n > self.max_tensor_bytes:
            raise NetTransportError(f"Tensor exceeds byte limit: {n}")
        return n

    def _validate_tensor_header(self, ndim, dims, dtype_code, payload_len):
        self._ensure_open()
        try:
            if not 0 <= ndim <= 5:
                raise NetTransportError(f"Invalid tensor ndim: {ndim}")
            shape = tuple(dims[:ndim])
            expected = self._tensor_nbytes(shape, dtype_code)
            if payload_len != expected:
                raise NetTransportError(f"Payload length {payload_len} does not match shape {shape}")
        except NetTransportError:
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
                    raise NetTransportError("Object exceeds byte limit")
                payload.extend(chunk)
        except (TypeError, ValueError) as e:
            raise NetTransportError(f"Object not JSON-serializable: {e}") from e
        return payload

    def send_tensor(self, t: torch.Tensor, stream: torch.cuda.Stream | None = None) -> None:
        """
        Send a tensor to the peer.

        Non-contiguous tensors are made contiguous before sending.
        CUDA tensors are staged through pinned host memory.

        Args:
            t: torch.Tensor to send (any shape, any supported dtype/device).
            stream: Optional CUDA stream for asynchronous staging.

        Raises:
            NetTransportError: If dtype is unsupported or send fails.
        """
        self._ensure_open()
        dtype = t.dtype
        if dtype not in self._DTYPE_TO_CODE:
            raise NetTransportError(f"Unsupported dtype: {dtype}")

        dtype_code = self._DTYPE_TO_CODE[dtype]
        ndim = len(t.shape)
        if ndim > 5:
            raise NetTransportError(f"Tensor has {ndim} dimensions; max 5 supported")

        shape_tuple = tuple(t.shape) + (0,) * (5 - ndim)
        payload_len = self._tensor_nbytes(tuple(t.shape), dtype_code)
        try:
            if not t.is_contiguous():
                t = t.contiguous()

            # Stage a CUDA tensor through the reused pinned buffer
            if t.is_cuda:
                staging = self._get_pinned_buffer(payload_len)
                send_data = staging[:payload_len].view(t.dtype).view(t.shape)
                if stream is not None:
                    with torch.cuda.stream(stream):
                        send_data.copy_(t, non_blocking = True)
                    stream.synchronize()
                else:
                    send_data.copy_(t, non_blocking = False)
                send_data = send_data.view(-1)
            else:
                send_data = t.view(-1)

            # Complete byte-view preparation before publishing any header. Contiguous
            # CPU tensors can still carry lazy negative/conjugate view bits.
            payload = None
            if payload_len > 0:
                send_data = send_data.resolve_conj().resolve_neg()
                payload = memoryview(send_data.view(torch.uint8).numpy())
        except Exception as e:
            raise NetTransportError(f"Tensor payload preparation failed: {e}") from e

        # Build and send header
        header = struct.pack(
            self._HEADER_FMT,
            self._MAGIC,
            self._MSG_TENSOR,
            dtype_code,
            ndim,
            *shape_tuple,
            payload_len,
        )
        try:
            self._sendall(header)
            if payload is not None:
                self._sendall(payload)
        except Exception as e:
            self._fail(f"Tensor send failed after header publication began: {e}")

    def recv_tensor(
        self,
        out: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """
        Receive a tensor from the peer.

        If `out` matches shape/dtype, it is reused; strided CPU outputs receive via
        a contiguous temporary and copy. All peer metadata is validated first.
        If targeting CUDA, reads directly into pinned memory via recv_into, then copies to GPU.

        Args:
            out: Optional pre-allocated tensor to receive into.
            device: Target device if out is None (defaults to out.device or CPU).
            stream: Optional CUDA stream for async host-to-device copy.

        Returns:
            The received tensor.

        Raises:
            NetTransportError: On framing/EOF/dtype errors.
        """
        header = self._recv_exact(self._HEADER_SIZE)
        (
            magic,
            msg_type,
            dtype_code,
            ndim,
            d0, d1, d2, d3, d4,
            payload_len,
        ) = struct.unpack(self._HEADER_FMT, header)

        if magic != self._MAGIC:
            self._fail(f"Bad magic: {hex(magic)}")
        if msg_type != self._MSG_TENSOR:
            self._fail(f"Expected tensor message, got {msg_type}")
        shape, dtype = self._validate_tensor_header(ndim, (d0, d1, d2, d3, d4), dtype_code, payload_len)

        # Determine target device
        if out is not None:
            if out.shape != shape or out.dtype != dtype:
                out = None
                target_device = torch.device(device) if device else torch.device("cpu")
            else:
                target_device = out.device
        else:
            target_device = torch.device(device) if device else torch.device("cpu")

        if out is None:
            out = torch.empty(shape, dtype = dtype, device = target_device)

        if payload_len == 0:
            return out

        if target_device.type == "cuda":
            # Fast zero-copy path: recv directly into reusable pinned host buffer
            staging = self._get_pinned_buffer(payload_len)
            mv = memoryview(staging[:payload_len].view(torch.uint8).numpy()).cast("B")
            self._recv_into_exact(mv)

            host_view = staging[:payload_len].view(dtype).view(shape)
            if stream is not None:
                with torch.cuda.stream(stream):
                    out.copy_(host_view, non_blocking = True)
                # The copy may still be reading the staging buffer when this returns; the next
                # send/recv that reuses the buffer waits for this event first
                self._staging_event = torch.cuda.Event()
                self._staging_event.record(stream)
            else:
                out.copy_(host_view, non_blocking = False)
        else:
            # Scalar views must be flattened before changing element size. A strided
            # output needs a contiguous receive buffer, then a stride-aware copy.
            buf = out if out.is_contiguous() else torch.empty(shape, dtype = dtype, device = target_device)
            mv = memoryview(buf.view(-1).view(torch.uint8).numpy()).cast("B")
            self._recv_into_exact(mv)
            if buf is not out:
                out.copy_(buf)

        return out

    def send_obj(self, obj: Any) -> None:
        """
        Send a JSON-serializable object to the peer.

        Args:
            obj: Any JSON-serializable Python object (dict, list, int, str, etc.).

        Raises:
            NetTransportError: On serialization or send failure.
        """
        payload = self._encode_obj(obj)

        payload_len = len(payload)

        # Build and send header (object message, no dtype/shape info)
        header = struct.pack(
            self._HEADER_FMT,
            self._MAGIC,
            self._MSG_OBJECT,
            0,
            0,
            0, 0, 0, 0, 0,
            payload_len,
        )
        try:
            self._sendall(header)
            if payload_len > 0:
                self._sendall(payload)
        except Exception as e:
            self._fail(f"Object send failed after header publication began: {e}")

    def recv_obj(self) -> Any:
        """
        Receive a JSON-serializable object from the peer.

        Returns:
            Decoded Python object.

        Raises:
            NetTransportError: On framing/EOF/parse errors.
        """
        header = self._recv_exact(self._HEADER_SIZE)
        (
            magic,
            msg_type,
            _,
            _,
            _, _, _, _, _,
            payload_len,
        ) = struct.unpack(self._HEADER_FMT, header)

        if magic != self._MAGIC:
            self._fail(f"Bad magic: {hex(magic)}")
        if msg_type != self._MSG_OBJECT:
            self._fail(f"Expected object message, got {msg_type}")

        self._validate_object_length(payload_len)
        payload = self._recv_exact(payload_len)

        try:
            return json.loads(payload.decode("utf-8"))
        except (ValueError, RecursionError, MemoryError) as e:
            # ValueError includes JSON/UTF-8 errors and integer conversion limits.
            self._fail(f"Object parse failed: {e}")

    def close(self) -> None:
        """Close the underlying socket and prevent further framing operations."""
        self._closed = True
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()

    def _sendall(self, data: memoryview | bytes) -> None:
        """
        Send all bytes of data over the socket, looping until complete.

        Args:
            data: Bytes or memoryview to send.

        Raises:
            NetTransportError: On socket errors.
        """
        self._ensure_open()
        deadline = time.monotonic() + self.io_timeout
        total_sent = 0
        total_len = len(data)
        while total_sent < total_len:
            try:
                self._set_io_deadline(deadline)
                sent = self.sock.send(data[total_sent:])
                if sent == 0:
                    self._fail("Socket send returned 0 (peer closed)")
                total_sent += sent
            except socket.error as e:
                self._fail(f"Send failed: {e}")

    def _recv_exact(self, n: int) -> bytes:
        """
        Receive exactly n bytes from the socket into a new bytes object.

        Args:
            n: Exact number of bytes to receive.

        Returns:
            Bytes received.

        Raises:
            NetTransportError: On EOF (partial read) or socket errors.
        """
        self._ensure_open()
        data = bytearray(n)
        mv = memoryview(data)
        self._recv_into_exact(mv)
        return bytes(data)

    def _recv_into_exact(self, buffer: memoryview) -> None:
        """
        Receive exactly len(buffer) bytes directly into an existing memoryview.
        Avoids extra memory allocations and copying.

        Args:
            buffer: Writable memoryview.

        Raises:
            NetTransportError: On EOF or socket errors.
        """
        self._ensure_open()
        deadline = time.monotonic() + self.io_timeout
        n = len(buffer)
        pos = 0
        while pos < n:
            try:
                self._set_io_deadline(deadline)
                nbytes = self.sock.recv_into(buffer[pos:])
                if nbytes == 0:
                    self._fail(f"EOF while expecting {n} bytes (got {pos})")
                pos += nbytes
            except socket.error as e:
                self._fail(f"Recv failed: {e}")

    def _get_pinned_buffer(self, size: int) -> torch.Tensor:
        """
        Allocate or reuse a pinned host buffer for CUDA staging.

        Grows the buffer if needed; never shrinks.

        Args:
            size: Required size in bytes.

        Returns:
            torch.Tensor (pinned, uint8) of at least `size` bytes.
        """
        if self._staging_event is not None:
            self._staging_event.synchronize()
            self._staging_event = None
        if self._pinned_buffer is None or self._pinned_size < size:
            new_size = min(self.max_tensor_bytes, size + size // 2 + 1024)
            self._pinned_buffer = torch.empty(
                new_size,
                dtype = torch.uint8,
                device = torch.device("cpu"),
            ).pin_memory()
            self._pinned_size = new_size

        return self._pinned_buffer[:size]
