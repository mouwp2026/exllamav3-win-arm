"""
Tests for exllamav3.model.net_transport: tensor and object transport over TCP.

Tests run on CPU only (no GPU required). Uses 127.0.0.1 and ephemeral ports.
Each test spawns a background thread for the peer to handle client/server roles.
"""

import sys
import os

# Direct import to avoid loading full exllamav3 package (which requires C++ compilation)
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

# Import directly from the module without loading exllamav3.__init__
import importlib.util
transport_path = os.path.join(repo_root, "exllamav3", "model", "net_transport.py")
spec = importlib.util.spec_from_file_location("net_transport", transport_path)
net_transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(net_transport)

NetEndpoint = net_transport.NetEndpoint
NetTransportError = net_transport.NetTransportError

import torch
import threading
import socket


def find_free_port():
    """Find an available port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


def test_dtype_roundtrip_float16():
    """Test round trip for float16 tensor (bit-exact equality)."""
    port = find_free_port()
    sent_tensor = torch.randn(10, 20, dtype = torch.float16)
    received_tensor = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        received_tensor[0] = ep.recv_tensor()
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 5.0)
    assert torch.equal(sent_tensor, received_tensor[0]), "float16 round trip failed"
    print("PASS: float16 round trip")


def test_dtype_roundtrip_bfloat16():
    """Test round trip for bfloat16 tensor (bit-exact via int16 view)."""
    port = find_free_port()
    sent_tensor = torch.randn(10, 20, dtype = torch.bfloat16)
    received_tensor = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        received_tensor[0] = ep.recv_tensor()
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 5.0)
    # For bfloat16, compare bit patterns directly
    assert torch.equal(sent_tensor, received_tensor[0]), "bfloat16 round trip failed"
    print("PASS: bfloat16 round trip")


def test_dtype_roundtrip_float32():
    """Test round trip for float32 tensor."""
    port = find_free_port()
    sent_tensor = torch.randn(10, 20, dtype = torch.float32)
    received_tensor = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        received_tensor[0] = ep.recv_tensor()
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 5.0)
    assert torch.equal(sent_tensor, received_tensor[0]), "float32 round trip failed"
    print("PASS: float32 round trip")


def test_dtype_roundtrip_int32():
    """Test round trip for int32 tensor."""
    port = find_free_port()
    sent_tensor = torch.randint(-1000, 1000, (10, 20), dtype = torch.int32)
    received_tensor = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        received_tensor[0] = ep.recv_tensor()
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 5.0)
    assert torch.equal(sent_tensor, received_tensor[0]), "int32 round trip failed"
    print("PASS: int32 round trip")


def test_dtype_roundtrip_int64():
    """Test round trip for int64 tensor."""
    port = find_free_port()
    sent_tensor = torch.randint(-1000, 1000, (10, 20), dtype = torch.int64)
    received_tensor = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        received_tensor[0] = ep.recv_tensor()
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 5.0)
    assert torch.equal(sent_tensor, received_tensor[0]), "int64 round trip failed"
    print("PASS: int64 round trip")


def test_dtype_roundtrip_uint8():
    """Test round trip for uint8 tensor."""
    port = find_free_port()
    sent_tensor = torch.randint(0, 255, (10, 20), dtype = torch.uint8)
    received_tensor = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        received_tensor[0] = ep.recv_tensor()
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 5.0)
    assert torch.equal(sent_tensor, received_tensor[0]), "uint8 round trip failed"
    print("PASS: uint8 round trip")


def test_dtype_roundtrip_bool():
    """Test round trip for bool tensor."""
    port = find_free_port()
    sent_tensor = torch.randint(0, 2, (10, 20), dtype = torch.bool)
    received_tensor = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        received_tensor[0] = ep.recv_tensor()
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 5.0)
    assert torch.equal(sent_tensor, received_tensor[0]), "bool round trip failed"
    print("PASS: bool round trip")


def test_noncontiguous_tensor():
    """Test that non-contiguous tensors (e.g., transposed) round trip correctly."""
    port = find_free_port()
    # Create a transposed (non-contiguous) tensor
    base = torch.randn(10, 20, dtype = torch.float32)
    sent_tensor = base.t()  # Transpose to make non-contiguous
    assert not sent_tensor.is_contiguous(), "Tensor should be non-contiguous"

    received_tensor = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        received_tensor[0] = ep.recv_tensor()
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 5.0)
    # After send, the receiver should get a contiguous tensor with the correct values
    assert torch.equal(sent_tensor, received_tensor[0]), "Non-contiguous round trip failed"
    print("PASS: non-contiguous tensor round trip")


def test_buffer_reuse_same_object():
    """Test that recv_tensor reuses the output buffer when shape/dtype/device match."""
    port = find_free_port()
    sent_tensor = torch.randn(10, 20, dtype = torch.float32)
    received_tensor = [None]
    reused = [False]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        out_buffer = torch.zeros(10, 20, dtype = torch.float32)
        result = ep.recv_tensor(out = out_buffer)
        # Check if the returned tensor is the same object as out_buffer
        reused[0] = result is out_buffer
        received_tensor[0] = result
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 5.0)
    assert reused[0], "Buffer was not reused (different object returned)"
    assert torch.equal(sent_tensor, received_tensor[0]), "Data incorrect after buffer reuse"
    print("PASS: buffer reuse (same object)")


def test_large_tensor():
    """Test that large tensors (>= 8 MiB) round trip correctly, proving recv loop handles many segments."""
    port = find_free_port()
    # 8 MiB = 8 * 1024 * 1024 bytes = 2097152 float32s
    size = 2097152
    sent_tensor = torch.randn(size, dtype = torch.float32)
    received_tensor = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 10.0)
        received_tensor[0] = ep.recv_tensor()
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 10.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 10.0)
    assert torch.equal(sent_tensor, received_tensor[0]), "Large tensor round trip failed"
    print("PASS: large tensor (8 MiB+) round trip")


def test_control_message_roundtrip():
    """Test that send_obj/recv_obj round trip for nested dicts and lists."""
    port = find_free_port()
    sent_obj = {
        "model": "qwen",
        "layers": [0, 1, 2],
        "config": {
            "hidden_size": 1024,
            "dtype": "bfloat16",
            "scales": [1.5, 2.0],
        },
        "count": 42,
    }
    received_obj = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        received_obj[0] = ep.recv_obj()
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_obj(sent_obj)
    ep.close()

    thread.join(timeout = 5.0)
    assert sent_obj == received_obj[0], "Control message round trip failed"
    print("PASS: control message (nested dict/list) round trip")


def test_eof_handling():
    """Test that closing the peer mid-stream raises NetTransportError, not a hang or truncated tensor."""
    port = find_free_port()
    exception_caught = [None]

    def server():
        try:
            ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
            # Try to receive, but the client will close abruptly
            _ = ep.recv_tensor()
            exception_caught[0] = None
        except NetTransportError as e:
            exception_caught[0] = e
        except Exception as e:
            exception_caught[0] = f"Unexpected error: {e}"

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    # Close without sending anything, simulating an abrupt disconnect
    ep.close()

    thread.join(timeout = 5.0)
    assert exception_caught[0] is not None, "No exception was raised for EOF"
    assert isinstance(exception_caught[0], NetTransportError), \
        f"Expected NetTransportError, got {type(exception_caught[0])}: {exception_caught[0]}"
    print("PASS: EOF handling (raises NetTransportError)")


def test_shape_dtype_mismatch_gets_new_buffer():
    """Test that shape/dtype mismatch with out buffer causes allocation of a new tensor."""
    port = find_free_port()
    sent_tensor = torch.randn(10, 20, dtype = torch.float32)
    received_tensor = [None]
    was_reallocated = [False]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
        # Provide a buffer with mismatched shape
        mismatch_buffer = torch.zeros(5, 10, dtype = torch.float32)
        result = ep.recv_tensor(out = mismatch_buffer)
        # The result should be a different object with the correct shape
        was_reallocated[0] = result is not mismatch_buffer
        received_tensor[0] = result
        ep.close()

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout = 5.0)
    assert was_reallocated[0], "Buffer was reused despite shape mismatch"
    assert received_tensor[0].shape == sent_tensor.shape, "Received shape incorrect"
    assert torch.equal(sent_tensor, received_tensor[0]), "Data incorrect after reallocation"
    print("PASS: shape/dtype mismatch causes reallocation")


def test_cuda_staging_consecutive_sends():
    """Two consecutive sends of CUDA tensors over one connection.

    CUDA tensors are staged through a pinned buffer that is over-allocated and never shrunk, so
    the payload must be taken from the view and not from that buffer's storage. When it is taken
    from the storage the first send still arrives, because the receiver reads only the length the
    header declares, but the surplus bytes stay in the stream and the next header read fails. One
    send therefore cannot catch this; two can.
    """
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        return

    port = find_free_port()
    sent_tensors = [
        torch.randn(1, 1, 2560, dtype = torch.float16, device = "cuda"),
        torch.randn(64, dtype = torch.float32, device = "cuda"),
    ]
    received_tensors = []
    server_error = [None]

    def server():
        try:
            ep = NetEndpoint.listen("127.0.0.1", port, timeout = 5.0)
            ep.sock.settimeout(5.0)
            for _ in sent_tensors:
                received_tensors.append(ep.recv_tensor())
            ep.close()
        except Exception as e:
            server_error[0] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target = server, daemon = True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout = 5.0)
    ep.sock.settimeout(5.0)
    for t in sent_tensors:
        ep.send_tensor(t)
    ep.close()

    thread.join(timeout = 10.0)
    assert server_error[0] is None, f"receiver failed: {server_error[0]}"
    assert len(received_tensors) == len(sent_tensors), \
        f"received {len(received_tensors)} of {len(sent_tensors)} tensors"
    for i, (s_, r_) in enumerate(zip(sent_tensors, received_tensors)):
        assert torch.equal(s_.cpu(), r_.cpu()), f"tensor {i} differs after round trip"
    print("PASS: CUDA staging, consecutive sends")


def test_dtype_roundtrip_float8():
    """Test that FP8 (float8_e4m3fn) round-trips correctly if supported by torch."""
    if not hasattr(torch, "float8_e4m3fn"):
        print("SKIP: FP8 not supported in this torch build")
        return
    port = find_free_port()
    # Create uint8 raw bytes and view as float8_e4m3fn
    raw = torch.randint(0, 256, (10, 20), dtype=torch.uint8)
    sent_tensor = raw.view(torch.float8_e4m3fn)
    received_tensor = [None]

    def server():
        ep = NetEndpoint.listen("127.0.0.1", port, timeout=5.0)
        received_tensor[0] = ep.recv_tensor()
        ep.close()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()

    ep = NetEndpoint.connect("127.0.0.1", port, timeout=5.0)
    ep.send_tensor(sent_tensor)
    ep.close()

    thread.join(timeout=5.0)
    assert received_tensor[0] is not None
    assert received_tensor[0].dtype == torch.float8_e4m3fn
    assert torch.equal(sent_tensor.view(torch.uint8), received_tensor[0].view(torch.uint8))
    print("PASS: dtype round trip (float8_e4m3fn)")


if __name__ == "__main__":
    # Run all tests
    test_dtype_roundtrip_float16()
    test_dtype_roundtrip_bfloat16()
    test_dtype_roundtrip_float8()
    test_dtype_roundtrip_float32()
    test_dtype_roundtrip_int32()
    test_dtype_roundtrip_int64()
    test_dtype_roundtrip_uint8()
    test_dtype_roundtrip_bool()
    test_noncontiguous_tensor()
    test_buffer_reuse_same_object()
    test_large_tensor()
    test_control_message_roundtrip()
    test_eof_handling()
    test_shape_dtype_mismatch_gets_new_buffer()
    test_cuda_staging_consecutive_sends()

    print("\nAll tests passed!")
