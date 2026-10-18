"""Final PR17 fail-closed regressions: real localhost TCP, CPU distributed mocks."""
import importlib.util
from pathlib import Path
import socket
import struct
import sys
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'exllamav3/model' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


net = load('net_transport')
nccl = load('nccl_transport')


@pytest.fixture
def tcp_pair():
    # AF_INET loopback, not socketpair: exercise the actual TCP byte stream.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(2)
        client = socket.create_connection(listener.getsockname(), timeout=2)
        server, _ = listener.accept()
    sender = net.NetEndpoint(client, io_timeout=.5)
    receiver = net.NetEndpoint(server, io_timeout=.5)
    try:
        yield sender, receiver
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize('layout', ['contiguous', 'scalar', 'strided'])
def test_tcp_negative_view_preserves_values_and_following_frame(tcp_pair, dtype, layout):
    sender, receiver = tcp_pair
    base = torch.tensor(1, dtype=dtype) if layout == 'scalar' else torch.arange(1, 7).to(dtype)
    source = torch._neg_view(base if layout != 'strided' else base[::2])
    assert source.is_neg()
    expected = -base if layout != 'strided' else -base[::2]
    sender.send_tensor(source)
    sender.send_obj({'next': True})
    actual = receiver.recv_tensor()
    assert actual.dtype == dtype and actual.shape == source.shape
    assert torch.equal(actual, expected)
    assert receiver.recv_obj() == {'next': True}
    assert source.is_neg()  # Sending does not mutate the caller's view.


@pytest.mark.parametrize('kind', ['meta', 'sparse'])
def test_tcp_payload_preparation_failure_does_not_publish_header(tcp_pair, kind):
    sender, receiver = tcp_pair
    source = torch.ones(1, device='meta') if kind == 'meta' else torch.eye(2).to_sparse()
    with pytest.raises(net.NetTransportError):
        sender.send_tensor(source)
    assert not sender._closed
    sender.send_tensor(torch.tensor([7.]))
    sender.send_obj({'next': True})
    assert torch.equal(receiver.recv_tensor(), torch.tensor([7.]))
    assert receiver.recv_obj() == {'next': True}


def assert_poisoned(endpoint, error):
    assert endpoint._closed
    for operation in (
        lambda: endpoint.send_obj(None),
        lambda: endpoint.send_tensor(torch.ones(1)),
        endpoint.recv_obj,
        endpoint.recv_tensor,
    ):
        with pytest.raises(error, match='closed or poisoned'):
            operation()


@pytest.mark.parametrize('kind', ['tensor', 'object'])
@pytest.mark.parametrize('phase', ['partial_header', 'partial_payload'])
@pytest.mark.parametrize('error_type', [RuntimeError, net.NetTransportError])
def test_tcp_post_publication_failure_is_typed_and_poisoned(tcp_pair, kind, phase, error_type):
    sender, receiver = tcp_pair
    sendall = sender._sendall
    calls = []

    def fail_after_bytes(data):
        calls.append(len(data))
        if phase == 'partial_header' or len(calls) == 2:
            # Send a real prefix before injecting a non-socket failure. The peer
            # must see EOF, never a later frame filling this incomplete one.
            sendall(data[:7] if phase == 'partial_header' else data[:1])
            raise error_type('injected after publication')
        sendall(data)

    with patch.object(sender, '_sendall', side_effect=fail_after_bytes):
        with pytest.raises(net.NetTransportError, match='injected after publication'):
            sender.send_tensor(torch.ones(2)) if kind == 'tensor' else sender.send_obj({'next': True})
    assert_poisoned(sender, net.NetTransportError)
    assert sender.sock.fileno() == -1
    with pytest.raises(net.NetTransportError, match='EOF'):
        receiver.recv_tensor() if kind == 'tensor' else receiver.recv_obj()
    assert_poisoned(receiver, net.NetTransportError)


def object_frame(payload):
    return struct.pack(net.NetEndpoint._HEADER_FMT, net.NetEndpoint._MAGIC,
                       1, 0, 0, 0, 0, 0, 0, 0, len(payload)) + payload


def distributed_reader(payloads, **limits):
    # CPU mocks inject the distributed wire buffers; the constructor, header
    # validation, tensor allocation and JSON decoder are production code.
    with patch.object(nccl.dist, 'is_initialized', return_value=True), \
            patch.object(nccl.dist, 'get_backend', return_value='gloo'):
        endpoint = nccl.NcclEndpoint(1, **limits)
    buffers = []
    for payload in payloads:
        buffers.extend((
            torch.tensor([endpoint._MAGIC, 1, 0, 0, 0, 0, 0, 0, 0, len(payload)]),
            torch.frombuffer(bytearray(payload), dtype=torch.uint8),
        ))
    frames = iter(buffers)

    def recv(target, *args, **kwargs):
        target.copy_(next(frames))

    return endpoint, recv


@pytest.fixture(params=['large_integer', 'deep_nesting', 'invalid_utf8', 'malformed_json'])
def bad_json_payload(request):
    old_limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    # Python 3.12's C decoder recursion guard can exceed the Python recursion
    # limit; this still-small frame exceeds both without changing either guard.
    depth = max(10000, sys.getrecursionlimit() + 100)
    payloads = {
        'large_integer': b'1' * 5000,
        'deep_nesting': b'[' * depth + b'0' + b']' * depth,
        'invalid_utf8': b'"\xff"',
        'malformed_json': b'{',
    }
    try:
        yield payloads[request.param]
    finally:
        sys.set_int_max_str_digits(old_limit)


def test_tcp_json_decoder_failure_is_typed_and_poisoned(tcp_pair, bad_json_payload):
    sender, receiver = tcp_pair
    assert len(bad_json_payload) < receiver.max_object_bytes
    sender.sock.sendall(object_frame(bad_json_payload) + object_frame(b'null'))
    with pytest.raises(net.NetTransportError, match='Object parse failed'):
        receiver.recv_obj()
    assert_poisoned(receiver, net.NetTransportError)
    assert receiver.sock.fileno() == -1


def test_distributed_json_decoder_failure_is_typed_and_poisoned(bad_json_payload):
    endpoint, recv = distributed_reader([bad_json_payload, b'null'])
    assert len(bad_json_payload) < endpoint.max_object_bytes
    try:
        with patch.object(nccl.dist, 'recv', side_effect=recv) as receive, \
                patch.object(nccl.dist, 'send', side_effect=AssertionError('poisoned send')):
            with pytest.raises(nccl.NcclTransportError, match='Object parse failed'):
                endpoint.recv_obj()
            assert_poisoned(endpoint, nccl.NcclTransportError)
            assert receive.call_count == 2  # Successor header was never consumed.
    finally:
        endpoint.close()


def test_tcp_exact_object_byte_limit_preserves_following_frame(tcp_pair):
    sender, receiver = tcp_pair
    sender.max_object_bytes = receiver.max_object_bytes = 5000
    value = 'x' * 4998  # JSON quotes bring the payload exactly to the cap.
    assert len(sender._encode_obj(value)) == sender.max_object_bytes
    sender.send_obj(value)
    sender.send_obj(None)
    assert receiver.recv_obj() == value
    assert receiver.recv_obj() is None
    assert not receiver._closed


def test_distributed_exact_object_byte_limit_preserves_following_frame():
    payload = b'"' + b'x' * 4998 + b'"'
    endpoint, recv = distributed_reader([payload, b'null'], max_object_bytes=len(payload))
    assert len(payload) == endpoint.max_object_bytes
    try:
        with patch.object(nccl.dist, 'recv', side_effect=recv) as receive:
            assert endpoint.recv_obj() == 'x' * 4998
            assert endpoint.recv_obj() is None
            assert receive.call_count == 4
            assert not endpoint._closed
    finally:
        endpoint.close()


def test_tcp_json_decoder_memory_failure_is_typed_and_poisoned(tcp_pair):
    sender, receiver = tcp_pair
    sender.send_obj(None)
    with patch.object(net.json, 'loads', side_effect=MemoryError('decoder allocation')):
        with pytest.raises(net.NetTransportError, match='Object parse failed'):
            receiver.recv_obj()
    assert_poisoned(receiver, net.NetTransportError)


def test_distributed_json_decoder_memory_failure_is_typed_and_poisoned():
    endpoint, recv = distributed_reader([b'null'])
    try:
        with patch.object(nccl.dist, 'recv', side_effect=recv), \
                patch.object(nccl.json, 'loads', side_effect=MemoryError('decoder allocation')):
            with pytest.raises(nccl.NcclTransportError, match='Object parse failed'):
                endpoint.recv_obj()
        assert_poisoned(endpoint, nccl.NcclTransportError)
    finally:
        endpoint.close()
