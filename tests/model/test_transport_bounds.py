"""CPU-only regression gates for PR17; direct imports avoid the CUDA extension."""
import importlib.util
from pathlib import Path
import socket
import struct
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'exllamav3/model' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

net = load('net_transport')
nccl = load('nccl_transport')

def pair(**kwargs):
    a, b = socket.socketpair()
    a.settimeout(.2)
    b.settimeout(.2)
    return net.NetEndpoint(a, **kwargs), net.NetEndpoint(b, **kwargs)

def header(ndim, dims, n, msg=0, dtype=2):
    return struct.pack(net.NetEndpoint._HEADER_FMT, net.NetEndpoint._MAGIC, msg, dtype, ndim,
                       *(tuple(dims) + (0,) * (5 - len(dims))), n)

def distributed_endpoint(h):
    # Invalid wire metadata is injected without initiating a distributed job.
    ep = nccl.NcclEndpoint.__new__(nccl.NcclEndpoint)
    ep.peer, ep.group, ep.comm_device = 1, None, torch.device('cpu')
    ep._hdr_send = torch.zeros(ep._HEADER_LEN, dtype=torch.long)
    ep._recv_header = lambda _: h
    return ep

@pytest.mark.parametrize('ndim,dims,n', [(1, (2,), 0), (1, (2,), 4), (6, (1,)*5, 4)])
def test_net_bad_shape_is_rejected_before_allocation_and_poisons(ndim, dims, n):
    a, b = pair()
    try:
        a.sock.sendall(header(ndim, dims, n) + b'\0'*4)
        with patch.object(torch, 'empty', side_effect=AssertionError('allocation before validation')):
            with pytest.raises(net.NetTransportError):
                b.recv_tensor(out=torch.full((2,), 123.))
        assert b.sock.fileno() == -1
        with pytest.raises(net.NetTransportError):
            b.recv_tensor()
    finally:
        a.close(); b.close()

@pytest.mark.parametrize('ndim,dims,n', [(1, (-2,0,0,0,0), 0), (6, (1,)*5, 4), (1, (2,0,0,0,0), 4)])
def test_distributed_bad_shape_is_rejected_before_allocation_and_poisons(ndim, dims, n):
    ep = distributed_endpoint([nccl.NcclEndpoint._MAGIC, 0, 2, ndim, *dims, n])
    with patch.object(torch, 'empty', side_effect=AssertionError('allocation before validation')):
        with pytest.raises(nccl.NcclTransportError):
            ep.recv_tensor()
    assert ep._closed
    with pytest.raises(nccl.NcclTransportError):
        ep.recv_tensor()

@pytest.mark.parametrize('transport', ['net', 'distributed'])
@pytest.mark.parametrize('kind', ['object', 'tensor', 'overflow', 'empty_overflow'])
def test_peer_allocation_is_bounded(transport, kind):
    dims = (2**38,) if kind == 'tensor' else ((2**40, 2**40) if kind == 'overflow' else (0, 2**63))
    n = 2**40 if kind in ('object', 'tensor') else 0
    a, b = pair()
    ep = b if transport == 'net' else distributed_endpoint([nccl.NcclEndpoint._MAGIC, int(kind == 'object'), 2,
        len(dims) if kind != 'object' else 0, *(dims + (0,)*(5-len(dims)) if kind != 'object' else (0,)*5), n])
    error = net.NetTransportError if transport == 'net' else nccl.NcclTransportError
    original = bytearray
    def bounded(n):
        assert n <= 1024, 'peer length reached bytearray allocation'
        return original(n)
    try:
        if transport == 'net':
            a.sock.sendall(header(0 if kind == 'object' else len(dims), () if kind == 'object' else dims, n, msg=int(kind == 'object')))
        with patch.object(net, 'bytearray', bounded, create=True), patch.object(torch, 'empty', side_effect=AssertionError('peer shape reached tensor allocation')):
            with pytest.raises(error):
                ep.recv_obj() if kind == 'object' else ep.recv_tensor()
        assert ep._closed
    finally:
        a.close(); b.close()

@pytest.mark.parametrize('transport', ['net', 'distributed'])
def test_limits_are_configurable_and_sends_validate_before_copy(transport):
    if transport == 'net':
        a, ep = pair(max_tensor_bytes=16, max_object_bytes=8)
        sender = patch.object(ep, '_sendall', side_effect=AssertionError('oversize send'))
        error = net.NetTransportError
    else:
        a = None
        with patch.object(nccl.dist, 'is_initialized', return_value=True), patch.object(nccl.dist, 'get_backend', return_value='gloo'):
            ep = nccl.NcclEndpoint(1, max_tensor_bytes=16, max_object_bytes=8)
        sender = patch.object(nccl.dist, 'send', side_effect=AssertionError('oversize send'))
        error = nccl.NcclTransportError
    try:
        t = torch.empty((3, 3), device='meta').T
        with sender, patch.object(torch.Tensor, 'contiguous', side_effect=AssertionError('copy before bound')):
            with pytest.raises(error): ep.send_tensor(t)
            with pytest.raises(error): ep.send_obj({'long': '123456789'})
        assert not ep._closed  # local validation before any wire bytes is recoverable
    finally:
        ep.close()
        if a: a.close()

@pytest.mark.parametrize('limit', [0, -1, 2**63, 1.5, True])
def test_limit_configuration_rejects_nonfinite_or_nonpositive_values(limit):
    a, b = socket.socketpair()
    try:
        with pytest.raises(net.NetTransportError): net.NetEndpoint(a, max_tensor_bytes=limit)
        with patch.object(nccl.dist, 'is_initialized', return_value=True), patch.object(nccl.dist, 'get_backend', return_value='gloo'):
            with pytest.raises(nccl.NcclTransportError): nccl.NcclEndpoint(1, max_object_bytes=limit)
    finally:
        a.close(); b.close()

@pytest.mark.parametrize('dtype', list(net.NetEndpoint._DTYPE_TO_CODE))
def test_cpu_scalar_roundtrip(dtype):
    a, b = pair()
    try:
        sent = torch.tensor(1, dtype=dtype)
        a.send_tensor(sent)
        got = b.recv_tensor()
        assert got.shape == sent.shape and got.dtype == dtype
        assert torch.equal(got.reshape(-1).view(torch.uint8), sent.reshape(-1).view(torch.uint8))
    finally:
        a.close(); b.close()

@pytest.mark.parametrize('layout', ['transpose', 'slice'])
def test_cpu_strided_receive_preserves_out(layout):
    a, b = pair()
    try:
        sent = torch.arange(6.).reshape(3, 2)
        out = torch.empty((2, 3)).T if layout == 'transpose' else torch.empty((3, 4))[:, ::2]
        assert not out.is_contiguous()
        a.send_tensor(sent)
        assert b.recv_tensor(out=out) is out
        assert torch.equal(out, sent)
        a.send_obj({'next': True})
        assert b.recv_obj() == {'next': True}
    finally:
        a.close(); b.close()

def _layout_worker(rank, port):
    import torch.distributed as dist
    nccl.NcclEndpoint.init_group(rank, 2, '127.0.0.1', port, backend='gloo', timeout_s=10)
    ep = nccl.NcclEndpoint(1-rank)
    try:
        for dtype in nccl.NcclEndpoint._DTYPE_TO_CODE:
            sent = torch.tensor(1, dtype=dtype)
            if rank == 0:
                ep.send_tensor(sent)
            else:
                out = torch.empty((), dtype=dtype)
                assert ep.recv_tensor(out=out) is out
                assert torch.equal(out.reshape(-1).view(torch.uint8), sent.reshape(-1).view(torch.uint8))
        for out in [torch.empty((2, 3)).T, torch.empty((3, 4))[:, ::2]]:
            sent = torch.arange(6.).reshape(3, 2)
            if rank == 0: ep.send_tensor(sent)
            else:
                assert ep.recv_tensor(out=out) is out
                assert torch.equal(out, sent)
        if rank == 0: ep.send_obj({'next': True})
        else: assert ep.recv_obj() == {'next': True}
    finally:
        dist.destroy_process_group()

def test_distributed_cpu_strided_and_scalar_actual_roundtrip():
    import multiprocessing as mp
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    ctx = mp.get_context('spawn')
    procs = [ctx.Process(target=_layout_worker, args=(rank, port)) for rank in (0, 1)]
    try:
        for p in procs: p.start()
        for p in procs: p.join(25)
        assert [p.exitcode for p in procs] == [0, 0]
    finally:
        for p in procs:
            if p.is_alive(): p.terminate(); p.join(5)

@pytest.mark.parametrize('part', ['header', 'payload'])
def test_eof_midframe_closes_endpoint(part):
    a, b = pair()
    try:
        frame = header(1, (2,), 8) + b'1234'
        a.sock.sendall(frame[:7] if part == 'header' else frame)
        a.close()
        with pytest.raises(net.NetTransportError, match='EOF'): b.recv_tensor()
        assert b.sock.fileno() == -1
        with pytest.raises(net.NetTransportError, match='closed'): b.recv_obj()
    finally:
        a.close(); b.close()

def test_connect_applies_remaining_deadline_and_rejects_late_success():
    import time
    class FakeSocket:
        timeouts = []
        closed = False
        def settimeout(self, value): self.timeouts.append(value)
        def connect(self, address): time.sleep(.03)
        def setsockopt(self, *args): pass
        def close(self): self.closed = True
    s = FakeSocket()
    with patch.object(net.socket, 'socket', return_value=s):
        with pytest.raises(net.NetTransportError): net.NetEndpoint.connect('127.0.0.1', 1, timeout=.01)
    assert s.closed
    assert s.timeouts and 0 < s.timeouts[0] <= .01

def test_localhost_connect_deadline_is_bounded():
    import time
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    start = time.monotonic()
    with pytest.raises(net.NetTransportError): net.NetEndpoint.connect('127.0.0.1', port, timeout=.05)
    assert time.monotonic() - start < .5

def test_accepted_and_connected_sockets_have_configured_io_timeout():
    import threading
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    box = {}
    def accept():
        try: box['ep'] = net.NetEndpoint.listen('127.0.0.1', port, timeout=1, io_timeout=.05, max_tensor_bytes=64, max_object_bytes=32)
        except Exception as e: box['error'] = e
    t = threading.Thread(target=accept, daemon=True); t.start()
    client = None
    try:
        client = net.NetEndpoint.connect('127.0.0.1', port, timeout=2, io_timeout=.05, max_tensor_bytes=64, max_object_bytes=32)
        t.join(1)
        assert 'error' not in box and 'ep' in box
        for ep in (client, box['ep']):
            assert ep.sock.gettimeout() == .05
            assert ep.max_tensor_bytes == 64 and ep.max_object_bytes == 32
        with pytest.raises(net.NetTransportError): box['ep'].recv_obj()
        assert box['ep']._closed
    finally:
        if client: client.close()
        if 'ep' in box: box['ep'].close()
        t.join(1)

def test_trickling_header_cannot_extend_receive_deadline():
    import threading, time
    a, b = pair(io_timeout=.06)
    def trickle():
        try:
            for byte in header(1, (2,), 8):
                a.sock.sendall(bytes([byte])); time.sleep(.02)
        except OSError: pass
    t = threading.Thread(target=trickle, daemon=True); t.start()
    start = time.monotonic()
    try:
        with pytest.raises(net.NetTransportError): b.recv_tensor()
        assert time.monotonic() - start < .3
        assert b._closed
    finally:
        a.close(); b.close(); t.join(1)

def test_blocked_send_is_bounded_and_poisoned():
    a, b = pair(io_timeout=.03)
    try:
        # Windows loopback can buffer a large write without a reader. Inject the
        # OS timeout here; real blocked/trickled reads exercise the live deadline.
        with patch.object(net.socket.socket, 'send', side_effect=socket.timeout('blocked')):
            with pytest.raises(net.NetTransportError): a.send_tensor(torch.zeros(2))
        assert a._closed
    finally:
        a.close(); b.close()

def test_close_cancels_an_inflight_receive():
    import threading, time
    a, b = pair(io_timeout=.5)
    errors = []
    def receive():
        try: b.recv_obj()
        except net.NetTransportError as e: errors.append(e)
    t = threading.Thread(target=receive, daemon=True); t.start()
    time.sleep(.02)
    b.close(); t.join(.3)
    a.close()
    assert not t.is_alive() and errors

@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
def test_invalid_timeouts_rejected(timeout):
    with patch.object(net.socket.socket, 'connect', return_value=None):
        with pytest.raises(net.NetTransportError): net.NetEndpoint.connect('127.0.0.1', 1, timeout=timeout)
    a, b = socket.socketpair()
    try:
        with pytest.raises(net.NetTransportError): net.NetEndpoint(a, io_timeout=timeout)
    finally:
        a.close(); b.close()

@pytest.mark.parametrize('method', ['send_tensor', 'recv_tensor'])
def test_distributed_explicit_stream_is_rejected_before_wire_activity(method):
    ep = distributed_endpoint([nccl.NcclEndpoint._MAGIC, 0, 2, 1, 2,0,0,0,0,8])
    ep._recv_header = lambda _: (_ for _ in ()).throw(AssertionError('header consumed'))
    with patch.object(nccl.dist, 'send', side_effect=AssertionError('header sent')):
        with pytest.raises(nccl.NcclTransportError, match='stream'):
            if method == 'send_tensor': ep.send_tensor(torch.ones(2), stream=object())
            else: ep.recv_tensor(stream=object())

@pytest.mark.parametrize('declared', [0,4])
def test_reviewer_malformed_tensor_never_consumes_following_frame(declared):
    a,b = pair()
    out = torch.full((2,), 123.)
    try:
        a.sock.sendall(header(1,(2,),declared) + struct.pack('f',1.))
        a.send_obj({'next':True})
        with pytest.raises(net.NetTransportError): b.recv_tensor(out=out)
        assert out.tolist() == [123.,123.]
        assert b._closed and b.sock.fileno() == -1
        with pytest.raises(net.NetTransportError): b.recv_obj()
    finally:
        a.close(); b.close()

@pytest.mark.parametrize('shape', [(), (0,), (2,0,3), (1,1,1,1,1)])
def test_valid_empty_and_scalar_shapes_keep_following_frame_aligned(shape):
    a,b = pair()
    try:
        sent = torch.ones(shape)
        a.send_tensor(sent); a.send_obj(None)
        got = b.recv_tensor()
        assert tuple(got.shape) == shape and torch.equal(got, sent)
        assert b.recv_obj() is None
    finally:
        a.close(); b.close()

def test_configured_exact_byte_boundary_and_local_rejection_keep_link_usable():
    a,b = pair(max_tensor_bytes=16, max_object_bytes=4)
    try:
        with pytest.raises(net.NetTransportError): a.send_tensor(torch.zeros(5))
        with pytest.raises(net.NetTransportError): a.send_obj('large')
        a.send_tensor(torch.arange(4.)); a.send_obj(None)
        assert torch.equal(b.recv_tensor(), torch.arange(4.))
        assert b.recv_obj() is None
    finally:
        a.close(); b.close()

@pytest.mark.parametrize('n', [-1,0,2**40])
def test_distributed_invalid_object_size_is_rejected_before_payload(n):
    ep = distributed_endpoint([nccl.NcclEndpoint._MAGIC,1,0,0,0,0,0,0,0,n])
    with patch.object(torch,'empty',side_effect=AssertionError('payload allocation')), patch.object(nccl.dist,'recv',side_effect=AssertionError('payload receive')):
        with pytest.raises(nccl.NcclTransportError): ep.recv_obj()
    assert ep._closed

@pytest.mark.parametrize('field,value', [(0,0), (1,99), (2,99), (3,-1)])
def test_distributed_wire_header_errors_poison_before_payload(field,value):
    ep = distributed_endpoint([])
    del ep._recv_header
    ep._hdr_recv = torch.zeros(ep._HEADER_LEN, dtype=torch.long)
    h = [ep._MAGIC,0,2,1,2,0,0,0,0,8]; h[field] = value
    def recv(buffer, *args, **kwargs): buffer.copy_(torch.tensor(h))
    with patch.object(nccl.dist,'recv',side_effect=recv), patch.object(torch,'empty',side_effect=AssertionError('payload allocation')):
        with pytest.raises(nccl.NcclTransportError): ep.recv_tensor()
    assert ep._closed
