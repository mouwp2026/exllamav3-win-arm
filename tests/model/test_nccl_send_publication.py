"""Distributed send publication regressions (CPU wire mocks and real Gloo)."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
import socket
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('nccl_publication', ROOT / 'exllamav3/model/nccl_transport.py')
nccl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nccl)


def endpoint():
    with patch.object(nccl.dist, 'is_initialized', return_value=True), \
            patch.object(nccl.dist, 'get_backend', return_value='gloo'):
        return nccl.NcclEndpoint(1)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64])
@pytest.mark.parametrize('layout', ['contiguous', 'scalar', 'strided'])
def test_negative_view_values_and_successor_frame(dtype, layout):
    ep = endpoint()
    base = torch.tensor(1, dtype=dtype) if layout == 'scalar' else torch.arange(1, 7).to(dtype)
    source = torch._neg_view(base if layout != 'strided' else base[::2])
    assert source.is_neg()
    expected = -base if layout != 'strided' else -base[::2]
    frames = []
    with patch.object(nccl.dist, 'send', side_effect=lambda t, *a, **k: frames.append(t.clone())):
        ep.send_tensor(source)
        ep.send_obj({'next': True})
    assert len(frames) == 4
    wire = iter(frames)
    with patch.object(nccl.dist, 'recv', side_effect=lambda t, *a, **k: t.copy_(next(wire))):
        actual = ep.recv_tensor()
        assert actual.dtype == dtype and actual.shape == source.shape
        assert torch.equal(actual, expected)
        assert ep.recv_obj() == {'next': True}
    assert source.is_neg() and not ep._closed


@pytest.mark.parametrize('phase', ['meta', 'sparse', 'byte_view', 'header_buffer', 'object_buffer'])
def test_prepublication_preparation_failure_is_typed_and_reusable(phase):
    ep = endpoint()
    source = torch.ones(1, device='meta') if phase == 'meta' else torch.ones(1)
    if phase == 'sparse':
        source = torch.eye(2).to_sparse()
    targets = {
        'byte_view': (torch.Tensor, 'view'),
        'header_buffer': (nccl.torch, 'tensor'),
        'object_buffer': (nccl.torch, 'frombuffer'),
    }

    inject = patch.object(*targets[phase], side_effect=MemoryError('injected preparation')) if phase in targets else nullcontext()
    frames = []
    with patch.object(nccl.dist, 'send', side_effect=lambda t, *a, **k: frames.append(t.clone())):
        with inject, pytest.raises(nccl.NcclTransportError):
            ep.send_obj({'next': True}) if phase == 'object_buffer' else ep.send_tensor(source)
        assert not frames and not ep._closed
        ep.send_tensor(torch.tensor([7.]))
        ep.send_obj({'next': True})
    wire = iter(frames)
    with patch.object(nccl.dist, 'recv', side_effect=lambda t, *a, **k: t.copy_(next(wire))):
        assert torch.equal(ep.recv_tensor(), torch.tensor([7.]))
        assert ep.recv_obj() == {'next': True}


@pytest.mark.parametrize(('kind', 'phase'), [
    ('tensor', 'header'), ('tensor', 'payload'),
    ('object', 'header'), ('object', 'payload'), ('empty_tensor', 'header'),
])
@pytest.mark.parametrize('error_type', [RuntimeError, nccl.NcclTransportError, MemoryError])
def test_publication_failure_is_typed_and_poisoned(kind, phase, error_type):
    ep = endpoint()
    calls = []
    failure = error_type('injected publication failure')

    def send(t, *args, **kwargs):
        calls.append(t.clone())
        if phase == 'header' or len(calls) == 2:
            # The backend may have published partial bytes before raising.
            raise failure

    with patch.object(nccl.dist, 'send', side_effect=send) as sending:
        with pytest.raises(nccl.NcclTransportError, match='injected publication failure'):
            if kind == 'object':
                ep.send_obj({'next': True})
            else:
                ep.send_tensor(torch.empty(0) if kind == 'empty_tensor' else torch.ones(1))
        assert len(calls) == (1 if phase == 'header' else 2)
        assert ep._closed
        for operation in (lambda: ep.send_tensor(torch.ones(1)), lambda: ep.send_obj(None), ep.recv_tensor, ep.recv_obj):
            with pytest.raises(nccl.NcclTransportError, match='closed or poisoned'):
                operation()
        assert sending.call_count == len(calls)


def roundtrip_payloads():
    for dtype in nccl.NcclEndpoint._DTYPE_TO_CODE:
        for layout in ('scalar', 'strided', 'empty'):
            base = torch.tensor(1) if layout == 'scalar' else torch.arange(6).reshape(2, 3)
            if layout == 'strided':
                base = base.t()
            if layout == 'empty':
                base = torch.empty(0)
            yield base.to(dtype)
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64):
        yield torch._neg_view(torch.ones(1, dtype=dtype))
        yield torch._neg_view(torch.tensor(1, dtype=dtype))
        yield torch._neg_view(torch.arange(6).to(dtype).reshape(2, 3).t())


def _gloo_worker(rank, port, q):
    nccl.NcclEndpoint.init_group(rank, 2, '127.0.0.1', port, backend='gloo', timeout_s=15)
    ep = nccl.NcclEndpoint(1 - rank)
    try:
        count = 0
        for source in roundtrip_payloads():
            expected = source.resolve_conj().resolve_neg().contiguous().view(-1).view(torch.uint8)
            if rank == 0:
                ep.send_tensor(source)
                ep.send_obj({'next': count})
                actual = ep.recv_tensor()
            else:
                out = torch.empty(tuple(reversed(source.shape)), dtype=source.dtype).t() if source.ndim == 2 else None
                actual = ep.recv_tensor(out=out)
                if out is not None:
                    assert actual is out
                assert ep.recv_obj() == {'next': count}
                ep.send_tensor(actual)
            assert actual.dtype == source.dtype and actual.shape == source.shape
            assert torch.equal(actual.contiguous().view(-1).view(torch.uint8), expected)
            count += 1
        q.put((rank, count))
    finally:
        ep.close()
        nccl.dist.destroy_process_group()


def test_real_gloo_negative_scalar_strided_dtypes_and_successor_frames():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = [ctx.Process(target=_gloo_worker, args=(rank, port, q)) for rank in range(2)]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(25)
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        results = dict(q.get(timeout=2) for _ in procs)
        assert results == {0: len(list(roundtrip_payloads())), 1: len(list(roundtrip_payloads()))}
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(5)
        q.close()
