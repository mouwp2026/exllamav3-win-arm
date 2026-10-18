"""
Tests for exllamav3.model.nccl_transport framing, run over the gloo backend on CPU (two spawned
processes, no GPU needed). The NCCL path uses the same framing with GPU buffers.
"""

import os
import sys
import socket
import importlib.util

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_path = os.path.join(repo_root, "exllamav3", "model", "nccl_transport.py")


def _load():
    spec = importlib.util.spec_from_file_location("nccl_transport", _path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _payloads():
    g = torch.Generator().manual_seed(0)
    out = [
        torch.randn(1, 1, 6144, generator = g),
        torch.randn(1, 37, 64, generator = g).half(),
        torch.randn(3, 5, generator = g).bfloat16(),
        torch.randint(-1000, 1000, (2, 3, 4), generator = g, dtype = torch.int32),
        torch.randint(0, 154880, (1, 513), generator = g, dtype = torch.long),
        torch.randint(0, 255, (17,), generator = g, dtype = torch.uint8),
        torch.rand(4, 4, generator = g) > 0.5,
        torch.empty(0, 6144),
        torch.randn(2, 1, 3, 1, 2, generator = g),
        torch.randn(8, 16, generator = g)[:, ::2],     # non-contiguous
    ]
    if hasattr(torch, "float8_e4m3fn"):
        out.append(torch.randn(1, 4, 6144, generator = g).to(torch.float8_e4m3fn))
    return out


_OBJS = [{"cmd": "fwd", "past_len": 512, "mode": "gen", "last_only": True, "want": None},
         [1, 2.5, "x", None], "plain string", 0]


def _worker(rank, port, q):
    mod = _load()
    mod.NcclEndpoint.init_group(rank, 2, "127.0.0.1", port, backend = "gloo", timeout_s = 120)
    ep = mod.NcclEndpoint(1 - rank)
    try:
        if rank == 0:
            for t in _payloads():
                ep.send_tensor(t)
            for o in _OBJS:
                ep.send_obj(o)
            back = ep.recv_tensor()
            q.put(("echo", torch.equal(back, _payloads()[0])))
        else:
            ok = []
            for t in _payloads():
                r = ep.recv_tensor()
                same = r.dtype == t.dtype and tuple(r.shape) == tuple(t.shape) and \
                    torch.equal(r.contiguous().view(-1).view(torch.uint8), t.contiguous().view(-1).view(torch.uint8))
                ok.append(same)
            objs = [ep.recv_obj() for _ in _OBJS]
            # recv into a preallocated buffer of the right shape
            ep.send_tensor(_payloads()[0])
            q.put(("tensors", ok))
            q.put(("objs", objs == _OBJS))
    finally:
        dist.destroy_process_group()


def test_gloo_roundtrip_all_dtypes_and_objects():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target = _worker, args = (r, port, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join(120)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    res = {}
    while not q.empty():
        k, v = q.get()
        res[k] = v
    assert res.get("tensors") and all(res["tensors"]), res
    assert res.get("objs") is True
    assert res.get("echo") is True


def test_requires_initialized_group():
    mod = _load()
    if dist.is_initialized():
        return
    try:
        mod.NcclEndpoint(1)
    except mod.NcclTransportError:
        return
    raise AssertionError("expected NcclTransportError")


if __name__ == "__main__":
    test_requires_initialized_group()
    test_gloo_roundtrip_all_dtypes_and_objects()
    print("ok")
