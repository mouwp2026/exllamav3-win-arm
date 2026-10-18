"""Regression gates for the async staging-buffer race fixed in PR #17.

The optional source override lets the same tests reproduce the pre-fix failure.
"""
import importlib.util
import os
from pathlib import Path
import socket
import threading
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(os.environ.get(
    "EXL3_TEST_NET_TRANSPORT_SOURCE",
    Path(__file__).resolve().parents[2] / "exllamav3/model/net_transport.py",
))
spec = importlib.util.spec_from_file_location("net_transport_staging", SOURCE)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def test_staging_reuse_waits_for_previous_copy():
    a, b = socket.socketpair()
    ep = transport.NetEndpoint(a)
    waited = []
    try:
        ep._pinned_buffer = torch.zeros(256, dtype=torch.uint8)
        ep._pinned_size = 256
        ep._staging_event = SimpleNamespace(synchronize=lambda: waited.append(True))
        result = ep._get_pinned_buffer(64)
        assert waited == [True], "staging overwritten before its previous H2D copy completed"
        assert ep._staging_event is None
        assert result.data_ptr() == ep._pinned_buffer.data_ptr()
    finally:
        ep.close()
        b.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA async H2D staging")
def test_async_cuda_receives_do_not_overwrite_inflight_staging():
    a, b = socket.socketpair()
    receiver, sender = transport.NetEndpoint(a), transport.NetEndpoint(b)
    errors = []
    values = [torch.full((1024,), n, dtype=torch.float32) for n in (1, 2)]

    def send():
        try:
            for value in values:
                sender.send_tensor(value)
        except Exception as exc:
            errors.append(exc)
        finally:
            sender.close()

    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    try:
        stream = torch.cuda.Stream()
        # Delay the first H2D so the next recv can expose premature host-buffer reuse.
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
        results = [receiver.recv_tensor(device="cuda:0", stream=stream) for _ in values]
        stream.synchronize()
        thread.join(timeout=10)
        assert not thread.is_alive(), "sender did not finish"
        assert not errors, errors
        for actual, expected in zip(results, values):
            assert torch.equal(actual.cpu(), expected), "an in-flight H2D read reused staging bytes"
    finally:
        receiver.close()
        sender.close()
        thread.join(timeout=10)
