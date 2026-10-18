"""CPU-only dispatch regression for the fork's decode int8 and upstream's tiled path."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch


def load_hc():
    """Load this one module without importing package startup or building the CUDA extension."""
    ext = types.SimpleNamespace()
    modules = {}
    for name in ("exllamav3", "exllamav3.modules", "exllamav3.util"):
        package = types.ModuleType(name)
        package.__path__ = []
        modules[name] = package
    module = types.ModuleType("exllamav3.modules.module")
    class Module:
        def __init__(self, config=None, key=None, qmap=None):
            self.config, self.key = config, key
    module.Module = Module
    modules[module.__name__] = module
    module = types.ModuleType("exllamav3.modules.rmsnorm")
    module.RMSNorm = type("RMSNorm", (), {})
    modules[module.__name__] = module
    module = types.ModuleType("exllamav3.model.config")
    module.Config = type("Config", (), {})
    modules[module.__name__] = module
    module = types.ModuleType("exllamav3.ext")
    module.exllamav3_ext = ext
    modules[module.__name__] = module
    module = types.ModuleType("exllamav3.util.tensor")
    module.g_tensor_cache = types.SimpleNamespace()
    modules[module.__name__] = module
    module = types.ModuleType("exllamav3.util.backend")
    module.HC_FOLD = False
    modules[module.__name__] = module
    name = "exllamav3.modules.hyperconnections"
    path = Path(__file__).resolve().parents[1] / "exllamav3/modules/hyperconnections.py"
    spec = importlib.util.spec_from_file_location(name, path)
    hc = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(hc)
    return hc


def test_decode_int8_dispatches_without_fp16_tables():
    hc = load_hc()
    calls = []
    hc.ext.gr_mix_int8 = lambda *args: calls.append("int8")
    hc.ext.gr_mix = lambda *args: calls.append("fp16")
    m = hc.GatedResidual(config=None, key="site", hc_mult=4, hidden_size=8, rms_norm_eps=1e-6)
    m.proj_m = 3
    m.proj_h = m.upx_h = None
    m.fn_q = torch.zeros((3, 32), dtype=torch.int8)
    m.fn_s = torch.ones(3)
    m.upx_q = torch.zeros((4, 2, 1, 4), dtype=torch.int8)
    m.upx_s = torch.ones((4, 8))
    m.w_h = torch.ones(32, dtype=torch.half)
    m.tiled = False
    m._mix(torch.zeros((1, 1, 4, 8)), cached=False)
    assert calls == ["int8"]


def test_int8_quantization_preserves_tiled_source_lifecycle():
    hc = load_hc()
    m = hc.GatedResidual(config=None, key="site", hc_mult=4, hidden_size=8, rms_norm_eps=1e-6)
    torch.manual_seed(123)
    m.proj_m = 5
    m.proj_h = torch.randn((64, 32), dtype=torch.half)
    m.norm_w = torch.linspace(0.1, 2.0, 32).reshape(4, 8)
    m.upx_h = torch.randn((4, 2, 1, 4), dtype=torch.half)
    original_proj = m.proj_h.clone()
    original_up = m.upx_h.clone()
    folded = (m.proj_h[:5].float() * m.norm_w.flatten()).half().float()
    scales = folded.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    expected = torch.round(folded / scales[:, None]).clamp_(-128, 127).to(torch.int8)
    m.tiled = True
    m._quantize_int8(4, 8)
    assert m.fn_q.shape == (5, 32) and m.upx_q.shape == (4, 2, 1, 4)
    assert torch.equal(m.fn_q, expected)
    assert torch.equal(m.fn_s, scales)
    assert torch.equal(m.proj_h, original_proj)
    assert torch.equal(m.upx_h, original_up)


def test_fp16_dispatch_uses_unfolded_projection_and_weighted_copy():
    hc = load_hc()
    calls = []
    hc.ext.gr_mix = lambda *args: calls.append(args)
    m = hc.GatedResidual(config=None, key="site", hc_mult=4, hidden_size=8, rms_norm_eps=1e-6)
    m.proj_m = 5
    m.proj_h = torch.randn(64, 32, dtype=torch.half)
    m.upx_h = torch.randn(4, 2, 1, 4, dtype=torch.half)
    m.w_h = torch.ones(32, dtype=torch.half)
    streams = torch.randn(1, 1, 4, 8)
    weighted = streams.reshape(1, 4, 8).clone()
    params = {"gr_weighted": (m, streams.data_ptr(), weighted)}
    m._mix(streams, cached=False, params=params)
    assert calls[0][1] is weighted
    assert torch.equal(calls[0][2], m.proj_h[:5])
    assert "gr_weighted" not in params
