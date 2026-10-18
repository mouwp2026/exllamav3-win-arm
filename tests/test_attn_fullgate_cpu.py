"""CPU-only, isolated Attention gating tests; no native extension, Triton, or GPU.

Run this file on its own: it imports attn.py and bc_attn.py under a private
package name with only their native dependencies stubbed, not their gate logic.
"""
import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1] / "exllamav3"
PREFIX = "k2_cpu_exllamav3"


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _module(name, **members):
    module = types.ModuleType(name)
    module.__dict__.update(members)
    sys.modules[name] = module
    return module


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def isolated_attn():
    # Use a private import namespace so other test modules cannot inherit these stubs.
    _package(PREFIX)
    modules = _package(PREFIX + ".modules")
    _package(PREFIX + ".model")
    _package(PREFIX + ".util")
    _package(PREFIX + ".modules.attention_fn")

    class Module:
        def __init__(self, config, key, parent):
            self.config = config
            self.key = key
            self.caps = {}
            self.device = torch.device("cpu")

        def register_submodule(self, module):
            pass

    modules.Module = Module
    modules.Linear = object
    modules.RMSNorm = object
    modules.LayerNorm = object
    _module(PREFIX + ".model.config", Config=object)
    _module(PREFIX + ".model.model_tp_alloc", TPAllocation=object)
    _module(PREFIX + ".util.rope", RopeSettings=object, RoPE=object)
    _module(PREFIX + ".util.tensor", g_tensor_cache=None,
            get_for_device=lambda params, key, device, default=None: params.get(key, default),
            to2=lambda x, *args: x)
    sys.modules[PREFIX + ".util"].profile_opt = None
    _module(PREFIX + ".util.backend", ROCM=False, ATTN_SPLIT_WARPS_STAGES=(4, 2),
            attn_decode_config=lambda *args, **kwargs: None, QKV_SLICE=True)
    _module(PREFIX + ".modules.attention_fn.smem", smem_limit=lambda *args: 98304)
    _module(PREFIX + ".constants", PAGE_SIZE=256)
    _module(PREFIX + ".modules.multilinear", MultiLinear=object, SlicedMultiLinear=object)
    ext = types.SimpleNamespace(
        mul_sigmoid_=lambda o, g: o.mul_(g.sigmoid()),
        mul_sigmoid_broadcast_=lambda o, g: o.mul_(g.unsqueeze(-1).sigmoid()),
        mul_softplus_broadcast_=lambda o, g: o.mul_(F.softplus(g.unsqueeze(-1))),
    )
    _module(PREFIX + ".ext", exllamav3_ext=ext)
    _module(PREFIX + ".modules.attention_fn.dispatch",
            attn_dispatch=lambda **kwargs: kwargs["q"].clone())
    bc = _load(PREFIX + ".modules.attention_fn.bc_attn", ROOT / "modules/attention_fn/bc_attn.py")
    bc.bc_attn_enable = True
    attn_fn = sys.modules[PREFIX + ".modules.attention_fn"]
    attn_fn.attn_dispatch = sys.modules[PREFIX + ".modules.attention_fn.dispatch"].attn_dispatch
    attn = _load(PREFIX + ".modules.attn", ROOT / "modules/attn.py")
    return attn, bc


class Projection:
    def __init__(self, result):
        self.result = result

    def forward(self, x, params, **kwargs):
        return self.result.clone()


def _attention(attn, gate, *, full_gate=True, gate_softplus=True, dtype=torch.float32):
    q = torch.tensor([[[[1.5, -2.0], [0.75, 3.0]],
                       [[-1.0, 2.5], [4.0, -0.5]]]], dtype=dtype)
    x = torch.zeros(1, 2, 4, dtype=dtype)
    proj = lambda t: Projection(t.reshape(1, 2, -1))
    module = attn.Attention(None, "layer.attn", 0, 4, 2, 2, 2, None,
                            q_proj=proj(q), k_proj=proj(q), v_proj=proj(q),
                            o_proj=Projection(torch.empty(0)), g_proj=proj(gate),
                            full_gate=full_gate, gate_softplus=gate_softplus)
    module.project_o = lambda o, bsz, seqlen, params: o
    return module, q, x


def test_full_gate_softplus_beta_ln2_applies_per_channel_after_attention(isolated_attn):
    attn, _ = isolated_attn
    gate = torch.tensor([[[-4.0, -1.0, 0.0, 1.0],
                          [2.0, 4.0, -2.0, 0.5]]])
    module, q, x = _attention(attn, gate)
    result = module.decode_flash_attn_nc(x, 1, 2, {})
    expected = q.reshape(1, 2, 4) * F.softplus(gate, beta=math.log(2))
    torch.testing.assert_close(result, expected)


def test_bc_declines_full_softplus_gate(isolated_attn, monkeypatch):
    _, bc = isolated_attn
    monkeypatch.setattr(bc, "_qsa_module_eligible", lambda m: True)
    projection = types.SimpleNamespace(out_features=16, out_features_unpadded=16,
                                       quant_type="exl3", inner=types.SimpleNamespace(bc=object()))
    module = types.SimpleNamespace(
        full_gate=True, gate_softplus=True, interleaved_gate=False,
        rope=None, q_norm=None, headwise_gate=False, g_proj=projection,
        multi_qg=object(), v_norm=None, tp_span_heads_norm=False,
        head_dim=8, num_q_heads=2, num_kv_heads=2,
        q_proj=projection, k_proj=projection, v_proj=projection,
        o_proj=projection, multi_kv=object(),
    )
    assert not bc._module_eligible(module)
    module.gate_softplus = False
    assert bc._module_eligible(module)  # existing full sigmoid graph remains enabled


def test_cached_full_softplus_uses_same_per_channel_gate(isolated_attn, monkeypatch):
    attn, _ = isolated_attn
    gate = torch.tensor([[[-6.0, -1.0, 0.0, 3.0],
                          [0.25, 2.0, -3.0, 7.0]]], dtype=torch.float16)
    module, q, x = _attention(attn, gate, dtype=torch.float16)
    # Force eager cached attention, not a fake fused result; no cache/GPU is needed.
    monkeypatch.setattr(attn, "_bc_attn_enable", False)
    result = module.decode_flash_attn(x, 1, 2, {
        "block_table": torch.empty(1, 1, dtype=torch.int32),
        "cache_seqlens": torch.zeros(1, dtype=torch.int32),
    })
    expected = q.reshape(1, 2, 4) * F.softplus(gate, beta=math.log(2))
    torch.testing.assert_close(result, expected)


def test_existing_full_gate_sigmoid_unchanged(isolated_attn):
    attn, _ = isolated_attn
    gate = torch.tensor([[[0.0, -1.0, 2.0, -2.0], [1.0, 3.0, 0.5, -3.0]]])
    module, q, x = _attention(attn, gate, gate_softplus=False)
    torch.testing.assert_close(module.decode_flash_attn_nc(x, 1, 2, {}),
                               q.reshape(1, 2, 4) * gate.sigmoid())


def test_existing_headwise_softplus_unchanged(isolated_attn):
    attn, _ = isolated_attn
    gate = torch.tensor([[[-3.0, 1.0], [0.0, 4.0]]])
    module, q, x = _attention(attn, gate, full_gate=False)
    expected = (q * F.softplus(gate.unsqueeze(-1))).reshape(1, 2, 4)
    torch.testing.assert_close(module.decode_flash_attn_nc(x, 1, 2, {}), expected)


def test_interleaved_softplus_remains_unsupported(isolated_attn):
    attn, _ = isolated_attn
    with pytest.raises(AssertionError, match="interleaved gate"):
        attn.Attention(None, "layer.attn", 0, 4, 2, 2, 2, None,
                       interleaved_gate=True, gate_softplus=True)
