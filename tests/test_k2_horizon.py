"""Small K2-Horizon model contracts from the public 6.50bpw config/index and modeling source."""
import json
import math
import sys
import types

# Importing exllamav3.ext normally compiles CUDA; these tests exercise CPU contracts only.
# The real extension's entrypoints are not called by model construction or torch helpers.
sys.modules.setdefault("exllamav3.ext", types.ModuleType("exllamav3.ext"))
sys.modules["exllamav3.ext"].exllamav3_ext = types.SimpleNamespace(
    silu_mul=lambda *args: None,
    gelu_mul=lambda *args: None,
    silu_oai_mul=lambda *args: None,
    relu2_mul=lambda *args: None,
    relu_mul=lambda *args: None,
    BC_LinearFP16=lambda *args: None,
)

paged = types.ModuleType("exllamav3.modules.attention_fn.triton_paged")
paged._qc_staging = 0
for name in (
    "fn_triton_paged_attn", "fn_triton_paged_attn_longq", "fn_triton_paged_attn_decode",
    "fn_triton_paged_attn_prefill", "fn_triton_varlen_attn",
    "fn_triton_paged_attn_decode_qc", "fn_triton_paged_attn_prefill_qc", "fn_triton_attn_nocache",
):
    setattr(paged, name, lambda *args: None)
sys.modules[paged.__name__] = paged

# Other attention/recurrent Triton kernels are not exercised by these CPU tests.
def _triton_stub(name):
    module = types.ModuleType(name)
    def missing(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return lambda *args, **kwargs: None
    module.__getattr__ = missing
    sys.modules[name] = module

paged.__getattr__ = lambda attr: _triton_stub_attr(attr)

def _triton_stub_attr(attr):
    if attr.startswith("__"):
        raise AttributeError(attr)
    return lambda *args, **kwargs: None

for name in (
    "exllamav3.modules.attention_fn.mla_triton",
    "exllamav3.modules.attention_fn.dsa_triton",
    "exllamav3.modules.attention_fn.qsa_triton",
    "exllamav3.modules.gated_delta_net_fn",
):
    _triton_stub(name)

import pytest
import torch
import torch.nn.functional as F

from exllamav3.model.config import Config


@pytest.fixture
def model_config(tmp_path):
    # Reduced version of the public 6.50bpw config; two dense layers then MoVA/MoE.
    config = {
        "architectures": ["K2HorizonForCausalLM"], "model_type": "k2_horizon",
        "hidden_size": 128, "head_dim": 64, "num_attention_heads": 2,
        "num_key_value_heads": 1, "num_hidden_layers": 3, "vocab_size": 128,
        "intermediate_size": 256, "moe_intermediate_size": 128,
        "num_experts": 4, "num_experts_per_tok": 2,
        "mova_num_experts": 4, "mova_num_experts_per_tok": 2,
        "mlp_only_layers": [0, 1], "decoder_sparse_step": 1,
        "num_shared_experts": 1, "layernorm_num_groups": 2,
        "norm_topk_prob": True, "router_score_func": "sigmoid",
        "router_scaling_factor": 2.5, "moe_gate_bias": True,
        "attention_gate_func": "softplus", "rms_norm_eps": 1e-6,
        "rope_head_dim": 64, "rope_parameters": {"rope_type": "default", "rope_theta": 1e7},
        "max_position_embeddings": 1024, "hidden_act": "silu",
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_registry_config_and_tensor_names(model_config):
    from exllamav3.architecture.k2_horizon import K2HorizonModel
    from exllamav3.modules import GatedMLP, BlockSparseMLP
    from exllamav3.modules.k2_horizon import MoVAValueProjection, K2GroupedRMSNorm

    cfg = Config.from_directory(str(model_config))
    model = K2HorizonModel(cfg)
    assert cfg.rope_settings.rope_theta == 1e7
    assert model.config is cfg
    assert len(model.modules) == 6
    dense = model.modules[1]
    sparse = model.modules[3]
    assert isinstance(dense.mlp, GatedMLP)
    assert dense.attn.v_proj.key == "model.layers.0.self_attn.v_proj"
    assert isinstance(sparse.mlp, BlockSparseMLP)
    assert sparse.mlp.router_type == "dots"
    assert sparse.mlp.routing_gate.key == "model.layers.2.mlp.gate"
    assert sparse.mlp.shared_experts.key == "model.layers.2.mlp.shared_experts"
    assert isinstance(sparse.attn.v_proj, MoVAValueProjection)
    assert sparse.attn.v_proj.router.key == "model.layers.2.self_attn.v_router"
    assert sparse.attn.v_proj.experts[0].key == "model.layers.2.self_attn.v_experts.0"
    assert sparse.attn.g_proj.key == "model.layers.2.self_attn.gate_proj"
    assert sparse.attn.full_gate and sparse.attn.gate_softplus
    assert dense.attn.full_gate and dense.attn.gate_softplus
    assert sparse.mlp.e_score_correction_bias_key == "gate.bias"
    assert isinstance(sparse.attn_norm, K2GroupedRMSNorm)
    assert model.modules[-1].key == "lm_head"


def test_all_expert_dots_calibration_ignores_selection_bias():
    from exllamav3.modules.block_sparse_mlp_routing import routing_dots

    logits = torch.tensor([[2., 0., -2.], [-2., 0., 2.]], dtype=torch.half)
    bias = torch.tensor([0., .9, -.4])
    cfg = types.SimpleNamespace(
        gate_tensor=torch.eye(3, dtype=torch.half), num_experts=3,
        e_score_correction_bias=bias, routed_scaling_factor=2.5,
    )
    selected, weights = routing_dots(2, cfg, logits, {"activate_all_experts": True})
    assert selected.tolist() == [[0, 1, 2], [0, 1, 2]]
    scores = logits.sigmoid().float()
    expected = (scores * (2.5 / scores.sum(dim=-1, keepdim=True))).half()
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    # Selection bias would produce substantially different contribution weights.
    biased = scores + bias
    biased_weights = (biased * (2.5 / biased.sum(dim=-1, keepdim=True))).half()
    assert not torch.allclose(weights, biased_weights, atol=.1)


def test_value_router_bias_selects_but_does_not_weight():
    from exllamav3.modules.k2_horizon import mova_routes

    logits = torch.tensor([[1.0, 0.0, -2.0], [-3.0, 0.0, 2.0]])
    bias = torch.tensor([0.0, 1.0, 0.0])
    selected, weights = mova_routes(logits, bias, 2, 2.5)
    expected = torch.sigmoid(logits)
    expected_choice = (expected + bias).topk(2, dim=-1).indices
    torch.testing.assert_close(selected, expected_choice)
    expected_weights = expected.gather(1, selected)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * 2.5
    torch.testing.assert_close(weights, expected_weights)


def test_single_value_expert_uses_raw_probability_times_scale():
    from exllamav3.modules.k2_horizon import mova_routes
    logits = torch.tensor([[2., 0., -2.]])
    bias = torch.tensor([0., .45, .4])
    picks, weights = mova_routes(logits, bias, 1, 2.5)
    assert picks.tolist() == [[1]]
    torch.testing.assert_close(weights, torch.tensor([[1.25]]))


def test_value_expert_silu_before_weighted_sum():
    from exllamav3.modules.k2_horizon import combine_mova_values

    hidden = torch.tensor([[1.0, 2.0], [3.0, -2.0]])
    selected = torch.tensor([[2, 0], [1, 2]])
    weights = torch.tensor([[0.4, 0.6], [0.8, 0.2]])
    matrices = [torch.tensor([[i + 1., 0.], [0., -i - 1.]]) for i in range(3)]
    actual = combine_mova_values(hidden, selected, weights, lambda i, rows: F.linear(rows, matrices[i]), 2)
    expected = torch.stack([
        sum(F.silu(F.linear(hidden[t], matrices[int(selected[t, k])])) * weights[t, k]
            for k in range(2)) for t in range(2)
    ])
    torch.testing.assert_close(actual, expected)


def test_final_grouped_norm_emits_half_for_quantized_lm_head(model_config, monkeypatch):
    from exllamav3.architecture.k2_horizon import K2HorizonModel
    from exllamav3.modules.rmsnorm import ext

    model = K2HorizonModel(Config.from_directory(str(model_config)))
    final_norm = model.modules[-2]
    block_norm = model.modules[1].attn_norm
    for norm in (final_norm, block_norm):
        norm.norm.weight = torch.nn.Parameter(torch.ones(128, dtype=torch.half), requires_grad=False)

    residual = torch.randn(2, 128, dtype=torch.float)
    assert block_norm.forward_torch(residual, {}).dtype == torch.float
    assert final_norm.forward_torch(residual, {}).dtype == torch.half
    assert final_norm.norm.out_dtype == torch.half
    assert model.modules[-1].key == "lm_head"

    # CPU stand-in for the CUDA kernel: check the real forward allocation and
    # grouped reshape on the FP32-residual -> half-precision-head boundary.
    def cpu_rms_norm(x, weight, y, eps, bias, scale, span_heads, residual_out, groups):
        assert not span_heads and not residual_out and bias == 0.0 and scale == 1.0
        grouped = x.float().reshape(-1, groups, x.shape[-1])
        normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
        y.copy_((normalized * weight.reshape(groups, -1)).reshape_as(y))

    monkeypatch.setattr(ext, "rms_norm", cpu_rms_norm, raising=False)
    torch.testing.assert_close(final_norm.forward(residual, {}), final_norm.forward_torch(residual, {}))
    assert final_norm.forward(residual, {}).dtype == torch.half
    assert block_norm.forward(residual, {}).dtype == torch.float


@pytest.mark.parametrize("out_dtype", [None, torch.half])
def test_grouped_norm_tp_roundtrip_preserves_output_dtype(out_dtype, monkeypatch):
    from exllamav3.modules.k2_horizon import K2GroupedRMSNorm

    norm = K2GroupedRMSNorm(None, "model.norm", 2, 1e-6, out_dtype=out_dtype)
    norm.norm.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.half), requires_grad=False)
    norm.norm.device = torch.device("cpu")

    class Pipe:
        def send(self, tensor):
            return tensor
        def recv(self, tensor, cuda):
            return tensor

    exported = norm.tp_export({}, Pipe())
    assert exported["norm"]["kwargs"]["out_dtype"] == out_dtype
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    imported = K2GroupedRMSNorm.tp_import(
        {"device": torch.device("cpu"), "consumer": Pipe()}, exported, {})
    assert imported.norm.out_dtype == out_dtype
    assert imported.groups == 2
    assert imported.forward_torch(torch.ones(1, 4, dtype=torch.float), {}).dtype == (out_dtype or torch.float)


def test_group_norm_normalizes_each_half_not_full_vector():
    from exllamav3.modules.k2_horizon import K2GroupedRMSNorm

    norm = K2GroupedRMSNorm(None, "test.norm", 2, 1e-6)
    norm.norm.weight = torch.nn.Parameter(torch.tensor([1., 2., 3., 4.]), requires_grad=False)
    x = torch.tensor([[[1., 2., 3., 6.]]])
    expected = (x.reshape(1, 1, 2, 2) * torch.rsqrt(x.reshape(1, 1, 2, 2).square().mean(-1, keepdim=True) + 1e-6)).reshape(1, 1, 4) * norm.norm.weight
    torch.testing.assert_close(norm.forward_torch(x, {}), expected)


def test_softplus_gate_is_per_channel_with_log2_beta():
    from exllamav3.modules.k2_horizon import k2_attention_gate

    output = torch.ones((1, 2, 2, 2))
    gate = torch.tensor([[[0., 1., -1., 3.], [2., -2., 0., 0.]]])
    result = k2_attention_gate(output, gate)
    torch.testing.assert_close(result, F.softplus(gate.reshape(1, 2, 2, 2), beta=math.log(2)))
    assert result.shape == output.shape


def test_rejects_unsupported_partial_rope(model_config):
    config_file = model_config / "config.json"
    data = json.loads(config_file.read_text())
    data["rope_head_dim"] = 32
    config_file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="rope_head_dim"):
        Config.from_directory(str(model_config))


@pytest.mark.parametrize("window_config", [
    {"use_sliding_window": True, "sliding_window": 4096},
    {"use_sliding_window": True, "sliding_window": None},
    {"use_sliding_window": True},
    {"use_sliding_window": False, "sliding_window": 4096},
    {"sliding_window": 4096},
])
def test_rejects_unsupported_sliding_window_before_model_creation(model_config, window_config):
    config_file = model_config / "config.json"
    data = json.loads(config_file.read_text())
    data.update(window_config)
    config_file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="K2 Horizon.*sliding.window"):
        Config.from_directory(str(model_config))


@pytest.mark.parametrize("window_config", [
    {"use_sliding_window": False, "sliding_window": None},
    {"sliding_window": None},
    {"use_sliding_window": False},
    {},
])
def test_global_attention_config_stays_accepted(model_config, window_config):
    from exllamav3.architecture.k2_horizon import K2HorizonModel
    config_file = model_config / "config.json"
    data = json.loads(config_file.read_text())
    data.update(window_config)
    config_file.write_text(json.dumps(data))
    model = K2HorizonModel(Config.from_directory(str(model_config)))
    assert model.modules[1].attn.sliding_window == -1


def test_missing_learned_router_biases_fail_before_weight_load(model_config):
    from exllamav3.architecture.k2_horizon import K2HorizonModel
    model = K2HorizonModel(Config.from_directory(str(model_config)))
    sparse = model.modules[3]
    with pytest.raises(ValueError, match=r"model.layers.2.self_attn.v_router.bias"):
        sparse.attn.v_proj.load(torch.device("cpu"))
    with pytest.raises(ValueError, match=r"model.layers.2.mlp.gate.bias"):
        sparse.mlp.load(torch.device("cpu"))


def test_mova_rejects_incorrect_selection_bias_shape_before_loading_weights(tmp_path):
    from safetensors.torch import save_file
    from exllamav3.loader import SafetensorsCollection
    from exllamav3.modules.k2_horizon import MoVAValueProjection
    key = "model.layers.0.self_attn"
    save_file({f"{key}.v_router.bias": torch.ones(2)}, str(tmp_path / "bad.safetensors"))
    class LocalConfig:
        stc = SafetensorsCollection(str(tmp_path), load_method="python")
    projection = MoVAValueProjection(LocalConfig(), key + ".v_proj", 4, 2, 3, 2, 2.5)
    with pytest.raises(ValueError, match=r"v_router.bias.*shape \[3\]"):
        projection.load(torch.device("cpu"))
    assert projection.router.inner is None


def test_router_projection_excludes_selection_bias(tmp_path):
    from safetensors.torch import save_file
    from exllamav3.modules import Linear
    save_file({"router.weight": torch.eye(2), "router.bias": torch.tensor([22., 27.])},
              str(tmp_path / "weights.safetensors"))
    class LocalConfig:
        from exllamav3.loader import SafetensorsCollection
        stc = SafetensorsCollection(str(tmp_path), load_method="python")
    router = Linear(LocalConfig(), "router", 2, 2, pad_to=1, load_bias=False)
    router.load(torch.device("cpu"))
    assert router.inner.bias is None
    torch.testing.assert_close(router.forward(torch.tensor([[1., 2.]], dtype=torch.half), {}),
                               torch.tensor([[1., 2.]], dtype=torch.half))


def test_separate_bias_overlay_is_indexed(model_config, tmp_path):
    from safetensors.torch import save_file
    overlay = tmp_path / "learned-biases.safetensors"
    save_file({
        "model.layers.2.self_attn.v_router.bias": torch.tensor([22., 23., 24., 25.]),
        "model.layers.2.mlp.gate.bias": torch.tensor([27., 28., 29., 30.]),
    }, str(overlay))
    cfg = Config.from_directory(str(model_config), routing_bias_overlay=str(overlay), load_method="python")
    assert cfg.stc.has_tensor("model.layers.2.self_attn.v_router.bias")
    assert cfg.stc.has_tensor("model.layers.2.mlp.gate.bias")
    torch.testing.assert_close(cfg.stc.get_tensor("model.layers.2.mlp.gate.bias"),
                               torch.tensor([27., 28., 29., 30.]))


def test_mova_loaded_projection_matches_independent_cpu_oracle(tmp_path):
    """Golden from k2_mova_oracle.py, pinned to HF modeling_k2_horizon.py c88277ce.

    The reference independently checks logits, biased picks, normalized weights,
    SiLU-before-reduction and KV ordering; this test goes through *loaded* linears.
    """
    from safetensors.torch import save_file
    from exllamav3.loader import SafetensorsCollection
    from exllamav3.modules.k2_horizon import MoVAValueProjection

    router_w = torch.tensor([[2, 0, -2, 0], [0, 2, 0, -2], [-2, 0, 2, 0]], dtype=torch.float16)
    expert_w = torch.tensor([
        [[1, 2, 3, 4], [-1, 0, 1, 2]],
        [[-1, 1, .5, 2], [2, -2, 1, .25]],
        [[.5, -.5, 2, -1], [1, 1, -1, -2]],
    ], dtype=torch.float16)
    prefix = "model.layers.0.self_attn"
    tensors = {f"{prefix}.v_router.weight": router_w,
               f"{prefix}.v_router.bias": torch.tensor([0, .45, .4], dtype=torch.float32)}
    for i, w in enumerate(expert_w):
        tensors[f"{prefix}.v_experts.{i}.weight"] = w
    save_file(tensors, str(tmp_path / "weights.safetensors"))

    class LocalConfig:
        stc = SafetensorsCollection(str(tmp_path), load_method="python")

    projection = MoVAValueProjection(LocalConfig(), f"{prefix}.v_proj", 4, 2, 3, 2, 2.5)
    projection.load(torch.device("cpu"))
    x = torch.eye(4, dtype=torch.float16).reshape(2, 2, 4)
    values = projection.forward(x, {}, out_dtype=torch.float32)
    # Independent oracle's float32 mixed_values golden (before KV transpose).
    expected = torch.tensor([
        [.9223722249416851, 1.1658379608373681],
        [.9949490432967847, .28161654052703783],
        [3.0910078047988194, .23292066820833945],
        [.30489382977797885, -.4136352073731773],
    ]).reshape(2, 2, 2)
    torch.testing.assert_close(values, expected, rtol=0, atol=.004)
    assert projection.router.inner.bias is None
    assert projection.bias is not None
    torch.testing.assert_close(projection.get_tensors()[f"{prefix}.v_router.bias"], projection.bias)
    projection.unload()
    assert projection.bias is None and projection.get_tensors() == {}


def test_mova_exl3_router_uses_physical_128_columns_but_routes_64_experts(tmp_path, monkeypatch):
    """CPU stand-in for the EXL3 C/B check on the 2560 -> 64 Spark router."""
    from safetensors.torch import save_file
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.loader import SafetensorsCollection
    from exllamav3.modules.k2_horizon import MoVAValueProjection

    key = "model.layers.3.self_attn"
    router = key + ".v_router"
    bias = torch.zeros(64)
    bias[1], bias[2] = .45, .4
    tensors = {
        router + ".suh": torch.ones(2560, dtype=torch.half),
        router + ".svh": torch.ones(128, dtype=torch.half),
        router + ".trellis": torch.zeros((160, 8, 128), dtype=torch.int16),
        router + ".bias": bias,
    }
    for idx in range(64):
        weight = torch.zeros((1, 2560), dtype=torch.half)
        if idx == 1:
            weight[0, 0] = 1
        tensors[f"{key}.v_experts.{idx}.weight"] = weight
    save_file(tensors, str(tmp_path / "router.safetensors"))

    class RouterBC:
        def __init__(self, trellis, suh, svh, K, bias, mcg, mul1, cache):
            assert bias is None  # learned bias must not enter EXL3 logits
            self.k, self.n = suh.numel(), svh.numel()
            assert trellis.shape[:2] == (self.k // 16, self.n // 16)

        def run_alloc(self, x, out_features, out_float):
            if x.shape[-1] != self.k or out_features != self.n:
                raise RuntimeError("C and B have incompatible shapes")
            logits = x.new_full((x.shape[0], self.n), 30)
            logits[:, :64] = -10
            logits[:, 0], logits[:, 1], logits[:, 2] = 2, 0, -2
            return logits

    monkeypatch.setattr(ext, "BC_LinearEXL3", RouterBC, raising=False)

    class LocalConfig:
        stc = SafetensorsCollection(str(tmp_path), load_method="python")

    projection = MoVAValueProjection(LocalConfig(), key + ".v_proj", 2560, 1, 64, 1, 2.5)
    projection.load(torch.device("cpu"))
    x = torch.ones((6, 2560), dtype=torch.half)
    values = projection.forward(x, {})
    assert projection.router.out_features == projection.router.inner.out_features == 128
    assert projection.router.out_features_unpadded == 64
    assert projection.router.trim_padded_out and not projection.router.load_bias
    assert values.shape == (6, 1)
    torch.testing.assert_close(values.float(), torch.full((6, 1), F.silu(torch.tensor(1.)).item() * 1.25),
                               rtol=0, atol=.002)


def test_mova_tp_allocation_never_silently_claims_expert_parallelism():
    from exllamav3.modules.k2_horizon import MoVAValueProjection

    key = "model.layers.0.self_attn"
    class LocalConfig:
        class stc:
            @staticmethod
            def get_tensor_sizes(key):
                return [64]
    projection = MoVAValueProjection(LocalConfig(), f"{key}.v_proj", 4, 2, 3, 2, 2.5)
    with pytest.raises(NotImplementedError, match="tensor.parallel|tensor parallel"):
        projection.make_tp_allocation({})
    with pytest.raises(NotImplementedError, match="tensor.parallel|tensor parallel"):
        projection.tp_import_split({}, {}, {}, (True, 0, 2))


def test_k2_model_disables_unimplemented_tensor_parallel_loading(model_config):
    from exllamav3.architecture.k2_horizon import K2HorizonModel
    config_file = model_config / "config.json"
    data = json.loads(config_file.read_text())
    # Isolate TP support from the unrelated full-gate softplus workstream.
    data["attention_gate_func"] = None
    config_file.write_text(json.dumps(data))
    model = K2HorizonModel(Config.from_directory(str(model_config)))
    assert model.caps["supports_tp"] is False


def test_mova_is_not_mistaken_for_fusible_linear_by_attention(model_config):
    from exllamav3.architecture.k2_horizon import K2HorizonModel
    config_file = model_config / "config.json"
    data = json.loads(config_file.read_text())
    data["attention_gate_func"] = None  # independent of full-gate workstream
    config_file.write_text(json.dumps(data))
    attn = K2HorizonModel(Config.from_directory(str(model_config))).modules[3].attn
    # The real load_local inspects K/V quant_type before it can fuse two linears.
    # A quantized K must not make the custom MoVA V look fusible.
    attn.rope_settings = None
    attn.q_norm = attn.k_norm = None  # not loaded: isolate fusion probe
    attn.k_proj.quant_type = "exl3"
    attn.load_local(torch.device("cuda:0"))
    assert attn.multi_kv is None


def test_sparse_attention_k_projection_trims_padding_before_head_reshape(model_config):
    from exllamav3.architecture.k2_horizon import K2HorizonModel
    config_file = model_config / "config.json"
    data = json.loads(config_file.read_text())
    data["attention_gate_func"] = None
    config_file.write_text(json.dumps(data))
    attn = K2HorizonModel(Config.from_directory(str(model_config))).modules[3].attn
    # One KV head x 64 channels gets padded to 128 by Linear. Attention
    # finish_qkv reshapes to exactly one head x 64 channels.
    assert attn.k_proj.out_features > attn.num_kv_heads * attn.head_dim
    assert attn.k_proj.trim_padded_out


def test_attention_project_qkv_uses_routed_values_in_kv_head_order(tmp_path):
    from safetensors.torch import save_file
    from exllamav3.loader import SafetensorsCollection
    from exllamav3.modules import Attention, Linear
    from exllamav3.modules.k2_horizon import MoVAValueProjection

    key = "model.layers.0.self_attn"
    router_w = torch.tensor([[2, 0, -2, 0], [0, 2, 0, -2], [-2, 0, 2, 0]], dtype=torch.half)
    experts = torch.tensor([
        [[1, 2, 3, 4], [-1, 0, 1, 2]],
        [[-1, 1, .5, 2], [2, -2, 1, .25]],
        [[.5, -.5, 2, -1], [1, 1, -1, -2]],
    ], dtype=torch.half)
    tensors = {f"{key}.q_proj.weight": torch.eye(4, dtype=torch.half),
               f"{key}.k_proj.weight": torch.eye(2, 4, dtype=torch.half),
               f"{key}.o_proj.weight": torch.eye(4, dtype=torch.half),
               f"{key}.v_router.weight": router_w,
               f"{key}.v_router.bias": torch.tensor([0, .45, .4])}
    tensors.update({f"{key}.v_experts.{i}.weight": w for i, w in enumerate(experts)})
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    class LocalConfig:
        stc = SafetensorsCollection(str(tmp_path), load_method="python")
        class infer_params:
            no_reconstruct = True
    cfg = LocalConfig()
    v_proj = MoVAValueProjection(cfg, key + ".v_proj", 4, 2, 3, 2, 2.5)
    attn = Attention(cfg, key, 0, 4, 2, 2, 1, None,
                     key_q="q_proj", k_proj=Linear(cfg, key + ".k_proj", 4, 2,
                                                   trim_padded_out=True),
                     v_proj=v_proj, key_o="o_proj")
    attn.load(torch.device("cpu"))
    x = torch.eye(4, dtype=torch.half).reshape(2, 2, 4)
    q, k, v, g = attn.project_qkv(x, {})
    assert q.shape == (2, 2, 2, 2) and k.shape == v.shape == (2, 2, 1, 2)
    assert g is None
    expected = torch.tensor([
        [.9223722249416851, 1.1658379608373681],
        [.9949490432967847, .28161654052703783],
        [3.0910078047988194, .23292066820833945],
        [.30489382977797885, -.4136352073731773],
    ]).reshape(2, 2, 1, 2)
    torch.testing.assert_close(v.float(), expected, atol=.004, rtol=0)
