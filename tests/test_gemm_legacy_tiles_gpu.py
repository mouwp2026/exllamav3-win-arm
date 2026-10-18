"""Saved real-pack BC experts prove the native legacy-row policy on GB10.

Run this file alone in a fresh Python process after the GPU owner stops serving.
Supply the own-byte-validated control artifacts produced by spark_bc_row_control:

  EXL3_INT8_GEMV=0 EXL3_GEMM_LEGACY_TILES=1 \
  SPARK_BC_MODEL=/home/cruzspark/models/flashnext-exl3-4.05bpw \
  SPARK_BC_GOLDEN=.../bc-row-control-405-b532-baseline.safetensors \
  SPARK_BC_TUNE_SNAPSHOT=.../coop_autotune_v1.bin \
  python -m pytest tests/test_gemm_legacy_tiles_gpu.py -q

Repeat with EXL3_GEMM_LEGACY_TILES=0 and the candidate b532 control to prove the
disabled path is unchanged. Each run uses a private copy of the preserved tuning
cache, loads one real MoE layer, and tests all saved expert rows1..32 through the
native eager/capture/replay lifecycle with changing expert pointers. No timing
claims, synthetic replacement weights, source edits, or shared-cache writes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

GOLDEN_REVISIONS = {
    "1": "94ba01d50a13fa9ff672473f2d0eef8b51a71e99",
    "0": "b5322c98c5760105de04f7cf7c28bece17fbc78e",
}


def exact(a,b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))


@pytest.fixture(scope="module")
@torch.inference_mode()
def real_layer(tmp_path_factory):
    names = ("SPARK_BC_MODEL","SPARK_BC_GOLDEN","SPARK_BC_TUNE_SNAPSHOT")
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Provide the actual model, validated golden control, and preserved cache snapshot")
    mode = os.environ.get("EXL3_GEMM_LEGACY_TILES")
    assert mode in GOLDEN_REVISIONS, "Set the native policy explicitly to0 or1 before process start"
    assert os.environ.get("EXL3_INT8_GEMV") == "0", "Control uses the unchanged FP16-activation path"
    assert "exllamav3" not in sys.modules, "Run this file alone so the private tune cache is selected before import"
    model_path = Path(os.environ["SPARK_BC_MODEL"]).resolve()
    golden_path = Path(os.environ["SPARK_BC_GOLDEN"])
    snapshot = Path(os.environ["SPARK_BC_TUNE_SNAPSHOT"])
    cache_data = snapshot.read_bytes()
    cache_hash = hashlib.sha256(cache_data).hexdigest()
    with safe_open(str(golden_path),framework="pt",device="cpu") as sf:
        meta=sf.metadata() or {}
        assert meta.get("bc_control_format") == "1" and meta.get("comparison_valid") == "true"
        assert meta.get("model") == str(model_path)
        assert meta.get("cache_snapshot_sha256") == cache_hash
        assert json.loads(meta["runtime"])["git_commit"] == GOLDEN_REVISIONS[mode]
        ids=json.loads(meta["bc_expert_ids"])
        assert ids == sorted(set(ids)) and ids
        records={}
        for expert in ids:
            x=sf.get_tensor(f"expert{expert:03d}.input").clone().contiguous()
            y=sf.get_tensor(f"expert{expert:03d}.output").clone().contiguous()
            assert x.ndim == 2 and x.dtype == torch.half and x.shape[1] == 2560
            assert y.shape == x.shape and y.dtype == torch.float32
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            records[expert]=(x,y)
    assert {len(x) for x,y in records.values()} == set(range(1,33)), "Golden data must cover every BC row count"
    private_cache=tmp_path_factory.mktemp("legacy-gemm-gpu")/"coop_autotune_v1.bin"
    private_cache.write_bytes(cache_data)
    assert private_cache.read_bytes() == cache_data
    os.environ["EXLLAMAV3_TUNE_CACHE"]=str(private_cache)
    from exllamav3 import Config,Model
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.util.tensor import g_tensor_cache
    torch.cuda.set_device(0)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    device=torch.device("cuda:0")
    prop=torch.cuda.get_device_properties(device)
    assert (prop.major,prop.minor,prop.multi_processor_count)==(12,1,48)
    config=Config.from_directory(str(model_path))
    model=Model.from_config(config,component="text")
    layer=model.find_module("model.language_model.layers.0.mlp")
    assert isinstance(layer,BlockSparseMLP)
    try:
        model._load_single(False,device,config,[layer],False)
        g_tensor_cache.drop_all()
        assert layer.hidden_size == layer.expert_size == 2560
        assert layer.intermediate_size == layer.intermediate_size_padded == 640
        assert layer.interm_dtype == torch.half and layer.bc is not None
        assert layer.support_quant_paths and layer.uniform_expert_q and not layer.support_fused
        assert layer.num_experts == layer.num_local_experts == 512 and layer.num_experts_per_tok == 10
        assert all(float(p.inner.K)==4 for group in (layer.gates,layer.ups,layer.downs) for p in group)
        yield SimpleNamespace(layer=layer,records=records,mode=mode,private_cache=private_cache)
    finally:
        torch.cuda.synchronize()
        layer.unload()
        g_tensor_cache.drop_all()
        assert snapshot.read_bytes() == cache_data, "A shared evidence cache was modified"


@pytest.mark.parametrize("minimum,maximum",[(1,8),(9,16),(17,24),(25,32)])
@torch.inference_mode()
def test_actual_experts_eager_capture_and_replay_equal_saved_runtime(real_layer,minimum,maximum):
    state=real_layer
    ids=[e for e,(x,y) in state.records.items() if minimum <= len(x) <= maximum]
    assert ids and {len(state.records[e][0]) for e in ids} == set(range(minimum,maximum+1))
    # Every row count sees changing expert and input pointers. The first encounter
    # is eager, the next records the graph, and later calls exercise graph patching.
    for order in (ids,list(reversed(ids)),ids[1:]+ids[:1]):
        for expert in order:
            x,expected=state.records[expert]
            current=x.to("cuda:0")
            state.layer.bc.run_single_expert(current,expert)
            actual=state.layer.experts_cfg.out_d2[:len(x)].detach().cpu().clone().contiguous()
            assert exact(expected,actual), (
                f"legacy={state.mode} expert={expert} rows={len(x)} differs from validated golden runtime; "
                f"maxabs={(expected-actual).abs().max().item()} changed={(expected!=actual).sum().item()}")
