"""CPU checks for opt-in historical CUDA generic-GEMM autotuning.

The fixture functions are byte-for-byte extracts from exl3_gemm.cu at fork
94ba01d (original) and b5322c98 (candidate before this option). They retain their
original function bodies, including both GEMM and sliced MGEMM hash domains.
Only declarations/headers and the independent driver below are supplied here.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "exllamav3/exllamav3_ext/quant"
FIXTURE_HASHES = {
    "94ba01d": "1635ffd30a04499f472a2e070af648f1ae772d9961a75d9bb1f49df83eba5880",
    "b5322c98": "59510f5451ae1447c5664b658b813e5d49a2a80f12883766fbe2c852102bd48a",
}


def extract(source, name):
    start = source.index(name)
    begin = source.index("{", start)
    depth, end = 1, begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_frozen_hash_source_proofs():
    for revision, expected in FIXTURE_HASHES.items():
        fixture = ROOT / f"tests/fixtures/gemm_hash_{revision}.h"
        data = fixture.read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        # When history is present, verify the independent extracts against the
        # real old source too. The byte-pinned fixtures also work in source ZIPs.
        result = subprocess.run(
            ["git","-C",str(ROOT),"show",
             f"{revision}:exllamav3/exllamav3_ext/quant/exl3_gemm.cu"],
            text=True,capture_output=True)
        if result.returncode == 0:
            selected = "\n\n".join(extract(result.stdout,name) for name in
                ("uint64_t roundup_pow2","uint64_t gemm_autotune_hash",
                 "uint64_t mgemm_autotune_hash")) + "\n"
            assert selected.encode() == data


@pytest.fixture(scope="module")
def host_programs(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required for source-faithful policy checks")
    folder = tmp_path_factory.mktemp("gemm-policy")
    source = (CUDA/"exl3_gemm.cu").read_text()
    current = "\n\n".join(extract(source,name) for name in
        ("uint64_t roundup_pow2","uint64_t gemm_autotune_hash",
         "uint64_t mgemm_autotune_hash"))
    old = (ROOT/"tests/fixtures/gemm_hash_94ba01d.h").read_text()
    baseline = (ROOT/"tests/fixtures/gemm_hash_b5322c98.h").read_text()
    cpp = """#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define MIN(a,b) ((a)<(b)?(a):(b))
#define CEIL_DIVIDE(a,b) (((a)+(b)-1)/(b))
#include "exl3_gemm_policy.h"
namespace old { """ + old + "\n}\nnamespace baseline { " + baseline + "\n}\n"
    cpp += "namespace actual { " + current + "\n}\n"
    cpp += """
int main(int argc,char** argv) {
  bool legacy=argc>1 && std::atoi(argv[1]);
  if (exl3_gemm_legacy_tiles_enabled()!=legacy) return 2;
  for (int available: {4,7})
    if (exl3_gemm_autotune_shape_count(available)!=(legacy?4:available)) return 3;
  int cases=0;
  for (int r=1;r<=257;r++) for (int dir=0;dir<2;dir++)
    for (int kbits=1;kbits<=8;kbits++) for (int fp=0;fp<2;fp++)
      for (int hk=0;hk<2;hk++) {
        int k=dir?640:2560, n=dir?2560:640, rr=r<2?2:r;
        auto ref=legacy ? old::gemm_autotune_hash(rr,k,n,kbits,fp,0,5,48,2,hk)
                        : baseline::gemm_autotune_hash(rr,k,n,kbits,fp,0,5,48,2,hk);
        auto got=actual::gemm_autotune_hash(rr,k,n,kbits,fp,0,5,48,2,hk);
        if(ref!=got) return 4;
        for (int slices: {1,6,24,32}) {
          ref=legacy ? old::mgemm_autotune_hash(rr,k,n,kbits,fp,0,5,48,2,slices,slices/2+1,hk)
                     : baseline::mgemm_autotune_hash(rr,k,n,kbits,fp,0,5,48,2,slices,slices/2+1,hk);
          got=actual::mgemm_autotune_hash(rr,k,n,kbits,fp,0,5,48,2,slices,slices/2+1,hk);
          if(ref!=got) return 5;
          cases++;
        }
        cases++;
      }
  std::printf("%d\\n",cases);
}
"""
    cpp = cpp.replace("#include <cstdlib>","#include <cstdlib>\n#include <initializer_list>")
    src = folder/"policy.cpp"
    src.write_text(cpp)
    binaries={}
    for rocm in (False,True):
        exe=folder/("policy-rocm" if rocm else "policy-cuda")
        command=[compiler,"-std=c++17","-O2","-I",str(CUDA),str(src),"-o",str(exe)]
        if rocm:
            command.insert(1,"-DUSE_ROCM")
        subprocess.run(command,check=True,capture_output=True,text=True)
        binaries[rocm]=exe
    return binaries


@pytest.mark.parametrize("value", [None,"","0","1"])
@pytest.mark.parametrize("rocm", [False,True])
def test_enabled_and_disabled_hash_domains_match_independent_sources(host_programs,value,rocm):
    env=dict(os.environ)
    env.pop("EXL3_GEMM_LEGACY_TILES",None)
    if value is not None:
        env["EXL3_GEMM_LEGACY_TILES"]=value
    expected=not rocm and value not in (None,"","0")
    result=subprocess.run([str(host_programs[rocm]),str(int(expected))],
                          env=env,text=True,capture_output=True,check=True)
    assert int(result.stdout)==257*2*8*2*2*5


def test_both_automatic_candidate_loops_use_the_same_policy():
    source=(CUDA/"exl3_gemm.cu").read_text()
    # The shared hash applies to GEMM and MGEMM. Both candidate loops must be
    # restricted with it, or a wider tile could populate a small-row cache key.
    assert source.count("candidate_shape_idx <= exl3_gemm_autotune_shape_count(EXL3_GEMM_NUM_SHAPES);")==2
    for function_name in ("int exl3_gemm_gr","int exl3_mgemm_gr"):
        body=extract(source,function_name)
        assert "exl3_gemm_autotune_shape_count(EXL3_GEMM_NUM_SHAPES)" in body
        assert "if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K, half_k)) continue;" in body


def test_cuda_graph_cache_miss_heuristic_remains_historical():
    # A sliced-MGEMM graph miss cannot tune while recording. It falls through to
    # the CUDA heuristic, which is still the exact 94ba01d body and only returns
    # shapes1..4. Pin that independent old body so this escape path is reviewed
    # if the fallback is ever widened.
    source=(CUDA/"exl3_kernel_map.cu").read_text()
    cuda_branch=source[source.index("#else\n\nint select_gemm_shape"):]
    body=extract(cuda_branch,"int select_gemm_shape")
    assert hashlib.sha256(body.encode()).hexdigest()=="e4b677a4c500a2f65a70e9e789e75d4d98dd9d0daf390240d999f7eb0e66bbd3"
