"""Execute the real NOSYNC eligibility expression at its safety boundaries."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "exllamav3/modules/block_sparse_mlp.py"


def expression():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_nosync" for t in n.targets))
    return compile(ast.Expression(body=node.value), str(SOURCE), "eval")


@pytest.mark.parametrize("change,expected", [
    ({}, True),
    ({"enabled": False}, False),
    ({"unified": False}, False),
    ({"local_experts": 511}, False),
    ({"rows": 1}, True),
    ({"rows": 16}, True),
    ({"rows": 17}, False),
    ({"rows": 24}, False),
    ({"rows": 16, "topk": 16}, True),
    ({"rows": 16, "topk": 17}, False),
    ({"buffers": False}, False),
    ({"topk": 5}, False),
    ({"topk": 6}, True),
    ({"capacity": 60}, True),
    ({"capacity": 59}, False),
    ({"concurrency": 11}, False),
])
def test_real_guard(change, expected):
    values = dict(enabled=True, unified=True, local_experts=512, rows=6, topk=10,
                  buffers=True, capacity=256, concurrency=6)
    values.update(change)
    layer = SimpleNamespace(
        num_local_experts=values["local_experts"], num_experts=512,
        _mkd_fused_rows=values["capacity"],
        _mkd_bufs=SimpleNamespace(temp_state_g=SimpleNamespace(
            shape=(values["concurrency"], values["capacity"], 2560))) if values["buffers"] else None,
    )
    assert bool(eval(expression(), {
        "MIXEDK_NOSYNC": values["enabled"], "mixedk_unified_ok": values["unified"],
        "num_tokens": values["rows"], "top_k": values["topk"], "self": layer,
        "MTILE_T1": 16, "TEMP_ROWS_FUSED": 128,
    })) is expected
