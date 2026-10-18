import ast
from pathlib import Path

import torch


_SOURCE = Path(__file__).parents[2] / "exllamav3" / "conversion" / "convert_model.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_NAMES = {"_row_batchable", "calib_chunks", "run_calib_chunks", "batch_states", "unbatch_states"}
_FUNCTIONS = [node for node in _TREE.body if isinstance(node, ast.FunctionDef) and node.name in _NAMES]
_NS = {"torch": torch, "num_ref_states": 2, "CALIB_BATCH": 3}
exec(compile(ast.Module(body=_FUNCTIONS, type_ignores=[]), str(_SOURCE), "exec"), _NS)


def _preserve(row, result):
    result["sentinel"] = row
    if row == 5:
        result["quant_preserve"] = {"cached": True}


def test_calib_chunks_keep_refs_bad_rows_and_preserve_on_one_row_path():
    chunks = list(_NS["calib_chunks"](range(10), {4}, _preserve))
    assert chunks == [
        ("one", 0), ("one", 1),
        ("batch", [2, 3]),
        ("one", 4), ("one", 5),
        ("batch", [6, 7, 8]),
        ("one", 9),
    ]


def test_calib_chunks_honors_allow_false_and_configured_batch_size():
    _NS["CALIB_BATCH"] = 2
    chunks = list(_NS["calib_chunks"](range(2, 8), set(), _preserve))
    assert chunks == [("batch", [2, 3]), ("one", 4), ("one", 5), ("batch", [6, 7])]
    assert list(_NS["calib_chunks"]([2, 3, 6], set(), _preserve, allow=False)) == [
        ("one", 2), ("one", 3), ("one", 6)
    ]


def test_oom_retries_same_rows_with_halved_batches():
    seen = []

    def one(row):
        seen.append([row])

    def batch(rows):
        seen.append(list(rows))
        if len(rows) > 2:
            raise torch.cuda.OutOfMemoryError("simulated")

    _NS["CALIB_BATCH"] = 4
    _NS["run_calib_chunks"]([2, 3, 6, 7, 8, 9, 10, 11], set(), _preserve, one, batch)
    assert seen == [[2, 3, 6, 7], [2, 3], [6, 7], [8, 9], [10, 11]]


def test_batch_states_and_unbatch_states_round_trip_3d_and_2d():
    state_3d = [torch.arange(12).reshape(1, 3, 4), torch.arange(12, 24).reshape(1, 3, 4)]
    batched_3d = _NS["batch_states"](state_3d, [0, 1])
    assert tuple(batched_3d.shape) == (2, 3, 4)
    unbatched_3d = _NS["unbatch_states"](batched_3d, [0, 1], state_3d)
    assert [tuple(x.shape) for x in unbatched_3d] == [(1, 3, 4), (1, 3, 4)]
    assert torch.equal(torch.cat(unbatched_3d, dim=0), batched_3d)

    state_2d = [torch.arange(12).reshape(3, 4), torch.arange(12, 24).reshape(3, 4)]
    batched_2d = _NS["batch_states"](state_2d, [0, 1])
    assert tuple(batched_2d.shape) == (2, 3, 4)
    unbatched_2d = _NS["unbatch_states"](batched_2d, [0, 1], state_2d)
    assert [tuple(x.shape) for x in unbatched_2d] == [(3, 4), (3, 4)]
    assert torch.equal(torch.stack(unbatched_2d, dim=0), batched_2d)
