"""CPU execution of the production MTP window and drafting methods.

The fake drafter enforces contiguous cache writes. This catches the important transition
from a zero-proposal wide batch to a smaller batch that resumes normal MTP sampling.
No CUDA kernels, model weights, or package startup are imported.
"""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "exllamav3/generator/generator.py"


def _class():
    root = ast.parse(SOURCE.read_text())
    return next(n for n in root.body if isinstance(n, ast.ClassDef) and n.name == "Generator")


def _resolve(value, env):
    init = next(n for n in _class().body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    start = next(i for i, n in enumerate(init.body)
                 if isinstance(n, ast.If) and ast.unparse(n.test) == "draft_row_budget is None")
    end = next(i for i in range(start, len(init.body))
               if isinstance(init.body[i], ast.Assign)
               and any(ast.unparse(t) == "self.draft_row_budget" for t in init.body[i].targets))
    code = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(init.body[start:end + 1]), type_ignores=[]))
    self = NS()
    exec(compile(code, str(SOURCE), "exec"),
         {"self": self, "draft_row_budget": value, "_os": NS(environ=env)})
    return self.draft_row_budget


@pytest.mark.parametrize("value,env,want", [
    (None, {}, 0), (None, {"EXL3_DRAFT_ROW_BUDGET": "8"}, 8),
    (12, {"EXL3_DRAFT_ROW_BUDGET": "8"}, 12), (0, {"EXL3_DRAFT_ROW_BUDGET": "8"}, 0),
])
def test_constructor_resolution_executes_source(value, env, want):
    assert _resolve(value, env) == want


@pytest.mark.parametrize("value,env", [
    (None, {"EXL3_DRAFT_ROW_BUDGET": "bad"}), (None, {"EXL3_DRAFT_ROW_BUDGET": "-1"}),
    (-1, {}), (True, {}), (8.0, {}),
])
def test_invalid_budget_rejected_early(value, env):
    with pytest.raises(ValueError, match="nonnegative integer"):
        _resolve(value, env)


def _methods():
    wanted = {"_mtp_window", "iterate_draftmodel_mtp_gen"}
    methods = [copy.deepcopy(n) for n in _class().body
               if isinstance(n, ast.FunctionDef) and n.name in wanted]
    cls = ast.ClassDef(name="MTPProbe", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {
        "torch": torch, "PAGE_SIZE": 256, "_MTP_DEVICE_DRAFT": False,
        "cuda_sync_active": lambda: None, "time": NS(time=lambda: 1.0),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["MTPProbe"]


class Job:
    def __init__(self, token, ready=True, position=4):
        self.token = token
        self.ready = ready
        self.sequences = [NS(kv_position=position, block_index_tensor=torch.arange(16).view(1, 16))]
        self.mtp_last_hidden = torch.tensor([[[float(token), 1.0]]])
        self.time_first_token = 1.0

    def is_prefill_done(self):
        return self.ready

    def get_max_seq_len(self):
        return self.sequences[0].kv_position + 1

    def get_input_ids_list(self):
        return [torch.tensor([[self.token]])]


class Draft:
    def __init__(self, jobs):
        self.written = {j.token: set(range(j.sequences[0].kv_position)) for j in jobs}
        self.prefills = []
        self.forwards = []
        self.samples = 0
        self.last_ids = None

    def _write(self, ids, params):
        for row in range(ids.shape[0]):
            token = int(ids[row, 0])
            pos = int(params["cache_seqlens"][row])
            assert set(range(pos)) <= self.written[token], f"draft cache hole before position {pos}"
            self.written[token].add(pos)

    def prefill(self, ids, params):
        self._write(ids, params)
        self.prefills.append((ids.clone(), params["target_hidden"].clone(),
                              params["cache_seqlens"].clone()))

    def forward(self, ids, params):
        self._write(ids, params)
        self.forwards.append(ids.clone())
        self.last_ids = ids.clone()
        return params["target_hidden"] + 0.125

    def sample_from_state(self, state, params):
        self.samples += 1
        return self.last_ids.clone()


def _probe(batch, budget=8, full=5):
    probe = _methods()()
    probe.num_draft_tokens = full
    probe.draft_row_budget = budget
    probe.active_jobs = [Job(i + 10) for i in range(batch)]
    probe.draft_model = Draft(probe.active_jobs)
    probe.draft_cache = object()
    probe.draft_calibrator = None
    probe.draft_input_ids_pinned = torch.zeros((max(batch, 1), 1), dtype=torch.long)
    probe.draft_ids_pinned = torch.zeros((max(batch, 1), max(full, 1)), dtype=torch.long)
    probe.model = NS(logit_layer_idx=0, modules=[NS(prepare_for_device=lambda state, params: state)])
    probe._staging = lambda name, *shape: torch.zeros(shape, dtype=torch.int32)
    return probe


@pytest.mark.parametrize("batch,budget,full,want", [
    (1, 1, 5, 5), (1, 8, 5, 5), (2, 8, 5, 3), (4, 8, 5, 1), (8, 8, 5, 0),
    (4, 0, 5, 5), (2, 1000, 5, 5), (2, 8, 0, 0), (4, 3, 5, 0),
])
def test_window_from_real_method(batch, budget, full, want):
    assert _probe(batch, budget, full)._mtp_window(batch) == want


@pytest.mark.parametrize("batch,want", [(1, 5), (2, 3), (4, 1)])
def test_real_mtp_loop_limits_proposals(batch, want):
    probe = _probe(batch)
    result = probe.iterate_draftmodel_mtp_gen([])
    assert result.shape == (batch, want)
    assert len(probe.draft_model.forwards) == want
    assert probe.draft_model.samples == want
    assert probe.draft_model.prefills == []


def test_zero_proposals_commit_cache_before_smaller_batch_resumes():
    probe = _probe(8)
    result = probe.iterate_draftmodel_mtp_gen([])
    assert result is None
    assert probe.draft_model.samples == 0
    assert probe.draft_model.forwards == []
    assert len(probe.draft_model.prefills) == 1
    ids, hidden, positions = probe.draft_model.prefills[0]
    assert ids.shape == (8, 1)
    assert hidden.shape == (8, 1, 2)
    assert positions.tolist() == [4] * 8
    assert probe._draft_conf_round is None
    # The target accepts one token, advances position, and supplies its new hidden carry.
    for job in probe.active_jobs:
        job.sequences[0].kv_position += 1
        job.mtp_last_hidden += 1
    probe.active_jobs = probe.active_jobs[:1]
    result = probe.iterate_draftmodel_mtp_gen([])
    assert result.shape[-1] == 5
    assert probe.draft_model.written[10] == set(range(10))


def test_batch_budget_counts_only_prefilled_jobs():
    probe = _probe(4)
    for job in probe.active_jobs[1:]:
        job.ready = False
    result = probe.iterate_draftmodel_mtp_gen([])
    assert result.shape[-1] == 5
    assert len(probe.draft_model.forwards) == 5
    assert all(ids.shape[0] == 1 for ids in probe.draft_model.forwards)


def _target_handoff(probe, old_positions, rewound=()):
    """Execute the production target-to-MTP handoff, with one accepted target token."""
    iterate = next(n for n in _class().body
                   if isinstance(n, ast.FunctionDef) and n.name == "iterate_gen")
    handoff = next(n for n in iterate.body
                   if isinstance(n, ast.If) and ast.unparse(n.test) == "self.mtp_draft")
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(handoff)], type_ignores=[]))
    batch = len(probe.active_jobs)
    target = torch.arange(batch * 2, dtype=torch.float).reshape(batch, 1, 2) + 100
    probe.mtp_draft = True
    namespace = {
        "self": probe, "torch": torch,
        "p_export_states": [target],
        "accepted_lengths": [1] * batch,
        "logit_mapping": list(range(batch + 1)),
        "rewound_jobs": set(rewound),
        "batch_ids": torch.cat([j.get_input_ids_list()[0] for j in probe.active_jobs]),
        "block_index": torch.zeros((batch, 16), dtype=torch.int32),
        "p_cache_seqlens": torch.tensor(old_positions, dtype=torch.int32),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return target


def test_missing_carry_maintains_every_batch_members_cache():
    probe = _probe(2)
    initial_carry = probe.active_jobs[0].mtp_last_hidden.clone()
    probe.active_jobs[1].mtp_last_hidden = None
    assert probe.iterate_draftmodel_mtp_gen([]) is None
    assert probe._mtp_skipped_round
    assert probe.draft_model.prefills == []
    for job in probe.active_jobs:
        job.sequences[0].kv_position += 1
    target = _target_handoff(probe, [4, 4])
    assert len(probe.draft_model.prefills) == 2
    assert torch.equal(probe.draft_model.prefills[0][1], initial_carry)
    assert torch.equal(probe.draft_model.prefills[1][1], torch.zeros_like(initial_carry))
    for row, job in enumerate(probe.active_jobs):
        assert probe.draft_model.written[job.token] == set(range(5))
        assert torch.equal(job.mtp_last_hidden, target[row:row + 1])
    assert probe.iterate_draftmodel_mtp_gen([]).shape == (2, 3)
    assert not probe._mtp_skipped_round
    assert all(probe.draft_model.written[j.token] == set(range(8)) for j in probe.active_jobs)


def test_one_token_prompt_bootstrap_writes_initial_zero_carry():
    probe = _probe(1)
    job = probe.active_jobs[0]
    job.sequences[0].kv_position = 0
    job.mtp_last_hidden = None
    probe.draft_model = Draft(probe.active_jobs)
    assert probe.iterate_draftmodel_mtp_gen([]) is None
    job.sequences[0].kv_position = 1
    _target_handoff(probe, [0])
    assert len(probe.draft_model.prefills) == 1
    _, hidden, positions = probe.draft_model.prefills[0]
    assert torch.count_nonzero(hidden) == 0
    assert positions.tolist() == [0]
    assert probe.iterate_draftmodel_mtp_gen([]).shape == (1, 5)
    assert probe.draft_model.written[job.token] == set(range(6))


def test_zero_budget_handoff_does_not_write_current_position_twice():
    probe = _probe(8)
    assert probe.iterate_draftmodel_mtp_gen([]) is None
    assert not probe._mtp_skipped_round
    assert len(probe.draft_model.prefills) == 1
    for job in probe.active_jobs:
        job.sequences[0].kv_position += 1
    _target_handoff(probe, [4] * 8)
    assert len(probe.draft_model.prefills) == 1
    probe.active_jobs = probe.active_jobs[:1]
    assert probe.iterate_draftmodel_mtp_gen([]).shape[-1] == 5
    assert probe.draft_model.forwards[-1].shape[0] == 1


def test_skipped_round_handoff_does_not_resurrect_rewound_carry():
    probe = _probe(2)
    for job in probe.active_jobs:
        job.mtp_last_hidden = None
    assert probe.iterate_draftmodel_mtp_gen([]) is None
    for job in probe.active_jobs:
        job.sequences[0].kv_position += 1
    _target_handoff(probe, [4, 4], rewound=[id(probe.active_jobs[0])])
    assert probe.active_jobs[0].mtp_last_hidden is None
    assert probe.draft_model.written[probe.active_jobs[0].token] == set(range(4))
    assert probe.draft_model.written[probe.active_jobs[1].token] == set(range(5))


class PretendCudaTensor(torch.Tensor):
    """Route the actual loop through device drafting using CPU-only operations."""
    @property
    def is_cuda(self):
        return True


class DeferredPositionDraft:
    """Record host sources as async DMA would, reading them after submission."""
    def __init__(self):
        self.sources = []
        self.last_ids = None

    def forward(self, ids, params):
        self.sources.append(params["cache_seqlens"])
        self.last_ids = ids
        return params["target_hidden"] + 0.125

    def sample_from_state(self, state, params):
        return self.last_ids


@pytest.mark.parametrize("batch,window", [(1, 5), (2, 3), (4, 1)])
def test_device_draft_positions_remain_immutable_until_final_readback(batch, window):
    probe = _probe(batch)
    probe.iterate_draftmodel_mtp_gen.__func__.__globals__["_MTP_DEVICE_DRAFT"] = True
    for job in probe.active_jobs:
        job.mtp_last_hidden = job.mtp_last_hidden.as_subclass(PretendCudaTensor)
    probe.draft_model = DeferredPositionDraft()
    result = probe.iterate_draftmodel_mtp_gen([])
    assert result.shape[-1] == window
    assert len(probe.draft_model.sources) == window
    # Read the original host sources only after the whole chain was submitted.
    # A single in-place-incremented source would now contain the final position
    # for every step, corrupting asynchronous H2D reads queued behind GPU work.
    for step, source in enumerate(probe.draft_model.sources):
        assert source.tolist() == [4 + step] * batch
    assert len({s.data_ptr() for s in probe.draft_model.sources}) == window
