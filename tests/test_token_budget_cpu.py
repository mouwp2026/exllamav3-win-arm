"""CPU execution of producer budgets, real token acceptance, MTP verification and async delivery.

The model/logits and cache allocations are fakes. Job token acceptance, forced-token
handling, stop/rewind/requeue behavior, Generator's speculative acceptance loop and
AsyncGenerator/AsyncJob are compiled directly from production source without importing
the CUDA extension. These tests do not claim GPU/model numerical or throughput coverage.
"""
from __future__ import annotations

import ast
import asyncio
import copy
import random
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
JOB_PATH = ROOT / "exllamav3/generator/job.py"
GEN_PATH = ROOT / "exllamav3/generator/generator.py"
ASYNC_PATH = ROOT / "exllamav3/generator/async_generator.py"
END = 9
VOCAB = 32


def _tree(path):
    return ast.parse(path.read_text())


def _class(path, name):
    return next(n for n in _tree(path).body if isinstance(n, ast.ClassDef) and n.name == name)


def _compile(nodes, path, namespace):
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
             + copy.deepcopy(nodes), type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def _load_class(path, name, namespace, methods=None):
    cls = copy.deepcopy(_class(path, name))
    if methods is not None:
        cls.body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name in methods]
        assert {n.name for n in cls.body} == methods
    _compile([cls], path, namespace)
    return namespace[name]


class Tokenizer:
    def __init__(self):
        self.pieces = [f"[{i}]" for i in range(VOCAB)]
        for i, piece in {
            0: "", 1: "a", 2: "b", 3: "c", 4: "<", 5: "/phase>", 6: "x",
            7: "<phase>", 8: "z", END: "</phase>", 10: "tail", 11: "X", 12: "Y",
            13: "�", 14: "STOP", 15: "ST", 16: "OP", 17: "",
        }.items():
            self.pieces[i] = piece

    def encode(self, text, **kwargs):
        assert kwargs.get("encode_special_tokens", True)
        if text == "</phase>tail":
            ids = [END, 10]
        else:
            ids = [self.pieces.index(text)]
        return torch.tensor([ids], dtype=torch.long)

    def get_id_to_piece_list(self, *_):
        return self.pieces

    def decode(self, ids, **_):
        return ["".join(self.pieces[i] for i in ids.flatten().tolist())]


class Sampler:
    reqs_past_ids = False
    supports_batch_verify = True

    def __init__(self):
        self.calls = []

    def forward(self, logits, past, seed, tokenizer, logit_mask=None):
        self.calls.append((logits.shape[0], logit_mask is not None))
        scores = logits.float()
        if logit_mask is not None:
            if logit_mask.dtype == torch.int32:
                positions = torch.arange(VOCAB)
                allowed = ((logit_mask[:, positions // 32].long() >> (positions % 32)) & 1).bool()
                scores = scores.masked_fill(~allowed, -torch.inf)
            else:
                scores = scores + logit_mask
        return scores.argmax(-1).view(logits.shape[0], 1)


class Filter:
    trigger_token = None
    is_active = True

    def __init__(self, allowed=3, stop=None, packed=False):
        self.allowed = allowed
        self.packed = packed
        self.stop = stop
        self.fed = []
        self.attached = None
        self.resets = 0

    def attach(self, job):
        self.attached = job

    def reset(self):
        self.resets += 1
        self.fed.clear()

    def feed(self, token_id):
        self.fed.append(token_id)
        return token_id == self.stop

    def rewind(self, n):
        del self.fed[-n:]

    def use_background_worker(self):
        return False

    def get_next_logit_mask(self):
        if self.packed:
            return torch.tensor([[1 << self.allowed]], dtype=torch.int32)
        mask = torch.full((1, VOCAB), -torch.inf, dtype=torch.half)
        mask[..., self.allowed] = 0
        return mask


def _partial_match(text_bytes, offsets_bytes, needle_bytes):
    # The native matcher contract: earliest full match, -2 for a suffix that may
    # complete a needle, or -1. Test strings are ASCII, represented as UTF32.
    text = bytes(text_bytes).decode("utf-32-le")
    offsets = np.frombuffer(offsets_bytes, dtype=np.int32).tolist()
    raw = bytes(needle_bytes)
    needles = [raw[a:b].decode("utf-32-le") for a, b in zip(offsets, offsets[1:])]
    found = [text.find(s) for s in needles if s in text]
    if found:
        return min(found)
    if any(text.endswith(s[:i]) for s in needles for i in range(1, len(s))):
        return -2
    return -1


NAMESPACE = {
    "torch": torch, "np": np, "random": random, "time": time, "lru_cache": lru_cache,
    "PAGE_SIZE": 256, "Sampler": Sampler, "DefaultSampler": Sampler,
    "ext": NS(partial_strings_match=_partial_match),
    "FIRST_MM_EMBEDDING_INDEX": 1000000,
}
SeqTensor = _load_class(ROOT / "exllamav3/util/tensor.py", "SeqTensor", NAMESPACE)
Sequence = _load_class(ROOT / "exllamav3/generator/pagetable.py", "Sequence", NAMESPACE, {"__init__"})
_compile([next(n for n in _tree(JOB_PATH).body
               if isinstance(n, ast.FunctionDef) and n.name == "_strings_to_utf32")],
         JOB_PATH, NAMESPACE)
JOB_METHODS = {
    "__init__", "constrain_output_now", "set_token_budget", "clear_token_budget",
    "_maybe_force_token_budget", "_advance_token_budget", "_token_budget_can_end", "_pop_forced_token",
    "receive_logits", "receive_sample", "set_sampler", "set_filters", "set_banned_strings",
    "_init_banned_strings", "_check_banned_strings", "_release_banned_hold",
    "hash_deferred_pages", "prepare_for_requeue", "prepare_logit_mask",
}
Job = _load_class(JOB_PATH, "Job", NAMESPACE, JOB_METHODS)


def _prepare_masks(job):
    # Match the Generator's initial per-step mask queue. Between accepted MTP
    # positions the production loop below queues these entries itself.
    assert job.filter_futures == [] and job.logit_masks == []
    for f in job.filters:
        if not f.is_active:
            continue
        job.filter_futures.append(None)
        job.logit_masks.append(f.get_next_logit_mask())
    job.prepare_logit_mask()


def _attach(job, generator, serial_number=0, rq=False):
    # Only scheduling, buffers and physical allocation are stubbed. The actual
    # constructor, token/stream logic and requeue state dictionary execute above.
    job.generator = generator
    job.serial_number = serial_number
    job.max_rq_tokens = 10000
    job.rq_margin = 0
    if not rq:
        job.held_tokens = SeqTensor((1, 0), torch.long, -1)
        job.held_probs = SeqTensor((1, 0), torch.float, -1)
        job.held_k_tokens = SeqTensor((1, 0, 0), torch.long, 1)
        job.held_k_probs = SeqTensor((1, 0, 0), torch.float, 1)
        job.held_logits = SeqTensor((1, 0, VOCAB), torch.float, 1)
    job.time_enqueue = job.time_first_prefill = job.time_first_token = time.time()
    seq = job.sequences[0]
    seq.kv_position = len(seq.sequence_ids) - 1
    seq.allocated_pages = [NS(kv_position=seq.kv_position, can_revert=False)]
    job.logits_device = "cpu"
    job.get_pinned_logit_mask = lambda: torch.empty((1, VOCAB), dtype=torch.half)
    job.get_pinned_logit_bitmask = lambda: torch.empty((1, VOCAB // 32), dtype=torch.int32)
    job.prepare_sampling_past_ids = lambda: None
    job.is_checkpoint_boundary = lambda: False
    job.deallocate_pages = lambda: setattr(job, "deallocated", True)
    job.deallocated = False


Job.prepare_for_queue = _attach


def make_job(**kwargs):
    kwargs.setdefault("input_ids", torch.tensor([[7, 1]]))
    kwargs.setdefault("max_new_tokens", 100)
    job = Job(**kwargs)
    generator = NS(tokenizer=Tokenizer(), draft_model=None, ngram_match_min=0, hybrid_draft_tokens=0,
                   recurrent_cache=None, padded_vocab_size=VOCAB)
    _attach(job, generator)
    return job


def logits_for(token_id=1, width=1):
    scores = torch.zeros((1, width, VOCAB))
    scores[..., token_id] = 5
    return scores


def sample(job, token_id=1, results=None):
    results = [] if results is None else results
    scores = logits_for(token_id)
    sampled = job.receive_logits(scores)
    result = job.receive_sample(scores, *sampled, results)
    return result, results


def emitted_ids(results):
    return [int(i) for r in results if r.get("token_ids") is not None
            for i in r["token_ids"].flatten().tolist()]


def arm(job, n, output=None, callback=None, guard=None):
    job.set_token_budget(n, torch.tensor([[END]]) if output is None else output,
                         end_token_id=END, on_end=callback, can_end=guard)


def _generator_node(name):
    return next(n for n in _class(GEN_PATH, "Generator").body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def run_mtp(job, proposals, scores=None, batch_verify=True):
    """Run the unmodified native acceptance branch and native rejection function."""
    proposals = list(proposals)
    width = len(proposals) + 1
    scores = logits_for(width=width) if scores is None else scores
    seq = job.sequences[0]
    initial_position = seq.kv_position
    seq.allocated_pages[0].kv_position = initial_position + width
    state = NS(rewinds=[])
    state.rewind = lambda count: state.rewinds.append(count)
    probe = NS(active_jobs=[job], tokenizer=job.generator.tokenizer,
               record_draft_stats=False, model=NS(prefetch_tokens=lambda _: None))
    node = _generator_node("iterate_gen")
    reject = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "reject_remainder")
    branch = next(n for n in node.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "draft_tokens is None and batch_logits.shape[1] == 1")
    namespace = {
        "self": probe, "torch": torch, "PAGE_SIZE": 256, "_BATCH_VERIFY": batch_verify,
        "draft_tokens": torch.tensor([proposals], dtype=torch.long),
        "batch_logits": scores, "logit_mapping": [0, 1], "batch_states": [state],
        "accepted_lengths": [], "completed_jobs": [], "requeuing_jobs": [],
        "rewound_jobs": set(), "j": 0, "results": [],
    }
    _prepare_masks(job)
    _compile([reject, *branch.orelse], GEN_PATH, namespace)
    namespace["state"] = state
    namespace["initial_position"] = initial_position
    return namespace


def test_zero_budget_and_prompt_do_not_count():
    job = make_job()
    calls = []
    arm(job, 0, callback=lambda j: calls.append(j.new_tokens))
    result, rows = sample(job)
    assert result[1].item() == END
    assert emitted_ids(rows) == [END]
    assert calls == [1] and job.token_budget is None


@pytest.mark.parametrize("budget", [1, 3, 24])
def test_exact_accepted_boundary_and_one_shot(budget):
    job = make_job()
    calls = []
    arm(job, budget, callback=lambda j: calls.append(j.new_tokens))
    rows = []
    for _ in range(budget + 3):
        sample(job, results=rows)
    assert emitted_ids(rows) == [1] * budget + [END, 1, 1]
    assert calls == [budget + 1]


def test_arm_mid_generation_is_relative_to_current_accepted_position():
    job = make_job()
    sample(job)
    sample(job)
    arm(job, 2)
    got = [sample(job)[0][1].item() for _ in range(4)]
    assert got == [1, 1, END, 1]


def test_natural_end_feeds_old_filter_then_restores_next_phase():
    job = make_job(filters=[Filter(allowed=1)])
    old_filter = job.filters[0]
    new_filter = Filter(allowed=3)
    order = []

    def end(j):
        order.append((j.new_tokens, list(old_filter.fed), j.forced_ids))
        j.set_filters([new_filter])

    arm(job, 8, callback=end)
    sample(job, END)
    assert order == [(1, [END], None)]
    assert new_filter.attached is job and new_filter.resets == 1
    _prepare_masks(job)
    result, _ = sample(job)
    assert result[1].item() == 3 and new_filter.fed == [3]


def test_forced_tail_drains_before_callback_and_filters():
    job = make_job(filters=[Filter(allowed=1)])
    old = job.filters[0]
    calls = []
    new = Filter(allowed=3)
    arm(job, 0, output="</phase>tail",
        callback=lambda j: (calls.append((j.new_tokens, j.forced_ids)), j.set_filters([new])))
    first, _ = sample(job)
    assert first[1].item() == END and calls == [] and not old.is_active
    second, _ = sample(job)
    assert second[1].item() == 10 and calls == [(2, None)]
    _prepare_masks(job)
    assert sample(job)[0][1].item() == 3
    assert old.fed == [] and new.fed == [3]


def test_existing_forced_queue_is_preserved():
    job = make_job()
    job.constrain_output_now(torch.tensor([[11, 12]]))
    arm(job, 0)
    got = [sample(job)[0][1].item() for _ in range(4)]
    assert got == [11, 12, END, 1]


def test_natural_end_in_existing_forced_tail_waits_for_all_pending_tokens():
    job = make_job()
    job.constrain_output_now(torch.tensor([[END, 10, 11]]))
    calls = []
    arm(job, 0, callback=lambda j: calls.append(j.new_tokens))
    got = [sample(job)[0][1].item() for _ in range(4)]
    assert got == [END, 10, 11, 1] and calls == [3]


def test_clear_preserves_scheduled_tail_without_callback():
    job = make_job()
    calls = []
    arm(job, 0, output="</phase>tail", callback=lambda _: calls.append(True))
    assert sample(job)[0][1].item() == END
    job.clear_token_budget()
    assert sample(job)[0][1].item() == 10
    assert sample(job)[0][1].item() == 1
    assert calls == [] and job.filters_suspended


@pytest.mark.parametrize("stop_kind", ["token", "string", "limit", "filter"])
def test_termination_wins_and_does_not_call_phase_callback(stop_kind):
    kwargs = {}
    if stop_kind == "token":
        kwargs["stop_conditions"] = [END]
    elif stop_kind == "string":
        kwargs["stop_conditions"] = ["</phase>"]
    elif stop_kind == "limit":
        kwargs["max_new_tokens"] = 1
    else:
        kwargs["filters"] = [Filter(stop=END)]
    job = make_job(**kwargs)
    calls = []
    # Natural closure lets the filter see the end marker; injection deliberately
    # suspends old filters according to existing constrain_output_now semantics.
    arm(job, 10, callback=lambda _: calls.append(True))
    result, rows = sample(job, END)
    assert result[0] and rows[-1]["eos"] and job.token_budget is None
    assert calls == []


def test_length_limit_before_deadline_never_injects():
    job = make_job(max_new_tokens=2)
    arm(job, 2)
    rows = []
    sample(job, results=rows)
    result, _ = sample(job, results=rows)
    assert result[0] and emitted_ids(rows) == [1, 1]
    assert job.token_budget is None


def test_partial_stop_and_unicode_hold_still_count_accepted_tokens():
    for special in (15, 13):
        job = make_job(stop_conditions=["STOP"])
        arm(job, 1)
        first, rows = sample(job, special)
        assert not first[0] and emitted_ids(rows) == []
        assert job.new_tokens == 1
        second, rows2 = sample(job)
        assert second[1].item() == END
        assert job.token_budget is None
        # Incomplete Unicode may keep the text buffered, but cannot move the
        # producer's native-token budget or make it depend on consumer delivery.
        assert job.new_tokens == 2


def test_literal_partial_marker_text_does_not_disarm_native_marker_budget():
    job = make_job()
    arm(job, 2)
    assert sample(job, 4)[0][1].item() == 4
    assert sample(job, 5)[0][1].item() == 5
    assert job.token_budget is not None
    assert sample(job)[0][1].item() == END


def test_healing_precedes_zero_budget_and_does_not_count():
    job = make_job(token_healing=True, input_ids=torch.tensor([[7, 1]]))
    calls = []
    arm(job, 0, callback=lambda j: calls.append(j.new_tokens))
    healed, _ = sample(job, 1)
    assert healed[1].item() == 1 and job.new_tokens == 0 and calls == []
    forced, _ = sample(job)
    assert forced[1].item() == END and calls == [1]


def test_banned_rewind_restores_accepted_deadline_and_does_not_close_phase():
    # First native end marker is part of a banned string and is rolled back.
    job = make_job(banned_strings=["</phase>"])
    calls = []
    arm(job, 2, callback=lambda j: calls.append(j.new_tokens))
    result, rows = sample(job, END)
    assert job.checkpoint_rewound and job.new_tokens == 0
    assert job.token_budget is not None and calls == []
    assert rows[-1]["suppressed_text"] == "</phase>"
    job.checkpoint_rewound = False  # Generator does this before the next iteration.
    assert sample(job, 1)[0][1].item() == 1
    assert sample(job, 1)[0][1].item() == 1
    assert sample(job)[0][1].item() == END
    assert calls == [3]


@pytest.mark.parametrize("callback_enabled", [False, True])
def test_partial_banned_hold_can_rewind_natural_end_without_losing_budget(callback_enabled):
    job = make_job(banned_strings=["</phase>x"])
    calls = []
    callback = (lambda j: calls.append(j.new_tokens)) if callback_enabled else None
    arm(job, 3, callback=callback)
    sample(job, END)
    assert job.checkpoint["offset"] == 1
    assert job.token_budget["end_seen"] and calls == []
    sample(job, 6)  # Completes banned text and removes both tokens.
    assert job.checkpoint_rewound and job.new_tokens == 0
    assert not job.token_budget["end_seen"] and calls == []
    job.checkpoint_rewound = False
    got = [sample(job)[0][1].item() for _ in range(4)]
    assert got == [1, 1, 1, END]
    assert calls == ([4] if callback_enabled else [])
    assert job.token_budget is None


def test_natural_end_callback_waits_until_partial_banned_hold_is_final():
    job = make_job(banned_strings=["</phase>x"])
    calls = []
    arm(job, 3, callback=lambda j: calls.append((j.new_tokens, j.checkpoint)))
    sample(job, END)
    assert calls == [] and job.token_budget is not None
    sample(job, 1)  # Disproves the banned continuation, making the end final.
    assert calls == [(2, None)] and job.token_budget is None


def test_guard_defers_due_injection_without_discarding_budget():
    job = make_job()
    observed = []

    def guard(j):
        observed.append(j.new_tokens)
        return j.new_tokens >= 3

    arm(job, 0, guard=guard)
    assert [sample(job)[0][1].item() for _ in range(4)] == [1, 1, 1, END]
    # Due forcing checks the accepted boundary; accepted END is checked too.
    assert observed == [0, 1, 2, 3, 4]
    assert job.token_budget is None


def test_approved_fixed_injection_finishes_if_guard_changes_inside_its_tail():
    job = make_job()
    calls = []
    observed = []

    def guard(j):
        observed.append(j.new_tokens)
        return j.new_tokens == 0

    arm(job, 0, output="</phase>tail", guard=guard,
        callback=lambda j: calls.append(j.new_tokens))
    assert [sample(job)[0][1].item() for _ in range(3)] == [END, 10, 1]
    assert calls == [2] and observed == [0, 1]
    assert job.token_budget is None


def test_guard_treats_native_end_inside_protected_data_as_literal():
    job = make_job()
    allowed = [False]
    calls = []
    arm(job, 10, guard=lambda _: allowed[0], callback=lambda j: calls.append(j.new_tokens))
    sample(job, END)
    assert job.token_budget is not None and not job.token_budget["end_seen"]
    assert calls == []
    allowed[0] = True
    sample(job, END)
    assert calls == [2] and job.token_budget is None


def test_guard_synchronizes_after_rewind_even_before_deadline():
    job = make_job(banned_strings=["</phase>x"])
    observed = []

    def guard(j):
        observed.append((j.new_tokens, j.sequences[0].sequence_ids.torch().clone()))
        return True

    arm(job, 8, guard=guard)
    sample(job, END)  # Native end check advances the guard's view.
    sample(job, 6)    # Banned rollback removes END and x.
    assert [n for n, _ in observed] == [1, 0]
    assert observed[-1][1].tolist() == [[7, 1]]
    assert job.token_budget is not None and not job.token_budget["end_seen"]


@pytest.mark.parametrize("boundary", ["before_sample", "native_end", "rewind"])
def test_guard_exception_is_contained_at_each_boundary(boundary):
    job = make_job(banned_strings=["</phase>x"] if boundary == "rewind" else None)
    before = job.sequences[0].sequence_ids.torch().clone()
    position = job.sequences[0].kv_position

    def guard(j):
        if boundary != "rewind" or j.checkpoint_rewound:
            raise RuntimeError("cannot verify safe phase boundary")
        return True

    arm(job, 0 if boundary == "before_sample" else 8, guard=guard)
    if boundary == "rewind":
        sample(job, END)
        result, rows = sample(job, 6)
    else:
        result, rows = sample(job, END)
    assert result[0] and rows[-1]["stage"] == "error"
    assert isinstance(rows[-1]["error"], RuntimeError)
    assert job.is_finished and job.token_budget is None
    if boundary == "before_sample":
        assert torch.equal(job.sequences[0].sequence_ids.torch(), before)
        assert job.sequences[0].kv_position == position and job.new_tokens == 0
        assert job.sampler.calls == [] and "token_ids" not in rows[-1]


def test_failed_budget_job_does_not_train_calibration_or_publish_mtp_carry():
    failed = make_job()

    def guard(_):
        raise RuntimeError("cannot decide phase boundary")

    arm(failed, 0, guard=guard)
    state = run_mtp(failed, [1, 1])
    assert failed._token_budget_failed and failed.new_tokens == 0

    peer = make_job()
    peer_state = run_mtp(peer, [1, 1])
    labels = []
    cal = NS(add_label=lambda confidence, accepted: labels.append((confidence, accepted)),
             decay_step=lambda: None)
    state["self"].active_jobs = [failed, peer]
    state["self"].draft_calibrator = cal
    state["self"]._draft_conf_round = {
        "conf": torch.tensor([[0.1, 0.2], [0.7, 0.8]]),
        "ids": torch.ones((2, 2), dtype=torch.long),
    }
    state["accepted_lengths"] = [1, peer_state["accepted_lengths"][0]]
    state["logit_mapping"] = [0, 1, 2]
    calibrate = next(n for n in _generator_node("iterate_gen").body
                     if isinstance(n, ast.If) and ast.unparse(n.test)
                     == "self.draft_calibrator is not None and self._draft_conf_round is not None")
    _compile([calibrate], GEN_PATH, state)
    assert len(labels) == 2
    assert [accepted for _, accepted in labels] == [True, True]
    assert [confidence for confidence, _ in labels] == pytest.approx([0.7, 0.8])

    handoff = next(n for n in _generator_node("iterate_gen").body
                   if isinstance(n, ast.If) and ast.unparse(n.test) == "self.mtp_draft")
    writes = []
    state["self"].active_jobs = [failed]
    state["self"].mtp_draft = True
    state["self"]._mtp_skipped_round = False
    state["self"].draft_model = NS(prefill=lambda *args: writes.append(args))
    state["self"].draft_cache = object()
    state.update(p_export_states=[torch.arange(6, dtype=torch.float).view(1, 3, 2)],
                 accepted_lengths=[1], logit_mapping=[0, 1])
    assert failed.mtp_last_hidden is None
    _compile([handoff], GEN_PATH, state)
    assert failed.mtp_last_hidden is None and writes == []


@pytest.mark.parametrize("result", [None, 1, "yes", torch.tensor(True)])
def test_non_bool_guard_return_fails_closed(result):
    job = make_job()
    arm(job, 0, guard=lambda _: result)
    ended, rows = sample(job)
    assert ended[0] and isinstance(rows[-1]["error"], TypeError)
    assert job.new_tokens == 0 and job.sampler.calls == []


def test_async_guard_is_rejected_and_returned_coroutine_is_closed():
    async def async_guard(_):
        return True

    job = make_job()
    with pytest.raises(ValueError, match="can_end"):
        arm(job, 0, guard=async_guard)

    async def late_result():
        return True

    arm(job, 0, guard=lambda _: late_result())
    result, rows = sample(job)
    assert result[0] and isinstance(rows[-1]["error"], TypeError)
    assert job.new_tokens == 0


def test_requeue_retains_absolute_deadline_and_callback():
    job = make_job()
    calls = []
    arm(job, 3, callback=lambda j: calls.append(j.rq_new_tokens + j.new_tokens))
    sample(job)
    sample(job)
    original_budget = job.token_budget
    assert job.prepare_for_requeue() is job
    assert job.token_budget is original_budget
    assert job.rq_new_tokens == 2 and job.new_tokens == 0
    assert sample(job)[0][1].item() == 1
    assert sample(job)[0][1].item() == END
    assert calls == [4]


def test_requeue_during_forced_tail_retains_end_seen_and_remaining_ids():
    job = make_job()
    calls = []
    arm(job, 0, output="</phase>tail", callback=lambda j: calls.append(j.rq_new_tokens + j.new_tokens))
    assert sample(job)[0][1].item() == END
    assert job.token_budget["end_seen"]
    job.prepare_for_requeue()
    assert sample(job)[0][1].item() == 10
    assert calls == [2]


@pytest.mark.parametrize("field,value", [
    ("max_tokens", -1), ("max_tokens", True), ("max_tokens", 1.5),
    ("end_token_id", -1), ("end_token_id", True),
    ("output", torch.tensor([END])), ("output", torch.empty((1, 0), dtype=torch.long)),
    ("output", torch.tensor([[1]])), ("output", 7), ("output", None), ("on_end", 7),
])
def test_invalid_api_values_fail_before_generation(field, value):
    job = make_job()
    options = dict(max_tokens=1, output=torch.tensor([[END]]), end_token_id=END)
    options[field] = value
    with pytest.raises((ValueError, TypeError)):
        job.set_token_budget(**options)
    assert job.token_budget is None and job.new_tokens == 0


def test_unenqueued_finished_and_async_callback_rejected():
    job = Job(torch.tensor([[1]]), max_new_tokens=10)
    with pytest.raises(ValueError):
        arm(job, 1)
    job = make_job()
    job.is_finished = True
    with pytest.raises(ValueError):
        arm(job, 1)

    async def callback(_):
        return None

    job = make_job()
    with pytest.raises(ValueError):
        arm(job, 1, callback=callback)


def test_tensor_output_is_owned_after_arming():
    job = make_job()
    ids = torch.tensor([[END, 10]])
    arm(job, 0, ids)
    ids.fill_(11)
    assert [sample(job)[0][1].item() for _ in range(2)] == [END, 10]


def test_callback_failure_is_per_job_error_and_stops_sampling():
    job = make_job()

    def fail(_):
        raise RuntimeError("phase filters unavailable")

    arm(job, 0, callback=fail)
    state = run_mtp(job, [END, 1, 1, 1, 1])
    assert state["completed_jobs"] == [job]
    assert state["results"][-1]["stage"] == "error"
    assert isinstance(state["results"][-1]["error"], RuntimeError)
    assert job.new_tokens == 1 and job.token_budget is None
    assert job.is_finished and job.sampler.calls == []


def test_awaitable_return_fails_closed_without_leaking_coroutine():
    job = make_job()

    async def later():
        return True

    arm(job, 0, callback=lambda _: later())
    result, rows = sample(job)
    assert result[0] and rows[-1]["stage"] == "error"
    assert isinstance(rows[-1]["error"], TypeError)


def test_mtp_budget_forcing_rejects_remaining_draft_and_keeps_accepted_carry():
    job = make_job()
    calls = []
    arm(job, 2, callback=lambda j: calls.append(j.new_tokens))
    state = run_mtp(job, [1, 1, 1, 1, 1])
    assert emitted_ids(state["results"]) == [1, 1, END]
    assert job.accepted_draft_tokens == 2 and job.rejected_draft_tokens == 3
    assert state["accepted_lengths"] == [3] and state["state"].rewinds == [3]
    assert job.sequences[0].allocated_pages[0].kv_position == job.sequences[0].kv_position
    assert calls == [3]
    assert all(rows == 1 for rows, _ in job.sampler.calls), "Budget must disable prebatched verification"

    # Execute the real target-to-MTP handoff as well: the next input is the forced
    # closing token, paired with the hidden state immediately preceding it.
    handoff = next(n for n in _generator_node("iterate_gen").body
                   if isinstance(n, ast.If) and ast.unparse(n.test) == "self.mtp_draft")
    hidden = torch.arange(12, dtype=torch.float).view(1, 6, 2)
    written = []
    state["self"].mtp_draft = True
    state["self"]._mtp_skipped_round = False
    state["self"].draft_model = NS(prefill=lambda ids, params: written.append((ids.clone(), params)))
    state["self"].draft_cache = object()
    state.update(p_export_states=[hidden], batch_ids=torch.tensor([[1, 1, 1, 1, 1, 1]]),
                 block_index=torch.zeros((1, 1), dtype=torch.int32),
                 p_cache_seqlens=torch.tensor([state["initial_position"]], dtype=torch.int32))
    _compile([handoff], GEN_PATH, state)
    assert torch.equal(job.mtp_last_hidden, hidden[:, 2:3])
    assert len(written) == 1 and written[0][0].tolist() == [[1, 1]]


@pytest.mark.parametrize("natural", [False, True, None])
@pytest.mark.parametrize("packed", [False, True])
def test_mtp_matched_closure_installs_grammar_before_next_same_window_sample(natural, packed):
    job = make_job()
    new = Filter(allowed=3, packed=packed)
    arm(job, None if natural is None else (10 if natural else 1),
        callback=lambda j: j.set_filters([new]))
    scores = logits_for(width=4)
    if natural is not False:
        scores[:, 1, :] = 0
        scores[:, 1, END] = 5
    state = run_mtp(job, [1, END, 3], scores=scores)
    assert emitted_ids(state["results"]) == [1, END, 3, 3]
    assert new.fed == [3, 3] and job.token_budget is None
    assert state["accepted_lengths"] == [4]
    assert state["state"].rewinds == [0]
    assert job.sampler.calls[-1][1], "Content sample must use the newly installed mask"
    assert job.device_logit_mask.dtype == (torch.int32 if packed else torch.half)
    assert job.filter_futures == [] and job.logit_masks == []


def test_no_budget_keeps_original_batch_verify_fast_path():
    job = make_job()
    state = run_mtp(job, [1] * 5)
    assert emitted_ids(state["results"]) == [1] * 6
    assert job.sampler.calls == [(6, False)]
    assert job.token_budget is None


class SyncGenerator:
    def __init__(self, width=5, **_):
        self.active_jobs = []
        self.pending_jobs = []
        self.tokenizer = Tokenizer()
        self.width = width
        self.draft_model = None
        self.ngram_match_min = 0
        self.hybrid_draft_tokens = 0
        self.recurrent_cache = None
        self.padded_vocab_size = VOCAB
        self.drained = 0
        self.closed = False

    def enqueue(self, job):
        _attach(job, self, len(self.active_jobs))
        self.active_jobs.append(job)

    def iterate(self):
        rows = []
        for job in list(self.active_jobs):
            state = run_mtp(job, [1] * self.width)
            rows.extend(state["results"])
            if job.is_finished:
                self.active_jobs.remove(job)
        return rows

    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs)

    def on_queue_drained(self):
        self.drained += 1

    def close(self):
        self.closed = True


cancel_space = {}
_compile([_generator_node("cancel")], GEN_PATH, cancel_space)
SyncGenerator.cancel = cancel_space["cancel"]
ASYNC_NS = {"Generator": SyncGenerator, "Job": Job, "torch": torch, "asyncio": asyncio}
async_tree = _tree(ASYNC_PATH)
_compile([n for n in async_tree.body if isinstance(n, ast.ClassDef)
          or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                                              and t.id == "_CANCELLED_SENTINEL" for t in n.targets)],
         ASYNC_PATH, ASYNC_NS)
AsyncGenerator = ASYNC_NS["AsyncGenerator"]
AsyncJob = ASYNC_NS["AsyncJob"]


@pytest.mark.parametrize("delay", [0, 1, 12])
def test_real_async_producer_boundary_is_independent_of_consumer_backpressure(delay):
    async def scenario():
        generator = AsyncGenerator()
        try:
            job = AsyncJob(generator, input_ids=torch.tensor([[7, 1]]), max_new_tokens=35)
            callback_counts = []
            job.set_token_budget(24, torch.tensor([[END]]), end_token_id=END,
                                 on_end=lambda native: callback_counts.append(native.new_tokens))
            for _ in range(delay):
                await asyncio.sleep(0)
            queued_before_read = job.queue.qsize()
            rows = [row async for row in job]
            assert emitted_ids(rows) == [1] * 24 + [END] + [1] * 10
            assert callback_counts == [25] and rows[-1]["eos"]
            if delay == 12:
                assert queued_before_read > 24
        finally:
            await generator.close()

    asyncio.run(scenario())


def test_async_callback_error_does_not_abort_an_unrelated_job():
    async def scenario():
        generator = AsyncGenerator()
        try:
            failed = AsyncJob(generator, input_ids=torch.tensor([[7, 1]]), max_new_tokens=12)
            peer = AsyncJob(generator, input_ids=torch.tensor([[7, 1]]), max_new_tokens=8)

            def fail(_):
                raise RuntimeError("phase not available")

            failed.set_token_budget(0, torch.tensor([[END]]), end_token_id=END, on_end=fail)
            with pytest.raises(RuntimeError, match="phase not available"):
                [r async for r in failed]
            rows = [r async for r in peer]
            assert emitted_ids(rows) == [1] * 8
            assert generator.error is None and generator.iteration_task.done() is False
        finally:
            await generator.close()

    asyncio.run(scenario())


def test_async_pre_sample_guard_error_preserves_peer_and_accepts_no_token():
    async def scenario():
        generator = AsyncGenerator()
        try:
            failed = AsyncJob(generator, input_ids=torch.tensor([[7, 1]]), max_new_tokens=12)
            peer = AsyncJob(generator, input_ids=torch.tensor([[7, 1]]), max_new_tokens=8)

            def guard(_):
                raise RuntimeError("protected-region parser failed")

            failed.set_token_budget(0, torch.tensor([[END]]), end_token_id=END, can_end=guard)
            with pytest.raises(RuntimeError, match="protected-region parser failed"):
                [r async for r in failed]
            assert failed.job.new_tokens == 0
            assert failed.job.sequences[0].sequence_ids.torch().tolist() == [[7, 1]]
            assert failed.job.sampler.calls == []
            rows = [r async for r in peer]
            assert emitted_ids(rows) == [1] * 8 and generator.error is None
        finally:
            await generator.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("when", ["pending", "forced_tail", "natural"])
def test_real_async_cancel_disarms_budget_and_wakes_waiting_consumer(when):
    async def scenario():
        generator = AsyncGenerator(width=0 if when == "forced_tail" else 5)
        job = AsyncJob(generator, input_ids=torch.tensor([[7, 1]]), max_new_tokens=100)
        calls = []
        job.set_token_budget(None if when == "natural" else (0 if when == "forced_tail" else 24),
                             torch.tensor([[END, 10, 11]]), end_token_id=END,
                             on_end=lambda _: calls.append(True))
        if when == "forced_tail":
            await asyncio.sleep(0)
            assert job.job.new_tokens == 1 and job.job.forced_ids is not None
        await job.cancel()
        assert job.job.token_budget is None
        assert generator.generator.num_remaining_jobs() == 0
        assert calls == [] and job.cancelled
        assert [r async for r in job] == []
        assert generator.generator.drained == 1
        await generator.close()

    asyncio.run(scenario())



def test_natural_only_observation_never_encodes_forces_or_changes_limits():
    job = make_job(max_new_tokens=100)
    limits = (job.min_new_tokens, job.max_new_tokens, set(job.stop_tokens))
    observed = []
    ended = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Natural-only observation must not encode or force output")

    job.generator.tokenizer.encode = forbidden
    job.constrain_output_now = forbidden
    job.set_token_budget(None, end_token_id=END,
                         can_end=lambda j: (observed.append(j.new_tokens), True)[1],
                         on_end=lambda j: ended.append(j.new_tokens))
    assert job.token_budget["deadline"] is None and job.token_budget["output"] is None
    rows = []
    for _ in range(64):
        sample(job, 1, rows)
    assert emitted_ids(rows) == [1] * 64
    assert observed == [] and ended == [] and job.forced_ids is None
    sample(job, END, rows)
    assert observed == [65] and ended == [65] and job.token_budget is None
    assert limits == (job.min_new_tokens, job.max_new_tokens, set(job.stop_tokens))


def test_natural_only_guard_keeps_literal_end_and_does_not_force_later():
    job = make_job()
    allowed = [False]
    ended = []
    job.set_token_budget(None, can_end=lambda _: allowed[0], end_token_id=END,
                         on_end=lambda j: ended.append(j.new_tokens))
    sample(job, END)
    assert not job.token_budget["end_seen"] and ended == []
    allowed[0] = True
    assert [sample(job)[0][1].item() for _ in range(4)] == [1] * 4
    assert ended == [] and job.forced_ids is None
    sample(job, END)
    assert ended == [6] and job.token_budget is None


def test_natural_only_requeue_retains_observer_without_creating_deadline():
    job = make_job()
    ended = []
    job.set_token_budget(None, end_token_id=END,
                         on_end=lambda j: ended.append(j.rq_new_tokens + j.new_tokens))
    for _ in range(5):
        sample(job)
    observer = job.token_budget
    job.prepare_for_requeue()
    assert job.token_budget is observer and observer["deadline"] is None
    assert job.rq_new_tokens == 5 and job.new_tokens == 0
    sample(job, END)
    assert ended == [6] and job.token_budget is None


def test_natural_only_banned_end_rewind_preserves_observation_without_forcing():
    job = make_job(banned_strings=["</phase>x"])
    ended = []
    observed = []
    job.set_token_budget(None, end_token_id=END,
                         can_end=lambda j: (observed.append(j.new_tokens), True)[1],
                         on_end=lambda j: ended.append(j.new_tokens))
    sample(job, END)
    assert job.token_budget["end_seen"] and ended == []
    sample(job, 6)
    assert job.checkpoint_rewound and job.new_tokens == 0
    assert observed == [1, 0] and not job.token_budget["end_seen"]
    job.checkpoint_rewound = False
    assert [sample(job)[0][1].item() for _ in range(4)] == [1] * 4
    assert job.token_budget["deadline"] is None and ended == []
    sample(job, END)
    assert ended == []
    sample(job, 2)
    assert ended == [6] and job.token_budget is None


@pytest.mark.parametrize("termination", ["length", "native_stop"])
def test_natural_only_respects_existing_termination_without_callback(termination):
    job = make_job(max_new_tokens=2, stop_conditions=[END] if termination == "native_stop" else None)
    ended = []
    job.set_token_budget(None, end_token_id=END, on_end=lambda _: ended.append(True))
    if termination == "native_stop":
        result, rows = sample(job, END)
    else:
        sample(job)
        result, rows = sample(job)
    assert result[0] and rows[-1]["eos"] and ended == []
    assert job.token_budget is None and job.forced_ids is None


@pytest.mark.parametrize("failure", ["guard", "callback"])
def test_natural_only_boundary_errors_are_contained(failure):
    job = make_job()
    peer = make_job()

    def fail(_):
        raise RuntimeError("natural handoff failed")

    job.set_token_budget(None, end_token_id=END,
                         can_end=fail if failure == "guard" else None,
                         on_end=fail if failure == "callback" else None)
    result, rows = sample(job, END)
    assert result[0] and rows[-1]["stage"] == "error"
    assert isinstance(rows[-1]["error"], RuntimeError)
    assert job._token_budget_failed and job.new_tokens == 1
    assert job.token_budget is None and sample(peer)[0][1].item() == 1


def test_natural_only_async_capability_and_no_deadline_under_backpressure():
    assert AsyncJob.supports_natural_token_budget is True

    async def scenario():
        generator = AsyncGenerator()
        try:
            job = AsyncJob(generator, input_ids=torch.tensor([[7, 1]]), max_new_tokens=35)
            callbacks = []
            job.set_token_budget(None, end_token_id=END,
                                 on_end=lambda _: callbacks.append(True))
            assert job.job.token_budget["deadline"] is None
            for _ in range(12):
                await asyncio.sleep(0)
            queued = job.queue.qsize()
            rows = [row async for row in job]
            assert emitted_ids(rows) == [1] * 35 and rows[-1]["eos"]
            assert queued > 24 and callbacks == [] and job.job.token_budget is None
        finally:
            await generator.close()

    asyncio.run(scenario())
