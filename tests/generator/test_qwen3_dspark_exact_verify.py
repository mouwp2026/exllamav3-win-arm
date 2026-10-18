from types import SimpleNamespace

import torch

from exllamav3.generator.generator import Generator
from exllamav3.architecture.qwen3_dspark import Qwen3DSparkModel


class FakeJob:
    def __init__(self, prefill_done=True):
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0
        self.draft_stats = []
        self.sequences = [object()]
        self.prefill_done = prefill_done

    def is_prefill_done(self):
        return self.prefill_done


def make_generator(target_tokens, active_jobs=1, include_counter_fields=True):
    gen = object.__new__(Generator)
    jobs = [FakeJob() for _ in range(active_jobs)]
    gen.active_jobs = jobs
    gen.record_draft_stats = True
    gen._draft_conf_round = None
    gen.draft_calibrator = None
    gen._exact_calls = []
    emitted = iter(target_tokens)

    def ordinary_target_step(results, draft_tokens=None):
        assert draft_tokens is None
        gen._exact_calls.append(None)
        for job in list(gen.active_jobs):
            token, eos = next(emitted)
            result = {
                "stage": "streaming",
                "job": job,
                "token_ids": torch.tensor([[token]]),
                "eos": eos,
            }
            if include_counter_fields:
                result.update({
                    "accepted_draft_tokens": job.accepted_draft_tokens,
                    "rejected_draft_tokens": job.rejected_draft_tokens,
                })
            results.append(result)
            if eos:
                gen.active_jobs.remove(job)

    gen.iterate_gen = ordinary_target_step
    return gen, jobs


def test_exact_serial_draft_accepts_prefix_then_rejects_remainder():
    gen, (job,) = make_generator([(11, False), (99, False)])
    results = []

    gen.iterate_exact_serial_draft(results, torch.tensor([[11, 12, 13]]))

    assert [int(r["token_ids"].item()) for r in results] == [11, 99]
    assert len(gen._exact_calls) == 2
    assert job.accepted_draft_tokens == 1
    assert job.rejected_draft_tokens == 2
    assert results[-1]["accepted_draft_tokens"] == 1
    assert results[-1]["rejected_draft_tokens"] == 2
    assert job.draft_stats == [(0, 3, 1)]


def test_exact_serial_draft_emits_bonus_after_full_acceptance():
    gen, (job,) = make_generator([(11, False), (12, False), (77, False)])
    results = []

    gen.iterate_exact_serial_draft(results, torch.tensor([[11, 12]]))

    assert [int(r["token_ids"].item()) for r in results] == [11, 12, 77]
    assert len(gen._exact_calls) == 3
    assert job.accepted_draft_tokens == 2
    assert job.rejected_draft_tokens == 0
    assert job.draft_stats == [(0, 2, 2)]


def test_exact_serial_draft_counts_current_and_tail_rejected_on_eos():
    gen, (job,) = make_generator([(11, True)])
    results = []

    gen.iterate_exact_serial_draft(results, torch.tensor([[11, 12, 13]]))

    assert job.accepted_draft_tokens == 0
    assert job.rejected_draft_tokens == 3
    assert results[-1]["accepted_draft_tokens"] == 0
    assert results[-1]["rejected_draft_tokens"] == 3


def test_exact_serial_draft_fails_closed_to_one_target_step_for_batched_jobs():
    gen, jobs = make_generator([(21, False), (22, False)], active_jobs=2)
    results = []

    gen.iterate_exact_serial_draft(results, torch.tensor([[11, 12], [13, 14]]))

    assert len(gen._exact_calls) == 1
    assert [j.accepted_draft_tokens for j in jobs] == [0, 0]
    assert [j.rejected_draft_tokens for j in jobs] == [0, 0]


def test_exact_serial_draft_fails_closed_when_another_active_job_is_prefilling():
    gen, jobs = make_generator(
        [(11, False), (22, False), (12, False), (23, False), (77, False), (24, False)],
        active_jobs=2,
    )
    jobs[1].prefill_done = False
    results = []

    gen.iterate_exact_serial_draft(results, torch.tensor([[11, 12], [13, 14]]))

    assert len(gen._exact_calls) == 1
    assert [j.accepted_draft_tokens for j in jobs] == [0, 0]
    assert [j.rejected_draft_tokens for j in jobs] == [0, 0]


def test_exact_serial_draft_adds_final_counters_to_result_events():
    gen, (job,) = make_generator([(99, False)], include_counter_fields=False)
    results = []

    gen.iterate_exact_serial_draft(results, torch.tensor([[11, 12]]))

    assert results[-1]["accepted_draft_tokens"] == 0
    assert results[-1]["rejected_draft_tokens"] == 2


def test_exact_serial_draft_ignores_unused_pinned_capacity_rows():
    gen, (job,) = make_generator([(11, False), (12, False), (77, False)])
    results = []

    gen.iterate_exact_serial_draft(
        results,
        torch.tensor([[11, 12], [999, 999], [998, 998]]),
    )

    assert [int(r["token_ids"].item()) for r in results] == [11, 12, 77]
    assert job.accepted_draft_tokens == 2
    assert job.rejected_draft_tokens == 0


def test_qwen3_dspark_requests_exact_serial_target_verification():
    model = object.__new__(Qwen3DSparkModel)
    model.caps = {}
    Qwen3DSparkModel.enable_exact_verification(model)
    assert model.caps["exact_serial_verify"] is True
