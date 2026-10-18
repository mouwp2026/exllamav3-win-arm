"""CPU source-level reservation regressions (not GPU correctness tests)."""
import unittest
from cpu_dflash_source_harness import generator, queue, draft, allocate


class NativeBlockReservationTests(unittest.TestCase):
    def test_known_request_end_boundary(self):
        gen = generator()
        job = queue(gen, prompt=253, max_new=1)
        result = draft(gen)
        self.assertEqual(result.shape, (4, 1))  # verification window remains shortened
        self.assertEqual(len(job.sequences[0].allocated_pages), 2)
        self.assertEqual(gen.draft_model.calls[-1]["native_rows"], 8)

    def test_requeue_before_next_native_block_overflows(self):
        gen = generator()
        job = queue(gen, prompt=240, max_new=1000, max_rq=16)
        while True:
            draft(gen)
            if job.accept_one():
                break
            self.assertLess(job.new_tokens, 20)
        self.assertEqual(job.new_tokens, 17)
        self.assertEqual(job.max_rq_tokens, 16)
        self.assertEqual(gen.draft_model.calls[-1]["end_exclusive"], 263)
        requeued = job.prepare_for_requeue()
        self.assertIs(requeued, job)
        self.assertEqual(requeued.serial_number, 17)
        self.assertEqual(requeued.last_init_kwargs["max_new_tokens"], 983)
        self.assertEqual(requeued.last_init_kwargs["max_rq_tokens"], 16)
        self.assertFalse(requeued.last_init_kwargs["token_healing"])
        allocate(gen, requeued)
        draft(gen)

    def test_implicit_output_limit_leaves_native_headroom(self):
        gen = generator(max_pages=2)
        job = queue(gen, prompt=253, max_new=None)
        self.assertEqual(job.max_new_tokens, gen.max_total_tokens - 253 - 1 - 7)
        self.assertEqual(len(job.sequences[0].allocated_pages), 2)
        draft(gen)

    def test_short_explicit_requeue_budget_fits_first_native_block(self):
        gen = generator()
        job = queue(gen, prompt=253, max_new=1000, max_rq=1)
        draft(gen)
        self.assertEqual(len(job.sequences[0].allocated_pages), 2)
        self.assertEqual((253 + job.max_rq_tokens) % 256, 0)
        self.assertEqual(job.orig_max_rq_tokens, 1)

    def test_ndt_four_still_reserves_eight_native_rows(self):
        gen = generator(ndt=4)
        queue(gen, prompt=250, max_new=1)
        self.assertEqual(draft(gen).shape[-1], 4)
        self.assertEqual(gen.draft_model.calls[-1]["native_rows"], 8)

    def test_ndt_two_still_reserves_eight_native_rows(self):
        gen = generator(ndt=2)
        queue(gen, prompt=252, max_new=1)
        self.assertEqual(draft(gen).shape[-1], 2)
        self.assertEqual(gen.draft_model.calls[-1]["native_rows"], 8)

    def test_last_round_of_longer_request(self):
        gen = generator(ndt=1)
        job = queue(gen, prompt=230, max_new=24)
        for _ in range(23):
            self.assertFalse(job.accept_one())
        draft(gen)
        self.assertEqual(gen.draft_model.calls[-1]["start"], 252)
        self.assertEqual(gen.draft_model.calls[-1]["end_exclusive"], 260)

    def test_capacity_rejects_before_allocating_missing_page(self):
        gen = generator(max_pages=1)
        with self.assertRaisesRegex(AssertionError, r"requires 2 pages \(only 1 available\)"):
            queue(gen)
        self.assertEqual(gen.pagetable.calls, [])
        self.assertEqual(gen.draft_model.calls, [])

    def test_confidence_zero_still_writes_full_block(self):
        gen = generator(conf_len=0, dynamic=True)
        queue(gen)
        self.assertIsNone(draft(gen))
        self.assertEqual(gen.draft_model.calls[-1]["native_rows"], 8)

    def test_healing_preserves_native_headroom(self):
        gen = generator()
        job = queue(gen, prompt=253, max_new=1, prefix=True)
        draft(gen)
        self.assertFalse(job.accept_one())  # healed token: new_tokens becomes zero
        draft(gen)
        self.assertEqual(gen.draft_model.calls[-1]["start"], 252)

    def test_native_block_can_span_more_than_one_additional_page(self):
        gen = generator(block_size=513)
        job = queue(gen, prompt=253, max_new=1, max_rq=1)
        draft(gen)
        self.assertEqual(len(job.sequences[0].allocated_pages), 3)
        self.assertEqual(gen.draft_model.calls[-1]["native_rows"], 513)

    def test_recurrent_requeue_boundary_is_still_aligned(self):
        gen = generator()
        gen.recurrent_cache = object()
        gen.recurrent_checkpoint_interval = 512
        job = queue(gen, prompt=509, max_new=1000, max_rq=1)
        self.assertEqual((509 + job.max_rq_tokens) % 512, 0)
        draft(gen)

    def test_verification_ceiling_above_native_size_is_not_reduced(self):
        gen = generator(ndt=12, block_size=8)
        job = queue(gen, max_new=5)
        self.assertEqual(gen.num_draft_tokens, 12)
        self.assertEqual(job.max_rq_tokens, 5 + 1 + 12)
        # Do not draft: ndt > native proposals is an existing unsupported shape case.


class CompatibilityTests(unittest.TestCase):
    """Default response budgets remain stable; explicit requeue lands on a full page."""


def compatibility_case(mode, ndt, prompt, max_new, max_rq):
    def test(self):
        gen = generator(mode=mode, ndt=ndt, max_pages=8)
        if mode == "none":
            expected_ndt = 0
        elif mode == "ngram":
            expected_ndt = 4 if ndt is None else ndt
        else:
            expected_ndt = ndt or (7 if mode == "dflash" else 4)
        self.assertEqual(gen.num_draft_tokens, expected_ndt)
        job = queue(gen, prompt=prompt, max_new=max_new, max_rq=max_rq)
        expected_new = max_new if max_new is not None else max(1, 8*256-prompt-1-expected_ndt)
        expected_budget = expected_new + 1 + expected_ndt if max_rq is None else (
            (prompt-1+max_rq+255)//256*256-prompt)
        self.assertEqual(job.max_new_tokens, expected_new)
        self.assertEqual(job.max_rq_tokens, expected_budget)
        headroom = expected_ndt if max_rq is not None else 0
        margin = 0 if max_rq is not None else expected_ndt
        self.assertEqual(len(job.sequences[0].allocated_pages), (prompt+expected_budget+headroom+255)//256)
        job.new_tokens = expected_budget-margin-1
        self.assertFalse(job.accept_one())
        self.assertTrue(job.accept_one())
    return test


COMPAT_CASES = []
for mode, ndt in (("none", None), ("none", 7), ("ar", None), ("ar", 1),
                  ("mtp", None), ("mtp", 1), ("ngram", None), ("ngram", 0),
                  ("ngram", 1), ("dflash", None), ("dflash", 0), ("dflash", 7)):
    for prompt, max_new, max_rq in ((253, 1, None), (257, 10, None),
                                    (253, None, None), (240, 1000, 512)):
        name = f"test_{mode}_ndt{ndt}_p{prompt}_new{max_new}_rq{max_rq}"
        setattr(CompatibilityTests, name, compatibility_case(mode, ndt, prompt, max_new, max_rq))
        COMPAT_CASES.append(name)


class BoundaryMatrixTests(unittest.TestCase):
    pass


def request_end_case(block_size, ndt, prompt, max_new):
    def test(self):
        gen = generator(block_size=block_size, ndt=ndt)
        job = queue(gen, prompt=prompt, max_new=max_new)
        for _ in range(max_new):
            result = draft(gen)
            self.assertEqual(result.shape[-1], ndt)
            self.assertEqual(gen.draft_model.calls[-1]["native_rows"], block_size)
            self.assertFalse(job.accept_one())
    return test


def requeue_case(block_size, ndt, prompt, max_rq):
    def test(self):
        gen = generator(block_size=block_size, ndt=ndt)
        job = queue(gen, prompt=prompt, max_new=10000, max_rq=max_rq)
        for _segment in range(2):
            for _round in range(600):
                draft(gen)
                if job.accept_one():
                    break
            else:
                self.fail("requeue did not trigger")
            job = job.prepare_for_requeue()
            self.assertEqual(job.orig_max_rq_tokens, max_rq)
            allocate(gen, job)
        draft(gen)
    return test


for block in (4, 8, 16):
    for ndt in (1, block-1):
        for prompt in (1, 249, 253, 255, 256, 257):
            for max_new in (1, 9):
                name = f"test_end_block{block}_ndt{ndt}_p{prompt}_new{max_new}"
                setattr(BoundaryMatrixTests, name, request_end_case(block, ndt, prompt, max_new))
    for ndt in (1, 2, block-1):
        for prompt in (240, 253, 256):
            for max_rq in (1, 16):
                name = f"test_requeue_block{block}_ndt{ndt}_p{prompt}_rq{max_rq}"
                setattr(BoundaryMatrixTests, name, requeue_case(block, ndt, prompt, max_rq))


if __name__ == "__main__":
    unittest.main(verbosity=2)
