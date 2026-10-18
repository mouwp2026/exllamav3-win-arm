import os

import pytest
import torch

from exllamav3.architecture.qwen3_dspark import Qwen3DSparkModel


def test_one_pass_verification_is_the_default_and_serial_is_opt_in(monkeypatch):
    # The default dSparK path verifies a whole draft window in one target forward; the per-token
    # reference verifier stays available behind EXL3_DSPARK_SERIAL_VERIFY=1
    src = open(Qwen3DSparkModel.__init__.__code__.co_filename, encoding = "utf8").read()
    assert 'os.environ.get("EXL3_DSPARK_SERIAL_VERIFY", "0") == "1"' in src
    assert '"max_verify_rows"' in src


TARGET = os.environ.get("MINICPM_TARGET_DIR")
DRAFT = os.environ.get("DSPARK_CHECKPOINT")


@pytest.mark.skipif(not (TARGET and DRAFT and torch.cuda.is_available()),
                    reason = "set MINICPM_TARGET_DIR and DSPARK_CHECKPOINT (GPU) to run")
def test_one_pass_dspark_output_equals_target_only_greedy():
    from exllamav3 import Config, Model, Tokenizer, Generator, Job, Cache, CacheLayer_fp16
    from exllamav3.generator.sampler import ArgmaxSampler

    tc = Config.from_directory(TARGET); tm = Model.from_config(tc); tok = Tokenizer.from_config(tc)
    dc = Config.from_directory(DRAFT); dm = Model.from_config(dc)
    tcache = Cache(tm, max_num_tokens = 4096, layer_type = CacheLayer_fp16, max_history = 64)
    dcache = Cache(dm, max_num_tokens = 4096, layer_type = CacheLayer_fp16, max_history = 64)
    tm.load(device = torch.device("cuda:0"), progressbar = False)
    dm.load(device = torch.device("cuda:0"), progressbar = False)
    assert dm.caps["exact_serial_verify"] is False

    prompt = "Write a Python function that merges two sorted lists, then explain its complexity."
    ids = tok.encode(prompt, add_bos = True)

    def run(**kw):
        gen = Generator(model = tm, cache = tcache, tokenizer = tok, max_batch_size = 1, **kw)
        job = Job(input_ids = ids, max_new_tokens = 160, stop_conditions = [], sampler = ArgmaxSampler())
        gen.enqueue(job)
        out = []
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                    out += r["token_ids"].flatten().tolist()
        return out, job

    plain, _ = run()
    for width in (4, 7):
        drafted, job = run(draft_model = dm, draft_cache = dcache, num_draft_tokens = width)
        assert drafted == plain, f"width {width}: first divergence at {next(i for i, (a, b) in enumerate(zip(plain, drafted)) if a != b)}"
        assert job.accepted_draft_tokens > 0
