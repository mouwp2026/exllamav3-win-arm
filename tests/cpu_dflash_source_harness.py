"""Stdlib-only source execution: real methods, explicit tensor/cache seams.

No torch/exllamav3 imports, CUDA, model weights, kernels or network calls.
Only receive_sample is sliced: execute its unchanged prefix through the actual
requeue decision, then return it before unrelated streaming/page bookkeeping.
All other extracted methods execute their complete, unchanged AST bodies.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS
import time

PAGE_SIZE = 256
# Source root: the checkout that owns exllamav3/ (this test lives in tests/).
ROOT = Path(os.environ.get("DFLASH_SOURCE_ROOT", Path(__file__).resolve().parents[1]))


class Tensor:
    """Minimal scalar/1D/2D tensor with real shared-storage slice views."""
    def __init__(self, shape, fill=0, storage=None, indices=None):
        self.shape = tuple(shape)
        size = 1
        for n in self.shape:
            size *= n
        self.storage = [fill] * size if storage is None else storage
        self.indices = list(range(size)) if indices is None else indices

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key += (slice(None),) * (len(self.shape) - len(key))
        axes = []
        outshape = []
        for n, k in zip(self.shape, key):
            axis = list(range(n))[k]
            if isinstance(axis, int):
                axes.append([axis])
            else:
                axes.append(axis)
                outshape.append(len(axis))
        import itertools
        indices = []
        for coords in itertools.product(*axes):
            flat = 0
            for coord, n in zip(coords, self.shape):
                flat = flat * n + coord
            indices.append(self.indices[flat])
        return Tensor(outshape, storage=self.storage, indices=indices)

    def __setitem__(self, key, value):
        self[key].copy_(value)

    def copy_(self, other):
        if isinstance(other, Tensor):
            assert self.shape == other.shape, (self.shape, other.shape)
            values = other.tolist()
        else:
            values = [other] * len(self.indices)
        for i, v in zip(self.indices, values):
            self.storage[i] = v
        return self

    def zero_(self):
        return self.copy_(0)

    def cpu(self):
        return self

    def item(self):
        assert len(self.indices) == 1
        return self.storage[self.indices[0]]

    def tolist(self):
        return [self.storage[i] for i in self.indices]


class Torch:
    long = int32 = int
    float = float
    Tensor = Tensor

    @staticmethod
    def empty(shape, **kwargs):
        return Tensor(shape)

    zeros = empty

    @staticmethod
    def full(shape, value, **kwargs):
        return Tensor(shape, value)

    @staticmethod
    def tensor(data, **kwargs):
        if isinstance(data, list) and data and isinstance(data[0], list):
            return Tensor((len(data), len(data[0])), storage=[v for row in data for v in row])
        if isinstance(data, list):
            return Tensor((len(data),), storage=data[:])
        return Tensor((), fill=data)

    @staticmethod
    def cat(tensors, dim=0):
        shape = list(tensors[0].shape)
        dim %= len(shape)
        shape[dim] = sum(t.shape[dim] for t in tensors)
        result = Tensor(shape)
        start = 0
        for t in tensors:
            key = [slice(None)] * len(shape)
            key[dim] = slice(start, start + t.shape[dim])
            result[tuple(key)].copy_(t)
            start += t.shape[dim]
        return result


class SeqTensor:
    def __init__(self, shape, **kwargs):
        self.tensor = Tensor(shape)

    @classmethod
    def ids(cls, length):
        return cls((1, length))

    def __len__(self):
        return self.tensor.shape[-1]

    def torch_slice(self, start, stop):
        return self.tensor[:, start:stop]

    def torch(self):
        return self.tensor


class CPUPageTable:
    """Allocate opaque CPU page records from actual Sequence.allocate_pages args."""
    def __init__(self, generator, cache):
        self.max_pages = cache.max_num_tokens // PAGE_SIZE
        self.referenced_pages = {}
        self.calls = []

    def allocate_pages(self, hashes, unique, recurrent, protected, restore):
        count = len(hashes) + unique
        assert count <= self.max_pages
        self.calls.append((len(hashes), unique))
        return [NS(page_index=40+i) for i in range(count)], 0, 0, False


def extract_class(relative, name, methods, namespace, requeue_prefix=False):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    bodies = []
    for method in methods:
        node = copy.deepcopy(next(n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name == method))
        if requeue_prefix and method == "receive_sample":
            stop = next(i for i, statement in enumerate(node.body) if isinstance(statement, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "requeue_now" for t in statement.targets))
            node.body = node.body[:stop+1] + [ast.Return(value=ast.Name(id="requeue_now", ctx=ast.Load()))]
        bodies.append(node)
    classnode = ast.ClassDef(name=name, bases=[], keywords=[], body=bodies, decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, classnode], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


NS_GLOBALS = {
    "torch": Torch, "PAGE_SIZE": PAGE_SIZE, "time": time, "_os": os,
    "PageTable": CPUPageTable, "SeqTensor": SeqTensor,
    "ThreadPoolExecutor": lambda **kw: None,
    "DraftConfidenceCalibrator": lambda confidence: NS(confidence=confidence),
    "ext": NS(BC_SAM=lambda: NS()), "cuda_sync_active": lambda: None,
    "tensor_hash_checksum": lambda tensor, prev: hashlib.blake2b(
        (prev or b"") + repr(tensor.tolist()).encode(), digest_size=16).digest(),
}
Generator = extract_class("exllamav3/generator/generator.py", "Generator",
                          ["__init__", "_staging", "iterate_draftmodel_dflash_gen"], NS_GLOBALS)
Sequence = extract_class("exllamav3/generator/pagetable.py", "Sequence",
                         ["prepare", "allocate_pages", "build_block_index_tensor"], NS_GLOBALS)
JobMethods = extract_class("exllamav3/generator/job.py", "Job",
                           ["prepare_for_queue", "receive_sample", "prepare_for_requeue",
                            "is_prefill_done", "get_max_seq_len", "get_input_ids_list", "_check_banned_strings"],
                           NS_GLOBALS, requeue_prefix=True)
InputLayer = extract_class("exllamav3/modules/arch_specific/dflash.py", "DFlashInputLayer", ["forward"], NS_GLOBALS)


class CPUJob(JobMethods):
    """Only Job.__init__/unrelated streaming state are stubbed, not reservations."""
    def __init__(self, input_ids, max_new_tokens=1, max_rq_tokens=None, **kwargs):
        prompt = input_ids.shape[-1]
        seq = Sequence()
        seq.input_ids = SeqTensor.ids(prompt)
        seq.sequence_ids = SeqTensor.ids(prompt)
        seq.max_cached_pages = None
        seq.kv_position = prompt - 1
        self.sequences = [seq]
        self.max_new_tokens = max_new_tokens
        self.max_rq_tokens = self.orig_max_rq_tokens = max_rq_tokens
        self.prefix_token = None
        self.banned_strings = []
        self.embeddings = []
        self.return_top_tokens = 0
        self.new_tokens = 0
        self.forced_sample = False
        # Job.__init__ is stubbed here; keep its optional phase state disabled.
        # The dedicated token-budget harness exercises guarded phase transitions.
        self.token_budget = kwargs.get("rq_state", {}).get("token_budget")
        self.token_budget_error = None
        self._token_budget_failed = False
        self.filters_suspended = False
        self.filters = []
        self.time_first_token = 1
        self.min_new_tokens = 0
        self.decode_special_tokens = False
        self.return_logits = self.return_probs = False
        self.identifier = self.sampler = self.stop_on_loop = None
        self.rng = self.stop_strings = self.stop_tokens = None
        self.stop_strings_utf32_buffer = self.stop_strings_utf32_offsets = None
        self.held_text = self.full_completion = ""
        self.held_tokens = self.held_probs = self.held_k_tokens = self.held_k_probs = self.held_logits = None
        self.time_enqueued = self.time_prefill = self.time_generate = 0
        self.accepted_draft_tokens = self.rejected_draft_tokens = 0
        self.rq_prompt_tokens = self.rq_cached = None
        self.rq_new_tokens = kwargs.get("rq_state", {}).get("rq_new_tokens", 0)
        self.cached_pages = self.cached_tokens = 0
        self.sam = self.forced_ids = None
        self.forced_index = 0
        self.last_init_kwargs = dict(input_ids=input_ids, max_new_tokens=max_new_tokens,
                                     max_rq_tokens=max_rq_tokens, **kwargs)

    def accept_one(self):
        requeue = self.receive_sample(None, Torch.tensor(1), None, None, None, [])
        for seq in self.sequences:
            seq.sequence_ids = SeqTensor.ids(len(seq.sequence_ids) + 1)
            seq.kv_position += 1
        return requeue


class CPUDraft:
    def __init__(self, block_size, mode, conf_len=None):
        self.config = NS(block_size=block_size) if mode == "dflash" else NS()
        self.caps = {"dflash_draft": mode == "dflash", "mtp_draft": mode == "mtp"}
        if mode == "dflash":
            self.caps["default_draft_size"] = block_size - 1
        self.conf_len = conf_len
        self.calls = []
        self.sequences = []
        self.input_layer = InputLayer()
        self.input_layer.native_draft_len = block_size
        self.input_layer.mask_token_id = 123
        self.input_layer.input_embedding_scale = 1.0
        self.input_layer.mask_embedding = None
        self.input_layer.attached_model = lambda: NS(loaded_tp=False, modules=[NS(forward=lambda x, params: x)])

    def forward(self, input_ids, params):
        state = self.input_layer.forward(input_ids, params)
        if self.conf_len is not None:
            params["draft_confidence_len"] = self.conf_len
        for row, seq in enumerate(self.sequences):
            start = params["cache_seqlens"][row].item()
            end = start + state.shape[-1]
            capacity = len(seq.allocated_pages) * PAGE_SIZE
            record = dict(start=start, end_exclusive=end, capacity=capacity,
                          native_rows=state.shape[-1], table_width=params["block_table"].shape[-1])
            self.calls.append(record)
            assert end <= capacity, f"native DFlash write exceeds allocated pages: {record}"
            for position in range(start, end):
                logical_page = position // PAGE_SIZE
                actual = params["block_table"][row, logical_page].item()
                expected = seq.allocated_pages[logical_page].page_index
                assert actual == expected, (position, actual, expected)
        return state

    def sample_from_state(self, state, params):
        return state


def generator(mode="dflash", ndt=1, block_size=8, max_pages=32, dynamic=False, conf_len=None):
    draft = CPUDraft(block_size, mode, conf_len) if mode in ("dflash", "ar", "mtp") else None
    cache = NS(max_num_tokens=max_pages * PAGE_SIZE)
    gen = Generator(model=NS(config=NS(vocab_size=32), caps={}), cache=cache, tokenizer=NS(),
                    max_batch_size=4, draft_model=draft,
                    draft_cache=NS(max_num_tokens=cache.max_num_tokens) if draft else None,
                    num_draft_tokens=ndt, ngram_match_min=2 if mode == "ngram" else 0,
                    dynamic_draft_tokens=dynamic)
    return gen


def queue(gen, prompt=253, max_new=1, max_rq=None, prefix=False):
    job = CPUJob(Tensor((1, prompt)), max_new_tokens=max_new, max_rq_tokens=max_rq)
    job.prefix_token = 1 if prefix else None
    if prefix:
        job.sequences[0].sequence_ids = SeqTensor.ids(prompt - 1)
        job.new_tokens = -1
    job.prepare_for_queue(gen, 17)
    allocate(gen, job)
    return job


def allocate(gen, job):
    for seq in job.sequences:
        seq.allocate_pages(gen.pagetable, None)
        seq.kv_position = len(seq.sequence_ids) - 1
    gen.active_jobs = [job]
    if gen.draft_model:
        gen.draft_model.sequences = job.sequences


def draft(gen):
    return gen.iterate_draftmodel_dflash_gen([])
