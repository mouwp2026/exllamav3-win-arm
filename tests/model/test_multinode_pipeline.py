"""CPU-only regression tests for examples/multinode_pipeline.py."""

import ast
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[2] / "examples" / "multinode_pipeline.py"
RECURRENT_UTIL_SOURCE = SOURCE.parents[1] / "exllamav3" / "cache" / "recurrent_util.py"


def _load_canonical_advance():
    tree = ast.parse(
        RECURRENT_UTIL_SOURCE.read_text(encoding = "utf-8"),
        filename = str(RECURRENT_UTIL_SOURCE),
    )
    node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "advance_recurrent_states"
    )
    namespace = {"torch": SimpleNamespace(Tensor = object)}
    exec(
        compile(ast.Module(body = [node], type_ignores = []), str(RECURRENT_UTIL_SOURCE), "exec"),
        namespace,
    )
    return namespace["advance_recurrent_states"]


CANONICAL_ADVANCE = _load_canonical_advance()


def _load_pipeline_slice(advance):
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"), filename = str(SOURCE))
    node = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PipelineSlice"),
        None,
    )
    assert node is not None, "PipelineSlice helper is missing"
    namespace = {"advance_recurrent_states": advance}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(SOURCE), "exec"), namespace)
    return namespace["PipelineSlice"]


def _load_cache_helpers(cache_cls, quant_layer_cls):
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"), filename = str(SOURCE))
    names = {"parse_cache_quant", "create_cache"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names, "cache quantization helpers are missing"
    namespace = {"Cache": cache_cls, "CacheLayer_quant": quant_layer_cls}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE), "exec"), namespace)
    return namespace


class FakeState:
    def __init__(self):
        self.position = 0
        self.last_history = None
        self.post_advance_calls = 0
        self.freed = 0

    def post_advance(self):
        self.post_advance_calls += 1

    def free(self):
        self.freed += 1


class FakeRecurrentModel:
    def prepare_inputs(self, ids, params):
        states = params.get("recurrent_states")
        if params["past_len"]:
            if states is None:
                raise ValueError("Past length given, but no previous state for recurrence in params")
            assert states[0].position == params["past_len"]
        elif states is None:
            params["recurrent_states"] = [FakeState()]
        else:
            assert states[0].position == 0


class FakeNonRecurrentModel:
    def __init__(self):
        self.params = []

    def prepare_inputs(self, ids, params):
        assert "recurrent_states" not in params
        self.params.append(dict(params))


class FakeModule:
    caps = {}

    def prepare_for_device(self, x, params):
        return x

    def forward(self, x, params):
        return x


def test_default_generation_reuses_state_and_advances_once_per_slice_forward():
    advance_calls = []

    def advance(ids, params, model):
        advance_calls.append((ids, params["recurrent_states"]))
        CANONICAL_ADVANCE(ids, params, model)

    pipeline_slice = _load_pipeline_slice(advance)(
        model = FakeRecurrentModel(),
        cache = object(),
        fwd_modules = [(FakeModule(), 0, None)],
        rank = 0,
        max_num_tokens = 32,
    )
    prefill_ids = SimpleNamespace(shape = (1, 3))
    decode_ids = SimpleNamespace(shape = (1, 1))

    pipeline_slice.forward(prefill_ids, None, past_len = 0, last_only = False)
    states = pipeline_slice.recurrent_states
    pipeline_slice.forward(decode_ids, None, past_len = 3, last_only = True)

    assert pipeline_slice.recurrent_states is states
    assert states[0].position == 4
    assert states[0].post_advance_calls == 2
    assert [call[0] for call in advance_calls] == [prefill_ids, decode_ids]


def test_non_recurrent_slice_keeps_forward_params_stateless():
    advance_calls = []

    def advance(ids, params, model):
        advance_calls.append((ids, params, model))
        CANONICAL_ADVANCE(ids, params, model)

    model = FakeNonRecurrentModel()
    pipeline_slice = _load_pipeline_slice(advance)(
        model = model,
        cache = object(),
        fwd_modules = [(FakeModule(), 0, None)],
        rank = 1,
        max_num_tokens = 32,
    )
    ids = SimpleNamespace(shape = (1, 1))
    hidden = object()

    result = pipeline_slice.forward(ids, hidden, past_len = 7, last_only = False)

    assert result is hidden
    assert pipeline_slice.recurrent_states is None
    assert "recurrent_states" not in model.params[0]
    assert len(advance_calls) == 1


def test_past_zero_starts_with_fresh_recurrent_state_after_nll():
    def advance(ids, params, model):
        CANONICAL_ADVANCE(ids, params, model)

    pipeline_slice = _load_pipeline_slice(advance)(
        model = FakeRecurrentModel(),
        cache = object(),
        fwd_modules = [(FakeModule(), 0, None)],
        rank = 0,
        max_num_tokens = 32,
    )

    pipeline_slice.forward(SimpleNamespace(shape = (1, 5)), None, past_len = 0, last_only = False)
    nll_state = pipeline_slice.recurrent_states[0]
    pipeline_slice.forward(SimpleNamespace(shape = (1, 3)), None, past_len = 0, last_only = False)

    assert nll_state.freed == 1
    assert pipeline_slice.recurrent_states[0] is not nll_state
    assert pipeline_slice.recurrent_states[0].position == 3


def test_close_frees_recurrent_state_slot_once():
    def advance(ids, params, model):
        CANONICAL_ADVANCE(ids, params, model)

    pipeline_slice = _load_pipeline_slice(advance)(
        model = FakeRecurrentModel(),
        cache = object(),
        fwd_modules = [(FakeModule(), 0, None)],
        rank = 1,
        max_num_tokens = 32,
    )
    pipeline_slice.forward(SimpleNamespace(shape = (1, 2)), object(), past_len = 0, last_only = False)
    state = pipeline_slice.recurrent_states[0]

    pipeline_slice.close()
    pipeline_slice.close()

    assert state.freed == 1
    assert pipeline_slice.recurrent_states is None


def test_cache_quant_selects_q4_and_accepts_one_or_two_bitrates():
    calls = []

    class FakeCache:
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))

    quant_layer = object()
    helpers = _load_cache_helpers(FakeCache, quant_layer)
    create_cache = helpers["create_cache"]
    model = object()

    create_cache(model, 8192, None)
    create_cache(model, 8192, "4")
    create_cache(model, 8192, "4,6")

    assert calls[0] == (model, {"max_num_tokens": 8192})
    assert calls[1] == (model, {
        "max_num_tokens": 8192,
        "layer_type": quant_layer,
        "k_bits": 4,
        "v_bits": 4,
    })
    assert calls[2] == (model, {
        "max_num_tokens": 8192,
        "layer_type": quant_layer,
        "k_bits": 4,
        "v_bits": 6,
    })
    try:
        helpers["parse_cache_quant"]("4,6,8")
    except ValueError as exc:
        assert str(exc) == "Specify either one or two bitrates for cache quantization"
    else:
        raise AssertionError("three cache bitrates should be rejected")


def test_entrypoint_frees_recurrent_state_on_shutdown():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"), filename = str(SOURCE))
    entrypoint = next(n for n in tree.body if isinstance(n, ast.Try))
    final_calls = [ast.unparse(n) for n in entrypoint.finalbody]
    assert final_calls[0] == "pipeline_slice.close()"


def test_cli_keeps_transport_defaults_and_adds_cache_quant():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"), filename = str(SOURCE))
    calls = [
        n.value for n in tree.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "add_argument"
    ]

    def argument(flag):
        return next(call for call in calls if any(
            isinstance(arg, ast.Constant) and arg.value == flag for arg in call.args
        ))

    def keywords(call):
        return {kw.arg: kw.value for kw in call.keywords}

    transport = argument("--transport")
    port = argument("--port")
    cache_quant = argument("--cache_quant")

    assert ast.literal_eval(keywords(transport)["choices"]) == ["tcp", "nccl"]
    assert ast.literal_eval(keywords(transport)["default"]) == "tcp"
    assert ast.literal_eval(keywords(port)["default"]) == 29650
    assert [ast.literal_eval(arg) for arg in cache_quant.args] == ["-cq", "--cache_quant"]
    assert "default" not in keywords(cache_quant)
