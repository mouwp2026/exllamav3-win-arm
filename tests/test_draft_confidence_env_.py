"""EXL3_DRAFT_CONFIDENCE fallback for Generator(draft_confidence=None).

API servers that build the Generator without exposing draft_confidence (TabbyAPI) get the
engine default unless the environment sets it. This checks the resolution rule only; it does
not construct a Generator (that needs a loaded model and a GPU).
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "exllamav3" / "generator" / "generator.py"


def _init_default():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Generator":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    args = fn.args.args[-len(fn.args.defaults):]
                    for a, d in zip(args, fn.args.defaults):
                        if a.arg == "draft_confidence":
                            return d
    raise AssertionError("Generator.__init__ has no draft_confidence argument")


def _resolve(value, env):
    # Mirror of the rule in Generator.__init__
    if value is None:
        value = float(env.get("EXL3_DRAFT_CONFIDENCE", "0.4"))
    return value


def test_signature_default_is_none():
    d = _init_default()
    assert isinstance(d, ast.Constant) and d.value is None


def test_env_read_in_source():
    src = SRC.read_text(encoding="utf-8")
    assert 'environ.get("EXL3_DRAFT_CONFIDENCE", "0.4")' in src


def test_resolution():
    assert _resolve(None, {}) == 0.4
    assert _resolve(None, {"EXL3_DRAFT_CONFIDENCE": "0.6"}) == 0.6
    assert _resolve(0.3, {"EXL3_DRAFT_CONFIDENCE": "0.6"}) == 0.3
