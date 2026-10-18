"""Execute production driver AST with CPU leaves; no model weights or CUDA imports."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'examples/multinode_pipeline.py'

def functions(namespace):
    nodes = [n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.FunctionDef) and n.name not in namespace]
    for node in nodes: node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), namespace)
    return namespace

def test_glm5_next_recurrent_family_is_allowed():
    ns = functions({})
    model = NS(
        caps={'recurrent_states': True},
        config=NS(architecture='Glm5NextForConditionalGeneration'),
    )

    ns['validate_model_family'](model)


@pytest.mark.parametrize('model', [
    NS(
        caps={'recurrent_states': True},
        config=NS(architecture='Qwen3_5ForConditionalGeneration'),
    ),
    NS(caps={'recurrent_states': True}),
])
def test_other_or_unknown_recurrent_family_is_rejected(model):
    ns = functions({})

    with pytest.raises(ValueError, match='recurrent-state architecture'):
        ns['validate_model_family'](model)


def test_recurrent_family_rejected_before_input_preparation():
    def forbidden(*args): raise AssertionError('recurrent input preparation reached')
    ns = functions({'model': NS(
                        caps={'recurrent_states': True},
                        config=NS(architecture='Qwen3_5ForConditionalGeneration'),
                        prepare_inputs=forbidden,
                    ),
                    'cache': None, 'args': NS(ctx=256), 'R': 0, 'my_fwd': [], 'torch': torch})
    with pytest.raises(ValueError, match='recurrent'): ns['run_slice'](torch.ones((1,3), dtype=torch.long), None, 0, False)

def test_startup_rejects_recurrent_before_cache_loading_and_network():
    def forbidden(*args, **kwargs): raise AssertionError('cache, device, load or connection reached')
    tree = ast.parse(PATH.read_text())
    # Complete startup nodes after args parsing through local slice construction;
    # parser/import leaves and actual weight load are intentionally not executed.
    start = next(i for i,n in enumerate(tree.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'R' for t in n.targets))
    end = next(i for i,n in enumerate(tree.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'my_files' for t in n.targets))
    config = NS(architecture='Qwen3_5ForConditionalGeneration')
    ns = functions({'args': NS(rank=0, addrs='a,b', splits='0:1,1:2', model='fake', transport='tcp', ctx=256, port=1, chunk=64, max_new=3),
                    'torch': NS(device=forbidden, cuda=NS(set_device=forbidden)),
                    'Config': NS(from_directory=lambda _: config),
                    'Model': NS(from_config=lambda cfg: NS(caps={'recurrent_states': True}, config=cfg)),
                    'Cache': forbidden, 'NetEndpoint': NS(listen=forbidden),
                    'threading': NS(Thread=forbidden), 'time': NS(time=lambda: 0)})
    with pytest.raises(ValueError, match='recurrent'):
        exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]), str(PATH), 'exec'), ns)


def driver(prompt_len=255, ctx=256, max_new=3, nll_file=None, nll_len=0, chunk=64, stop_token=1):
    import time, math
    steps, messages, logs = [], [], []
    class Tokenizer:
        eos_token_id = 999
        def single_id(self, _): return 998
        def encode(self, text, **kwargs):
            return torch.ones((1, nll_len if text == 'NLL' else prompt_len), dtype=torch.long)
        def decode(self, *args, **kwargs): return ['dummy']
    class Down:
        stopped = False
        def send_obj(self, msg):
            messages.append(msg)
            self.stopped = msg['cmd'] == 'stop'
        def send_tensor(self, _): pass
    down = Down()
    class Up:
        def recv_obj(self):
            if down.stopped: return {'ranks': [{'compute_ms_p50': 0}]}
            return {'token': stop_token, 'nll_sum': 0, 'n': 1, 'top1': 1}
    def timed(ids, x, past, last):
        time.sleep(.002)
        steps.append((past, ids.shape[-1]))
        return torch.zeros((1, ids.shape[-1], 1))
    ns = functions({'torch': torch, 'args': NS(nll_file=nll_file, prompt='P', max_new=max_new, ctx=ctx, chunk=chunk),
                    'Tokenizer': NS(from_config=lambda _: Tokenizer()), 'config': None,
                    'model': NS(default_chat_prompt=lambda x:x), 'dev': torch.device('cpu'),
                    'timed': timed, 'down': down, 'up': Up(), 'time': time, 'math': math,
                    'log': lambda *a: logs.append(' '.join(map(str,a))),
                    'compute_summary': lambda: {'compute_ms_p50': 0}})
    return ns['drive'], steps, messages, logs


def test_driver_stops_before_context_overflow():
    drive, steps, messages, logs = driver()
    drive()
    assert steps == [(0,254), (254,1), (255,1)]
    assert all(p+q <= 256 for p,q in steps)
    assert messages[-1]['cmd'] == 'stop'
    assert any('context_limit' in s for s in logs)

@pytest.mark.parametrize('length', [0, 257])
def test_driver_rejects_invalid_prompt_before_forward(length):
    drive, steps, _, _ = driver(prompt_len=length)
    with pytest.raises(ValueError, match='prompt'): drive()
    assert not steps

@pytest.mark.parametrize('length', [1,257])
def test_driver_rejects_invalid_nll_before_forward(tmp_path, length):
    path = tmp_path / 'nll.txt'; path.write_text('NLL')
    drive, steps, _, _ = driver(nll_file=str(path), nll_len=length)
    with pytest.raises(ValueError, match='NLL'): drive()
    assert not steps

@pytest.mark.parametrize('past,qlen', [(256,1), (255,2), (-1,1), (0,0)])
def test_run_slice_checks_capacity_before_model_or_cache(past, qlen):
    def forbidden(*args): raise AssertionError('forward preparation reached')
    ns = functions({'torch': torch, 'model': NS(caps={}, prepare_inputs=forbidden), 'args':NS(ctx=256),
                    'cache':None, 'R':0, 'my_fwd':[]})
    with pytest.raises(ValueError, match='context'): ns['run_slice'](torch.ones((1,qlen)), None, past, True)

@pytest.mark.parametrize('max_new', [0,1,3])
def test_driver_preserves_eos_and_zero_generation(max_new):
    drive, steps, messages, logs = driver(prompt_len=4, max_new=max_new, stop_token=999)
    drive()
    assert len(steps) == (2 if max_new else 0)
    assert messages[-1]['cmd'] == 'stop'
    assert any(('eos' if max_new else 'max_new') in s for s in logs)

@pytest.mark.parametrize('kwargs', [{'ctx':0}, {'max_new':-1}, {'chunk':0}])
def test_driver_rejects_invalid_options(kwargs):
    drive, steps, _, _ = driver(**kwargs)
    with pytest.raises(ValueError): drive()
    assert not steps

@pytest.mark.parametrize('ctx,length,expected', [(1,1,[(0,1)]), (256,256,[(0,255),(255,1)]), (256,254,[(0,253),(253,1),(254,1),(255,1)])])
def test_driver_exact_capacity_and_remaining_steps(ctx,length,expected):
    drive,steps,_,logs = driver(prompt_len=length,ctx=ctx,max_new=10)
    drive()
    assert steps == expected
    assert any('context_limit' in s for s in logs)

def test_nll_exact_context_chunks_are_not_truncated(tmp_path):
    path=tmp_path/'nll.txt'; path.write_text('NLL')
    drive,steps,_,_ = driver(prompt_len=1,max_new=0,nll_file=str(path),nll_len=256,chunk=64)
    drive()
    assert steps == [(0,64),(64,64),(128,64),(192,64)]
