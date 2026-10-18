"""CPU-only constructor/requirement regressions from exact source AST.
Usage: python test_sampler_requirements.py PATH/TO/exllamav3/generator/sampler
Does not import torch/exllamav3 or execute sampling kernels.
"""
import ast
import copy
import math
import sys
import unittest
from enum import Enum
from pathlib import Path
ROOT=Path(sys.argv.pop(1)) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'exllamav3/generator/sampler'

def load_classes(fused):
    ns={'Enum':Enum,'math':math,'fused_sampler_enable':fused}
    for name in ('sampler.py','custom.py'):
        nodes=[]
        for node in ast.parse((ROOT/name).read_text(encoding='utf-8')).body:
            if isinstance(node,ast.ClassDef) and (node.name in ('Sampler','CustomSampler','SS') or node.name.startswith('SS_')):
                node=copy.deepcopy(node)
                if node.name!='SS':
                    node.body=[m for m in node.body if isinstance(m,(ast.Assign,ast.AnnAssign)) or
                               isinstance(m,ast.FunctionDef) and m.name in ('__init__','alt','prep','reqs_past_ids','reqs_torch_seed')]
                    if not node.body:node.body=[ast.Pass()]
                nodes.append(node)
            elif isinstance(node,ast.FunctionDef) and node.name in ('_match_fused_tail','clamp','conditional'):
                nodes.append(copy.deepcopy(node))
        tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree),str(ROOT/name),'exec'),ns)
    return ns

class RequirementsTests(unittest.TestCase):
    def check(self,factory,want_history,want_seed=False):
        for fused in (False,True):
            with self.subTest(fused=fused):
                ns=load_classes(fused)
                sampler=ns['CustomSampler'](factory(ns))
                self.assertEqual(sampler.reqs_past_ids,want_history)
                self.assertEqual(sampler.reqs_torch_seed,want_seed)
                self.assertFalse(any(isinstance(s,ns['SS_NoOp']) for s in sampler.steps))

    def test_neutral_penalties_do_not_require_history(self):
        self.check(lambda n:[n['SS_RepP'](),n['SS_PresFreqP'](),n['SS_Argmax']()],False)

    def test_active_repetition_retains_history(self):
        self.check(lambda n:[n['SS_RepP'](1.1),n['SS_Argmax']()],True)

    def test_active_presence_retains_history(self):
        self.check(lambda n:[n['SS_PresFreqP'](pres_p=.3),n['SS_Argmax']()],True)

    def test_active_frequency_retains_history(self):
        self.check(lambda n:[n['SS_PresFreqP'](freq_p=.3),n['SS_Argmax']()],True)

    def test_empty_penalty_range_drops_history(self):
        self.check(lambda n:[n['SS_RepP'](1.1,sustain_range=0),n['SS_PresFreqP'](pres_p=.3,sustain_range=0),n['SS_Argmax']()],False)

    def test_multinomial_retains_seed(self):
        self.check(lambda n:[n['SS_Sample_mn']()],False,True)

    def test_categorical_does_not_require_torch_seed(self):
        self.check(lambda n:[n['SS_Sample']()],False,False)

    def test_dry_requirement_matches_enabled_state(self):
        # DRY construction uses torch only for float32 rounding. These cases
        # isolate its real alt/requirement methods using explicit constructor state.
        for multiplier,base,enabled in [(0.,1.75,False),(.5,.5,False),(.5,1.75,True)]:
            with self.subTest(multiplier=multiplier,base=base):
                def factory(n):
                    dry=object.__new__(n['SS_DRY']);dry.dry_multiplier=multiplier;dry.dry_base=base
                    return [dry,n['SS_Argmax']()]
                self.check(factory,enabled)

    def test_replacement_step_requirements_win(self):
        def factory(n):
            class Replaced(n['SS_Base']):
                def alt(self):return n['SS_Sample_mn']()
            return [Replaced()]
        self.check(factory,False,True)

    def test_removed_step_does_not_force_seed(self):
        def factory(n):
            class Removed(n['SS_Base']):
                def reqs_torch_seed(self):return True
                def alt(self):return n['SS_NoOp']()
            return [Removed(),n['SS_Argmax']()]
        self.check(factory,False,False)

    def test_actual_generator_gate_opens_only_for_neutral_sampler(self):
        from types import SimpleNamespace
        path=Path(__file__).resolve().parents[1]/'exllamav3/generator/generator.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        tests=[n.test for n in ast.walk(tree) if isinstance(n,ast.If)
               and '_BATCH_VERIFY' in ast.unparse(n.test) and 'reqs_past_ids' in ast.unparse(n.test)]
        self.assertEqual(len(tests),1)
        gate=compile(ast.Expression(tests[0]),str(path),'eval')
        for fused in (False,True):
            for rep,expected in ((1.0,True),(1.1,False)):
                with self.subTest(fused=fused,rep=rep):
                    n=load_classes(fused)
                    sampler=n['CustomSampler']([n['SS_RepP'](rep),n['SS_PresFreqP'](),n['SS_Argmax']()])
                    job=SimpleNamespace(sequences=[None],filters=[],forced_ids=None,return_probs=False,
                                        return_top_tokens=0,new_tokens=0,sampler=sampler,device_logit_mask=None)
                    self.assertEqual(eval(gate,{'_BATCH_VERIFY':True,'draft_tokens':object(),
                                     'batch_logits':SimpleNamespace(shape=(1,8,16)),'job':job}),expected)

if __name__=='__main__':unittest.main(verbosity=2)
