"""CPU/source-AST tests of the installed batched-verification gate."""
import ast,importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace
_arg=Path(sys.argv[1]) if len(sys.argv)>1 else None
SOURCE=_arg.resolve() if _arg is not None and (_arg/'exllamav3/generator/sampler').is_dir() else Path(__file__).resolve().parents[1]
saved=sys.argv[:]
sys.argv=[__file__,str(SOURCE/'exllamav3/generator/sampler')]
spec=importlib.util.spec_from_file_location('requirements_harness',Path(__file__).parent/'cpu_requirements_harness.py')
harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
sys.argv=saved

def gate(sampler,**changes):
 p=SOURCE/'exllamav3/generator/generator.py'
 tests=[n.test for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.If) and '_BATCH_VERIFY' in ast.unparse(n.test) and 'reqs_past_ids' in ast.unparse(n.test)]
 assert len(tests)==1
 job=SimpleNamespace(sequences=[None],filters=[],forced_ids=None,return_probs=False,return_top_tokens=0,new_tokens=0,sampler=sampler,device_logit_mask=None)
 for key,value in changes.items():setattr(job,key,value)
 return eval(compile(ast.Expression(tests[0]),str(p),'eval'),{'_BATCH_VERIFY':True,'draft_tokens':object(),'batch_logits':SimpleNamespace(shape=(1,5,152576)),'job':job})

class BatchGateTests(unittest.TestCase):
 def test_neutral_greedy_is_batch_eligible(self):
  for fused in (False,True):
   with self.subTest(fused=fused):
    n=harness.load_classes(fused)
    s=n['CustomSampler']([n['SS_RepP'](),n['SS_PresFreqP'](),n['SS_Argmax']()])
    self.assertFalse(s.reqs_past_ids)
    self.assertTrue(gate(s),'Neutral greedy rows are independent')

 def test_stateful_adaptive_sampler_is_not_batch_eligible(self):
  for fused in (False,True):
   with self.subTest(fused=fused):
    n=harness.load_classes(fused)
    s=n['CustomSampler']([n['SS_RepP'](),n['SS_PresFreqP'](),n['SS_AdaptiveP'](.5,.9)])
    self.assertFalse(s.reqs_past_ids)
    self.assertFalse(gate(s),'Adaptive-P feedback cannot advance across batch rows')

 def test_unknown_sampler_fails_closed(self):
  self.assertFalse(gate(SimpleNamespace(reqs_past_ids=False)))

 def test_base_sampler_defaults_to_serial(self):
  n=harness.load_classes(False)
  self.assertFalse(n['Sampler']().supports_batch_verify)

 def test_categorical_sampling_stays_serial(self):
  for fused in (False,True):
   n=harness.load_classes(fused)
   for tail in [[n['SS_Sample']()],[n['SS_Temperature'](.8),n['SS_TopK'](40),n['SS_TopP'](.95),n['SS_Sample']()]]:
    with self.subTest(fused=fused,tail=[type(x).__name__ for x in tail]):
     s=n['CustomSampler']([n['SS_RepP'](),n['SS_PresFreqP'](),*tail])
     self.assertFalse(s.reqs_past_ids)
     self.assertFalse(gate(s))

 def test_multinomial_stays_serial_and_keeps_seed_requirement(self):
  for fused in (False,True):
   n=harness.load_classes(fused)
   s=n['CustomSampler']([n['SS_Sample_mn']()])
   self.assertTrue(s.reqs_torch_seed)
   self.assertFalse(gate(s))

 def test_active_penalties_stay_serial(self):
  for fused in (False,True):
   n=harness.load_classes(fused)
   for penalty in [n['SS_RepP'](1.1),n['SS_PresFreqP'](pres_p=.3),n['SS_PresFreqP'](freq_p=.3)]:
    s=n['CustomSampler']([penalty,n['SS_Argmax']()])
    self.assertTrue(s.reqs_past_ids)
    self.assertFalse(gate(s))

 def test_active_dry_stays_serial(self):
  for fused in (False,True):
   n=harness.load_classes(fused);dry=object.__new__(n['SS_DRY'])
   dry.dry_multiplier=.5;dry.dry_base=1.75
   s=n['CustomSampler']([dry,n['SS_Argmax']()])
   self.assertTrue(s.reqs_past_ids)
   self.assertFalse(gate(s))

 def test_custom_argmax_subclass_fails_closed(self):
  for fused in (False,True):
   n=harness.load_classes(fused)
   class Unknown(n['SS_Argmax']):pass
   self.assertFalse(gate(n['CustomSampler']([Unknown()])))

 def test_greedy_with_extra_steps_is_conservatively_serial(self):
  for fused in (False,True):
   n=harness.load_classes(fused)
   s=n['CustomSampler']([n['SS_Normalize'](),n['SS_Argmax']()])
   self.assertFalse(gate(s))

 def test_existing_job_constraints_still_close_greedy_gate(self):
  for fused in (False,True):
   n=harness.load_classes(fused);s=n['CustomSampler']([n['SS_Argmax']()])
   self.assertTrue(gate(s))
   for key,value in [('sequences',[None,None]),('filters',[object()]),('forced_ids',object()),('return_probs',True),('return_top_tokens',1),('new_tokens',-1),('device_logit_mask',object())]:
    with self.subTest(fused=fused,constraint=key):self.assertFalse(gate(s,**{key:value}))

 def test_inserted_preparation_requirements_are_retained(self):
  for fused in (False,True):
   n=harness.load_classes(fused)
   class Needs(n['SS_Base']):
    def reqs_past_ids(self):return True
    def reqs_torch_seed(self):return True
   class Terminal(n['SS_Base']):
    def prep(self,state):return [Needs]
   s=n['CustomSampler']([Terminal()])
   self.assertTrue(s.reqs_past_ids)
   self.assertTrue(s.reqs_torch_seed)
   self.assertFalse(gate(s))

if __name__=='__main__':unittest.main(verbosity=2,argv=[sys.argv[0]])
