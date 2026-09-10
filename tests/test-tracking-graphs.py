"""CPU contract tests for replay ownership; actual GPU qualification is separate."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spring_turret.sam31_tracking_graph import TrackingGraphStage, tracking_graph_status
from spring_turret.tracking_graph_cache import install_nonblocking_cache, finish_cache_warmup
from spring_turret.sam31_tracking_regions import PropagationRegion, mux_inputs, pack_features, tensor_indexed_memory


@dataclass(frozen=True)
class Device:
    type: str


class Tensor:
    def __init__(self, value, kind='cuda'):
        self.value, self.device = value, Device(kind)
        self.shape, self.dtype = (1,), 'float32'
    def stride(self): return (1,)
    def is_floating_point(self): return False
    def clone(self): return Tensor(self.value, self.device.type)
    def copy_(self, other): self.value = other.value


class Tree:
    @staticmethod
    def tree_flatten(value):
        leaves = []
        def walk(v):
            if type(v) in (tuple, list): return (type(v), tuple(walk(x) for x in v))
            if type(v) is dict: return (dict, tuple((k, walk(x)) for k,x in v.items()))
            leaves.append(v)
            return None
        return leaves, walk(value)
    @staticmethod
    def tree_map_only(kind, fn, v):
        if isinstance(v, kind): return fn(v)
        if type(v) in (tuple, list): return type(v)(Tree.tree_map_only(kind, fn, x) for x in v)
        if type(v) is dict: return {k:Tree.tree_map_only(kind, fn, x) for k,x in v.items()}
        return v


class Values(list):
    def __getitem__(self, key):
        if isinstance(key, tuple): return self
        if isinstance(key, list): return Values(list.__getitem__(self,i) for i in key)
        return list.__getitem__(self,key)
    def __setitem__(self, key, value):
        if isinstance(key, list):
            for j,i in enumerate(key): list.__setitem__(self,i,value[j] if isinstance(value,list) else value)
        else: list.__setitem__(self,key,value)


class MemoryFixture:
    add_object_conditional_embeddings = condition_as_mask_input = True
    condition_as_mask_input_fg = 1
    def encode(self, *, conditioning_objects=None, multiplex_state):
        if self.add_object_conditional_embeddings or self.condition_as_mask_input:
            if conditioning_objects is None:
                conditioning_objects = []
                unconditioning_objects = sorted(list(multiplex_state.get_all_valid_object_idx()))
            else:
                conditioning_objects = sorted(list(conditioning_objects))
                unconditioning_objects = sorted(i for i in multiplex_state.get_all_valid_object_idx() if i not in conditioning_objects)
        cond_values = Values([0,0,0])
        if len(conditioning_objects) > 0:
            cond_values[conditioning_objects] = self.condition_as_mask_input_fg
        obj_merged_embed = Values([10,11,12])
        obj_non_cond_embed = Values([20,21,22])
        obj_merged_embed[unconditioning_objects] = obj_non_cond_embed[unconditioning_objects]
        return cond_values,obj_merged_embed


class TrackingGraphTests(unittest.TestCase):
    def test_global_warmup_closes_even_unused_stages(self):
        stage, _ = self.stage('xpu')
        stage.torch.compiler = SimpleNamespace(set_stance=lambda value:nullcontext())
        install_nonblocking_cache({'additional_prompt_detection':stage})
        finish_cache_warmup({'additional_prompt_detection':stage})
        self.assertEqual(stage(Tensor(3,'xpu'))['memory'].value,6)
        self.assertEqual(stage.captures,0)
        self.assertEqual(stage.direct_calls,1)

    def test_temporal_capture_waits_for_normal_memory_fill(self):
        stage, _ = self.stage('xpu')
        stage.torch.compiler = SimpleNamespace(set_stance=lambda value:nullcontext())
        install_nonblocking_cache({'memory_attention_and_mask':stage})
        for index in range(8):
            self.assertEqual(stage(Tensor(index,'xpu'),mode=index)['memory'].value,2*index)
            self.assertEqual(stage.captures,0)
        stage(Tensor(8,'xpu'),mode=7)
        self.assertEqual(stage.captures,0)
        stage(Tensor(9,'xpu'),mode=7)
        self.assertEqual(stage.captures,1)
        stage(Tensor(10,'xpu'),mode=7)
        self.assertEqual(stage.replay_calls,1)
        self.assertEqual(stage.direct_calls,9)
        stage.calls = 64
        for _ in range(4): stage(Tensor(11,'xpu'),mode='new')
        self.assertEqual(stage.captures,1)
        self.assertFalse(stage.warmup_signatures)

    def test_nonblocking_misses_preserve_graphs_and_owned_outputs(self):
        stage, _ = self.stage('xpu')
        stances = []
        @contextmanager
        def stance(value):
            stances.append(value)
            yield
        stage.torch.compiler = SimpleNamespace(set_stance=stance)
        install_nonblocking_cache({'memory':stage})
        first = stage(Tensor(2,'xpu'), mode=False)
        for mode in (True,None,3,4,5,6):
            output = stage(Tensor(3,'xpu'), mode=mode)
            self.assertEqual(output['memory'].value,6)
            self.assertEqual(output['flag'],mode)
            output['memory'].value = -100
        last = stage(Tensor(7,'xpu'), mode=False)
        self.assertEqual(first['memory'].value,4)
        self.assertEqual(last['memory'].value,14)
        self.assertEqual((stage.calls,stage.captures,len(stage.variants)),(8,1,1))
        self.assertEqual((stage.direct_calls,stage.replay_calls),(6,1))
        self.assertEqual(stances,['eager_on_recompile']*6)
        install_nonblocking_cache({'memory':stage})
        self.assertEqual(stage.direct_calls,6) # Idempotent, preserves telemetry.
        with self.assertRaises(ValueError): stage(Tensor(1,'cpu'))
        with self.assertRaises(TypeError): stage(Tensor(1,'xpu'),mode=object())
        stage.compiled = lambda *args,**kw: (_ for _ in ()).throw(RuntimeError('real failure'))
        with self.assertRaisesRegex(RuntimeError,'real failure'):
            stage(Tensor(1,'xpu'),mode='new')

    def stage(self, kind, *, corrupt=False, composed=False, **cache):
        captures, compiling = [], []
        class Graph:
            def replay(self):
                self.outputs['memory'].value = self.inputs[0].value * 2 + int(corrupt)
        @contextmanager
        def graph_context(graph, stream):
            captures.append(graph)
            try: yield
            finally: captures.pop()
        def compiled(x, *, mode=False):
            result = {'memory':Tensor(x.value * 2, kind), 'flag':mode}
            if captures:
                captures[-1].inputs, captures[-1].outputs = (x,), result
            return result
        runtime = SimpleNamespace(Stream=object, synchronize=lambda:None,
            stream=lambda s:nullcontext(), graph=graph_context,
            **{('CUDAGraph' if kind=='cuda' else 'XPUGraph'):Graph})
        def compile(fn, **kwargs):
            compiling.append(kwargs)
            return compiled
        def close(a,b,**kwargs):
            self.assertEqual(a.value if isinstance(a,Tensor) else a,
                             b.value if isinstance(b,Tensor) else b)
        torch = SimpleNamespace(Tensor=Tensor, compile=compile, **{kind:runtime},
            compiler=SimpleNamespace(set_stance=lambda value:nullcontext()),
            testing=SimpleNamespace(assert_close=close))
        with patch.dict(sys.modules, {'torch.utils':SimpleNamespace(_pytree=Tree)}):
            stage = TrackingGraphStage(torch, compiled, 'memory', kind, composed=composed, **cache)
        return stage, compiling

    def test_replay_never_overwrites_retained_temporal_outputs(self):
        for kind in ('cuda','xpu'):
            stage, compiling = self.stage(kind)
            source = Tensor(3, kind)
            first = stage(source)
            second = stage(Tensor(7, kind))
            self.assertEqual(first['memory'].value, 6)
            self.assertEqual(second['memory'].value, 14)
            second['memory'].value = 1000  # Caller mutations cannot poison replay.
            third = stage(Tensor(8, kind))
            self.assertEqual(third['memory'].value,16)
            self.assertEqual(source.value,3)
            self.assertEqual(stage.captures,1)
            self.assertEqual(stage.calls,3)
            self.assertTrue(compiling[0]['fullgraph'])
            self.assertFalse(compiling[0]['dynamic'])
            self.assertTrue(compiling[0]['options']['emulate_precision_casts'])
            if kind=='cuda': self.assertFalse(compiling[0]['options']['triton.cudagraphs'])

    def test_exact_signature_specializes_constants_and_bounds_graph_cache(self):
        stage,_ = self.stage('cuda')
        stage(Tensor(1), mode=False)
        stage(Tensor(1), mode=True)
        stage(Tensor(1), mode=None)
        self.assertEqual(len(stage.variants),2)
        self.assertEqual(stage.captures,3)
        stage(Tensor(1), mode=False)
        self.assertEqual(len(stage.variants),2)
        self.assertEqual(stage.captures,4)

    def test_failed_capture_is_not_reported_as_successful(self):
        stage,_ = self.stage('cuda', corrupt=True)
        with self.assertRaises(AssertionError): stage(Tensor(1))
        self.assertFalse(stage.variants)
        self.assertEqual(stage.calls,0)
        self.assertEqual(stage.captures,0)

    def test_repeated_temporal_shapes_stop_recapturing(self):
        stage, _ = self.stage('cuda', max_variants=4, cache_policy='retain', capture_repetitions=3)
        for i in range(8): stage(Tensor(i), mode='startup-' + str(i))
        self.assertEqual(stage.captures, 0)  # Do not capture growing startup memories.
        for i in range(60):
            result = stage(Tensor(i), mode=i % 3)
            self.assertEqual(result['memory'].value, 2*i)
            result['memory'].value = -1
        self.assertEqual(stage.captures, 3)
        self.assertEqual(stage.evictions, 0)
        self.assertEqual(stage.replay_calls, 51)
        self.assertEqual(stage.direct_calls, 14)
        self.assertEqual(stage.calls, 68)

    def test_full_retained_cache_runs_exact_uncached_operation_without_recapture(self):
        stage, _ = self.stage('cuda', max_variants=2, cache_policy='retain', capture_repetitions=2)
        stage.torch.compiler.set_stance = lambda value: (_ for _ in ()).throw(AssertionError('No eager fallback'))
        for mode in (0,1):
            stage(Tensor(1), mode=mode)
            stage(Tensor(2), mode=mode)
        first = stage(Tensor(4), mode=0)
        for mode in range(2,200):
            result = stage(Tensor(mode), mode=mode)
            self.assertEqual(result['memory'].value, 2*mode)
            self.assertEqual(result['flag'], mode)
        self.assertEqual(stage(Tensor(8), mode=0)['memory'].value, 16)
        self.assertEqual(first['memory'].value, 8)
        self.assertEqual((stage.captures,stage.evictions,len(stage.variants)), (2,0,2))
        self.assertFalse(stage.admission_counts)

    def test_temporal_admission_metadata_is_bounded_and_failed_direct_call_propagates(self):
        stage, _ = self.stage('cuda', cache_policy='retain', capture_repetitions=3)
        for i in range(200): stage(Tensor(i), mode=i)
        self.assertEqual(stage.captures, 0)
        self.assertEqual(len(stage.admission_counts), 128)
        stage.compiled = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('real error'))
        with self.assertRaisesRegex(RuntimeError, 'real error'): stage(Tensor(1), mode='failure')
        self.assertEqual(stage.calls, 200)

    def test_no_cpu_tensors_or_unknown_python_objects_hidden_in_capture(self):
        stage,_ = self.stage('xpu')
        with self.assertRaises(ValueError): stage(Tensor(1,'cpu'))
        with self.assertRaises(TypeError): stage(Tensor(1,'xpu'), mode=object())

    def test_flags_report_observed_replay_not_just_installed_wrappers(self):
        stage,_ = self.stage('xpu')
        self.assertFalse(tracking_graph_status({'memory':stage}, 'xpu')['torch_compile'])
        stage(Tensor(1,'xpu'))
        status = tracking_graph_status({'memory':stage}, 'xpu')
        self.assertTrue(status['torch_compile'])
        self.assertTrue(status['sycl_graph'])
        self.assertFalse(status['cuda_graph'])
        self.assertEqual(status['compilation_scope'], 'tensor-stages')
        self.assertEqual(status['graph_stages']['memory']['calls'],1)

    def test_invoke_does_not_reuse_lists_mutated_by_sam_encoder(self):
        stage,_ = self.stage('cuda')
        original = Tensor(4)
        inputs = (([original],), {})
        def replace_views(src):
            self.assertIs(src[0],original)
            src[0] = Tensor(100)
            return src[0]
        stage.compiled = replace_views
        stage.invoke(inputs)
        stage.invoke(inputs)
        self.assertIs(inputs[0][0][0],original)

    def test_composed_region_does_not_recompile_its_compiled_components(self):
        stage, compiling = self.stage('xpu', composed=True)
        self.assertEqual(compiling, [])
        stage(Tensor(2,'xpu'))
        stage(Tensor(3,'xpu'))
        self.assertEqual(stage.captures,1)
        self.assertEqual(tracking_graph_status({'region':stage}, 'xpu')['compilation_scope'], 'tensor-regions')

    def test_output_aliases_owned_once_and_not_borrowed_from_replay(self):
        stage,_ = self.stage('cuda')
        original = Tensor(2)
        copies = stage.clone({'a':original, 'b':original})
        self.assertIs(copies['a'], copies['b'])
        self.assertIsNot(copies['a'], original)

    def test_nested_tensor_storage_is_explicit_not_hidden_from_cloning(self):
        a, b = Tensor(1), Tensor(2)
        backbone = {'vision_mask':None, 'backbone_fpn':[SimpleNamespace(tensors=a, mask=None)],
                    'vision_pos_enc':[b]}
        stage,_ = self.stage('cuda')
        packed = stage.clone(pack_features(backbone))
        self.assertIsNot(packed[0][0], a)
        self.assertIsNot(packed[1][0], b)
        backbone['backbone_fpn'][0].mask = object()
        with self.assertRaises(AssertionError): pack_features(backbone)

    def propagation(self, fail=False):
        encoder = SimpleNamespace(forward=lambda **kw: {'memory':'real-result'})
        def prepare(**kw):
            result = encoder.forward(src=Tensor(1))['memory']
            if fail: raise ValueError('prepare failed')
            return result
        tracker = SimpleNamespace(transformer=SimpleNamespace(encoder=encoder),
            _prepare_memory_conditioned_features=prepare,
            _forward_sam_heads=lambda *a, **kw:'interactive-result')
        calls = []
        def region(name, fn, backend):
            def call(*args):
                calls.append((name, args))
                return 'captured-result'
            return call
        return PropagationRegion(tracker, region), calls

    def test_deferred_attention_requires_exact_placeholder_and_is_consumed_once(self):
        p, calls = self.propagation()
        state = SimpleNamespace(mux_matrix=Tensor(1), demux_matrix=Tensor(1),
            num_buckets=1, multiplex_count=16, get_all_valid_object_idx=lambda:{0})
        placeholder = p.prepare_memory()
        with self.assertRaises(AssertionError): p.prepare_memory()
        self.assertEqual(p.heads(placeholder, multiplex_state=state), 'captured-result')
        self.assertIsNone(p.pending)
        self.assertEqual(calls[0][0], 'memory_attention_and_mask')
        self.assertEqual(calls[0][1][3], mux_inputs(state))
        p.prepare_memory()
        with self.assertRaises(AssertionError): p.heads(Tensor(1), multiplex_state=state)

    def test_failed_memory_preparation_restores_real_encoder(self):
        p,_ = self.propagation(fail=True)
        with self.assertRaises(ValueError): p.prepare_memory()
        self.assertIs(p.tracker.transformer.encoder.forward, p.encoder)
        self.assertIsNone(p.pending)

    def test_interactive_path_stays_native_and_cannot_consume_deferred_attention(self):
        p,calls = self.propagation()
        self.assertEqual(p.heads(Tensor(1), multiplex_state=object(), point_inputs={}), 'interactive-result')
        self.assertFalse(calls)
        placeholder = p.prepare_memory()
        with self.assertRaises(AssertionError):
            p.heads(placeholder, multiplex_state=object(), point_inputs={})

    def test_fixed_shape_conditioning_mask_preserves_all_assignment_cases(self):
        original = MemoryFixture().encode
        fake_torch = SimpleNamespace(
            full_like=lambda values,v:Values([v]*len(values)),
            where=lambda c,a,b:Values(x if keep else y for keep,x,y in zip(c,a,b)))
        with patch.dict(original.__func__.__globals__, {'torch':fake_torch}):
            adapted = tensor_indexed_memory(original)
        state = SimpleNamespace(get_all_valid_object_idx=lambda:{0,1,2})
        for conditions in (None,[],[0],[1,2],[0,1,2]):
            expected = original(conditioning_objects=conditions,multiplex_state=state)
            actual = adapted(conditioning_objects=None,multiplex_state=state,
                region_condition_mask=Values(i in (conditions or []) for i in range(3)))
            self.assertEqual(expected,actual)
        with self.assertRaises(AssertionError): tensor_indexed_memory(self.test_flags_report_observed_replay_not_just_installed_wrappers)


if __name__=='__main__': unittest.main()
