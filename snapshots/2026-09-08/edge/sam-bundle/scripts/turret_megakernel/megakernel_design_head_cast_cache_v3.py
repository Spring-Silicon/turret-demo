"""Exact, deliberately scoped cast cache for the retained packed SAM3.1 head.

prepare_cached_head(compiled_head, four_example_inputs, generated_module=None)
returns a callable accepting features, position, text_memory, text_mask. All four
remain dynamic. Construction must be outside graph capture and concurrent head
execution. It temporarily instruments the existing generated call to obtain its
lifted parameters, restoring it in finally. No retained module/source is edited.

Two existing decoder coordinate vectors remain unchanged static arguments; they
are bound to the live decoder coordinate-cache storage and never cast-cached.
The returned object owns its half caches and keeps the original FP32 parameters
alive. It removes only pure original cast launches and changes only GEMM READ
arguments; every original scratch allocation, alias and write remains in place.
No graph output aliases a cache. EngineGraph owns its ordinary output allocations.

Lifetime: immutable inference parameters, same device/dtype/shape/layout and
compiled program. Rebuild this object AND captured graphs after changing/loading
parameters or changing device/layout. validate_cache() is an explicit setup/audit
check, never a per-frame GPU operation. Parameter tensor versions do not detect
unsupported .data writes; callers must invalidate on any parameter update. The
screen additionally hashes all master/cache bytes before and after replay.
"""
import ast
import copy
from collections import Counter
import hashlib
import math
from pathlib import Path
import re
import types


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linear(node):
    """Decode static pointer arithmetic, rejecting every unknown expression."""
    if isinstance(node, ast.Name):return {node.id:1},0
    if isinstance(node, ast.Constant) and isinstance(node.value,int):return {},node.value
    assert isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add),ast.unparse(node)
    a,x=linear(node.left);b,y=linear(node.right)
    return dict(Counter(a)+Counter(b)),x+y


def kernel_catalog(tree):
    kernels={};casts={}
    for node in tree.body:
        if not isinstance(node,ast.Assign) or not isinstance(node.value,ast.Call) or ast.unparse(node.value.func)!='async_compile.triton':continue
        name=node.targets[0].id
        fn=next(x for x in ast.parse(node.value.args[1].value).body if isinstance(x,ast.FunctionDef))
        meta=next(kw.value for dec in fn.decorator_list if isinstance(dec,ast.Call) for kw in dec.keywords if kw.arg=='triton_meta')
        signature=ast.literal_eval(next(v for k,v in zip(meta.keys,meta.values) if k.value=='signature'))
        formals=[v.arg for v in fn.args.args];kernels[name]=formals
        pointers={k:v for k,v in signature.items() if 'ptr' in k}
        if pointers.get('in_ptr0')!='*fp32' or len(pointers)<2 or not all(k.startswith('out_ptr') and v=='*fp16' for k,v in pointers.items() if k!='in_ptr0'):continue
        temps=[v for v in fn.body if isinstance(v,ast.Assign) and isinstance(v.targets[0],ast.Name) and v.targets[0].id.startswith('tmp')]
        if not temps or not all(isinstance(v.value,ast.Call) and (ast.unparse(v.value.func)=='tl.load' or isinstance(v.value.func,ast.Attribute) and v.value.func.attr=='to') for v in temps):continue
        loads=[v for v in temps if ast.unparse(v.value.func)=='tl.load'];assert len(loads)==1
        mask=ast.unparse(loads[0].value.args[1]);assert mask in ('xmask','None')
        coefficients,offset=linear(loads[0].value.args[0]);assert coefficients=={'in_ptr0':1,'x0':1}
        previous=loads[0].targets[0].id
        for temp in temps[1:]:
            assert ast.unparse(temp.value.func.value)==previous
            assert len(temp.value.args)==1 and ast.unparse(temp.value.args[0]) in ('tl.float16','tl.float32')
            previous=temp.targets[0].id
        stores=[v.value for v in fn.body if isinstance(v,ast.Expr) and isinstance(v.value,ast.Call) and ast.unparse(v.value.func)=='tl.store']
        outputs=[v for v in formals if v.startswith('out_ptr')];assert len(stores)==len(outputs)
        for store,out in zip(stores,outputs,strict=True):
            assert linear(store.args[0])==({out:1,'x0':1},0)
            assert ast.unparse(store.args[1])==previous and ast.unparse(store.args[2])==mask
        # No hidden arithmetic statements or auxiliary memory operations.
        assignments={v.targets[0].id:ast.unparse(v.value) for v in fn.body if isinstance(v,ast.Assign)}
        assert assignments['x0']=='xindex' and assignments['xmask']==('xindex < xnumel' if mask=='xmask' else 'tl.full([XBLOCK], True, tl.int1)[:]')
        assert assignments['xoffset']=='tl.program_id(0) * XBLOCK'
        assert assignments['xindex']=='xoffset + tl.arange(0, XBLOCK)[:]'
        assert set(assignments)=={'xnumel','xoffset','xindex','xmask','x0'}|{v.targets[0].id for v in temps}
        assert len(fn.body)==len(assignments)+len(stores)
        casts[name]={'kernel':name,'offset':offset,'length':int(assignments['xnumel']),'outputs':len(outputs),'formals':formals}
    return kernels,casts


def analyze_source(source):
    """CPU-only closed-world alias/version proof and generated-call transform."""
    tree=ast.parse(source);kernels,casts=kernel_catalog(tree)
    fns=[v for v in ast.walk(tree) if isinstance(v,ast.FunctionDef) and v.name=='call'];assert len(fns)==1
    original=copy.deepcopy(fns[0]);fn=copy.deepcopy(original)
    unpack=fn.body[0];assert isinstance(unpack,ast.Assign) and isinstance(unpack.targets[0],ast.Tuple)
    argument_names=[v.id for v in unpack.targets[0].elts];assert argument_names==[f'arg{i}_1' for i in range(360)]
    env={};objects=[];keys={};entries=[];producers=[];reads=[];invalidations=[];aliases=[];allocations=[]
    def new_view(shape=None,stride=None,dtype=None):
        obj={'id':len(objects),'tag':None,'children':{}};objects.append(obj)
        return {'obj':obj,'shape':shape,'stride':stride,'offset':0,'dtype':dtype}
    def view(value):
        if isinstance(value,ast.Name):return env.get(value.id)
        if isinstance(value,ast.Call) and ast.unparse(value.func)=='reinterpret_tensor':
            base=view(value.args[0]);assert base is not None,ast.unparse(value)
            return {**base,'shape':tuple(ast.literal_eval(value.args[1])),'stride':tuple(ast.literal_eval(value.args[2])),
                'offset':base['offset']+ast.literal_eval(value.args[3])}
        if isinstance(value,ast.Subscript):
            base=view(value.value)
            if base is not None:
                index=ast.literal_eval(value.slice)
                if index not in base['obj']['children']:base['obj']['children'][index]=new_view()
                return base['obj']['children'][index]
        return None
    def assert_no_cached(node):
        for item in ast.walk(node):
            if isinstance(item,ast.Name) and item.id in env:assert env[item.id]['obj']['tag'] is None,('unclassified cached read',node.lineno,item.id)
    def clear(value,line):
        item=view(value)
        if item is not None and item['obj']['tag'] is not None:
            invalidations.append({'line':line,'storage':item['obj']['id'],'producer':item['obj']['tag']['producer']})
            item['obj']['tag']=None
    def replace_read(value,line,gemm,index):
        item=view(value)
        if item is None or item['obj']['tag'] is None:return value
        tag=item['obj']['tag'];offset=item['offset']-tag['offset'];shape=item['shape'];stride=item['stride']
        assert shape is not None and stride is not None and len(shape)==len(stride)
        assert item['dtype']=='torch.float16' and min(stride)>=0 and min(shape)>0
        upper=offset+sum((n-1)*s for n,s in zip(shape,stride));assert 0<=offset<=upper<entries[tag['key']]['length'],(line,offset,upper)
        replacement=ast.parse(f"reinterpret_tensor(self._cast_cache[{tag['key']}], {shape!r}, {stride!r}, {offset})",mode='eval').body
        reads.append({'line':line,'gemm':gemm,'operand':index,'producer':tag['producer'],'cache_key':tag['key'],
            'original':ast.unparse(value),'replacement':ast.unparse(replacement),'shape':shape,'stride':stride,'offset':offset,'last_element':upper})
        return ast.copy_location(replacement,value)
    def statements(body):
        result=[]
        for st in body:
            if isinstance(st,ast.With):
                for context in st.items:assert_no_cached(context.context_expr)
                st.body=statements(st.body)
            elif isinstance(st,ast.Assign) and len(st.targets)==1 and isinstance(st.targets[0],ast.Name):
                target=st.targets[0].id;value=st.value;old=view(value)
                if old is not None:
                    env[target]=old;aliases.append({'line':st.lineno,'target':target,'source':ast.unparse(value),'storage':old['obj']['id']})
                elif isinstance(value,ast.Call) and 'empty_strided' in ast.unparse(value.func):
                    assert_no_cached(value)
                    env[target]=new_view(tuple(ast.literal_eval(value.args[0])),tuple(ast.literal_eval(value.args[1])),ast.unparse(value.args[2]));allocations.append(ast.unparse(st))
                elif isinstance(value,ast.Call) and ast.unparse(value.func).startswith('torch.ops.'):
                    assert_no_cached(value);env[target]=new_view()
                else:assert_no_cached(value)
            elif isinstance(st,ast.Delete):
                for target in st.targets:
                    if isinstance(target,ast.Name):env.pop(target.id,None)
            elif isinstance(st,ast.Expr) and isinstance(st.value,ast.Call):
                call=st.value;name=ast.unparse(call.func);kernel=name.removesuffix('.run')
                if kernel in casts:
                    spec=casts[kernel];src=ast.unparse(call.args[0]);match=re.fullmatch(r'arg(\d+)_1',src);assert match and int(match[1])>=4
                    assert len(call.args)==2+spec['outputs'] and ast.literal_eval(call.args[-1])==spec['length']
                    key=(kernel,int(match[1]))
                    if key not in keys:
                        keys[key]=len(entries);entries.append({**spec,'argument':int(match[1]),'key':len(entries)})
                    producer=len(producers);producer_row={'producer':producer,'line':st.lineno,'cache_key':keys[key],'source':src,'outputs':[]}
                    for value in call.args[1:-1]:
                        item=view(value);assert item is not None and item['dtype']=='torch.float16'
                        assert math.prod(item['shape'])==spec['length'] and sum((n-1)*s for n,s in zip(item['shape'],item['stride']))+1==spec['length']
                        # The kernel writes [0,length) from this tensor's data pointer.
                        item['obj']['tag']={'producer':producer,'key':keys[key],'offset':item['offset']}
                        producer_row['outputs'].append({'expression':ast.unparse(value),'storage':item['obj']['id'],'offset':item['offset']})
                    producers.append(producer_row);continue
                elif kernel in kernels:
                    for formal,value in zip(kernels[kernel],call.args):
                        if formal.startswith('in'):assert_no_cached(value)
                    for formal,value in zip(kernels[kernel],call.args):
                        if formal.startswith(('out_ptr','in_out')):clear(value,st.lineno)
                elif name in ('extern_kernels.mm','extern_kernels.addmm','extern_kernels.bmm','extern_kernels.baddbmm'):
                    call.args=[replace_read(value,st.lineno,name,i) for i,value in enumerate(call.args)]
                    for kw in call.keywords:
                        if kw.arg=='out':clear(kw.value,st.lineno)
                        else:assert_no_cached(kw.value)
                elif name=='assert_tensor_metadata':
                    item=view(call.args[0]);assert item is not None
                    assert_no_cached(call)
                    item.update(shape=tuple(ast.literal_eval(call.args[1])),stride=tuple(ast.literal_eval(call.args[2])),dtype=ast.unparse(call.args[3]))
                else:assert_no_cached(call)
            elif isinstance(st,(ast.Return,ast.Assign)):assert_no_cached(st)
            else:raise AssertionError(('unsupported statement',st.lineno,ast.unparse(st)))
            result.append(st)
        return result
    fn.body=statements(fn.body);ast.fix_missing_locations(fn)
    assert set(v['producer'] for v in reads)==set(range(len(producers)))
    assert len(producers)==293 and len(reads)==308 and len(entries)==243
    def counts(node):return Counter(ast.unparse(v.func) for v in ast.walk(node) if isinstance(v,ast.Call))
    before=counts(original);after=counts(fn)
    assert sum(v for k,v in before.items() if k.startswith('triton_') and k.endswith('.run'))==563
    assert sum(v for k,v in after.items() if k.startswith('triton_') and k.endswith('.run'))==270
    for name,count in before.items():
        if name not in {v['kernel']+'.run' for v in entries}|{'reinterpret_tensor'}:assert after[name]==count,(name,count,after[name])
    # Structural diff permits only removed pure cast calls and the enumerated
    # GEMM reads. This proves allocations, aliases, out= and other calls unchanged.
    restored=copy.deepcopy(fn);byline={v['line']:v for v in producers};readmap={(v['line'],v['operand']):v for v in reads}
    for call in [v for v in ast.walk(restored) if isinstance(v,ast.Call) and ast.unparse(v.func).startswith('extern_kernels.')]:
        for i in range(len(call.args)):
            if (call.lineno,i) in readmap:call.args[i]=ast.parse(readmap[call.lineno,i]['original'],mode='eval').body
    expected=copy.deepcopy(original)
    def strip(body):
        out=[]
        for st in body:
            if st.lineno in byline:continue
            if isinstance(st,ast.With):st.body=strip(st.body)
            out.append(st)
        return out
    expected.body=strip(expected.body)
    assert ast.dump(restored,include_attributes=False)==ast.dump(expected,include_attributes=False)
    transformed=ast.unparse(fn)+'\n';compile(transformed,'<retained_head_cast_cache>','exec')
    report={'kind':'retained_packed_head_cast_only_static_proof','original_triton_launches':563,'cached_triton_launches':270,
        'gemm_calls':sum(before[k] for k in before if k.startswith('extern_kernels.')),'removed_cast_launches':len(producers),
        'rewritten_gemm_read_operands':len(reads),'unique_casts':len(entries),'unique_parameter_inputs':len({v['argument'] for v in entries}),
        'cache_bytes':sum(v['length']*2 for v in entries),'logical_source_bytes_removed':sum(entries[v['cache_key']]['length']*4 for v in producers),
        'logical_store_bytes_removed':sum(entries[v['cache_key']]['length']*2*len(v['outputs']) for v in producers),
        'entries':entries,'producers':producers,'reads':reads,'subsequent_write_invalidations':invalidations,'aliases_tracked':aliases,
        'preserved_scratch_allocations':len(allocations),'exact_structural_diff_passed':True,
        'scope':'Only GEMM positional reads can reference owned caches. All original allocations, aliases, returns, Triton noncast calls, GEMM writes and scalar arguments preserved. Logical traffic is not measured DRAM traffic.'}
    return report,transformed


def _capture_call(self,args):
    self._cast_cache_captured_args=tuple(args)
    return self._cast_cache_original_call(args)


def tensor_signature(value):
    try:version=value._version
    except RuntimeError:version=None
    return (value.data_ptr(),value.untyped_storage().data_ptr(),str(value.device),str(value.dtype),tuple(value.shape),tuple(value.stride()),value.storage_offset(),version)


class CachedHead:
    def __call__(self,features,position,text_memory,text_mask):
        values=(features,position,text_memory,text_mask)
        assert [(str(v.device),str(v.dtype),tuple(v.shape),tuple(v.stride())) for v in values]==self._dynamic_metadata
        flat=list(self._flat_parameters)
        flat[:4]=[text_mask,features,text_memory,position]
        return self._call(flat)

    def validate_cache(self):
        assert [tensor_signature(v) for v in self._flat_parameters[4:]]==self._parameter_signatures,'Lifted storage changed; rebuild cache and graphs'
        current=dict(self._retained.named_parameters())
        assert {name:tensor_signature(current[name]) for name in self._master_signatures}==self._master_signatures,'Named master changed'
        modules=dict(self._retained.named_modules())
        for name,index,expected in self._coordinate_signatures:
            module=modules[name]
            assert tuple(module.compilable_stored_size)==(72,72)
            assert tensor_signature(module.compilable_cord_cache[index])==expected,'Existing coordinate cache changed'
        assert [tensor_signature(v) for v in self.cache_tensors]==self._cache_signatures,'Cached storage/version changed'
        return {'master_signatures_unchanged':True,'cache_signatures_unchanged':True}


def prepare_cached_head(retained_callable,example_inputs,generated_module=None):
    """Construct outside capture; see module docstring for invalidation rules."""
    import torch
    if generated_module is None:
        from torch._inductor.codecache import PyCodeCache
        retained_callable(*example_inputs)
        candidates=[]
        for module in PyCodeCache.modules:
            if not hasattr(module,'runner') or not hasattr(module.runner,'call'):continue
            source=Path(module.__file__).read_text()
            if 'torch.ops.sam31_megakernel.packed_decoder_attention.default' in source and 'arg359_1' in source:candidates.append(module)
        assert len(candidates)==1,('Pass actual retained generated_module explicitly',len(candidates))
        generated_module=candidates[0]
    module=generated_module;source=Path(module.__file__).read_text();report,transformed=analyze_source(source)
    runner=module.runner;func=runner.call.__func__;assert not func.__code__.co_freevars and not func.__closure__
    original_code=func.__code__;original=types.FunctionType(original_code,func.__globals__,func.__name__,func.__defaults__)
    assert not hasattr(runner,'_cast_cache_original_call')
    runner._cast_cache_original_call=types.MethodType(original,runner)
    try:
        func.__code__=_capture_call.__code__
        retained_callable(*example_inputs)
        assert hasattr(runner,'_cast_cache_captured_args'),'Selected generated module was not called'
        flat=runner._cast_cache_captured_args
    finally:
        func.__code__=original_code
        del runner._cast_cache_original_call
        if hasattr(runner,'_cast_cache_captured_args'):del runner._cast_cache_captured_args
    assert len(flat)==360 and len(example_inputs)==4
    for captured,provided in zip(flat[:4],[example_inputs[i] for i in (3,0,2,1)],strict=True):assert tensor_signature(captured)==tensor_signature(provided)
    assert all(isinstance(v,torch.Tensor) and v.dtype==torch.float32 for v in flat[4:])
    # This is a fixed inference module: every static lifted tensor must actually
    # be one of its FP32 parameters, not an accidentally captured dynamic tensor.
    assert hasattr(retained_callable,'named_parameters')
    named_masters=list(retained_callable.named_parameters())
    master_storage={tensor_signature(v)[:-1]:name for name,v in named_masters}
    assert all(tensor_signature(v)[:-1] not in master_storage for v in flat[:4])
    coordinates={}
    for module_name,submodule in retained_callable.named_modules():
        values=getattr(submodule,'compilable_cord_cache',None)
        if values is not None:
            assert tuple(submodule.compilable_stored_size)==(72,72) and len(values)==2
            for axis,value in enumerate(values):
                assert value.shape==(72,) and value.dtype==torch.float32
                coordinates[tensor_signature(value)[:-1]]=(module_name,axis)
    matched=[];coordinate_matches=[]
    for index,value in enumerate(flat[4:],4):
        signature=tensor_signature(value)[:-1]
        if index in (180,181):
            assert signature in coordinates and signature not in master_storage,('Unbound retained coordinates',index,signature)
            assert index not in {entry['argument'] for entry in report['entries']}
            module_name,axis=coordinates[signature]
            coordinate_matches.append({'argument':index,'module_name':module_name,'axis':axis,
                'origin':'Existing decoder.compilable_cord_cache from _get_coords(72,72,device), unchanged by cast caching'})
            continue
        assert signature in master_storage,('Lifted tensor is not an exact FP32 master view',index,signature)
        matched.append({'argument':index,'master_name':master_storage[signature],
            'same_python_object':any(value is master for _,master in named_masters)})
    assert len(matched)==354 and len(coordinate_matches)==2
    assert {entry['argument'] for entry in report['entries']} <= {entry['argument'] for entry in matched}
    report['lifted_master_storage_matches']=matched
    report['existing_coordinate_storage_matches']=coordinate_matches
    device=example_inputs[0].device;assert device.type=='xpu' and all(v.device==device for v in flat)
    result=CachedHead();result._flat_parameters=(None,None,None,None,*flat[4:]);result.cache_tensors=[]
    result.coordinate_tensors=tuple(flat[row['argument']] for row in coordinate_matches)
    result._dynamic_metadata=[(str(v.device),str(v.dtype),tuple(v.shape),tuple(v.stride())) for v in example_inputs]
    report['cache_construction_checks']=[]
    with torch.xpu.device(device):
        stream=module.get_raw_stream(device.index)
        for spec in report['entries']:
            parameter=flat[spec['argument']]
            assert parameter.is_contiguous() and spec['offset']+spec['length']<=parameter.numel()
            outputs=[torch.empty((spec['length'],),device=device,dtype=torch.float16) for _ in range(spec['outputs'])]
            getattr(module,spec['kernel']).run(parameter,*outputs,spec['length'],stream=stream)
            equal=all(torch.equal(outputs[0],v) for v in outputs[1:]);assert equal
            # Independent pointwise conversion comparison; no geometry folding.
            eager=parameter.reshape(-1)[spec['offset']:spec['offset']+spec['length']].to(torch.float16)
            assert torch.equal(outputs[0],eager)
            result.cache_tensors.append(outputs[0]);report['cache_construction_checks'].append({'key':spec['key'],'fanout_exact':equal,'pointwise_half_exact':True})
    result.cache_tensors=tuple(result.cache_tensors)
    namespace={};exec(compile(transformed,str(Path(__file__).with_name('megakernel_design_head_cast_cache_generated.py')),'exec'),module.__dict__,namespace)
    new_runner=types.SimpleNamespace(partitions=runner.partitions,_cast_cache=result.cache_tensors)
    result._call=types.MethodType(namespace['call'],new_runner);result._module=module;result._retained=retained_callable
    result._parameter_signatures=[tensor_signature(v) for v in result._flat_parameters[4:]]
    result._master_signatures={row['master_name']:tensor_signature(flat[row['argument']]) for row in matched}
    result._coordinate_signatures=[(row['module_name'],row['axis'],tensor_signature(flat[row['argument']])) for row in coordinate_matches]
    result._cache_signatures=[tensor_signature(v) for v in result.cache_tensors]
    result.transformed_source=transformed;result.report={**report,'original_source_path':str(module.__file__),'original_source_sha256':digest(module.__file__),
        'transformed_source_sha256':hashlib.sha256(transformed.encode()).hexdigest(),'lifted_parameter_count':354,'existing_coordinate_count':2,
        'dynamic_argument_mapping':{'features':1,'position':3,'text_memory':2,'text_mask':0},'device':str(device)}
    result.validate_cache()
    return result
