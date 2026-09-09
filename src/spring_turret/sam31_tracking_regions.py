"""Larger SAM 3.1 replay regions; never capture mutable tracking sessions.

The stock CPU NMS, association, conditioning-frame selection and ID management
remain between regions. Region outputs own their storage. Each replay receives
current memory and multiplex matrices, not a captured Python session snapshot.
"""
from types import SimpleNamespace

try:
    from .sam31_tracking_graph import TrackingGraphStage
except ImportError:
    from sam31_tracking_graph import TrackingGraphStage


class TensorMultiplex:
    """Immutable tensor-only view, with exactly upstream's mux/demux arithmetic."""
    def __init__(self, mux_matrix, demux_matrix, num_buckets, multiplex_count, valid_indices):
        self.mux_matrix, self.demux_matrix = mux_matrix, demux_matrix
        self.num_buckets, self.multiplex_count = num_buckets, multiplex_count
        self.total_valid_entries = demux_matrix.shape[0]
        self.valid_indices = valid_indices

    def mux(self, x):
        assert x.shape[0] == self.total_valid_entries
        return (self.mux_matrix @ x.reshape(x.shape[0], -1)).view(
            (self.num_buckets, self.multiplex_count) + x.shape[1:])

    def demux(self, x):
        assert x.shape[:2] == (self.num_buckets, self.multiplex_count)
        return (self.demux_matrix @ x.reshape(self.num_buckets * self.multiplex_count, -1)).view(
            (self.total_valid_entries,) + x.shape[2:])

    def get_valid_object_mask(self):
        return (self.mux_matrix.sum(dim=1) > 0).reshape(self.num_buckets, self.multiplex_count)

    def get_all_valid_object_idx(self):
        return set(self.valid_indices)


def mux_inputs(state):
    return (state.mux_matrix, state.demux_matrix, state.num_buckets,
            state.multiplex_count, tuple(sorted(state.get_all_valid_object_idx())))


def tensor_indexed_memory(encode):
    """Use one fixed-shape GPU conditioning mask, not host-list indexing.

    Preserve upstream's values with where instead of indexed assignment. Only
    object/bucket count affects specialization; changing which objects are
    conditioned no longer recaptures a graph. Reject unknown source contracts.
    """
    import ast
    import inspect
    import textwrap
    from types import MethodType
    tree = ast.parse(textwrap.dedent(inspect.getsource(encode)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef) and not function.decorator_list
    class DeviceIndices(ast.NodeTransformer):
        planning = conditions = embeddings = 0
        def visit_If(self, node):
            test = ast.unparse(node.test)
            if test == 'self.add_object_conditional_embeddings or self.condition_as_mask_input':
                self.planning += 1
                return ast.Pass()
            if test == 'len(conditioning_objects) > 0':
                assert len(node.body) == 1 and not node.orelse
                assert ast.unparse(node.body[0]) == 'cond_values[conditioning_objects] = self.condition_as_mask_input_fg'
                self.conditions += 1
                return ast.parse('cond_values = torch.where(region_condition_mask, '
                    'torch.full_like(cond_values, self.condition_as_mask_input_fg), cond_values)').body[0]
            return self.generic_visit(node)
        def visit_Assign(self, node):
            if ast.unparse(node) == 'obj_merged_embed[unconditioning_objects] = obj_non_cond_embed[unconditioning_objects]':
                self.embeddings += 1
                return ast.parse('obj_merged_embed = torch.where(region_condition_mask[:, None], '
                                 'obj_merged_embed, obj_non_cond_embed)').body[0]
            return node
    replace = DeviceIndices()
    replace.visit(tree)
    assert (replace.planning, replace.conditions, replace.embeddings) == (1,1,1), 'Upstream memory indexing contract changed'
    function.args.kwonlyargs.append(ast.arg(arg='region_condition_mask'))
    function.args.kw_defaults.append(None)
    namespace = dict(encode.__func__.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<sam31-region-memory>', 'exec'), namespace)
    return MethodType(namespace[function.name], encode.__self__)


def pack_features(backbone):
    # Explicitly expose NestedTensor storage to the replay ownership machinery.
    assert backbone['vision_mask'] is None
    assert all(x.mask is None for x in backbone['backbone_fpn'])
    return ([x.tensors for x in backbone['backbone_fpn']], backbone['vision_pos_enc'])


def unpack_features(packed):
    from sam3.model.data_misc import NestedTensor
    fpn, pos = packed
    return {'vision_features':fpn[-1], 'vision_mask':None,
            'backbone_fpn':[NestedTensor(x, None) for x in fpn], 'vision_pos_enc':pos}


class DetectionRegion:
    def __init__(self, torch, detector, region, *, share_image):
        self.torch, self.detector = torch, detector
        self.original = detector.forward_grounding
        self.share_image = share_image
        self.image = self.features = None
        self.image_calls = self.shared_image_hits = 0
        self.full = region('image_and_detection', self.forward_image, 'inductor-image+aot_eager-heads')
        self.prompt = (region('additional_prompt_detection', self.forward_prompt, 'aot_eager-heads')
                       if share_image else None)

    def forward_prompt(self, packed, language, img_ids, text_ids, geometry):
        from sam3.model.geometry_encoders import Prompt
        backbone = unpack_features(packed['detection'])
        backbone.update(language)
        for key in ('interactive', 'sam2_backbone_out'):
            backbone[key] = unpack_features(packed[key])
        out = self.original(backbone, SimpleNamespace(img_ids=img_ids, text_ids=text_ids),
                            None, Prompt(**geometry))
        return {k:out[k] for k in ('pred_logits', 'pred_boxes', 'pred_boxes_xyxy', 'pred_masks')}

    def forward_image(self, image, language, img_ids, text_ids, geometry):
        backbone = self.detector.backbone.forward_image(image.unsqueeze(0).float())
        packed = {'detection':pack_features(backbone),
                  **{k:pack_features(backbone[k]) for k in ('interactive', 'sam2_backbone_out')}}
        out = self.forward_prompt(packed, language, img_ids, text_ids, geometry)
        if not self.share_image:
            del packed['detection']
        return out, packed

    def __call__(self, backbone_out, find_input, find_target, geometric_prompt, **kwargs):
        # This boundary is specifically the causal online, one-image adapter.
        # Refuse offline batches, interactive detection or unknown call contracts.
        assert find_target is None and not self.detector.training
        assert not (set(kwargs) - {'feature_cache'})
        assert 'backbone_fpn' not in backbone_out
        images = backbone_out['img_batch_all_stages'].tensors
        assert hasattr(images, 'image') and len(images) == 1
        assert find_input.img_ids.numel() == 1
        geometry = {k:getattr(geometric_prompt, k) for k in (
            'box_embeddings', 'box_mask', 'point_embeddings', 'point_mask',
            'box_labels', 'point_labels', 'mask_embeddings', 'mask_mask', 'mask_labels')}
        assert all(geometry[k] is None or geometry[k].numel() == 0
                   for k in ('box_embeddings', 'point_embeddings', 'mask_embeddings'))
        language = {k:backbone_out[k] for k in ('language_features', 'language_mask')}
        # Absolute video time stays in the session; this tensor is always slot 0.
        img_ids = self.torch.zeros_like(find_input.img_ids)
        image = images.image
        if self.share_image and image is self.image:
            out = self.prompt(self.features, language, img_ids, find_input.text_ids, geometry)
            packed = self.features
            self.shared_image_hits += 1
        else:
            out, packed = self.full(image, language, img_ids, find_input.text_ids, geometry)
            self.image_calls += 1
            if self.share_image:
                # The worker creates one immutable pixels tensor per frame and
                # passes that same object to every prompt session. Retain a
                # strong reference: allocator address reuse cannot yield a hit.
                self.image, self.features = image, packed
        out['prev_encoder_out'] = {'backbone_out':{
            k:unpack_features(packed[k]) for k in ('interactive', 'sam2_backbone_out')}}
        return out


class PropagationRegion:
    """Run CPU memory selection, then replay attention through final mask/pointer.

    Upstream _prepare_memory_conditioned_features only reshapes the attention
    result before passing it to _forward_sam_heads. Intercept that one boundary
    to defer attention. A guarded, identity-matched placeholder may ONLY reach
    our head wrapper; it is never used as an actual prediction.
    """
    def __init__(self, tracker, region):
        self.tracker = tracker
        self.prepare = tracker._prepare_memory_conditioned_features
        self.head = tracker._forward_sam_heads
        self.encoder = tracker.transformer.encoder.forward
        self.pending = None
        self.combined = region('memory_attention_and_mask', self.forward, 'aot_eager-attention+decoder')
        self.no_memory = region('propagation_mask', self.forward_head, 'aot_eager-decoder')

    def prepare_memory(self, **kwargs):
        assert self.pending is None, 'Unconsumed deferred memory attention'
        captured = []
        def defer(**inputs):
            assert not captured, 'Expected one memory-attention call'
            captured.append(inputs)
            return {'memory':inputs['src']}
        module = self.tracker.transformer.encoder
        module.forward = defer
        try:
            result = self.prepare(**kwargs)
        finally:
            module.forward = self.encoder
        if captured:
            self.pending = result, captured[0]
        return result

    def forward_head(self, features, kwargs, mux):
        return self.head(backbone_features=features, multiplex_state=TensorMultiplex(*mux), **kwargs)

    def forward(self, attention, shape, kwargs, mux):
        features = self.encoder(**attention)['memory'].permute(1, 2, 0).view(shape)
        return self.forward_head(features, kwargs, mux)

    def heads(self, backbone_features, *, multiplex_state, **kwargs):
        pending, self.pending = self.pending, None
        interactive = kwargs.get('point_inputs') is not None or kwargs.get('mask_inputs') is not None
        if pending is not None:
            placeholder, attention = pending
            assert backbone_features is placeholder and not interactive, 'Deferred attention contract changed'
            return self.combined(attention, tuple(backbone_features.shape), kwargs, mux_inputs(multiplex_state))
        if interactive:
            return self.head(backbone_features, multiplex_state=multiplex_state, **kwargs)
        return self.no_memory(backbone_features, kwargs, mux_inputs(multiplex_state))


def install_tracking_regions(torch, model, device_type, *, progress=None):
    """Thor baseline: joint detection; Arc: joint detection/propagation/memory.

    Compile the SAME component functions/backends as the qualified stage path,
    but do not allocate/replay/clone their intermediate outputs individually.
    """
    tracker, detector = model.tracker.model, model.detector
    regions = {}
    def region(name, fn, backend):
        stage = TrackingGraphStage(torch, fn, name, device_type, composed=True,
            progress=progress, backend=backend, max_variants=4 if device_type == 'xpu' else 2)
        regions[name] = stage
        return stage

    def compiled(module, backend='aot_eager'):
        options = {'emulate_precision_casts':True}
        if device_type == 'cuda': options['triton.cudagraphs'] = False
        module.forward = torch.compile(module.forward, backend=backend, fullgraph=True,
            dynamic=False, **({'options':options} if backend == 'inductor' else {}))

    # Text is computed only when prompts change, outside the per-frame region.
    text_module = detector.backbone.language_backbone.encoder
    text_stage = TrackingGraphStage(torch, text_module.forward, 'text_encoder', device_type,
                                    progress=progress, backend='aot_eager')
    text_module.forward = text_stage
    regions['text_encoder'] = text_stage
    compiled(detector.backbone.vision_backbone, 'inductor')
    # Upstream allocates pinned host ROI scales even for ZERO box prompts.
    # CUDA disallows pin_memory during capture. Empty boxes contain no values;
    # return the identical empty embedding/mask without ROI work. Nonempty
    # geometry is explicitly outside this online text-only region's contract.
    geometry = detector.geometry_encoder
    original_boxes = geometry._encode_boxes
    def encode_boxes(boxes, boxes_mask, boxes_labels, img_feats):
        if boxes.shape[0] == 0:
            return geometry.label_embed(boxes_labels.long()), boxes_mask
        return original_boxes(boxes, boxes_mask, boxes_labels, img_feats)
    geometry._encode_boxes = encode_boxes
    for module in (detector.transformer.encoder, detector.transformer.decoder, detector.segmentation_head):
        compiled(module)
    detection = DetectionRegion(torch, detector, region, share_image=device_type == 'xpu')
    detector.forward_grounding = detection
    if device_type == 'cuda':
        # Baseline: retain the already-qualified temporal stage boundaries.
        for name, module in (('memory_encoder', tracker.maskmem_backbone),
                             ('memory_attention', tracker.transformer.encoder),
                             ('tracking_masks', tracker.sam_mask_decoder)):
            stage = TrackingGraphStage(torch, module.forward, name, device_type,
                                       progress=progress, backend='aot_eager')
            module.forward = stage
            regions[name] = stage
    else:
        for module in (tracker.maskmem_backbone, tracker.transformer.encoder, tracker.sam_mask_decoder):
            compiled(module)
        propagation = PropagationRegion(tracker, region)
        tracker._prepare_memory_conditioned_features = propagation.prepare_memory
        tracker._forward_sam_heads = propagation.heads
        encode = tensor_indexed_memory(tracker._encode_new_memory)
        def memory(args, kwargs, mux):
            return encode(*args, **kwargs, multiplex_state=TensorMultiplex(*mux))
        memory_stage = region('memory_update', memory, 'aot_eager-memory-encoder')
        def encode_memory(*args, multiplex_state, **kwargs):
            conditioning = kwargs.get('conditioning_objects')
            conditioning = set() if conditioning is None else set(conditioning)
            assert conditioning <= set(range(multiplex_state.total_valid_entries))
            kwargs['conditioning_objects'] = None  # Replaced by the device mask.
            kwargs['region_condition_mask'] = torch.tensor(
                [i in conditioning for i in range(multiplex_state.total_valid_entries)],
                dtype=torch.bool, device=multiplex_state.mux_matrix.device)
            return memory_stage(args, kwargs, mux_inputs(multiplex_state))
        tracker._encode_new_memory = encode_memory
    return regions
