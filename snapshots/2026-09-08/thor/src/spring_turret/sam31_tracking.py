"""Causal SAM 3.1 Object Multiplex tracking, separate from detector-only SAM.

One shared dense model, independent temporal state per text prompt. The upstream
CUDA defaults are retargeted only inside SAM's imports in this isolated worker;
no Torch functions or on-disk model sources are changed.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys


class XpuSource(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if node.attr == "major" and (
            ast.unparse(node.value) in ("device_props", "props")
            or isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "get_device_properties"
        ):
            # NVIDIA architecture/TF32 gates; FA3 is explicitly disabled.
            return ast.copy_location(ast.Constant(0), node)
        self.generic_visit(node)
        if node.attr == "cuda" and ast.unparse(node.value) != "torch.backends":
            node.attr = "xpu"
        elif node.attr == "is_cuda":
            node.attr = "is_xpu"
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str) and (
            node.value == "cuda" or node.value.startswith("cuda:")
        ):
            node.value = node.value.replace("cuda", "xpu", 1)
        return node


def configure_source(bundle: Path, device_type: str = "xpu"):
    if device_type not in ("xpu", "cuda"):
        raise ValueError("Tracking device must be xpu or cuda")
    root = bundle.resolve()
    if not (root / "sam3/model/sam3_multiplex_tracking.py").is_file():
        raise ValueError("SAM 3.1 tracking source bundle is missing")
    if "sam3" in sys.modules:
        raise RuntimeError("Tracking must run in its own fresh worker")
    # Use upstream Torch mask operations, not NVIDIA-specific perflib kernels.
    os.environ["USE_PERFLIB"] = "0"
    fingerprints = {}

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            source = self.get_data(self.path)
            fingerprints[str(Path(self.path).relative_to(root))] = hashlib.sha256(source).hexdigest()
            tree = ast.parse(source, self.path)
            if device_type == "xpu":
                tree = XpuSource().visit(tree)
            return compile(ast.fix_missing_locations(tree), self.path, "exec", dont_inherit=True)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "sam3" and not fullname.startswith("sam3."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.endswith(".py"):
                if not Path(spec.origin).resolve().is_relative_to(root):
                    raise RuntimeError("Mixed SAM tracking source packages")
                spec.loader = Loader(fullname, spec.origin)
            return spec

    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, Finder())
    return fingerprints


def build_model(torch, checkpoint: Path, device=None):
    import sam3
    from sam3 import model_builder as b
    from sam3.model.sam3_multiplex_base import Sam3MultiplexPredictorWrapper
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTracking
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    tracker = b.build_sam3_multiplex_video_model(
        checkpoint_path=None, load_from_HF=False, device="cpu",
        use_fa3=False, use_rope_real=True, compile=False,
    )
    # Exactly the upstream shared-encoder architecture, not a second image ViT.
    tracker.backbone = None
    wrapper = Sam3MultiplexPredictorWrapper(model=tracker)
    wrapper.bf16_context.__exit__(None, None, None)
    bpe = str(Path(sam3.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz")
    detector = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=SAM3VLBackboneTri(scalp=0,
            visual=b._create_multiplex_tri_backbone(compile_mode=None, use_fa3=False, use_rope_real=True),
            text=b._create_text_encoder(bpe)),
        transformer=b._create_sam3_transformer(use_fa3=False),
        segmentation_head=b._create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None, input_geometry_encoder=b._create_geometry_encoder(),
        use_early_fusion=True, use_dot_prod_scoring=True,
        dot_prod_scoring=b._create_dot_product_scoring(), supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    # Official SAM3.1 tracking thresholds and memory policy. Only scheduling is
    # causal: no lookahead, reverse pass, or 16-frame batch in a live camera.
    model = Sam3MultiplexTracking(
        tracker=wrapper, detector=detector,
        score_threshold_detection=.4, det_nms_thresh=.1, det_nms_use_iom=True,
        assoc_iou_thresh=.1, new_det_thresh=.65,
        hotstart_delay=15, hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=False,
        suppress_overlapping_based_on_recent_occlusion_threshold=.7,
        suppress_det_close_to_boundary=True, fill_hole_area=0,
        recondition_every_nth_frame=16, use_iom_recondition=True, iom_thresh_recondition=.5,
        masklet_confirmation_enable=True, reconstruction_bbox_iou_thresh=-1,
        reconstruction_bbox_det_score=.8, max_num_objects=16,
        postprocess_batch_size=1, use_batched_grounding=False,
        max_num_kboxes=0, sprinkle_removal_area=0, is_multiplex=True,
        image_size=1008, image_mean=(.5, .5, .5), image_std=(.5, .5, .5), compile_model=False,
    )
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    weights = weights.get("model", weights)
    missing, extra = model.load_state_dict(weights, strict=False, assign=True)
    # Real RoPE sin/cos buffers are deterministically derived, not checkpoint
    # parameters. Every learned parameter and every other buffer is required.
    unexpected_missing = [key for key in missing if not key.endswith((".freqs_cis_real", ".freqs_cis_imag"))]
    if unexpected_missing or extra:
        raise RuntimeError(f"Incomplete SAM tracker weights: {unexpected_missing}, {extra}")
    if (tracker.use_memory_selection or tracker.memory_temporal_stride_for_eval != 1
            or tracker.max_obj_ptrs_in_encoder != 16 or tracker.num_maskmem != 7
            or tracker.max_cond_frames_in_attn != 4):
        raise RuntimeError("Unsupported temporal memory policy for the online adapter")
    if device is None:
        device = torch.device("xpu", torch.xpu.current_device())
    return model.eval().requires_grad_(False).to(device)


class CurrentImage:
    """One actual image; absolute tracker time is separate from image slot 0."""
    def __init__(self, image, frame_idx):
        self.image, self.frame_idx = image, frame_idx

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index not in (0, self.frame_idx):
            raise IndexError("The online tracker requested an unavailable frame")
        return self.image


def prune_tracker_state(state, frame_idx, *, max_cond=4, keep_first=False, history=32):
    """Evict only memories unreachable by future forward-only inference.

    Keep 32 recent frames (>16 object pointers and 7 mask memories), plus the
    latest four conditioning frames (and first if configured). No session reset
    or ID replacement is used to limit memory. Reverse/interactive edits are not
    exposed by this live adapter.
    """
    cond = state["output_dict"]["cond_frame_outputs"]
    retained = set(sorted(cond)[-max_cond:])
    if keep_first and cond:
        retained.add(min(cond))
    floor = frame_idx - history + 1

    def trim(mapping, protected=()):
        for key in list(mapping):
            if key < floor and key not in protected:
                del mapping[key]

    stores = [state["output_dict"]]
    stores += list(state.get("output_dict_per_obj", {}).values())
    stores += list(state.get("temp_output_dict_per_obj", {}).values())
    for store in stores:
        trim(store["cond_frame_outputs"], retained)
        trim(store["non_cond_frame_outputs"])
    for key in ("point_inputs_per_obj", "mask_inputs_per_obj"):
        for mapping in state.get(key, {}).values():
            trim(mapping, retained)
    trim(state.get("frames_already_tracked", {}), retained)
    for inds in state.get("consolidated_frame_inds", {}).values():
        inds.intersection_update({i for i in inds if i >= floor or i in retained})


class OnlineSession:
    def __init__(self, model, image, width, height, prompt):
        self.model, self.index = model, 0
        self.state = {"device": model.device, "constants": {},
                      "orig_width": width, "orig_height": height}
        model._construct_initial_input_batch(self.state, image.unsqueeze(0))
        self.state["input_batch"].find_text_batch[0] = prompt
        self.trackers, self.metadata, self.cache = [], {}, {}

    def step(self, image, *, prune=True):
        from sam3.model.sam3_multiplex_base import MaskletConfirmationStatus
        index = self.index
        previous_ids = set(int(i) for i in self.metadata.get("obj_ids_all_gpu", ()))
        self.state["input_batch"].img_batch.tensors = CurrentImage(image, index)
        for tracker in self.trackers:
            # SAM normalizes pointer time encodings by min(video length, 16).
            # A live stream is not a succession of 1-, 2-, 3-frame clips.
            tracker["num_frames"] = max(self.model.tracker.max_obj_ptrs_in_encoder, index + 1)
        masks, scores, self.trackers, self.metadata, stats, _ = self.model._det_track_one_frame(
            frame_idx=index, num_frames=index + 1, reverse=False,
            input_batch=self.state["input_batch"],
            geometric_prompt=self.state["constants"]["empty_geometric_prompt"],
            tracker_states_local=self.trackers, tracker_metadata_prev=self.metadata,
            feature_cache=self.cache, orig_vid_height=self.state["orig_height"],
            orig_vid_width=self.state["orig_width"], is_image_only=False,
        )
        meta = self.metadata
        rank = meta["rank0_metadata"]
        ids = meta["obj_ids_all_gpu"]
        unconfirmed = ids[rank["masklet_confirmation"]["status"] == MaskletConfirmationStatus.UNCONFIRMED.value].tolist()
        out = {"obj_id_to_mask": masks, "obj_id_to_score": scores,
               "obj_id_to_sam2_score": meta["obj_id_to_sam2_score_frame_wise"][index],
               "frame_stats": stats}
        processed = self.model._postprocess_output(
            self.state, out, removed_obj_ids=rank["removed_obj_ids"],
            suppressed_obj_ids=rank["suppressed_obj_ids"][index], unconfirmed_obj_ids=unconfirmed,
        )
        active = set(int(i) for i in ids)
        processed["active_ids"] = sorted(active)
        processed["memory_frames"] = sum(len(t["output_dict"]["cond_frame_outputs"]) + len(t["output_dict"]["non_cond_frame_outputs"]) for t in self.trackers)
        processed["propagated_ids"] = sorted(int(i) for i in out["obj_id_to_sam2_score"] if int(i) in previous_ids)
        if prune:
            for tracker in self.trackers:
                prune_tracker_state(tracker, index, max_cond=self.model.tracker.max_cond_frames_in_attn,
                                    keep_first=self.model.tracker.keep_first_cond_frame)
            # These are reporting histories, not inputs to temporal attention.
            for history in (meta["obj_id_to_sam2_score_frame_wise"], rank["suppressed_obj_ids"]):
                for key in list(history):
                    if key < index - 31:
                        del history[key]
            for field in ("obj_id_to_score", "obj_id_to_last_occluded"):
                for key in list(meta[field]):
                    if key not in active:
                        del meta[field][key]
            for field in ("obj_first_frame_idx", "unmatched_frame_inds", "trk_keep_alive"):
                for key in list(rank[field]):
                    if key not in active:
                        del rank[field][key]
            for pair in list(rank["overlap_pair_to_frame_inds"]):
                if not set(pair).issubset(active):
                    del rank["overlap_pair_to_frame_inds"][pair]
            # Removed IDs are never reused (max_obj_id stays monotonic).
            rank["removed_obj_ids"].intersection_update(active)
        self.index += 1
        return processed
