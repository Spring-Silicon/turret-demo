"""Sleepy-joe 2026-09-08 mask layers; only construction imports relocated.

Original mask_model.py SHA256:
f375733900fd601c8eac60122d30c9cda9aab4fd8593adf533b2c190b51d14f4
"""

import copy
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F


class ImageFeatures(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        assert backbone.scalp == 0 and len(backbone.vision_backbone.convs) == 3
        self.backbone = backbone

    def forward(self, pixels):
        neck = self.backbone.vision_backbone
        trunk = neck.trunk(pixels)[-1]
        trunk = getattr(trunk, "tensors", trunk)
        low = neck.convs[-1](trunk)
        return low, neck.position_encoding(low).to(low.dtype), trunk


class ExtraFeatures(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.convs = backbone.vision_backbone.convs[:2]

    def forward(self, trunk):
        return self.convs[0](trunk), self.convs[1](trunk)


def grounding_with_mask_inputs(head):
    graph = copy.deepcopy(head.graph)
    graph.set_codegen(torch.fx.graph.CodeGen())
    nodes = {node.name: node for node in graph.nodes}
    contracts = {
        "permute_211": ((5184, 1, 256), "output.transpose(0, 1)"),
        "add_268": ((200, 1, 256), "intermediate.append(out_norm(output))"),
        "cat": ((33, 1, 256), "prompt = torch.cat"),
        "cat_1": ((1, 33), "prompt_mask = torch.cat"),
    }
    for name, (shape, source) in contracts.items():
        node = nodes[name]
        assert tuple(node.meta["val"].shape) == shape
        assert source in node.meta["stack_trace"]
    output = next(n for n in graph.nodes if n.op == "output")
    original = tuple(output.args[0])
    assert len(original) == 3
    with graph.inserting_before(output):
        queries = graph.call_function(
            torch.ops.aten.permute.default, (nodes["add_268"], [1, 0, 2])
        )
        queries = graph.call_function(torch.ops.aten.unsqueeze.default, (queries, 0))
    output.args = (
        original + (nodes["permute_211"], queries, nodes["cat"], nodes["cat_1"]),
    )
    graph.lint()
    return torch.fx.GraphModule(head, graph)


class TiledGroupNorm(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, value):
        batch, channels, height, width = value.shape
        groups = self.norm.num_groups
        assert batch == 1 and height * width % 256 == 0
        x = (
            value.float()
            .permute(0, 2, 3, 1)
            .reshape(batch, height * width // 256, 256, groups, channels // groups)
        )
        variance, mean = torch.var_mean(x, (2, 4), correction=0, keepdim=True)
        average = mean.mean(1, keepdim=True)
        variance = (variance + (mean - average).square()).mean(1, keepdim=True)
        normalized = (x - average) * torch.rsqrt(variance + self.norm.eps)
        normalized = normalized.reshape(batch, height, width, channels).permute(
            0, 3, 1, 2
        )
        return (
            normalized * self.norm.weight[None, :, None, None]
            + self.norm.bias[None, :, None, None]
        )


class InstanceMasks(torch.nn.Module):
    def __init__(self, checkpoint, device, tiled_norm=False):
        super().__init__()
        if __package__:
            from .sam31_worker import _redirect_cuda_construction
        else:
            from sam31_worker import _redirect_cuda_construction
        from sam3.model_builder import _create_segmentation_head

        with (_redirect_cuda_construction(torch) if torch.device(device).type == 'xpu' else nullcontext()):
            self.head = _create_segmentation_head(use_fa3=False)
        state = torch.load(
            Path(checkpoint), map_location="cpu", weights_only=True, mmap=True
        )
        state = state.get("model", state)
        prefix = "detector.segmentation_head."
        self.head.load_state_dict(
            {
                k.removeprefix(prefix): v
                for k, v in state.items()
                if k.startswith(prefix)
            },
            strict=True,
        )
        self.head.act_ckpt = False
        if tiled_norm:
            self.head.pixel_decoder.norms = torch.nn.ModuleList(
                TiledGroupNorm(norm) for norm in self.head.pixel_decoder.norms
            )
        self.to(device).eval().requires_grad_(False)

    def forward(self, high, middle, low, memory, queries, prompt, prompt_mask):
        return self.head(
            backbone_feats=[high, middle, low],
            obj_queries=queries,
            image_ids=torch.zeros(1, dtype=torch.int64, device=low.device),
            encoder_hidden_states=memory,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )["pred_masks"]


class BinaryMaskOutput(torch.nn.Module):
    def __init__(self, capacity=10, confidence=0.5):
        super().__init__()
        if not isinstance(capacity, int) or not 1 <= capacity <= 10:
            raise ValueError("Mask capacity must be between 1 and 10")
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between zero and one")
        self.capacity = capacity
        self.confidence = confidence

    def forward(self, logits, presence, native_masks):
        scores = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).reshape(200)
        indices = torch.argsort(scores, descending=True, stable=True)[: self.capacity]
        selected_scores = scores[indices]
        valid = selected_scores > self.confidence
        total = (scores > self.confidence).sum(dtype=torch.int32)
        retained = valid.sum(dtype=torch.int32)
        masks = native_masks[0, indices].unsqueeze(0)
        probabilities = F.interpolate(
            masks, (1008, 1008), mode="bilinear", align_corners=False
        ).sigmoid()
        binary = (probabilities[0] > 0.5) & valid[:, None, None]
        return (
            binary,
            torch.where(valid, selected_scores, 0),
            torch.where(valid, indices, -1),
            retained,
            total - retained,
        )


class ReferenceMaskHead(torch.nn.Module):
    """Original SAM grounding/segmentation call for independent boundary checks."""

    def __init__(self, model, segmentation):
        super().__init__()
        self.model = model
        self.model.segmentation_head = segmentation

    def forward(self, low, position, high, middle, text, text_mask):
        from sam3.model.data_misc import FindStage
        from sam3.model.geometry_encoders import Prompt

        backbone = {
            "vision_features": low,
            "backbone_fpn": [high, middle, low],
            "vision_pos_enc": [position],
            "language_features": text,
            "language_mask": text_mask,
        }
        find = FindStage(
            img_ids=torch.zeros(1, dtype=torch.int64, device=low.device),
            text_ids=torch.zeros(1, dtype=torch.int64, device=low.device),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometry = Prompt(
            box_embeddings=low.new_zeros((0, 1, 4)),
            box_mask=torch.zeros((1, 0), dtype=torch.bool, device=low.device),
        )
        out = self.model.forward_grounding(
            backbone_out=backbone,
            find_input=find,
            geometric_prompt=geometry,
            find_target=None,
        )
        return (
            out["pred_logits"],
            out["pred_boxes"],
            out["presence_logit_dec"],
            out["pred_masks"],
        )
