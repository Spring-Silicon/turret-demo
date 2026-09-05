"""Model choices shared by the HTTP process and inference workers."""

if __package__:
    from .prompts import normalize_prompts
else:
    from prompts import normalize_prompts

# Ultralytics v8.4.140 cfg/datasets/coco.yaml, in checkpoint class-ID order.
COCO_CLASSES = tuple("person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush".split("|"))
MODELS = {
    "sam3.1": {"label": "SAM 3.1", "worker": "sam31_worker.py", "classes": None},
    "yolo26x": {"label": "YOLO26x", "worker": "yolo26_worker.py", "classes": COCO_CLASSES},
}


def model_prompts(model, prompts):
    prompts = normalize_prompts(prompts)
    if model == "yolo26x":
        prompts = [prompt.lower() for prompt in prompts]
        if any(prompt not in COCO_CLASSES for prompt in prompts):
            raise ValueError("YOLO26x only supports the 80 COCO classes; use SAM 3.1 for free text")
    return prompts
