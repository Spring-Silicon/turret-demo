"""Model choices shared by the HTTP process and inference workers."""

if __package__:
    from .prompts import normalize_prompts
else:
    from prompts import normalize_prompts

MODELS = {
    "sam3.1": {"label": "SAM 3.1", "worker": "sam31_worker.py", "classes": None},
    "sam3.1-tracking": {"label": "SAM 3.1 Tracking", "worker": "sam31_tracking_worker.py", "classes": None},
    "sam3.1-mask": {"label": "SAM 3.1 Mask", "worker": "sam31_mask_worker.py", "classes": None},
}


def model_available(model, config):
    if model not in MODELS or not config.get("enabled", False):
        return False
    if model == "sam3.1-tracking":
        return config.get("device_type", "xpu") in ("xpu", "cuda") and bool(config.get("sam31_tracking_bundle"))
    if model == "sam3.1-mask":
        return config.get("device_type", "xpu") == "cuda" or bool(config.get("sam31_mask_bundle"))
    return model == "sam3.1"


def model_prompts(model, prompts):
    if model not in MODELS:
        raise ValueError("model must be one of " + ", ".join(MODELS))
    return normalize_prompts(prompts)
