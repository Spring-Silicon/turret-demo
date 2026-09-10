"""Model choices shared by the HTTP process and inference workers."""

if __package__:
    from .prompts import normalize_prompts
else:
    from prompts import normalize_prompts

MODELS = {
    "sam3.1": {"label": "SAM 3.1", "worker": "sam31_worker.py", "classes": None},
    "sam3.1-tracking": {"label": "SAM 3.1 Tracking", "worker": "sam31_tracking_worker.py", "classes": None},
    "sam3.1-v18": {"label": "SAM 3.1 v18", "worker": "sam31_tracking_v18_worker.py", "classes": None},
    "sam3.1-mask": {"label": "SAM 3.1 Mask", "worker": "sam31_mask_worker.py", "classes": None},
    "efficient-nomem": {"label": "Efficient NoMem", "worker": "proteus_worker.py", "classes": None},
    "hybrid-nomem": {"label": "Hybrid NoMem", "worker": "proteus_worker.py", "classes": None},
    "efficient-tracking": {"label": "Efficient Tracking", "worker": "proteus_worker.py", "classes": None},
    "efficient-memory": {"label": "Efficient Memory", "worker": "proteus_worker.py", "classes": None},
    "sam3.1-nomem": {"label": "SAM NoMem", "worker": "sam31_mask_worker.py", "classes": None},
}

PROTEUS_MODELS = ("efficient-nomem", "hybrid-nomem", "efficient-tracking", "efficient-memory")


def is_session_model(model):
    """Native IDs/session lifetime do not necessarily imply neural memory."""
    return is_tracking_model(model) or model in PROTEUS_MODELS


def max_prompts(model):
    return 1 if model in ("efficient-nomem", "hybrid-nomem", "efficient-memory") else 8


def is_tracking_model(model):
    return model in ("sam3.1-tracking", "sam3.1-v18", "efficient-tracking", "efficient-memory")


def model_available(model, config):
    if model not in MODELS or not config.get("enabled", False):
        return False
    if model in PROTEUS_MODELS:
        return (config.get("device_type", "xpu") == "xpu" and bool(config.get("proteus_bundle"))
                and (model != "efficient-memory" or bool(config.get("proteus_memory_bundle"))))
    if model == "sam3.1-nomem":
        return config.get("device_type") == "cuda"
    if model == "sam3.1-tracking":
        return config.get("device_type", "xpu") in ("xpu", "cuda") and bool(config.get("sam31_tracking_bundle"))
    if model == "sam3.1-v18":
        return (config.get("device_type", "xpu") == "xpu"
                and bool(config.get("sam31_tracking_bundle"))
                and bool(config.get("sam31_tracking_v18_bundle")))
    if model == "sam3.1-mask":
        return config.get("device_type", "xpu") == "cuda" or bool(config.get("sam31_mask_bundle"))
    return model == "sam3.1"


def model_prompts(model, prompts):
    if model not in MODELS:
        raise ValueError("model must be one of " + ", ".join(MODELS))
    result = normalize_prompts(prompts)
    if len(result) > max_prompts(model):
        raise ValueError(f"{MODELS[model]['label']} accepts one prompt; remove the extra object rows")
    return result
