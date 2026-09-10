"""Device-independent public response fields for every inference implementation.

Workers may report extra diagnostics, but viewers consume this common contract.
Missing measurements are null, never invented zero-latency/GPU-only numbers.
"""
from spring_turret.models import MODELS, MENU_MODELS, public_model_id

API_VERSION = 1


def detection_result(result, device_type):
    fields = {
        "boxes":[], "categories":[], "frame_sequence":None, "frame_url":None,
        "mask_overlay":None, "temporal_tracking":False,
        "active_instance_ids":[], "retired_instance_ids":[], "propagated_instance_ids":[],
        "latency_ms":None, "fps":None,
        **result,
    }
    fields["timing"] = {"preprocess_ms":None, "tracking_ms":None,
                        "model_ms":None, "model_gpu_ms":None, "mask_overlay_ms":None,
                        "worker_total_ms":None, **result.get("timing", {})}
    fields["execution"] = {
        "device_type":device_type, "device":result.get("device"),
        "precision":result.get("precision"),
        "torch_compile":result.get("torch_compile"),
        "graph_replay":("cuda" if result.get("cuda_graph") is True else
                        "sycl" if result.get("sycl_graph") is True else None),
        "compilation_scope":result.get("compilation_scope"),
        **result.get("execution", {}),
    }
    return fields


def detection_status(result):
    """Normalize legacy/device-specific snapshots at the public HTTP boundary.

    Keep worker state/progress strings for diagnostics and older clients. The
    additive progress.phase is the shared UI vocabulary, not a claim that every
    preparation is a fresh compile. No inference/control state is mutated here.
    """
    device_type = result.get("device_type") or result.get("execution", {}).get("device_type")
    if device_type is None:
        if result.get("sycl_graph") is True: device_type = "xpu"
        elif result.get("cuda_graph") is True: device_type = "cuda"
    fields = detection_result(result, device_type)
    state = result.get("state", "idle")
    stage = result.get("progress_stage") or state
    # A failed/stopped runtime must not retain a previous preparation message.
    if state in ("error", "disabled", "idle", "waiting_for_camera"):
        stage = state
    if stage.startswith("compiling"):
        phase = "preparing"
    elif stage.startswith("loading"):
        phase = "loading"
    elif stage.startswith("capturing"):
        phase = "capturing"
    elif stage.startswith("validating"):
        phase = "validating"
    else:
        phase = stage if stage in ("running", "error", "disabled", "idle", "waiting_for_camera") else "preparing"
    fields.update(api_version=API_VERSION, device_type=device_type,
                  progress_stage=result.get("progress_stage"),
                  progress={"phase":phase, "stage":stage})
    fields["pipeline_timing"] = {"cycle_ms":None, **result.get("pipeline_timing", {})}
    if "model" in result:
        fields["implementation_model"] = result.get("implementation_model", result["model"])
        fields["model"] = public_model_id(result["model"])
    if "models" in result:
        # Also cover an older, still-running hardware process during deployment.
        choices = {item.get("id"): item for item in result["models"]}
        fields["models"] = [{**choices[key], "label":MODELS[key]["label"]}
                            for key in MENU_MODELS if key in choices]
    return fields


def public_status(state):
    """Normalize an already-owned full or section-only viewer snapshot."""
    fields = {"api_version":API_VERSION, **state}
    if "detection" in state:
        fields["detection"] = detection_status(state["detection"])
    if "tracking" in state:
        # Null means unspecified policy, not an invented hold/reacquire promise.
        fields["tracking"] = {"continuity":None, "hold_reason":None, **state["tracking"]}
    return fields
