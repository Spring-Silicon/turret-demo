"use strict";

// Each mounted panel owns all mutable UI state and its fixed API namespace.
function mountTurret(document, apiPrefix = "", options = {}) {
const sharedControls = options.sharedControls === true;
let sharedBusy = false;
const apiUrl = path => apiPrefix + path;
const sliders = { x: document.getElementById("x-slider"), y: document.getElementById("y-slider") };
const motorToggle = document.getElementById("motor-toggle");
const recalibrateButton = document.getElementById("recalibrate");
const message = document.getElementById("message");
const editingAxes = new Set();
const pendingAngles = new Map();
let movingAxis = null;
let moveTimer = null;
let arming = false;
let stopping = false;
let recalibrating = false;
let status = null;
let detectionSending = false;
let feedSource = "";
let displayedDetection = null;
let polling = false;
const promptRows = document.getElementById("prompt-rows");
const addPromptButton = document.getElementById("add-prompt");
const updatePromptsButton = document.getElementById("update-prompts");
const detectionMessage = document.getElementById("detection-status");
const fpsCounter = document.getElementById("fps-counter");
const fpsValue = document.getElementById("fps-value");
const feedLoading = document.getElementById("feed-loading");
const feedLoadingLabel = document.getElementById("feed-loading-label");
if (options.loadingLabel) feedLoadingLabel.textContent = options.loadingLabel;
feedLoading.hidden = false;
fpsCounter.hidden = true;
const modelSelector = document.getElementById("detection-model");
let activeModel = null;
let renderedModel = null;
let modelSending = false;
const modelDrafts = new Map();
let draftInitialized = false;
let draftVersion = 0;
let promptError = "";
let targetSending = false;
let cameraFeed = document.getElementById("camera-feed");
const boxTargets = document.getElementById("box-targets");
const boxButtons = new Map();
const maskOverlay = document.getElementById("mask-overlay");
const frameBuffer = document.createElement("canvas");
const frameBufferContext = frameBuffer.getContext("2d", {alpha: false});
let displayContext = null;
let frameDetection = null;
let frameBackendPid = null;
let visibleFrame = null;
let maskPointer = null;
let focusedMaskId = null;
let loadingFrame = null;
let streamFrame = null;
let detectionStreamOpen = false;
let fpsRevision = null;
let fpsSamples = [];
let pressedBox = null;
let messageExpiresAt = 0;

function detectionFps(detection, now = performance.now()) {
  if (!isDetectionFresh(detection) || !Number.isInteger(detection.frame_sequence)) {
    fpsRevision = null;
    fpsSamples = [];
    return null;
  }
  const revision = `${detection.model}:${detection.revision}`;
  const sequence = detection.frame_sequence;
  if (revision !== fpsRevision || sequence < fpsSamples.at(-1)?.sequence) {
    fpsRevision = revision;
    fpsSamples = [];
  }
  // Count completed worker frames, including those between browser polls.
  // Repeated responses add no frames; their elapsed time lets stalls reach 0.
  fpsSamples.push({time: now, sequence});
  while (fpsSamples.length > 1 && fpsSamples[1].time <= now - 2000) fpsSamples.shift();
  const first = fpsSamples[0];
  const elapsed = now - first.time;
  return elapsed >= 500 ? 1000 * (sequence - first.sequence) / elapsed : null;
}

function renderFps(fps) {
  const value = fps === null ? "—" : fps.toFixed(1);
  if (fpsValue.textContent === value) return;
  fpsValue.textContent = value;
  fpsCounter.setAttribute("aria-label", fps === null ? "Frames per second unavailable" : `${value} frames per second`);
}

function clickableFrame() {
  return frameDetection?.state === "running" && status?.camera?.online &&
    frameBackendPid === (status?.runtime?.backend_pid ?? null) &&
    frameDetection.revision === status?.detection?.revision &&
    status?.detection?.state === "running";
}

function hasTrackingMask(detection) {
  return ((detection?.model === "sam3.1-tracking" && detection?.temporal_tracking === true) ||
      (detection?.model === "sam3.1-mask" && detection?.mask_detection === true)) &&
    detection.mask_overlay?.format === "indexed-png" &&
    typeof detection.mask_overlay.png === "string" && detection.mask_overlay.png.length > 0;
}

function isMaskMode(detection) {
  return ["sam3.1-mask", "sam3.1-tracking"].includes(detection?.model);
}

function trackedBox(detection, backendPid = frameBackendPid) {
  const tracking = status?.tracking;
  if (!tracking?.target || !status?.camera?.online || detection?.state !== "running" ||
      status?.detection?.state !== "running" || detection.revision !== status.detection.revision ||
      backendPid !== (status?.runtime?.backend_pid ?? null)) return null;
  if (tracking.continuity === "hold-reacquire" &&
      !["tracking", "centered", "limited"].includes(tracking.state)) return null;
  if (tracking.instance_id != null) return detection.boxes.find(b => b.instance_id === tracking.instance_id);
  // Prefer the controller's actual target over independently picking a nearer
  // instance. Never reuse an ID from a different prompt/model revision.
  const active = tracking.frame?.[0] === detection.revision && tracking.box?.prompt === tracking.target
    ? detection.boxes.find(b => b.instance_id === tracking.box.instance_id) : null;
  return active || nearestDisplayedBox(detection, tracking.target);
}

function prepareMaskSurface(pending) {
  if (!pending.mask) return null;
  const canvas = document.createElement("canvas");
  canvas.width = pending.mask.naturalWidth; canvas.height = pending.mask.naturalHeight;
  const context = canvas.getContext("2d", {willReadFrequently: true});
  context.drawImage(pending.mask, 0, 0);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
  const original = new Uint8ClampedArray(pixels.data);
  const labels = new Uint16Array(canvas.width * canvas.height);
  const boxes = pending.detection.boxes;
  const colors = boxes.map((box, index) => ({index: index + 1, box,
    rgb: /^#[0-9a-f]{6}$/i.test(box.color || "")
      ? [1, 3, 5].map(i => parseInt(box.color.slice(i, i+2), 16)) : null}));
  const palette = new Map();
  for (let i = 0; i < labels.length; i++) {
    const p = i * 4;
    if (!original[p+3]) continue;
    const rgb = (original[p] << 16) | (original[p+1] << 8) | original[p+2];
    if (!palette.has(rgb)) {
      // Canvas un-premultiplication can round palette channels by 1-2. Resolve
      // each color once; ambiguous colors remain visible but cannot mis-target.
      let best = 13, label = 0;
      for (const candidate of colors) {
        if (!candidate.rgb || !Number.isInteger(candidate.box.instance_id)) continue;
        const distance = candidate.rgb.reduce((sum, c, n) => sum + (c-original[p+n]) ** 2, 0);
        if (distance < best) { best = distance; label = candidate.index; }
        else if (distance === best) label = 0;
      }
      palette.set(rgb, label);
    }
    labels[i] = palette.get(rgb);
  }
  return {canvas, context, pixels, original, labels, boxes, key: null};
}

function maskInstanceAt(surface, point) {
  if (!surface || !point) return null;
  const rect = boxTargets.getBoundingClientRect();
  const displayX = Math.floor((point.x - rect.left) / rect.width * surface.canvas.width);
  const y = Math.floor((point.y - rect.top) / rect.height * surface.canvas.height);
  if (displayX < 0 || y < 0 || displayX >= surface.canvas.width || y >= surface.canvas.height) return null;
  // CSS mirrors the camera, while mask labels keep their original image coordinates.
  const x = surface.canvas.width - 1 - displayX;
  return surface.boxes[surface.labels[y * surface.canvas.width + x] - 1]?.instance_id ?? null;
}

function paintMask(pending) {
  const surface = pending.maskSurface;
  if (!surface) return null;
  const target = trackedBox(pending.detection, pending.backendPid)?.instance_id ?? null;
  const hover = focusedMaskId ?? maskInstanceAt(surface, maskPointer);
  const key = `${target}:${hover}`;
  if (key !== surface.key) {
    surface.pixels.data.set(surface.original);
    const selectedLabel = target == null ? -1 : surface.boxes.findIndex(b => b.instance_id === target) + 1;
    const hoveredLabel = hover == null ? -1 : surface.boxes.findIndex(b => b.instance_id === hover) + 1;
    for (let i = 0; i < surface.labels.length; i++) {
      const label = surface.labels[i];
      if (!label || (label !== selectedLabel && label !== hoveredLabel)) continue;
      const p = i * 4, selected = label === selectedLabel;
      surface.pixels.data[p] = 255;
      surface.pixels.data[p+1] = selected ? 32 : 132;
      surface.pixels.data[p+2] = selected ? 48 : 142;
      surface.pixels.data[p+3] = 144;
    }
    surface.context.putImageData(surface.pixels, 0, 0);
    surface.key = key;
  }
  return surface.canvas;
}

function renderMaskOverlay() {
  // Mask pixels are composited into the persistent camera canvas, not a
  // separately uploaded image layer that can paint out of sync with the JPEG.
  maskOverlay.hidden = true;
  if (!visibleFrame?.maskSurface || visibleFrame.detection !== frameDetection) return;
  const previous = visibleFrame.maskSurface.key;
  paintMask(visibleFrame);
  if (previous !== visibleFrame.maskSurface.key) {
    composeProcessedFrame(visibleFrame);
    displayContext.drawImage(frameBuffer, 0, 0);
  }
}

function composeProcessedFrame(pending) {
  const width = pending.image.naturalWidth, height = pending.image.naturalHeight;
  if (!width || !height || !frameBufferContext) throw new Error("Cannot render camera frame");
  if (frameBuffer.width !== width) frameBuffer.width = width;
  if (frameBuffer.height !== height) frameBuffer.height = height;
  // Compose offscreen first. A mask draw failure leaves the visible surface
  // untouched; the opaque JPEG overwrites the previous offscreen pixels.
  frameBufferContext.globalCompositeOperation = "copy";
  frameBufferContext.drawImage(pending.image, 0, 0, width, height);
  frameBufferContext.globalCompositeOperation = "source-over";
  if (pending.mask) frameBufferContext.drawImage(paintMask(pending), 0, 0, width, height);
}

function presentProcessedFrame(pending) {
  pending.maskSurface = prepareMaskSurface(pending);
  composeProcessedFrame(pending);
  const width = pending.image.naturalWidth, height = pending.image.naturalHeight;
  if (!displayContext || cameraFeed.width !== width || cameraFeed.height !== height) {
    // Create a surface only on first frame or a genuine resolution change.
    // Draw before insertion; never clear/replace the visible surface per frame.
    const canvas = document.createElement("canvas");
    canvas.id = "camera-feed";
    canvas.width = width; canvas.height = height;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "Mirrored processed turret camera frame");
    const context = canvas.getContext("2d", {alpha: false});
    if (!context) throw new Error("Cannot render camera frame");
    context.globalCompositeOperation = "copy";
    context.drawImage(frameBuffer, 0, 0);
    cameraFeed.replaceWith(canvas);
    cameraFeed = canvas; displayContext = context;
    cameraFeed.parentElement.style.aspectRatio = `${width} / ${height}`;
  } else {
    displayContext.drawImage(frameBuffer, 0, 0);
  }
  // Reveal only after the camera image and matching masks have been committed.
  cameraFeed.hidden = false;
  feedLoading.hidden = true;
  fpsCounter.hidden = false;
  cameraFeed.setAttribute("data-frame-key", processedFrameKey(pending.detection));
  cameraFeed.setAttribute("data-frame-sequence", String(pending.detection.frame_sequence));
  cameraFeed.setAttribute("data-mask-present", String(Boolean(pending.mask)));
  maskOverlay.replaceChildren();
  frameDetection = pending.detection;
  frameBackendPid = pending.backendPid;
  visibleFrame = pending;
  renderTracking();
}

function processedFrameKey(detection) {
  return `${status?.runtime?.backend_pid ?? ""}:${detection.model}:${detection.revision}:${detection.frame_sequence}:${detection.frame_url}`;
}

function loadProcessedFrame(detection, source, streamed) {
  const pending = {detection: {...detection}, source, streamed, image: document.createElement("img"),
    mask: hasTrackingMask(detection) ? document.createElement("img") : null, remaining: 0,
    backendPid: status?.runtime?.backend_pid ?? null};
  loadingFrame = pending;
  feedSource = processedFrameKey(detection);
  pending.image.alt = "Processed turret camera frame";
  for (const image of [pending.image, pending.mask].filter(Boolean)) {
    image.setAttribute("data-frame-key", processedFrameKey(detection));
    image.setAttribute("data-frame-sequence", String(detection.frame_sequence));
  }
  if (pending.mask) pending.mask.alt = "";
  pending.remaining = pending.mask ? 2 : 1;
  const finish = (failed = false) => {
    if (loadingFrame !== pending) return;
    if (!failed && --pending.remaining > 0) return;
    loadingFrame = null;
    if (!failed && pending.detection.revision === displayedDetection?.revision &&
        pending.detection.model === displayedDetection?.model) {
      try { presentProcessedFrame(pending); }
      catch (_) { showMessage("Frame display failed; retaining the last result", true); }
    }
    // At most one decode plus the latest worker result, never a video backlog.
    // Errors retain the previous complete pair and wait for a newer result.
    if (displayedDetection && status?.camera) renderDetection(displayedDetection);
  };
  const decode = (image, url) => {
    image.addEventListener("load", () => {
      if (typeof image.decode === "function") image.decode().then(() => finish(), () => finish(true));
      else finish();
    }, {once: true});
    image.addEventListener("error", () => finish(true), {once: true});
    image.src = url;
  };
  decode(pending.image, source);
  if (pending.mask) decode(pending.mask, `data:image/png;base64,${detection.mask_overlay.png}`);
}

async function selectInstance(selection) {
  if (targetSending || detectionSending || modelSending || sharedBusy) return;
  targetSending = true;
  pendingAngles.clear();
  renderTracking();
  try {
    await request("/api/tracking/instance", {method: "POST", body: JSON.stringify(selection)});
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    targetSending = false;
    renderTracking();
  }
}

// Capture on the stable overlay, not an individual box: IDs and buttons may
// change between pointerdown/up at camera rate. Keep the exact down-frame.
boxTargets.addEventListener("pointerup", (event) => {
  if (!pressedBox || pressedBox.pointerId !== event.pointerId) return;
  const picked = pressedBox;
  pressedBox = null;
  boxTargets.releasePointerCapture(event.pointerId);
  if (Math.hypot(event.clientX - picked.x, event.clientY - picked.y) > 12) return;
  if (!clickableFrame() || picked.selection.revision !== frameDetection.revision) {
    showMessage("That camera frame is no longer available; click a displayed box", true);
    return;
  }
  return selectInstance(picked.selection);
});
boxTargets.addEventListener("pointercancel", () => { pressedBox = null; });
boxTargets.addEventListener("lostpointercapture", () => { pressedBox = null; });
boxTargets.addEventListener("pointermove", event => {
  if (!isMaskMode(frameDetection)) return;
  maskPointer = {x: event.clientX, y: event.clientY};
  focusedMaskId = null;
  boxTargets.style.cursor = maskInstanceAt(visibleFrame?.maskSurface, maskPointer) == null ? "default" : "pointer";
  renderMaskOverlay();
});
boxTargets.addEventListener("pointerleave", () => {
  maskPointer = null;
  renderMaskOverlay();
});
boxTargets.addEventListener("pointerdown", event => {
  if (!isMaskMode(frameDetection) || !clickableFrame() || event.button !== 0 || !event.isPrimary ||
      targetSending || detectionSending || modelSending || sharedBusy) return;
  const id = maskInstanceAt(visibleFrame?.maskSurface, {x: event.clientX, y: event.clientY});
  if (id == null) return;
  pressedBox = {selection: {revision: frameDetection.revision,
    frame_sequence: frameDetection.frame_sequence, instance_id: id},
    pointerId: event.pointerId, x: event.clientX, y: event.clientY};
  boxTargets.setPointerCapture(event.pointerId);
});

function renderBoxTargets() {
  const masked = isMaskMode(frameDetection);
  boxTargets.classList.toggle("masked", masked);
  if (!masked) { maskPointer = null; focusedMaskId = null; boxTargets.style.cursor = ""; }
  boxTargets.hidden = !frameDetection;
  if (boxTargets.hidden) return;
  const ids = new Set();
  for (const box of frameDetection.boxes) {
    if (!Number.isInteger(box.instance_id)) continue;
    const id = box.instance_id;
    ids.add(id);
    let button = boxButtons.get(id);
    if (!button) {
      button = document.createElement("button");
      button.type = "button";
      button.className = "box-target";
      const label = document.createElement("span");
      label.className = "box-label";
      button.append(label);
      const selection = () => ({revision: frameDetection.revision,
        frame_sequence: frameDetection.frame_sequence, instance_id: id});
      button.addEventListener("pointerdown", (event) => {
        if (isMaskMode(frameDetection)) return; // Mask pixels, not rectangles, own pointer hit testing.
        if (event.button !== 0 || !event.isPrimary || button.disabled || button.hidden || !clickableFrame()) return;
        pressedBox = {selection: selection(), pointerId: event.pointerId,
          x: event.clientX, y: event.clientY};
        boxTargets.setPointerCapture(event.pointerId);
      });
      button.addEventListener("click", (event) => {
        // Pointer activation was handled above. Retain keyboard/assistive clicks.
        if (event.detail !== 0 || !clickableFrame() || button.disabled || button.hidden) return;
        return selectInstance(selection());
      });
      button.addEventListener("focus", () => { focusedMaskId = id; renderMaskOverlay(); });
      button.addEventListener("blur", () => { focusedMaskId = null; renderMaskOverlay(); });
      boxButtons.set(id, button);
      boxTargets.append(button);
    }
    const [x1, y1, x2, y2] = box.xyxy;
    Object.assign(button.style, {left: `${(1-x2)*100}%`, top: `${y1*100}%`,
      width: `${(x2-x1)*100}%`, height: `${(y2-y1)*100}%`,
      zIndex: String(Math.round(1000000*(1-(x2-x1)*(y2-y1))))});
    button.disabled = !clickableFrame() || targetSending || detectionSending || modelSending || sharedBusy;
    button.setAttribute("aria-label", `Track ${box.prompt} object ${id}`);
    button.setAttribute("aria-pressed", String(trackedBox(frameDetection)?.instance_id === id));
    button.title = masked ? "" : `Track this ${box.prompt}`;
    button.classList.toggle("client-overlay", frameDetection.client_overlay === true && !masked);
    button.classList.toggle("mask-instance", masked);
    button.style.setProperty("--box-color", box.color || "#55e8ce");
    button.children[0].textContent = frameDetection.client_overlay === true && !masked
      ? `${box.prompt.slice(0, 48)} ${Math.round(box.score * 100)}%` : "";
  }
  for (const [id, button] of boxButtons) if (!ids.has(id)) {
    if (focusedMaskId === id) focusedMaskId = null;
    button.remove();
    boxButtons.delete(id);
  }
}

function updatePromptControls() {
  if (sharedControls) return;
  const rows = [...promptRows.children];
  const enabled = Boolean(status?.detection?.enabled) && !modelSending;
  rows.forEach((row, index) => {
    row.querySelector(".detection-prompt").disabled = !enabled;
    row.querySelector(".detection-prompt").setAttribute("aria-label", `Object ${index + 1} to find`);
    row.querySelector(".remove-prompt").disabled = !enabled;
  });
  addPromptButton.disabled = !enabled || rows.length >= (status?.detection?.max_prompts || 8);
  updatePromptsButton.disabled = !enabled || detectionSending;
  const disabled = !enabled || detectionSending || targetSending;
  if (modelSelector.disabled !== disabled) modelSelector.disabled = disabled;
  for (const option of modelSelector.options || []) {
    const disabled = !status?.detection?.models?.find(model => model.id === option.value)?.available;
    if (option.disabled !== disabled) option.disabled = disabled;
  }
  updatePromptColors(displayedDetection || status?.detection);
  updateTargetControls();
}

function updateTargetControls() {
  if (sharedControls) return;
  const target = status?.tracking?.target;
  const prompts = status?.detection?.prompts || [];
  const seen = new Set();
  for (const row of promptRows.children) {
    const prompt = row.querySelector(".detection-prompt").value.trim();
    const button = row.querySelector(".target-prompt");
    const applied = Boolean(prompt) && prompts.includes(prompt) && !seen.has(prompt);
    seen.add(prompt);
    const selected = applied && prompt === target;
    button.disabled = !status?.detection?.enabled || !applied || targetSending || detectionSending || modelSending;
    button.setAttribute("aria-pressed", String(selected));
    const label = selected && status?.tracking?.instance_id != null ? `Track nearest ${prompt}` : selected ? `Stop tracking ${prompt}` : `Track ${prompt || "object"}`;
    button.setAttribute("aria-label", label);
    button.title = !applied ? "Update prompts before tracking this class" : label;
  }
}

async function selectTarget(target) {
  if (targetSending || modelSending) return;
  targetSending = true;
  pendingAngles.clear();
  updateTargetControls();
  try {
    await request("/api/tracking/target", { method: "POST", body: JSON.stringify({ target }) });
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    targetSending = false;
    updateTargetControls();
  }
}

function nearestDisplayedBox(detection, target) {
  const width = (displayContext && cameraFeed.width) || cameraFeed.naturalWidth || status?.camera?.width || 1280;
  const height = (displayContext && cameraFeed.height) || cameraFeed.naturalHeight || status?.camera?.height || 720;
  let nearest = null, distance = Infinity;
  for (const box of detection?.boxes || []) {
    if (box.prompt !== target || !Array.isArray(box.xyxy) || box.xyxy.length !== 4) continue;
    const [x1, y1, x2, y2] = box.xyxy;
    if (!box.xyxy.every(v => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1) || x1 >= x2 || y1 >= y2) continue;
    const point = Object.hasOwn(box, "mask_centroid") ? box.mask_centroid : [(x1+x2)/2, (y1+y2)/2];
    if (!Array.isArray(point) || point.length !== 2 || !point.every(v => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1)) continue;
    const d = ((point[0] - .5) * width) ** 2 + ((point[1] - .5) * height) ** 2;
    if (d < distance || (d === distance && box.score > nearest.score)) { nearest = box; distance = d; }
  }
  return nearest;
}

function renderTracking() {
  const tracking = status?.tracking, target = tracking?.target;
  const element = document.getElementById("tracking-status");
  element.hidden = !tracking?.error;
  element.textContent = tracking?.error || "";
  element.classList.toggle("error", Boolean(tracking?.error));
  document.getElementById("frame-center").toggleAttribute("hidden", !target);
  const detection = frameDetection;
  const box = trackedBox(detection);
  document.getElementById("tracking-overlay").toggleAttribute("hidden", !box || isMaskMode(detection));
  if (box) {
    const [x1, y1, x2, y2] = box.xyxy;
    const rect = document.getElementById("tracked-box");
    for (const [key, value] of Object.entries({x: x1 * 1000, y: y1 * 1000, width: (x2 - x1) * 1000, height: (y2 - y1) * 1000})) rect.setAttribute(key, value);
  }
  updateTargetControls();
  renderMaskOverlay();
  renderBoxTargets();
}

function isDetectionFresh(detection) {
  return detection?.state === "running" &&
    Number.isFinite(detection.frame_age_ms) &&
    detection.frame_age_ms < Math.max(5000, 2 * (Number.isFinite(detection.latency_ms) ? detection.latency_ms : 0) + 1000);
}

function detectionPhase(detection) {
  if (detection?.progress?.phase) return detection.progress.phase;
  // Compatibility with a backend during a rolling viewer update.
  const stage = String(detection?.progress_stage || detection?.state || "idle");
  if (stage.startsWith("compiling")) return "preparing";
  if (stage.startsWith("loading")) return "loading";
  return stage;
}

function updatePromptColors(detection) {
  if (sharedControls) return;
  [...promptRows.children].forEach((row, index) => {
    const prompt = row.querySelector(".detection-prompt").value.trim();
    // Match the category, not its old row index: drafts can remove/edit rows
    // without applying them to the detector yet.
    const category = detection?.categories?.find((item) => item.prompt === prompt);
    row.style.setProperty("--prompt-color", category?.color || detection?.colors?.[index] || "#55e8ce");
  });
}

function addPromptRow(value = "", focus = false) {
  const row = document.getElementById("prompt-row-template").content.firstElementChild.cloneNode(true);
  const input = row.querySelector(".detection-prompt");
  input.value = value;
  input.addEventListener("input", () => {
    if (row.querySelector(".target-prompt").getAttribute("aria-pressed") === "true") selectTarget(null);
    draftVersion += 1;
    promptError = "";
    updatePromptColors(displayedDetection);
    updateTargetControls();
  });
  row.querySelector(".remove-prompt").addEventListener("click", () => {
    if (row.querySelector(".target-prompt").getAttribute("aria-pressed") === "true") selectTarget(null);
    draftVersion += 1;
    promptError = "";
    if (promptRows.children.length === 1) input.value = "";
    else row.remove();
    updatePromptControls();
    const remaining = promptRows.querySelector(".detection-prompt");
    remaining.focus();
  });
  row.querySelector(".target-prompt").addEventListener("click", () => {
    if (row.querySelector(".target-prompt").disabled) return;
    const prompt = input.value.trim();
    return selectTarget(status?.tracking?.target === prompt && status?.tracking?.instance_id == null ? null : prompt);
  });
  promptRows.append(row);
  updatePromptControls();
  if (focus) input.focus();
}

function setPromptRows(prompts) {
  promptRows.replaceChildren();
  (prompts.length ? prompts : [""]).forEach((prompt) => addPromptRow(prompt));
}

function renderDetection(detection) {
  if (displayedDetection && detection &&
      (detection.revision < displayedDetection.revision ||
       (detection.revision === displayedDetection.revision && detection.state === "running" &&
        detection.frame_sequence < displayedDetection.frame_sequence))) return;
  if (detection && (!draftInitialized || activeModel !== (detection.model || "sam3.1"))) {
    if (activeModel && !sharedControls) modelDrafts.set(activeModel, readPromptRows());
    activeModel = detection.model || "sam3.1";
    draftVersion += 1;
    if (!sharedControls) setPromptRows(modelDrafts.get(activeModel) || detection.prompts || (detection.prompt ? [detection.prompt] : []));
    draftInitialized = true;
    loadingFrame = null;
    promptError = "";
  }
  const selection = detection?.model || "sam3.1";
  // Do not reset Firefox's open native menu on every inference result. Compare
  // committed model state, not the popup's temporary hover/keyboard selection.
  if (!sharedControls && renderedModel !== selection) {
    if (modelSelector.value !== selection) modelSelector.value = selection;
    renderedModel = selection;
  }
  displayedDetection = detection;
  updatePromptControls();
  const phase = detectionPhase(detection);
  const preparing = ["loading", "preparing", "capturing", "validating"].includes(phase);
  renderFps(detectionFps(preparing ? null : detection));
  detectionMessage.textContent = promptError || (frameDetection ? detection?.error : "") || "";
  detectionMessage.hidden = !detectionMessage.textContent;
  detectionMessage.classList.toggle("error", Boolean(promptError || detection?.error));
  // The viewer is a sink for completed worker results, never the raw camera.
  // Hold the last pair during loading, stalls, camera loss and model changes.
  // A status poll may already describe N+1 while the paired SSE image is N.
  // That is still a new, valid display frame: never require SSE to catch the
  // status poll exactly (which can starve slower backends at the poll cadence).
  const candidate = detectionStreamOpen ? streamFrame?.detection : detection;
  if (candidate?.state !== "running" || !candidate.frame_url ||
      !Number.isInteger(candidate.frame_sequence) || candidate.revision !== detection?.revision ||
      candidate.model !== detection?.model) return;
  if (frameDetection && frameBackendPid === (status?.runtime?.backend_pid ?? null) &&
      candidate.revision === frameDetection.revision && candidate.model === frameDetection.model &&
      candidate.frame_sequence <= frameDetection.frame_sequence) return;
  const source = candidate.frame_url;
  const streamed = streamFrame?.frame_url === source && streamFrame?.revision === candidate.revision;
  // A queued initial HTTP JPEG must not block already-arrived SSE frames.
  // Upgrade it once to an in-memory JPEG; thereafter still decode one at a time.
  const upgradeToStream = streamed && loadingFrame && !loadingFrame.streamed;
  if ((processedFrameKey(candidate) !== feedSource || upgradeToStream) && (!loadingFrame || upgradeToStream))
    loadProcessedFrame(candidate, streamed ? streamFrame.dataUrl : apiUrl(source), streamed);
}

function shouldStop(servo) {
  return (servo.run_requested ?? servo.armed) || servo.armed ||
    Object.values(servo.axes).some(axis => servo.run_requested === undefined ? axis.torque !== false : axis.torque === true);
}

function renderMotors() {
  const servo = status?.servo;
  if (!servo) return;
  for (const name of ["x", "y"]) {
    const axis = servo.axes[name];
    const slider = sliders[name];
    slider.min = axis.min_degrees;
    slider.max = axis.max_degrees;
    slider.disabled = !servo.armed || !servo.online || stopping;
    if (!editingAxes.has(name) && !pendingAngles.has(name) && movingAxis !== name) {
      const value = servo.armed ? axis.goal_degrees ?? axis.degrees : axis.degrees;
      if (value !== null) slider.value = value;
      document.getElementById(name + "-degrees").textContent = value === null ? "—" : Number(value).toFixed(1) + "°";
    }
  }
  const stop = shouldStop(servo);
  motorToggle.disabled = arming || stopping || (!stop && (recalibrating || !(servo.can_start ?? servo.ready)));
  recalibrateButton.disabled = arming || stopping || recalibrating || !servo.can_recalibrate;
  recalibrateButton.textContent = recalibrating ? "Saving zeros…" : "Recalibrate zeros";
  motorToggle.setAttribute("aria-label", stop ? "Stop motors" : "Start motors");
  motorToggle.title = stop ? "Stop motors (Escape)" : "Start motors (Enter)";
  motorToggle.classList.toggle("stopping", stop);
  // SVGElement does not reflect a .hidden property into the HTML attribute.
  document.getElementById("start-icon").toggleAttribute("hidden", stop);
  document.getElementById("stop-icon").toggleAttribute("hidden", !stop);
}

function showMessage(text, error = false) {
  message.textContent = text;
  message.hidden = !text;
  message.classList.toggle("error", error);
  messageExpiresAt = error && text ? performance.now() + 5000 : 0;
}

function render(next) {
  // Command replies can omit process diagnostics supplied by status polling.
  if (!next.runtime && status?.runtime) next = {...next, runtime: status.runtime};
  if (status?.runtime?.backend_pid && next.runtime?.backend_pid &&
      status.runtime.backend_pid !== next.runtime.backend_pid) {
    // Revisions/sequences restart with the backend. Retain the old picture,
    // but never reject all results from the new process as out of order.
    status = {...status, detection: null};
    displayedDetection = loadingFrame = streamFrame = null;
    feedSource = "";
    fpsRevision = null;
  }
  if (status?.detection && next.detection && (
      next.detection.revision < status.detection.revision ||
      (next.detection.revision === status.detection.revision &&
       next.detection.frame_sequence < status.detection.frame_sequence))) {
    next = {...next, detection: status.detection};
  }
  status = next;
  renderDetection(next.detection);
  const { camera, servo } = next;
  if (!frameDetection) cameraFeed.parentElement.style.aspectRatio = `${camera.width} / ${camera.height}`;
  if (!servo.armed) pendingAngles.clear();
  renderMotors();
  renderTracking();
  options.onStatus?.(next);
  // A 30 FPS detection update must not erase a rejected click's error instantly.
  if (performance.now() >= messageExpiresAt || (frameDetection && (servo.error || camera.error)))
    showMessage(frameDetection ? servo.error || camera.error || "" : "", Boolean(frameDetection && (servo.error || camera.error)));
}

async function request(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    cache: "no-store",
    signal: AbortSignal.timeout(5000),
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  render(body);
  return body;
}

async function startMotors() {
  if (arming || stopping || recalibrating || !(status?.servo?.can_start ?? status?.servo?.ready)
      || (status?.servo?.run_requested ?? status?.servo?.armed)) return;
  arming = true;
  renderMotors();
  try {
    await request("/api/servo/arm", { method: "POST" });
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    arming = false;
    renderMotors();
  }
}

async function recalibrateZeros() {
  if (arming || stopping || recalibrating || !status?.servo?.can_recalibrate) return;
  if (!window.confirm("Set the current X and Y positions as 0°? Position the camera at your intended zero first. Motors stay off; angle limits remain relative to the new zero.")) return;
  recalibrating = true;
  pendingAngles.clear();
  editingAxes.clear();
  renderMotors();
  try {
    await request("/api/servo/recalibrate", {method: "POST", body: "{}"});
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    recalibrating = false;
    renderMotors();
  }
}
recalibrateButton.addEventListener("click", recalibrateZeros);

async function stopMotors() {
  if (stopping) return;
  stopping = true;
  pendingAngles.clear();
  editingAxes.clear();
  renderMotors();
  try {
    await request("/api/servo/disable", { method: "POST" });
  } catch (error) {
    showMessage(error.message + "; torque-off not confirmed", true);
  } finally {
    stopping = false;
    renderMotors();
  }
}

async function flushAngles() {
  moveTimer = null;
  if (movingAxis || stopping || !status?.servo?.armed) return;
  const entry = pendingAngles.entries().next().value;
  if (!entry) return;
  const [axis, degrees] = entry;
  pendingAngles.delete(axis);
  movingAxis = axis;
  try {
    await request("/api/servo/position", { method: "POST", body: JSON.stringify({ axis, degrees }) });
  } catch (error) {
    pendingAngles.clear();
    showMessage(error.message, true);
  } finally {
    movingAxis = null;
    renderMotors();
    if (pendingAngles.size) moveTimer = setTimeout(flushAngles, 80);
  }
}

for (const [axis, slider] of Object.entries(sliders)) {
  slider.addEventListener("pointerdown", () => editingAxes.add(axis));
  const finishEdit = () => { editingAxes.delete(axis); };
  slider.addEventListener("pointerup", finishEdit);
  slider.addEventListener("pointercancel", finishEdit);
  slider.addEventListener("blur", finishEdit);
  slider.addEventListener("input", () => {
    document.getElementById(axis + "-degrees").textContent = Number(slider.value).toFixed(1) + "°";
    if (!status?.servo?.armed || stopping) return;
    pendingAngles.set(axis, Number(slider.value));
    if (!movingAxis && moveTimer === null) moveTimer = setTimeout(flushAngles, 80);
  });
}

motorToggle.addEventListener("click", () => {
  if (motorToggle.disabled || !status?.servo) return;
  const stop = shouldStop(status.servo);
  if (stop) stopMotors();
  else startMotors();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") stopMotors();
  if (event.key === "Enter" && !event.defaultPrevented && !event.repeat && !event.isComposing
      && event.target?.tagName !== "BUTTON") {
    event.preventDefault?.();
    startMotors();
  }
});

async function poll() {
  if (polling) return;
  polling = true;
  try {
    await request("/api/status");
  } catch (error) {
    // A backend still starting is expected; keep the initial loader quiet.
    if (frameDetection) showMessage(error.message, true);
    options.onOffline?.(error);
    updatePromptColors(null);
    renderFps(detectionFps(null));
    detectionMessage.textContent = "Detector connection lost";
    detectionMessage.hidden = !frameDetection;
    detectionMessage.classList.add("error");
    document.getElementById("tracking-overlay").toggleAttribute("hidden", true);
    boxTargets.hidden = true;
  } finally {
    polling = false;
  }
}

async function setPrompts(startAfter = false) {
  if (detectionSending || modelSending) return;
  const submittedVersion = draftVersion;
  const prompts = readPromptRows();
  detectionSending = true;
  promptError = "";
  updatePromptControls();
  try {
    const body = await request("/api/detection/prompts", { method: "POST", body: JSON.stringify({ prompts }) });
    if (draftVersion === submittedVersion) setPromptRows(body.detection.prompts);
    if (startAfter) await startMotors();
  } catch (error) {
    promptError = error.message;
    detectionMessage.textContent = error.message;
    detectionMessage.classList.add("error");
  } finally {
    detectionSending = false;
    updatePromptControls();
  }
}
async function pollDetection() {
  try {
    const current = status?.detection;
    if (current?.enabled && !detectionStreamOpen) {
      const response = await fetch(apiUrl(`/api/detection/status?revision=${current.revision}&sequence=${current.frame_sequence ?? -1}`), {
        cache: "no-store", signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) throw new Error("Detection stream unavailable");
      const detection = await response.json();
      render({...status, detection});
    }
    setTimeout(pollDetection, current?.enabled && !detectionStreamOpen ? 0 : 250);
  } catch (_) {
    // The normal status poll reports failures and keeps controls responsive.
    setTimeout(pollDetection, 1000);
  }
}
const detectionEvents = new EventSource(apiUrl("/api/detection/events"));
detectionEvents.onopen = () => { detectionStreamOpen = true; };
detectionEvents.onerror = () => { detectionStreamOpen = false; }; // Automatic reconnect; HTTP fallback meanwhile.
detectionEvents.onmessage = (event) => {
  const detection = JSON.parse(event.data);
  if (detection.jpeg) {
    const dataUrl = `data:image/jpeg;base64,${detection.jpeg}`;
    delete detection.jpeg;
    streamFrame = {frame_url: detection.frame_url, revision: detection.revision,
      dataUrl, detection};
  }
  if (status) render({...status, detection});
};
function readPromptRows() {
  return [...promptRows.querySelectorAll(".detection-prompt")].map(input => input.value.trim());
}

if (!sharedControls) {
modelSelector.addEventListener("change", async () => {
  if (modelSending || detectionSending || targetSending) return;
  const model = modelSelector.value;
  modelSending = true;
  pendingAngles.clear();
  updatePromptControls();
  renderBoxTargets();
  try {
    await request("/api/detection/model", {method: "POST", body: JSON.stringify({model})});
  } catch (error) {
    if (modelSelector.value !== activeModel) modelSelector.value = activeModel;
    renderedModel = activeModel;
    promptError = error.message;
    detectionMessage.textContent = error.message;
    detectionMessage.classList.add("error");
  } finally {
    modelSending = false;
    updatePromptControls();
  }
});
document.getElementById("detection-form").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.repeat && !event.isComposing && ["SELECT", "INPUT"].includes(event.target.tagName)) {
    event.preventDefault();
    setPrompts(true);
  }
});
document.getElementById("detection-form").addEventListener("submit", (event) => {
  event.preventDefault();
  setPrompts();
});
addPromptButton.addEventListener("click", () => {
  if (addPromptButton.disabled) return;
  draftVersion += 1;
  promptError = "";
  addPromptRow("", true);
});
setPromptRows([]);
}
poll();
setInterval(poll, 200);
pollDetection();
return {
  command: (path, body) => request(path, {method: "POST", body: JSON.stringify(body)}),
  setSharedBusy(value) {
    sharedBusy = value;
    if (value) pendingAngles.clear();
    renderBoxTargets();
  },
};
}

// Preserve the standalone page used by the remote viewer and single-device CLI.
if (document.getElementById("motor-toggle")) {
  mountTurret(document);
  window.addEventListener("load", () => window.springDemoReady?.(), {once: true});
}
