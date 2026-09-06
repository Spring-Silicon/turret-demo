"use strict";

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
let keepaliveSending = false;
let status = null;
let detectionSending = false;
let feedSource = "";
let displayedDetection = null;
let polling = false;
const promptRows = document.getElementById("prompt-rows");
const addPromptButton = document.getElementById("add-prompt");
const updatePromptsButton = document.getElementById("update-prompts");
const detectionMessage = document.getElementById("detection-status");
const modelSelector = document.getElementById("detection-model");
let activeModel = null;
let modelSending = false;
const modelDrafts = new Map();
let draftInitialized = false;
let draftVersion = 0;
let promptError = "";
let targetSending = false;
const cameraFeed = document.getElementById("camera-feed");
const boxTargets = document.getElementById("box-targets");
const boxButtons = new Map();
let frameDetection = null;
let loadingDetection = null;
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

function clickableFrame() {
  return frameDetection?.state === "running" && status?.camera?.online &&
    frameDetection.revision === status?.detection?.revision &&
    status?.detection?.state === "running" &&
    frameDetection.frame_age_ms + performance.now() - frameDetection.receivedAt <= 750;
}

async function selectInstance(selection) {
  if (targetSending || detectionSending || modelSending) return;
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
  if (!clickableFrame() || picked.selection.revision !== frameDetection.revision ||
      performance.now() > picked.expiresAt) {
    showMessage("That camera frame is stale; click a box in a fresh frame", true);
    return;
  }
  return selectInstance(picked.selection);
});
boxTargets.addEventListener("pointercancel", () => { pressedBox = null; });
boxTargets.addEventListener("lostpointercapture", () => { pressedBox = null; });

function renderBoxTargets() {
  boxTargets.hidden = !clickableFrame();
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
        if (event.button !== 0 || !event.isPrimary || button.disabled || !clickableFrame()) return;
        pressedBox = {selection: selection(), pointerId: event.pointerId,
          x: event.clientX, y: event.clientY,
          expiresAt: frameDetection.receivedAt + 750 - frameDetection.frame_age_ms};
        boxTargets.setPointerCapture(event.pointerId);
      });
      button.addEventListener("click", (event) => {
        // Pointer activation was handled above. Retain keyboard/assistive clicks.
        if (event.detail !== 0 || !clickableFrame() || button.disabled) return;
        return selectInstance(selection());
      });
      boxButtons.set(id, button);
      boxTargets.append(button);
    }
    const [x1, y1, x2, y2] = box.xyxy;
    Object.assign(button.style, {left: `${x1*100}%`, top: `${y1*100}%`,
      width: `${(x2-x1)*100}%`, height: `${(y2-y1)*100}%`,
      zIndex: String(Math.round(1000000*(1-(x2-x1)*(y2-y1))))});
    button.disabled = targetSending || detectionSending || modelSending;
    button.setAttribute("aria-label", `Track ${box.prompt} object ${id}`);
    button.setAttribute("aria-pressed", String(status?.tracking?.instance_id === id));
    button.title = `Track this ${box.prompt}`;
    button.classList.toggle("client-overlay", frameDetection.client_overlay === true);
    button.style.setProperty("--box-color", box.color || "#55e8ce");
    button.children[0].textContent = frameDetection.client_overlay === true
      ? `${box.prompt.slice(0, 48)} ${Math.round(box.score * 100)}%` : "";
  }
  for (const [id, button] of boxButtons) if (!ids.has(id)) {
    button.remove();
    boxButtons.delete(id);
  }
}

cameraFeed.addEventListener("load", () => {
  // A click must refer to the JPEG actually on screen, not the newest API
  // result while that image is still downloading. Only one JPEG loads at once.
  frameDetection = loadingDetection;
  loadingDetection = null;
  renderTracking();
  // If a newer pair arrived during JPEG decoding, load it immediately.
  if (detectionStreamOpen && streamFrame && status?.camera) renderDetection(status.detection);
});
cameraFeed.addEventListener("error", () => {
  loadingDetection = frameDetection = null;
  feedSource = "";
  renderTracking();
});

function updatePromptControls() {
  const rows = [...promptRows.children];
  const enabled = Boolean(status?.detection?.enabled) && !modelSending;
  rows.forEach((row, index) => {
    row.querySelector(".detection-prompt").disabled = !enabled;
    row.querySelector(".detection-prompt").setAttribute("aria-label", `Object ${index + 1} to find`);
    row.querySelector(".remove-prompt").disabled = !enabled;
  });
  addPromptButton.disabled = !enabled || rows.length >= (status?.detection?.max_prompts || 8);
  updatePromptsButton.disabled = !enabled || detectionSending;
  modelSelector.disabled = !enabled || detectionSending || targetSending;
  for (const option of modelSelector.options || []) {
    option.disabled = !status?.detection?.models?.find(model => model.id === option.value)?.available;
  }
  updatePromptCounts(displayedDetection || status?.detection);
  updateTargetControls();
}

function updateTargetControls() {
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
  const width = status?.camera?.width || 1280, height = status?.camera?.height || 720;
  let nearest = null, distance = Infinity;
  for (const box of detection?.boxes || []) {
    if (box.prompt !== target || !Array.isArray(box.xyxy) || box.xyxy.length !== 4) continue;
    const [x1, y1, x2, y2] = box.xyxy;
    if (!box.xyxy.every(v => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1) || x1 >= x2 || y1 >= y2) continue;
    const d = (((x1 + x2) / 2 - .5) * width) ** 2 + (((y1 + y2) / 2 - .5) * height) ** 2;
    if (d < distance || (d === distance && box.score > nearest.score)) { nearest = box; distance = d; }
  }
  return nearest;
}

function renderTracking() {
  const tracking = status?.tracking, target = tracking?.target;
  const element = document.getElementById("tracking-status");
  const labels = { stopped: "press Start", waiting: "waiting for fresh detections", lost: "not found",
    centered: "centered", tracking: "tracking", limited: "angle limit", uncalibrated: "camera directions not calibrated" };
  element.hidden = !target && !tracking?.error;
  element.textContent = tracking?.error || (target ? `${target}${tracking.instance_id != null ? " · retargeting" : ""} · ${tracking.state === "uncalibrated" ? labels.uncalibrated : !status?.servo?.armed ? "press Start" : labels[tracking.state] || "waiting"}` : "");
  element.classList.toggle("error", Boolean(tracking?.error));
  document.getElementById("frame-center").toggleAttribute("hidden", !target);
  const detection = frameDetection;
  const box = target && clickableFrame()
    ? tracking.instance_id != null ? detection.boxes.find(b => b.instance_id === tracking.instance_id)
      : nearestDisplayedBox(detection, target) : null;
  document.getElementById("tracking-overlay").toggleAttribute("hidden", !box);
  if (box) {
    const [x1, y1, x2, y2] = box.xyxy;
    const rect = document.getElementById("tracked-box");
    for (const [key, value] of Object.entries({x: x1 * 1000, y: y1 * 1000, width: (x2 - x1) * 1000, height: (y2 - y1) * 1000})) rect.setAttribute(key, value);
  }
  updateTargetControls();
  renderBoxTargets();
}

function isDetectionFresh(detection) {
  return detection?.state === "running" &&
    detection.frame_age_ms < Math.max(5000, 2 * detection.latency_ms + 1000);
}

function updatePromptCounts(detection) {
  const fresh = isDetectionFresh(detection);
  [...promptRows.children].forEach((row, index) => {
    const prompt = row.querySelector(".detection-prompt").value.trim();
    // Match the category, not its old row index: drafts can remove/edit rows
    // without applying them to the detector yet.
    const category = detection?.categories?.find((item) => item.prompt === prompt);
    const count = fresh && category && Number.isInteger(category.count) && category.count >= 0
      ? category.count : null;
    const output = row.querySelector(".prompt-count");
    output.textContent = count === null ? "—" : String(count);
    output.setAttribute("aria-label", count === null ? `${prompt || "Object"}: count unavailable` : `${prompt}: ${count} detected`);
    output.title = count === null
      ? (prompt && !detection?.prompts?.includes(prompt) ? "Update prompts to count this object" : "Waiting for detection")
      : `${count} detected`;
    output.classList.toggle("unavailable", count === null);
    row.style.setProperty("--prompt-color", category?.color || detection?.colors?.[index] || "#55e8ce");
  });
}

function addPromptRow(value = "", focus = false) {
  const row = document.getElementById("prompt-row-template").content.firstElementChild.cloneNode(true);
  let input = row.querySelector(".detection-prompt");
  if (status?.detection?.model === "yolo26x") {
    const select = document.createElement("select");
    select.className = "detection-prompt";
    for (const name of ["", ...status.detection.classes]) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name || "Select COCO class";
      select.append(option);
    }
    input.replaceWith(select);
    input = select;
  }
  input.value = value;
  input.addEventListener("input", () => {
    if (row.querySelector(".target-prompt").getAttribute("aria-pressed") === "true") selectTarget(null);
    draftVersion += 1;
    promptError = "";
    updatePromptCounts(displayedDetection);
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
    if (activeModel) modelDrafts.set(activeModel, readPromptRows());
    activeModel = detection.model || "sam3.1";
    draftVersion += 1;
    setPromptRows(modelDrafts.get(activeModel) || detection.prompts || (detection.prompt ? [detection.prompt] : []));
    draftInitialized = true;
    frameDetection = loadingDetection = null;
    promptError = "";
  }
  modelSelector.value = detection?.model || "sam3.1";
  displayedDetection = detection;
  updatePromptControls();
  const name = detection?.model === "yolo26x" ? "YOLO26x · 80 COCO classes" : "SAM 3.1";
  const labels = { disabled: "Inference not configured", idle: name, loading: `Loading ${name}…`,
    compiling: `Compiling ${name}…`, validating: "Validating detector…", capturing: "Capturing SYCL graph…",
    waiting_for_camera: "Waiting for camera…" };
  const fresh = isDetectionFresh(detection);
  const fps = detectionFps(detection);
  const imageMode = detection?.image_backend === "israel-w8a8-development"
    ? "W8A8 dev (accuracy unqualified) · torch.compile"
    : detection?.image_backend === "graphs-native-sycl" ? "native image + compiled grounding" : "torch.compile";
  const loopMs = detection?.pipeline_timing?.cycle_ms;
  const loopTiming = Number.isFinite(loopMs) ? ` · ${Math.round(loopMs)} ms loop` : "";
  detectionMessage.textContent = promptError || detection?.error || (fresh
    ? `${detection.boxes.length} ${detection.boxes.length === 1 ? "box" : "boxes"} · ${fps === null ? "—" : fps.toFixed(1)} FPS · ${detection.latency_ms} ms model${loopTiming} · ${imageMode} + SYCL graphs`
    : labels[detection?.state] || "Waiting for detection…");
  detectionMessage.classList.toggle("error", Boolean(promptError || detection?.error));
  const source = fresh ? detection.frame_url : "/stream.mjpg";
  const streamed = streamFrame?.frame_url === source && streamFrame?.revision === detection?.revision;
  // Full hardware polls may be ahead of the stream. Wait for the paired JPEG
  // instead of downloading it again; disconnected streams retain HTTP fallback.
  if (fresh && detectionStreamOpen && !streamed) return;
  if (source !== feedSource && (!loadingDetection || !fresh)) {
    feedSource = source;
    loadingDetection = fresh ? {...detection, receivedAt: performance.now()} : null;
    if (!fresh) frameDetection = null;
    cameraFeed.src = streamed ? streamFrame.dataUrl : source;
  }
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
  const stop = servo.armed || Object.values(servo.axes).some(axis => axis.torque !== false);
  motorToggle.disabled = arming || stopping || (!stop && (recalibrating || !servo.ready));
  recalibrateButton.disabled = arming || stopping || recalibrating || !servo.can_recalibrate;
  recalibrateButton.textContent = recalibrating ? "Saving zeros…" : "Recalibrate zeros";
  motorToggle.setAttribute("aria-label", stop ? "Stop motors" : "Start motors");
  motorToggle.title = stop ? "Stop motors (Escape)" : "Start motors";
  motorToggle.classList.toggle("stopping", stop);
  // SVGElement does not reflect a .hidden property into the HTML attribute.
  document.getElementById("start-icon").toggleAttribute("hidden", stop);
  document.getElementById("stop-icon").toggleAttribute("hidden", !stop);
}

function showDevice(id, name, online) {
  const element = document.getElementById(id);
  element.textContent = `${name}: ${online ? "online" : "offline"}`;
  element.classList.toggle("online", online);
}

function showMessage(text, error = false) {
  message.textContent = text;
  message.hidden = !text;
  message.classList.toggle("error", error);
  messageExpiresAt = error && text ? performance.now() + 5000 : 0;
}

function render(next) {
  if (status?.detection && next.detection && (
      next.detection.revision < status.detection.revision ||
      (next.detection.revision === status.detection.revision &&
       next.detection.frame_sequence < status.detection.frame_sequence))) {
    next = {...next, detection: status.detection};
  }
  status = next;
  renderDetection(next.detection);
  const { camera, servo } = next;
  cameraFeed.parentElement.style.aspectRatio = `${camera.width} / ${camera.height}`;
  showDevice("camera-status", "Camera", camera.online);
  showDevice("servo-status", "X/Y", servo.online);
  document.getElementById("camera-offline").hidden = camera.online;
  if (!servo.armed) pendingAngles.clear();
  renderMotors();
  renderTracking();
  // A 30 FPS detection update must not erase a rejected click's error instantly.
  if (servo.error || camera.error || performance.now() >= messageExpiresAt)
    showMessage(servo.error || camera.error || "", Boolean(servo.error || camera.error));
}

async function request(path, options = {}) {
  const response = await fetch(path, {
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
  if (arming || stopping || recalibrating || !status?.servo?.ready) return;
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
  const stop = status.servo.armed || Object.values(status.servo.axes).some(axis => axis.torque !== false);
  if (stop) stopMotors();
  else startMotors();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") stopMotors();
});

async function keepMotorsAlive() {
  if (!status?.servo?.armed || stopping || keepaliveSending) return;
  keepaliveSending = true;
  try {
    await request("/api/servo/keepalive", { method: "POST" });
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    keepaliveSending = false;
  }
}
setInterval(keepMotorsAlive, 700);

async function poll() {
  if (polling) return;
  polling = true;
  try {
    await request("/api/status");
  } catch (error) {
    showMessage(error.message, true);
    updatePromptCounts(null);
    detectionFps(null);
    detectionMessage.textContent = "Detector connection lost";
    detectionMessage.classList.add("error");
    document.getElementById("tracking-overlay").toggleAttribute("hidden", true);
    boxTargets.hidden = true;
  } finally {
    polling = false;
  }
}

async function setPrompts() {
  if (detectionSending || modelSending) return;
  const submittedVersion = draftVersion;
  const prompts = readPromptRows();
  detectionSending = true;
  promptError = "";
  updatePromptControls();
  try {
    const body = await request("/api/detection/prompts", { method: "POST", body: JSON.stringify({ prompts }) });
    if (draftVersion === submittedVersion) setPromptRows(body.detection.prompts);
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
      const response = await fetch(`/api/detection/status?revision=${current.revision}&sequence=${current.frame_sequence ?? -1}`, {
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
const detectionEvents = new EventSource("/api/detection/events");
detectionEvents.onopen = () => { detectionStreamOpen = true; };
detectionEvents.onerror = () => { detectionStreamOpen = false; }; // Automatic reconnect; HTTP fallback meanwhile.
detectionEvents.onmessage = (event) => {
  const detection = JSON.parse(event.data);
  if (detection.jpeg) {
    streamFrame = {frame_url: detection.frame_url, revision: detection.revision,
      dataUrl: `data:image/jpeg;base64,${detection.jpeg}`};
    delete detection.jpeg;
  }
  if (status) render({...status, detection});
};
function readPromptRows() {
  return [...promptRows.querySelectorAll(".detection-prompt")].map(input => input.value.trim());
}

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
    modelSelector.value = activeModel;
    promptError = error.message;
    detectionMessage.textContent = error.message;
    detectionMessage.classList.add("error");
  } finally {
    modelSending = false;
    updatePromptControls();
  }
});
document.getElementById("detection-form").addEventListener("keydown", event => {
  if (event.key === "Enter" && event.target.tagName === "SELECT") {
    event.preventDefault();
    setPrompts();
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
poll();
setInterval(poll, 200);
pollDetection();
