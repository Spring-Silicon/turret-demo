"use strict";

const sliders = { x: document.getElementById("x-slider"), y: document.getElementById("y-slider") };
const motorToggle = document.getElementById("motor-toggle");
const message = document.getElementById("message");
const editingAxes = new Set();
const pendingAngles = new Map();
let movingAxis = null;
let moveTimer = null;
let arming = false;
let stopping = false;
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
let draftInitialized = false;
let draftVersion = 0;
let promptError = "";
let targetSending = false;
const cameraFeed = document.getElementById("camera-feed");
const boxTargets = document.getElementById("box-targets");
const boxButtons = new Map();
let frameDetection = null;
let loadingDetection = null;

function clickableFrame() {
  return frameDetection?.state === "running" && status?.camera?.online &&
    frameDetection.revision === status?.detection?.revision &&
    status?.detection?.state === "running" &&
    frameDetection.frame_age_ms + performance.now() - frameDetection.receivedAt <= 750;
}

async function selectInstance(selection) {
  if (targetSending || detectionSending) return;
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
      let pressedSelection = null;
      const selection = () => ({revision: frameDetection.revision,
        frame_sequence: frameDetection.frame_sequence, instance_id: id});
      button.addEventListener("pointerdown", () => { pressedSelection = clickableFrame() ? selection() : null; });
      button.addEventListener("pointercancel", () => { pressedSelection = null; });
      button.addEventListener("keydown", () => { pressedSelection = null; });
      button.addEventListener("click", () => {
        if (!clickableFrame() || button.disabled) { pressedSelection = null; return; }
        const picked = pressedSelection || selection();
        pressedSelection = null;
        return selectInstance(picked);
      });
      boxButtons.set(id, button);
      boxTargets.append(button);
    }
    const [x1, y1, x2, y2] = box.xyxy;
    Object.assign(button.style, {left: `${x1*100}%`, top: `${y1*100}%`,
      width: `${(x2-x1)*100}%`, height: `${(y2-y1)*100}%`,
      zIndex: String(Math.round(1000000*(1-(x2-x1)*(y2-y1))))});
    button.disabled = targetSending || detectionSending;
    button.setAttribute("aria-label", `Track ${box.prompt} object ${id}`);
    button.setAttribute("aria-pressed", String(status?.tracking?.instance_id === id));
    button.title = `Track this ${box.prompt}`;
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
});
cameraFeed.addEventListener("error", () => {
  loadingDetection = frameDetection = null;
  feedSource = "";
  renderTracking();
});

function updatePromptControls() {
  const rows = [...promptRows.children];
  const enabled = Boolean(status?.detection?.enabled);
  rows.forEach((row, index) => {
    row.querySelector("input").disabled = !enabled;
    row.querySelector("input").setAttribute("aria-label", `Object ${index + 1} to find`);
    row.querySelector(".remove-prompt").disabled = !enabled;
  });
  addPromptButton.disabled = !enabled || rows.length >= (status?.detection?.max_prompts || 8);
  updatePromptsButton.disabled = !enabled || detectionSending;
  updatePromptCounts(displayedDetection || status?.detection);
  updateTargetControls();
}

function updateTargetControls() {
  const target = status?.tracking?.target;
  const prompts = status?.detection?.prompts || [];
  const seen = new Set();
  for (const row of promptRows.children) {
    const prompt = row.querySelector("input").value.trim();
    const button = row.querySelector(".target-prompt");
    const applied = Boolean(prompt) && prompts.includes(prompt) && !seen.has(prompt);
    seen.add(prompt);
    const selected = applied && prompt === target;
    button.disabled = !status?.detection?.enabled || !applied || targetSending || detectionSending;
    button.setAttribute("aria-pressed", String(selected));
    const label = selected && status?.tracking?.instance_id != null ? `Track nearest ${prompt}` : selected ? `Stop tracking ${prompt}` : `Track ${prompt || "object"}`;
    button.setAttribute("aria-label", label);
    button.title = !applied ? "Update prompts before tracking this class" : label;
  }
}

async function selectTarget(target) {
  if (targetSending) return;
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
  element.textContent = tracking?.error || (target ? `${target}${tracking.instance_id != null ? " · selected object" : ""} · ${tracking.state === "uncalibrated" ? labels.uncalibrated : !status?.servo?.armed ? "press Start" : labels[tracking.state] || "waiting"}` : "");
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
    const prompt = row.querySelector("input").value.trim();
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
  const input = row.querySelector("input");
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
    const remaining = promptRows.querySelector("input");
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
  if (!draftInitialized && detection) {
    setPromptRows(detection.prompts || (detection.prompt ? [detection.prompt] : []));
    draftInitialized = true;
  }
  displayedDetection = detection;
  updatePromptControls();
  const labels = { disabled: "SAM not configured", idle: "SAM 3.1", loading: "Loading SAM 3.1…",
    compiling: "Compiling SAM 3.1…", validating: "Validating detector…", capturing: "Capturing SYCL graph…",
    waiting_for_camera: "Waiting for camera…" };
  const fresh = isDetectionFresh(detection);
  detectionMessage.textContent = promptError || detection?.error || (fresh
    ? `${detection.boxes.length} ${detection.boxes.length === 1 ? "box" : "boxes"} · ${detection.latency_ms} ms · torch.compile + SYCL graphs`
    : labels[detection?.state] || "Waiting for detection…");
  detectionMessage.classList.toggle("error", Boolean(promptError || detection?.error));
  const source = fresh ? detection.frame_url : "/stream.mjpg";
  if (source !== feedSource && (!loadingDetection || !fresh)) {
    feedSource = source;
    loadingDetection = fresh ? {...detection, receivedAt: performance.now()} : null;
    if (!fresh) frameDetection = null;
    cameraFeed.src = source;
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
  motorToggle.disabled = arming || stopping || (!stop && !servo.ready);
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
}

function render(next) {
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
  if (arming || stopping || !status?.servo?.ready) return;
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
    document.getElementById("tracking-overlay").toggleAttribute("hidden", true);
    boxTargets.hidden = true;
  } finally {
    polling = false;
  }
}

async function setPrompts() {
  if (detectionSending) return;
  const submittedVersion = draftVersion;
  const prompts = [...promptRows.querySelectorAll("input")].map((input) => input.value.trim());
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
