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
    draftVersion += 1;
    promptError = "";
    updatePromptCounts(displayedDetection);
  });
  row.querySelector(".remove-prompt").addEventListener("click", () => {
    draftVersion += 1;
    promptError = "";
    if (promptRows.children.length === 1) input.value = "";
    else row.remove();
    updatePromptControls();
    const remaining = promptRows.querySelector("input");
    remaining.focus();
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
  if (source !== feedSource) {
    feedSource = source;
    document.getElementById("camera-feed").src = source;
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
  showDevice("camera-status", "Camera", camera.online);
  showDevice("servo-status", "X/Y", servo.online);
  document.getElementById("camera-offline").hidden = camera.online;
  if (!servo.armed) pendingAngles.clear();
  renderMotors();
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
