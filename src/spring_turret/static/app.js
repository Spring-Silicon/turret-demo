"use strict";

const slider = document.getElementById("position-slider");
const degrees = document.getElementById("position-degrees");
const raw = document.getElementById("position-raw");
const message = document.getElementById("message");
const armButton = document.getElementById("arm-button");
const stopButton = document.getElementById("stop-button");
const centerButton = document.getElementById("center-button");
const jogButtons = [...document.querySelectorAll("[data-step]")];
let status = null;
let sending = false;
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

function showPosition(position, servo) {
  const bounded = Math.max(servo.min_position, Math.min(servo.max_position, position));
  slider.min = servo.min_position;
  slider.max = servo.max_position;
  slider.value = bounded;
  degrees.textContent = (((bounded - servo.center_position) * 360) / 4096).toFixed(1);
  raw.textContent = bounded;
}

function showDevice(id, name, online) {
  const element = document.getElementById(id);
  element.textContent = `${name}: ${online ? "online" : "offline"}`;
  element.classList.toggle("online", online);
}

function showMessage(text, error = false) {
  message.textContent = text;
  message.classList.toggle("error", error);
}

function render(next) {
  status = next;
  renderDetection(next.detection);
  const { camera, servo } = next;
  showDevice("camera-status", "Camera", camera.online);
  showDevice("servo-status", "Servo", servo.online);
  document.getElementById("camera-offline").hidden = camera.online;
  if (!sending) showPosition(servo.position ?? servo.center_position, servo);

  const canMove = servo.online && servo.armed && !sending;
  slider.disabled = !canMove;
  centerButton.disabled = !canMove;
  jogButtons.forEach((button) => { button.disabled = !canMove; });
  armButton.disabled = !servo.online || servo.armed || sending;
  stopButton.disabled = !servo.online || sending;
  armButton.textContent = servo.armed ? "Armed" : "Arm";

  if (servo.error) showMessage(servo.error, true);
  else if (camera.error) showMessage(camera.error, true);
  else showMessage(servo.armed ? "Servo armed" : "Servo disarmed");
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  render(body);
  return body;
}

async function action(path, options = {}) {
  if (sending) return;
  sending = true;
  try {
    await request(path, { method: "POST", ...options });
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    sending = false;
  }
}

async function sendPosition(position) {
  if (!status) return;
  const servo = status.servo;
  const bounded = Math.max(servo.min_position, Math.min(servo.max_position, Math.round(position)));
  showPosition(bounded, servo);
  await action("/api/servo/position", { body: JSON.stringify({ position: bounded }) });
}

slider.addEventListener("input", () => {
  if (status) showPosition(Number(slider.value), status.servo);
});
slider.addEventListener("change", () => sendPosition(Number(slider.value)));
jogButtons.forEach((button) => button.addEventListener("click", () => {
  sendPosition(Number(slider.value) + Number(button.dataset.step));
}));
armButton.addEventListener("click", () => action("/api/servo/arm"));
stopButton.addEventListener("click", () => action("/api/servo/disable"));
centerButton.addEventListener("click", () => action("/api/servo/center"));

document.addEventListener("keydown", (event) => {
  if (!status || sending) return;
  if (event.key === "Escape") action("/api/servo/disable");
  if (event.target.matches("input, textarea, button") || event.target.isContentEditable) return;
  if (event.key === " " && status.servo.armed) {
    event.preventDefault();
    action("/api/servo/center");
  }
  if (event.key === "ArrowLeft" && status.servo.armed) sendPosition(Number(slider.value) - 11);
  if (event.key === "ArrowRight" && status.servo.armed) sendPosition(Number(slider.value) + 11);
});

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
