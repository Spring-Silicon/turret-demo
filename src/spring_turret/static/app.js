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
const promptInput = document.getElementById("detection-prompt");
const detectionMessage = document.getElementById("detection-status");

function renderDetection(detection) {
  if (displayedDetection && detection &&
      (detection.revision < displayedDetection.revision ||
       (detection.revision === displayedDetection.revision && detection.state === "running" &&
        detection.frame_sequence < displayedDetection.frame_sequence))) return;
  if (!displayedDetection && detection?.prompt && document.activeElement !== promptInput) {
    promptInput.value = detection.prompt;
  }
  displayedDetection = detection;
  promptInput.disabled = !detection?.enabled;
  document.getElementById("detect-button").disabled = !detection?.enabled || detectionSending;
  document.getElementById("clear-detection").disabled = !detection?.enabled || detectionSending;
  const labels = { disabled: "SAM not configured", idle: "SAM 3.1", loading: "Loading SAM 3.1…",
    compiling: "Compiling SAM 3.1…", validating: "Validating detector…", capturing: "Capturing SYCL graph…",
    waiting_for_camera: "Waiting for camera…" };
  const fresh = detection?.state === "running" &&
    detection.frame_age_ms < Math.max(5000, 2 * detection.latency_ms + 1000);
  detectionMessage.textContent = detection?.error || (fresh
    ? `${detection.boxes.length} boxes · ${detection.latency_ms} ms · torch.compile + SYCL graphs`
    : labels[detection?.state] || "Waiting for detection…");
  detectionMessage.classList.toggle("error", Boolean(detection?.error));
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
  } finally {
    polling = false;
  }
}

async function setPrompt(prompt) {
  detectionSending = true;
  try {
    await request("/api/detection/prompt", { method: "POST", body: JSON.stringify({ prompt }) });
  } catch (error) {
    detectionMessage.textContent = error.message;
    detectionMessage.classList.add("error");
  } finally {
    detectionSending = false;
  }
}
document.getElementById("detection-form").addEventListener("submit", (event) => {
  event.preventDefault();
  setPrompt(promptInput.value.trim());
});
document.getElementById("clear-detection").addEventListener("click", () => {
  promptInput.value = "";
  setPrompt("");
});
poll();
setInterval(poll, 200);
