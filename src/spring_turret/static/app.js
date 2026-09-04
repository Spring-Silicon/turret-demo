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
  if (event.key === " " && status.servo.armed) {
    event.preventDefault();
    action("/api/servo/center");
  }
  if (event.key === "ArrowLeft" && status.servo.armed) sendPosition(Number(slider.value) - 11);
  if (event.key === "ArrowRight" && status.servo.armed) sendPosition(Number(slider.value) + 11);
});

async function poll() {
  try {
    await request("/api/status");
  } catch (error) {
    showMessage(error.message, true);
  }
}

window.addEventListener("load", () => {
  window.setTimeout(() => {
    document.getElementById("camera-feed").src = "/stream.mjpg";
  }, 0);
});
poll();
setInterval(poll, 1000);
