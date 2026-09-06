"use strict";

// Exercise the actual client functions without a browser or GPU.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor() {
    this.children = [];
    this.events = {};
    this.value = "";
    this.attributes = {};
    this.style = { setProperty() {} };
    this.classList = { toggle() {}, add() {} };
  }
  addEventListener(name, action) { this.events[name] = action; }
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) { return this.attributes[name] ?? null; }
  hasAttribute(name) { return Object.hasOwn(this.attributes, name); }
  toggleAttribute(name, force) {
    const present = force ?? !this.hasAttribute(name);
    if (present) this.attributes[name] = "";
    else delete this.attributes[name];
    return present;
  }
  append(child) { child.parent = this; this.children.push(child); }
  replaceChildren() { this.children = []; }
  focus() {}
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  querySelector(selector) {
    if (this.fields) return this.fields[selector === ".detection-prompt" ? "input" : selector];
    return this.children[0]?.querySelector(selector);
  }
  querySelectorAll(selector) { return this.children.map(child => child.querySelector(selector)); }
  cloneNode() {
    const row = new Element();
    row.fields = { input: new Element(), ".prompt-count": new Element(), ".remove-prompt": new Element(), ".target-prompt": new Element() };
    row.fields.input.replaceWith = replacement => { row.fields.input = replacement; };
    return row;
  }
}

const elements = new Map();
const get = (id) => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
get("prompt-row-template").content = { firstElementChild: new Element() };
// Match the SVG markup: .hidden is not reflected on SVGElement, unlike HTML.
get("stop-icon").setAttribute("hidden", "");
const context = vm.createContext({
  document: { getElementById: get, createElement: () => new Element(), querySelectorAll: () => [], addEventListener() {} },
  performance: {now: () => 1000},
  fetch: () => new Promise(() => {}),
  AbortSignal,
  window: {confirm: () => false},
  setTimeout() { return 1; },
  setInterval() {},
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/spring_turret/static/app.js"), "utf8"), context);
const run = (code) => vm.runInContext(code, context);
const rows = () => get("prompt-rows").children;
const counts = () => rows().map(row => row.fields[".prompt-count"].textContent);
run(`status = { detection: { enabled: true, state: "running", revision: 1,
  frame_age_ms: 200, latency_ms: 150, max_prompts: 8, prompts: ["finger", "person"],
  categories: [{prompt: "finger", count: 9, color: "green"}, {prompt: "person", count: 2, color: "orange"}]
}}; displayedDetection = status.detection;
setPromptRows(["finger", "person"]);`);
assert.deepEqual(counts(), ["9", "2"]);
assert.equal(rows()[0].fields[".prompt-count"].attributes["aria-label"], "finger: 9 detected");
run('setPromptRows(["person", "finger"]);');
assert.deepEqual(counts(), ["2", "9"]); // Not tied to original row indexes.
rows()[0].fields.input.value = "unapplied object";
rows()[0].fields.input.events.input();
assert.deepEqual(counts(), ["—", "9"]);
run('displayedDetection.categories[0].count = 0; updatePromptCounts(displayedDetection);');
assert.deepEqual(counts(), ["—", "0"]);
run('displayedDetection.state = "loading"; updatePromptCounts(displayedDetection);');
assert.deepEqual(counts(), ["—", "—"]);
run('displayedDetection.state = "running"; displayedDetection.frame_age_ms = 6000; updatePromptCounts(displayedDetection);');
assert.deepEqual(counts(), ["—", "—"]);
run('updatePromptCounts(null);');
assert.deepEqual(counts(), ["—", "—"]);
get("add-prompt").events.click();
assert.equal(rows().length, 3);
assert.equal(rows()[2].fields.input.value, "");
assert.equal(counts()[2], "—");
while (rows().length < 8) get("add-prompt").events.click();
assert.equal(get("add-prompt").disabled, true);
get("add-prompt").events.click();
assert.equal(rows().length, 8);
rows()[7].fields[".remove-prompt"].events.click();
assert.equal(rows().length, 7);
assert.equal(get("add-prompt").disabled, false);
console.log("validated per-category counts, draft/stale states and single add button");

run(`globalThis.fpsDetection = {state: "running", model: "sam3.1", revision: 1,
  frame_sequence: 100, latency_ms: 1, frame_age_ms: 100};`);
assert.equal(run('detectionFps(fpsDetection, 0)'), null);
assert.equal(run('detectionFps({...fpsDetection, frame_sequence: 102}, 200)'), null);
assert.equal(run('detectionFps({...fpsDetection, frame_sequence: 106}, 600)'), 10);
assert.equal(run('detectionFps({...fpsDetection, frame_sequence: 106}, 800)'), 7.5); // Duplicate poll, not a new frame.
assert.equal(run('detectionFps({...fpsDetection, frame_sequence: 106}, 3000)'), 0); // Stalled worker.
assert.equal(run('detectionFps({...fpsDetection, revision: 2, frame_sequence: 107}, 3200)'), null);
assert.equal(run('detectionFps({...fpsDetection, model: "yolo26x", revision: 3, frame_sequence: 108}, 3400)'), null);
assert.equal(run('detectionFps({...fpsDetection, model: "yolo26x", revision: 3, frame_sequence: 118}, 4400)'), 10);
assert.equal(run('detectionFps({...fpsDetection, state: "compiling"}, 4600)'), null);
assert.equal(run('fpsSamples.length'), 0);
assert.equal(run('detectionFps({...fpsDetection, frame_age_ms: 6000}, 4800)'), null);
assert.equal(run('detectionFps(fpsDetection, 5000)'), null);
assert.equal(run('detectionFps({...fpsDetection, frame_sequence: 99}, 5600)'), null); // Restarted sequence.
run('detectionFps(null);');
console.log("validated measured detection FPS, skipped/duplicate polls, stalls and model/prompt resets");

run(`status.servo = {online: true, ready: true, armed: false, axes: {
  x: {degrees: 2, goal_degrees: null, min_degrees: -45, max_degrees: 45, torque: false},
  y: {degrees: -3, goal_degrees: null, min_degrees: -30, max_degrees: 30, torque: false}
}}; renderMotors();`);
assert.equal(get("x-slider").disabled, true);
assert.equal(get("motor-toggle").attributes["aria-label"], "Start motors");
assert.equal(get("motor-toggle").disabled, false);
assert.equal(get("start-icon").hasAttribute("hidden"), false);
assert.equal(get("stop-icon").hasAttribute("hidden"), true);
assert.equal(get("x-degrees").textContent, "2.0°");
assert.equal(get("y-slider").min, -30);
run('status.servo.axes.y.min_degrees = -90; status.servo.axes.y.max_degrees = 90; renderMotors();');
assert.equal(get("y-slider").min, -90);
assert.equal(get("y-slider").max, 90);
run('status.servo.armed = true; status.servo.axes.x.torque = true; status.servo.axes.y.torque = true; renderMotors();');
assert.equal(get("x-slider").disabled, false);
assert.equal(get("motor-toggle").attributes["aria-label"], "Stop motors");
assert.equal(get("start-icon").hasAttribute("hidden"), true);
assert.equal(get("stop-icon").hasAttribute("hidden"), false);
get("x-slider").value = 12.5;
get("x-slider").events.input();
assert.equal(get("x-degrees").textContent, "12.5°");
assert.equal(run('pendingAngles.get("x")'), 12.5);
get("y-slider").value = -10;
get("y-slider").events.input();
assert.equal(run('pendingAngles.get("y")'), -10);
run('renderMotors();');
assert.equal(Number(get("x-slider").value), 12.5); // A poll cannot undo a pending drag.
run('movingAxis = "x"; renderMotors();');
assert.equal(get("motor-toggle").disabled, false); // Stop remains available during moves.
run('status.servo.online = false; status.servo.armed = false; status.servo.axes.x.torque = null; renderMotors();');
assert.equal(get("motor-toggle").attributes["aria-label"], "Stop motors");
assert.equal(get("motor-toggle").disabled, false); // Unknown torque must not hide Stop.
assert.equal(get("x-slider").disabled, true);
run('status.servo.online = true; status.servo.axes.x.torque = false; status.servo.axes.y.torque = false; renderMotors();');
assert.equal(get("motor-toggle").attributes["aria-label"], "Start motors");
assert.equal(get("start-icon").hasAttribute("hidden"), false);
assert.equal(get("stop-icon").hasAttribute("hidden"), true);
console.log("validated dual degree sliders, pending edits and start/stop states");

(async () => {
  run(`status.camera = {online: true, width: 1280, height: 720};
    status.detection = {enabled: true, state: "running", revision: 4, frame_sequence: 10,
      frame_age_ms: 200, latency_ms: 150, prompts: ["cup", "bottle"], categories: [],
      boxes: [{prompt: "cup", xyxy: [.56, .46, .64, .54], score: .9},
              {prompt: "cup", xyxy: [.46, .61, .54, .69], score: .9},
              {prompt: "bottle", xyxy: [.46, .46, .54, .54], score: .9}],
      frame_url: "/test.jpg"};
    displayedDetection = status.detection;
    frameDetection = {...status.detection, receivedAt: 1000};
    status.tracking = {target: null, state: "off"};
    setPromptRows(["cup", "bottle", "new class"]);
    globalThis.targetRequests = [];
    request = async (path, options) => {
      targetRequests.push([path, JSON.parse(options.body)]);
      status.tracking = {target: JSON.parse(options.body).target, state: "waiting"};
      renderTracking();
      return status;
    };`);
  assert.equal(rows()[2].fields[".target-prompt"].disabled, true);
  await rows()[0].fields[".target-prompt"].events.click();
  assert.equal(rows()[0].fields[".target-prompt"].attributes["aria-pressed"], "true");
  assert.equal(rows()[1].fields[".target-prompt"].attributes["aria-pressed"], "false");
  assert.equal(get("tracking-overlay").hasAttribute("hidden"), false);
  assert.equal(get("frame-center").hasAttribute("hidden"), false);
  assert.equal(get("tracked-box").attributes.y, 610); // Pixel distance selects the vertical cup.
  await rows()[1].fields[".target-prompt"].events.click();
  assert.equal(rows()[0].fields[".target-prompt"].attributes["aria-pressed"], "false");
  assert.equal(rows()[1].fields[".target-prompt"].attributes["aria-pressed"], "true");
  await rows()[1].fields[".target-prompt"].events.click();
  assert.equal(run("status.tracking.target"), null);
  assert.equal(get("tracking-overlay").hasAttribute("hidden"), true);
  assert.equal(run('targetRequests.every(([path]) => path === "/api/tracking/target")'), true); // Never auto-start.
  await run('selectTarget("cup");');
  run('setPromptRows(["cup", "cup", "bottle"]);');
  assert.equal(rows().filter(row => row.fields[".target-prompt"].attributes["aria-pressed"] === "true").length, 1);
  assert.equal(rows()[1].fields[".target-prompt"].disabled, true);
  run('frameDetection.frame_age_ms = 1000; renderTracking();');
  assert.equal(get("tracking-overlay").hasAttribute("hidden"), true);
  rows()[0].fields.input.value = "edited class";
  rows()[0].fields.input.events.input();
  await Promise.resolve();
  assert.equal(run("status.tracking.target"), null); // Editing selected class cancels tracking.
  console.log("validated single-class target selector, stale overlay and nearest-box highlight");

  run(`status.detection.frame_sequence = 11;
    status.detection.boxes = [{prompt: "cup", xyxy: [.1,.2,.3,.4], score: .9, instance_id: 7},
      {prompt: "cup", xyxy: [.45,.45,.55,.55], score: .9, instance_id: 8}];
    status.servo.armed = false;
    frameDetection = {...status.detection, receivedAt: 1000};
    status.tracking = {target: "cup", instance_id: 7, state: "stopped"};
    renderTracking();
    request = async (path, options) => {
      const body = JSON.parse(options.body);
      targetRequests.push([path, body]);
      status.tracking = {target: "cup", instance_id: body.instance_id, state: "stopped"};
      return status;
    };`);
  assert.equal(get("tracked-box").attributes.x, 100); // Selected, not the centered cup.
  assert.equal(get("box-targets").children.length, 2);
  assert.equal(get("box-targets").hidden, false);
  const clickBox = get("box-targets").children[0];
  clickBox.events.pointerdown();
  run('status.detection = {...status.detection, frame_sequence: 12};');
  await clickBox.events.click();
  assert.equal(run('JSON.stringify(targetRequests.at(-1))'),
    '["/api/tracking/instance",{"revision":4,"frame_sequence":11,"instance_id":7}]');
  assert.equal(run('status.servo.armed'), false);
  await get("box-targets").children[1].events.click();
  assert.equal(run('status.tracking.instance_id'), 8);
  const before = run('targetRequests.length');
  run('frameDetection.receivedAt = 0; renderTracking();');
  await clickBox.events.click();
  assert.equal(run('targetRequests.length'), before);
  assert.equal(get("box-targets").hidden, true);
  run(`loadingDetection = {...status.detection, frame_sequence: 12, receivedAt: 1000};
    frameDetection = null;`);
  get("camera-feed").events.load();
  assert.equal(run('frameDetection.frame_sequence'), 12);
  assert.equal(run('loadingDetection'), null);
  get("camera-feed").events.error();
  assert.equal(get("box-targets").hidden, true);
  console.log("validated clickable instances, exact displayed-frame selection, stale clicks and no auto-start");
  run(`displayedDetection = null; activeModel = null; draftInitialized = false;
    status.detection = {enabled: true, model: "sam3.1", revision: 10, state: "idle", prompts: ["face"]};
    renderDetection(status.detection);`);
  rows()[0].fields.input.value = "unapplied SAM text";
  run(`status.detection = {enabled: true, model: "yolo26x", revision: 11, state: "loading",
    prompts: ["person"], classes: ["person", "cup"], models: [{id: "sam3.1", available: true}, {id: "yolo26x", available: true}]};
    renderDetection(status.detection);`);
  assert.equal(get("detection-model").value, "yolo26x");
  assert.equal(rows()[0].fields.input.value, "person");
  assert.deepEqual(rows()[0].fields.input.children.map(option => option.value), ["", "person", "cup"]);
  assert.equal(run("frameDetection"), null);
  rows()[0].fields.input.value = "cup";
  run(`status.detection = {...status.detection, model: "sam3.1", revision: 12, prompts: ["face"], classes: null};
    renderDetection(status.detection);`);
  assert.equal(rows()[0].fields.input.value, "unapplied SAM text");
  run(`request = async (path, options) => {
    targetRequests.push([path, JSON.parse(options.body)]);
    status.detection = {...status.detection, model: JSON.parse(options.body).model,
      revision: 13, prompts: ["person"], classes: ["person", "cup"]};
    renderDetection(status.detection);
    return status;
  };`);
  get("detection-model").value = "yolo26x";
  await get("detection-model").events.change();
  assert.equal(run("JSON.stringify(targetRequests.at(-1))"), '["/api/detection/model",{"model":"yolo26x"}]');
  assert.equal(rows()[0].fields.input.value, "cup"); // YOLO draft preserved too.
  assert.equal(run("status.servo.armed"), false);
  run(`globalThis.submissions = 0; setPrompts = () => { submissions++; };`);
  get("detection-form").events.keydown({key: "Enter", target: {tagName: "SELECT"}, preventDefault() {}});
  assert.equal(run("submissions"), 1);
  console.log("validated model selection, fixed class choices, independent drafts and Enter submission");
  run(`globalThis.zeroRequests = [];
    status.servo.can_recalibrate = true; status.servo.ready = false;
    status.servo.armed = false; renderMotors();
    request = async (path, options) => {zeroRequests.push([path, options]); return status;};`);
  assert.equal(get("recalibrate").disabled, false); // Outside range is precisely why zeros may need resetting.
  await get("recalibrate").events.click();
  assert.equal(run('zeroRequests.length'), 0); // Cancel leaves calibration untouched.
  run('window.confirm = () => true;');
  await get("recalibrate").events.click();
  assert.equal(run('JSON.stringify(zeroRequests)'), '[["/api/servo/recalibrate",{"method":"POST","body":"{}"}]]');
  assert.equal(run('status.servo.armed'), false);
  run('status.servo.can_recalibrate = false; renderMotors();');
  assert.equal(get("recalibrate").disabled, true);
  await get("recalibrate").events.click();
  assert.equal(run('zeroRequests.length'), 1);
  run('status.servo.can_recalibrate = true; recalibrating = true; renderMotors();');
  assert.equal(get("recalibrate").disabled, true);
  assert.equal(get("motor-toggle").disabled, true);
  await get("recalibrate").events.click();
  assert.equal(run('zeroRequests.length'), 1);
  console.log("validated zero calibration confirmation, no auto-arm and pending-request exclusion");
})().catch(error => { console.error(error); process.exitCode = 1; });
