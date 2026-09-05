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
    if (this.fields) return this.fields[selector];
    return this.children[0]?.querySelector(selector);
  }
  querySelectorAll(selector) { return this.children.map(child => child.querySelector(selector)); }
  cloneNode() {
    const row = new Element();
    row.fields = { input: new Element(), ".prompt-count": new Element(), ".remove-prompt": new Element(), ".target-prompt": new Element() };
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
  document: { getElementById: get, querySelectorAll: () => [], addEventListener() {} },
  fetch: () => new Promise(() => {}),
  AbortSignal,
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
  run('displayedDetection.frame_age_ms = 1000; renderTracking();');
  assert.equal(get("tracking-overlay").hasAttribute("hidden"), true);
  rows()[0].fields.input.value = "edited class";
  rows()[0].fields.input.events.input();
  await Promise.resolve();
  assert.equal(run("status.tracking.target"), null); // Editing selected class cancels tracking.
  console.log("validated single-class target selector, stale overlay and nearest-box highlight");
})().catch(error => { console.error(error); process.exitCode = 1; });
