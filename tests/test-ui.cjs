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
    row.fields = { input: new Element(), ".prompt-count": new Element(), ".remove-prompt": new Element() };
    return row;
  }
}

const elements = new Map();
const get = (id) => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
get("prompt-row-template").content = { firstElementChild: new Element() };
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
assert.equal(get("x-degrees").textContent, "2.0°");
assert.equal(get("y-slider").min, -30);
run('status.servo.armed = true; status.servo.axes.x.torque = true; status.servo.axes.y.torque = true; renderMotors();');
assert.equal(get("x-slider").disabled, false);
assert.equal(get("motor-toggle").attributes["aria-label"], "Stop motors");
assert.equal(get("stop-icon").hidden, false);
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
console.log("validated dual degree sliders, pending edits and start/stop states");
