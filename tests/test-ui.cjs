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
    this.style = { setProperty(name, value) { this[name] = value; } };
    this.naturalWidth = 1280; this.naturalHeight = 720;
    this.classList = { toggle() {}, add() {} };
  }
  addEventListener(name, action) { this.events[name] = action; }
  getContext() {
    return this.context ||= {getImageData: () => ({data: new Uint8ClampedArray(this.pixelData || this.width * this.height * 4)}),
    putImageData: pixels => {this.pixelData = new Uint8ClampedArray(pixels.data);},
    drawImage: image => {
      if (image.failDraw) throw new Error('draw failed');
      if (this.context.globalCompositeOperation === 'copy') this.raster = [];
      this.raster ||= [];
      this.raster.push(...(image.raster || [image.src]));
      this.src = this.raster[0]; // Test-only trace of the committed JPEG pixels.
      this.pixelData = image.pixelData;
    }};
  }
  getBoundingClientRect() { return this.rect || {left: 0, top: 0, width: 1280, height: 720}; }
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
  setPointerCapture(id) { this.capturedPointer = id; }
  releasePointerCapture(id) { if (this.capturedPointer === id) this.capturedPointer = null; }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  replaceWith(element) {
    element.parentElement = this.parentElement;
    if (this.ownerMap) {
      element.ownerMap = this.ownerMap;
      element.ownerKey = this.ownerKey;
      this.ownerMap.set(this.ownerKey, element);
    }
  }
  querySelector(selector) {
    if (this.fields) return this.fields[selector === ".detection-prompt" ? "input" : selector];
    return this.children[0]?.querySelector(selector);
  }
  querySelectorAll(selector) { return this.children.map(child => child.querySelector(selector)); }
  cloneNode() {
    const row = new Element();
    row.fields = { input: new Element(), ".remove-prompt": new Element(), ".target-prompt": new Element() };
    row.fields.input.replaceWith = replacement => { row.fields.input = replacement; };
    return row;
  }
}

const elements = new Map();
const documentEvents = {};
const get = (id) => {
  if (["camera-status", "servo-status", "camera-offline"].includes(id)) return null;
  if (!elements.has(id)) {
    const element = new Element(); element.ownerMap = elements; element.ownerKey = id;
    elements.set(id, element);
  }
  return elements.get(id);
};
get("prompt-row-template").content = { firstElementChild: new Element() };
// Match the SVG markup: .hidden is not reflected on SVGElement, unlike HTML.
get("stop-icon").setAttribute("hidden", "");
const context = vm.createContext({
  document: { getElementById: get, createElement: () => new Element(), querySelectorAll: () => [], addEventListener(name, handler) {documentEvents[name] = handler;} },
  performance: {now: () => 1000},
  fetch: () => new Promise(() => {}),
  AbortSignal,
  EventSource: class {constructor(url) {this.url = url;}},
  window: {confirm: () => false, localStorage: {getItem: () => "25"}},
  setTimeout() { return 1; },
  setInterval() {},
});
const appSource = fs.readFileSync(path.join(__dirname, "../src/spring_turret/static/app.js"), "utf8");
assert.doesNotMatch(appSource, /keepMotorsAlive|\/api\/servo\/keepalive/);
// Exercise the exact mounted implementation, exposing closure internals only
// inside this VM so the existing fine-grained state-transition tests remain.
const mountedSource = appSource.split('function mountTurret(document, apiPrefix = "", options = {}) {')[1]
  .split('\n}\n\n// Preserve the standalone page')[0];
vm.runInContext('const apiPrefix = "", options = {};\n' + mountedSource.replace('\nreturn {\n  command:', '\nglobalThis.panelClient = {\n  command:'), context);
const run = (code) => vm.runInContext(code, context);
const completeFrame = () => {
  const pending = run('loadingFrame');
  assert.ok(pending);
  pending.image.events.load();
  if (pending.mask) pending.mask.events.load();
  if (run('frameDetection') === pending.detection) {
    assert.equal(get('feed-loading').hidden, true);
    assert.equal(get('camera-feed').hidden, false);
    assert.equal(get('fps-counter').hidden, false);
  }
};
assert.equal(get('feed-loading').hidden, false);
assert.equal(get('fps-counter').hidden, true);
const rows = () => get("prompt-rows").children;
assert.match(fs.readFileSync(path.join(__dirname,'../src/spring_turret/static/index.html'),'utf8'),
  /<option value="sam3\.1-mask">SAM 3\.1 Mask<\/option>/);
assert.doesNotMatch(fs.readFileSync(path.join(__dirname, '../src/spring_turret/static/index.html'), 'utf8'), /yolo/i);
assert.equal(run(`hasTrackingMask({model:'sam3.1-mask',mask_detection:true,temporal_tracking:false,
  mask_overlay:{format:'indexed-png',png:'AAAA'}})`),true);
assert.equal(run(`hasTrackingMask({model:'sam3.1',mask_detection:true,
  mask_overlay:{format:'indexed-png',png:'AAAA'}})`),false);
const colors = () => rows().map(row => row.style["--prompt-color"]);
for (const id of ["camera-feed", "box-targets", "tracking-overlay"])
  assert.equal(get(id).style["--view-zoom"], undefined);
assert.doesNotMatch(fs.readFileSync(path.join(__dirname, "../src/spring_turret/static/index.html"), "utf8"), /fov-slider/);
assert.match(fs.readFileSync(path.join(__dirname, "../src/spring_turret/static/app.css"), "utf8"), /object-fit: contain/);
console.log("validated uncropped full-frame display regardless of old saved FOV");
assert.match(fs.readFileSync(path.join(__dirname, "../src/spring_turret/static/index.html"), "utf8"),
  /<option value="sam3\.1-tracking">SAM 3\.1 Tracking<\/option>/);
run(`status = { detection: { enabled: true, state: "running", revision: 1,
  frame_age_ms: 200, latency_ms: 150, max_prompts: 8, prompts: ["finger", "person"],
  categories: [{prompt: "finger", count: 9, color: "green"}, {prompt: "person", count: 2, color: "orange"}]
}}; displayedDetection = status.detection;
setPromptRows(["finger", "person"]);`);
assert.deepEqual(colors(), ["green", "orange"]);
run('setPromptRows(["person", "finger"]);');
assert.deepEqual(colors(), ["orange", "green"]); // Not tied to original row indexes.
rows()[0].fields.input.value = "unapplied object";
rows()[0].fields.input.events.input();
assert.deepEqual(colors(), ["#55e8ce", "green"]);
run('updatePromptColors(null);');
assert.deepEqual(colors(), ["#55e8ce", "#55e8ce"]);
get("add-prompt").events.click();
assert.equal(rows().length, 3);
assert.equal(rows()[2].fields.input.value, "");
while (rows().length < 8) get("add-prompt").events.click();
assert.equal(get("add-prompt").disabled, true);
get("add-prompt").events.click();
assert.equal(rows().length, 8);
rows()[7].fields[".remove-prompt"].events.click();
assert.equal(rows().length, 7);
assert.equal(get("add-prompt").disabled, false);
console.log("validated prompt colors, draft edits and single add button");

run(`globalThis.fpsDetection = {state: "running", model: "sam3.1", revision: 1,
  frame_sequence: 100, latency_ms: 1, frame_age_ms: 100};`);
assert.equal(run('detectionFps(fpsDetection, 0)'), null);
assert.equal(run('detectionFps({...fpsDetection, frame_sequence: 102}, 200)'), null);
assert.equal(run('detectionFps({...fpsDetection, frame_sequence: 106}, 600)'), 10);
assert.equal(run('detectionFps({...fpsDetection, frame_sequence: 106}, 800)'), 7.5); // Duplicate poll, not a new frame.
assert.equal(run('detectionFps({...fpsDetection, frame_sequence: 106}, 3000)'), 0); // Stalled worker.
assert.equal(run('detectionFps({...fpsDetection, revision: 2, frame_sequence: 107}, 3200)'), null);
assert.equal(run('detectionFps({...fpsDetection, model: "sam3.1-mask", revision: 3, frame_sequence: 108}, 3400)'), null);
assert.equal(run('detectionFps({...fpsDetection, model: "sam3.1-mask", revision: 3, frame_sequence: 118}, 4400)'), 10);
assert.equal(run('detectionFps({...fpsDetection, state: "compiling"}, 4600)'), null);
assert.equal(run('fpsSamples.length'), 0);
assert.equal(run('detectionFps({...fpsDetection, frame_age_ms: 6000}, 4800)'), null);
assert.equal(run('detectionFps(fpsDetection, 5000)'), null);
assert.equal(run('detectionFps({...fpsDetection, frame_sequence: 99}, 5600)'), null); // Restarted sequence.
run('detectionFps(null);');
console.log("validated measured detection FPS, skipped/duplicate polls, stalls and model/prompt resets");
run('renderFps(14.36);');
assert.equal(get('fps-value').textContent, '14.4');
assert.equal(get('fps-counter').getAttribute('aria-label'), '14.4 frames per second');
run('renderFps(0);');
assert.equal(get('fps-value').textContent, '0.0');
run('renderFps(null);');
assert.equal(get('fps-value').textContent, '—');
assert.equal(get('fps-counter').getAttribute('aria-label'), 'Frames per second unavailable');

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
run(`status.servo = {...status.servo, run_requested:true, armed:false, online:false, can_start:true, recovering:true}; renderMotors();`);
assert.equal(get('motor-toggle').attributes['aria-label'], 'Stop motors');
assert.equal(get('motor-toggle').disabled, false);
run('delete status.servo.run_requested; delete status.servo.recovering; status.servo.online = true;');
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
  assert.equal(get("tracking-overlay").hasAttribute("hidden"), false);
  rows()[0].fields.input.value = "edited class";
  rows()[0].fields.input.events.input();
  await Promise.resolve();
  assert.equal(run("status.tracking.target"), null); // Editing selected class cancels tracking.
  console.log("validated single-class target selector, delayed overlay and nearest-box highlight");
  assert.equal(run(`nearestDisplayedBox({boxes:[
    {prompt:'cup',xyxy:[.45,.45,.55,.55],score:.9,instance_id:1,mask_centroid:[.8,.8]},
    {prompt:'cup',xyxy:[.3,.3,.6,.6],score:.9,instance_id:2,mask_centroid:[.5,.5]}]},'cup').instance_id`),2);
  assert.equal(run(`nearestDisplayedBox({boxes:[
    {prompt:'cup',xyxy:[.45,.45,.55,.55],score:.9,mask_centroid:null}]},'cup')`),null);
  console.log("validated centroid-based preview and empty-mask exclusion");

  run(`status.tracking = {target:'cup',instance_id:1,continuity:'hold-reacquire',state:'holding'};
    status.servo.armed = true;
    frameDetection.boxes = [{prompt:'cup',xyxy:[.45,.45,.55,.55],score:.9,instance_id:1}];
    renderTracking();`);
  assert.equal(run('trackedBox(frameDetection)'), null);
  assert.equal(get('tracking-status').hidden, true);
  assert.equal(get('tracking-status').textContent, '');
  run(`status.tracking.state='reacquiring'; renderTracking();`);
  assert.equal(run('trackedBox(frameDetection)'), null);
  assert.equal(get('tracking-status').hidden, true);
  run(`status.tracking.state='centered'; renderTracking();`);
  assert.equal(run('trackedBox(frameDetection).instance_id'), 1);
  console.log("validated persistent-target hold/reacquire without status clutter or false red highlights");

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
  run('frameDetection.client_overlay = true; renderBoxTargets();');
  assert.equal(clickBox.children[0].textContent, "cup 90%");
  assert.equal(clickBox.style.left, "70%"); // Original x=.1–.3 appears at mirrored x=.7–.9.
  assert.ok(Math.abs(parseFloat(clickBox.style.width) - 20) < 1e-10);
  run('frameDetection.client_overlay = false; renderBoxTargets();');
  assert.equal(clickBox.children[0].textContent, ""); // No duplicate labels on baked overlays.
  const pointer = {button: 0, isPrimary: true, pointerId: 1, clientX: 100, clientY: 200};
  clickBox.events.pointerdown(pointer);
  run('status.detection = {...status.detection, frame_sequence: 12};');
  await get("box-targets").events.pointerup(pointer);
  assert.equal(run('JSON.stringify(targetRequests.at(-1))'),
    '["/api/tracking/instance",{"revision":4,"frame_sequence":11,"instance_id":7}]');
  assert.equal(run('status.servo.armed'), false);
  await get("box-targets").children[1].events.click({detail: 0});
  assert.equal(run('status.tracking.instance_id'), 8);
  const beforeGesture = run('targetRequests.length');
  await clickBox.events.click({detail: 1}); // No duplicate after pointerup.
  clickBox.events.pointerdown({...pointer, button: 2});
  await get("box-targets").events.pointerup(pointer);
  clickBox.events.pointerdown(pointer);
  get("box-targets").events.pointercancel(pointer);
  await get("box-targets").events.pointerup(pointer);
  clickBox.events.pointerdown(pointer);
  await get("box-targets").events.pointerup({...pointer, clientX: 150}); // Drag, not click.
  assert.equal(run('targetRequests.length'), beforeGesture);
  clickBox.events.pointerdown(pointer);
  run(`frameDetection = {...frameDetection, frame_sequence: 13,
    boxes: frameDetection.boxes.map(b => ({...b, instance_id: b.instance_id + 10}))}; renderBoxTargets();`);
  assert.equal(get("box-targets").children.includes(clickBox), false);
  assert.equal(get("box-targets").capturedPointer, 1); // Stable overlay survives button replacement.
  await get("box-targets").events.pointerup(pointer);
  assert.equal(run('JSON.stringify(targetRequests.at(-1))'),
    '["/api/tracking/instance",{"revision":4,"frame_sequence":11,"instance_id":7}]');
  assert.equal(get("box-targets").capturedPointer, null);
  const freshBox = get("box-targets").children[0];
  freshBox.events.pointerdown(pointer);
  run('frameDetection.frame_age_ms = 10000; frameDetection.receivedAt = 0;');
  await get("box-targets").events.pointerup(pointer);
  assert.equal(run('targetRequests.length'), beforeGesture + 2); // No pointer age timeout.
  freshBox.events.pointerdown(pointer);
  run('frameDetection = {...frameDetection, revision: 5}; status.detection.revision = 5;');
  await get("box-targets").events.pointerup(pointer);
  assert.equal(run('targetRequests.length'), beforeGesture + 2); // Prompt revision changed mid-press.
  assert.match(get("message").textContent, /no longer available/);
  run('frameDetection.revision = 4; status.detection.revision = 4;');
  const before = run('targetRequests.length');
  run('frameDetection.receivedAt = 0; renderTracking();');
  await freshBox.events.click({detail: 0});
  assert.equal(run('targetRequests.length'), before + 1);
  assert.equal(get("box-targets").hidden, false);
  run('status.camera.online = false; renderTracking();');
  assert.equal(get("box-targets").hidden, false);
  assert.equal(freshBox.disabled, true);
  run('status.camera.online = true; status.detection.state = "error"; renderTracking();');
  assert.equal(get("box-targets").hidden, false);
  assert.equal(freshBox.disabled, true);
  run('status.detection.state = "running";');
  console.log("validated clickable instances, 30 FPS button replacement, keyboard/cancel/drag, exact frame and no auto-start");
  run(`displayedDetection = null; activeModel = null; draftInitialized = false;
    status.detection = {enabled: true, model: "sam3.1", revision: 10, state: "idle", prompts: ["face"]};
    renderDetection(status.detection);`);
  rows()[0].fields.input.value = "unapplied SAM text";
  run(`status.detection = {enabled: true, model: "sam3.1-mask", revision: 11, state: "loading",
    prompts: ["person"], classes: null, models: [{id: "sam3.1", available: true}, {id: "sam3.1-mask", available: true}]};
    renderDetection(status.detection);`);
  assert.equal(get("detection-model").value, "sam3.1-mask");
  assert.equal(rows()[0].fields.input.value, "person");
  assert.equal(rows()[0].fields.input.children.length, 0);
  assert.ok(run("frameDetection")); // Retain the last processed picture while switching.
  rows()[0].fields.input.value = "cup";
  run(`status.detection = {...status.detection, model: "sam3.1", revision: 12, prompts: ["face"], classes: null};
    renderDetection(status.detection);`);
  assert.equal(rows()[0].fields.input.value, "unapplied SAM text");
  const menu = get('detection-model'), menuWrites = [];
  menu.options = ['sam3.1','sam3.1-tracking','sam3.1-mask'].map(value => {
    const option = new Element(); option.value = value; return option;
  });
  run('updatePromptControls();');
  for (const [node, properties] of [[menu,['value','disabled']],...menu.options.map(o=>[o,['disabled']])]) {
    for (const property of properties) {
      let value = node[property];
      Object.defineProperty(node,property,{get:()=>value,set:next=>{menuWrites.push(property);value=next;},configurable:true});
    }
  }
  menu.value = 'sam3.1-mask'; menuWrites.length = 0;
  run('for(let i=0;i<100;i++) renderDetection({...status.detection});');
  assert.deepEqual(menuWrites,[]);
  assert.equal(menu.value,'sam3.1-mask'); // Polls cannot overwrite uncommitted keyboard navigation.
  menu.value = 'sam3.1';
  console.log('validated standalone model popup remains untouched by repeated status updates');
  run(`request = async (path, options) => {
    targetRequests.push([path, JSON.parse(options.body)]);
    status.detection = {...status.detection, model: JSON.parse(options.body).model,
      revision: 13, prompts: ["person"], classes: ["person", "cup"]};
    renderDetection(status.detection);
    return status;
  };`);
  get("detection-model").value = "sam3.1-mask";
  await get("detection-model").events.change();
  assert.equal(run("JSON.stringify(targetRequests.at(-1))"), '["/api/detection/model",{"model":"sam3.1-mask"}]');
  assert.equal(rows()[0].fields.input.value, "cup"); // Mask draft preserved too.
  assert.equal(run("status.servo.armed"), false);
  run(`globalThis.submissions = 0; setPrompts = () => { submissions++; };`);
  get("detection-form").events.keydown({key: "Enter", target: {tagName: "SELECT"}, preventDefault() {}});
  assert.equal(run("submissions"), 1);
  console.log("validated model selection, free-text SAM prompts, independent drafts and Enter submission");
  run(`globalThis.enterCalls = []; status.servo.armed=false; status.servo.run_requested=false;
    request = async path => {enterCalls.push(path); status.servo.armed=true; status.servo.run_requested=true; return status;};`);
  documentEvents.keydown({key:'Enter',target:{tagName:'DIV'},preventDefault(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(run('JSON.stringify(enterCalls)'), '["/api/servo/arm"]');
  documentEvents.keydown({key:'Enter',target:{tagName:'DIV'},preventDefault(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(run('enterCalls.length'),1);
  run('delete status.servo.run_requested;');
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
  run(`cameraFeed.parentElement = {style: {}};
    status.camera = {online: true, width: 1280, height: 720};
    status.detection = {...status.detection, frame_sequence: 200, revision: 20};
    render({...status, servo: {...status.servo, armed: false},
      detection: {...status.detection, frame_sequence: 199}});`);
  assert.equal(run('status.detection.frame_sequence'), 200);
  assert.equal(run('status.servo.armed'), false);
  run(`render({...status, detection: {...status.detection, revision: 19, frame_sequence: 999}});`);
  assert.equal(run('status.detection.revision'), 20);
  run('showMessage("That camera frame is stale", true); render(status);');
  assert.equal(get("message").textContent, "That camera frame is stale");
  run('messageExpiresAt = 999; render(status);');
  assert.equal(get("message").hidden, true);
  console.log("validated client overlays and out-of-order status without losing motor updates");
  run(`render({...status, detection: {...status.detection, model: "sam3.1", state: "running", frame_age_ms: 10, latency_ms: 100,
    timing: {model_ms:100}, pipeline_timing: {cycle_ms: 121.4}, boxes: [], image_backend: "israel-w8a8-development"}});`);
  assert.equal(get("detection-status").textContent, "");
  assert.equal(get("detection-status").hidden, true);
  run(`render({...status, detection: {...status.detection,
    image_backend: "israel-w8a8-packed-development"}});`);
  assert.doesNotMatch(get("detection-status").textContent, /W8A8|accuracy unqualified|torch\.compile|graphs/);
  assert.equal(get("detection-status").textContent, "");
  assert.equal(get("detection-status").hidden, true);
  run('render({...status, detection: {...status.detection, pipeline_timing: undefined}});');
  assert.doesNotMatch(get("detection-status").textContent, /ms total|NaN/);
  run('status.detection.image_backend = "torch.compile";');
  run('render({...status, detection: {...status.detection, cuda_graph: true, sycl_graph: false}});');
  assert.equal(get("detection-status").textContent, "");
  assert.equal(get("detection-status").hidden, true);
  assert.equal(run('isDetectionFresh({state:"running",frame_age_ms:null,latency_ms:null})'), false);
  assert.equal(run('isDetectionFresh({state:"running",frame_age_ms:20})'), true);
  for (const raw of ['compiling', 'compiling_tracker_memory_update']) {
    assert.equal(run(`detectionPhase({state:${JSON.stringify(raw)}})`), 'preparing');
  }
  assert.equal(run('detectionPhase({state:"running",progress:{phase:"preparing"}})'), 'preparing');
  console.log('validated common progress vocabulary with routine latency and counts hidden');
  assert.doesNotMatch(get("detection-status").textContent, /CUDA|SYCL|torch\.compile|graphs/);
  run('render({...status, detection: {...status.detection, cuda_graph: false, sycl_graph: true}});');
  assert.equal(get("detection-status").textContent, "");
  assert.equal(get("detection-status").hidden, true);
  run(`loadingFrame = null; feedSource = "";
    detectionEvents.onopen();
    detectionEvents.onmessage({data: JSON.stringify({...status.detection, state: "running",
      frame_age_ms: 10, latency_ms: 9, frame_url: "/api/detection/frame/20-201.jpg",
      frame_sequence: 201, boxes: [], jpeg: "YWJj"})});`);
  assert.equal(run('loadingFrame.image.src'), "data:image/jpeg;base64,YWJj");
  assert.equal(run('status.detection.jpeg'), undefined);
  run(`detectionEvents.onmessage({data: JSON.stringify({...status.detection,
    frame_sequence: 202, frame_url: "/api/detection/frame/20-202.jpg", jpeg: "ZGVm"})});`);
  assert.equal(run('loadingFrame.image.src'), "data:image/jpeg;base64,YWJj"); // Decode one at a time.
  completeFrame();
  assert.equal(run('frameDetection.frame_sequence'), 201); // Boxes remain on the displayed image.
  assert.equal(run('loadingFrame.detection.frame_sequence'), 202);
  assert.equal(get("camera-feed").src, "data:image/jpeg;base64,YWJj");
  completeFrame();
  assert.equal(run('frameDetection.frame_sequence'), 202);
  run('detectionEvents.onerror();');
  assert.equal(run('detectionStreamOpen'), false);
  run(`loadProcessedFrame({...status.detection, frame_sequence:203}, '/pending-http.jpg', false);
    detectionEvents.onopen();
    detectionEvents.onmessage({data: JSON.stringify({...status.detection, frame_sequence:203,
      frame_url:"/api/detection/frame/20-203.jpg", jpeg:"bmV3"})});`);
  assert.equal(get('camera-feed').src, 'data:image/jpeg;base64,ZGVm');
  assert.equal(run('loadingFrame.detection.frame_sequence'), 203);
  assert.equal(run('loadingFrame.streamed'), true);
  run(`detectionEvents.onmessage({data: JSON.stringify({...status.detection,frame_sequence:204,
    frame_url:"/api/detection/frame/20-204.jpg", jpeg:"bmV4dA=="})});`);
  assert.equal(get('camera-feed').src, 'data:image/jpeg;base64,ZGVm');
  completeFrame();
  assert.equal(run('frameDetection.frame_sequence'),203);
  assert.equal(get('camera-feed').src, 'data:image/jpeg;base64,bmV3');
  completeFrame();
  // Status polling is allowed to run ahead of SSE; never starve paired frames.
  run(`render({...status,detection:{...status.detection,frame_sequence:206,
    frame_url:'/api/detection/frame/20-206.jpg'}});
    detectionEvents.onmessage({data:JSON.stringify({...status.detection,frame_sequence:205,
      frame_url:'/api/detection/frame/20-205.jpg',jpeg:'MjA1'})});`);
  assert.equal(run('status.detection.frame_sequence'),206);
  assert.equal(run('loadingFrame.detection.frame_sequence'),205);
  completeFrame();
  assert.equal(run('frameDetection.frame_sequence'),205);
  assert.equal(get('camera-feed').src,'data:image/jpeg;base64,MjA1');
  run(`detectionEvents.onmessage({data:JSON.stringify({...status.detection,frame_sequence:204,
    frame_url:'/api/detection/frame/20-204.jpg',jpeg:'old'})});`);
  assert.equal(run('loadingFrame'),null); // Late packets cannot rewind the display.
  console.log("validated paired image streaming, bounded decode queue and reconnect fallback");

  run(`detectionStreamOpen = false; streamFrame = null;
    status.detection = {...status.detection, model:'sam3.1-tracking', state:'running', revision:30,
      temporal_tracking:true, frame_sequence:1, frame_url:'/30-1.jpg',
      mask_overlay:{format:'indexed-png',png:'first'}, boxes:[]}; renderDetection(status.detection);`);
  const firstPair = run('loadingFrame'), previousImage = get('camera-feed');
  firstPair.image.events.load();
  assert.equal(get('camera-feed'),previousImage); // JPEG must wait for its mask.
  firstPair.mask.events.load();
  assert.equal(get('mask-overlay').hidden,true);
  assert.equal(get('camera-feed'),previousImage); // Persistent presentation surface.
  assert.equal(get('camera-feed').raster[1],firstPair.mask.src);
  run('renderDetection(status.detection);');
  assert.equal(run('loadingFrame'),null); // Status polling cannot repeatedly decode.
  run(`status.detection = {...status.detection,frame_sequence:2,frame_url:'/30-2.jpg',
    mask_overlay:{format:'indexed-png',png:'second'}}; renderDetection(status.detection);`);
  const secondPair = run('loadingFrame');
  secondPair.mask.events.load(); // Reverse decode order also holds the old pair.
  assert.equal(get('camera-feed'),previousImage);
  assert.equal(get('camera-feed').raster[1],firstPair.mask.src);
  assert.equal(get('mask-overlay').hidden,true);
  secondPair.image.events.load();
  assert.equal(get('camera-feed'),previousImage);
  assert.equal(get('camera-feed').raster[1],secondPair.mask.src);
  run(`status.detection = {...status.detection,frame_sequence:3,frame_url:'/30-3.jpg'}; renderDetection(status.detection);`);
  const failedPair = run('loadingFrame');
  failedPair.image.events.load(); failedPair.mask.events.error();
  assert.equal(get('camera-feed'),previousImage);
  assert.equal(get('camera-feed').raster[1],secondPair.mask.src);
  run(`status.detection = {...status.detection,frame_sequence:4,frame_url:'/30-4.jpg'}; renderDetection(status.detection);`);
  const obsoletePair = run('loadingFrame');
  run(`status.detection = {...status.detection,revision:31,model:'sam3.1-mask',state:'loading',frame_url:null};
    renderDetection(status.detection);`);
  obsoletePair.image.events.load(); obsoletePair.mask.events.load();
  assert.equal(get('camera-feed'),previousImage);
  assert.equal(get('camera-feed').raster[1],secondPair.mask.src);
  console.log('validated atomic JPEG/mask pairing, reverse decode order, failures and obsolete model results');

  run(`status.detection = {...status.detection, model:'sam3.1-mask', state:'running',
    temporal_tracking:false, mask_detection:true, frame_sequence:5, frame_url:'/31-5.jpg',
    mask_overlay:{format:'indexed-png',png:'per-frame-mask'},mask_overflow:{person:3},boxes:[]};
    renderDetection(status.detection);`);
  const detectorMaskPair = run('loadingFrame');
  assert.ok(detectorMaskPair.mask); // Per-frame masks do not require temporal memory.
  detectorMaskPair.image.events.load(); detectorMaskPair.mask.events.load();
  assert.equal(get('camera-feed'),previousImage);
  assert.equal(get('camera-feed').raster[1],detectorMaskPair.mask.src);
  assert.equal(get('detection-status').textContent, '');
  assert.equal(get('detection-status').hidden, true);
  console.log('validated non-temporal masks without overflow text');

  for (const model of ['sam3.1-mask', 'sam3.1-tracking']) {
    run(`request = async (path, options) => {
        const body=JSON.parse(options.body); targetRequests.push([path,body]);
        status.tracking={target:'cup',instance_id:body.instance_id,state:'stopped'}; return status;
      };
      status.detection = {...status.detection,model:${JSON.stringify(model)},revision:31,
      mask_detection:true,temporal_tracking:true,client_overlay:true,frame_sequence:6,
      frame_url:'/31-6.jpg',prompts:['cup'],boxes:[
        {instance_id:7,prompt:'cup',xyxy:[0,0,1,1],color:'#20b080',score:.9},
        {instance_id:8,prompt:'cup',xyxy:[0,0,1,1],color:'#40c090',score:.8}]};
      status.tracking = {target:'cup',instance_id:null,box:status.detection.boxes[0],frame:[31,6]};
      feedSource=''; renderDetection(status.detection);`);
    const pair = run('loadingFrame');
    pair.mask.naturalWidth = 4; pair.mask.naturalHeight = 2;
    pair.mask.pixelData = [32,176,128,112, 0,0,0,0, 65,191,144,112, 0,0,0,0,
      32,176,128,112, 65,191,144,112, 65,191,144,112, 0,0,0,0];
    completeFrame();
    const surface = run('visibleFrame.maskSurface');
    const pixel = i => [...surface.pixels.data.slice(i*4, i*4+4)];
    assert.deepEqual([...surface.labels], [1,0,2,0,1,2,2,0]); // Rounded PNG colors, not overlapping boxes.
    assert.deepEqual(pixel(0), [255,32,48,144]);
    assert.deepEqual(pixel(2), [65,191,144,112]); // Other instances keep their original color.
    assert.equal(get('tracking-overlay').hasAttribute('hidden'), true);
    assert.equal(get('box-targets').children[0].children[0].textContent, '');
    const stableCanvas = get('camera-feed'), frameKey = stableCanvas.getAttribute('data-frame-key');
    const requestsBefore = run('targetRequests.length');
    get('box-targets').rect = {left: 20, top: 30, width: 800, height: 400};
    const mouse = {button:0,isPrimary:true,pointerId:10,clientX:320,clientY:130};
    assert.equal(run('maskInstanceAt(visibleFrame.maskSurface, {x:20,y:130})'), null); // Mirrored leftmost pixel is background.
    assert.equal(run('maskInstanceAt(visibleFrame.maskSurface, {x:819.9,y:130})'), 7);
    assert.equal(run('maskInstanceAt(visibleFrame.maskSurface, {x:19.9,y:130})'), null);
    assert.equal(run('maskInstanceAt(visibleFrame.maskSurface, {x:820,y:130})'), null);
    get('box-targets').events.pointermove(mouse);
    assert.deepEqual(pixel(2), [255,132,142,144]);
    assert.deepEqual(pixel(0), [255,32,48,144]); // Hover cannot lighten the selected target.
    assert.equal(get('camera-feed'), stableCanvas);
    assert.equal(stableCanvas.getAttribute('data-frame-key'), frameKey); // Same processed frame, no raw camera refresh.
    assert.equal(run('targetRequests.length'), requestsBefore); // Hover never moves motors.
    get('box-targets').events.pointermove({...mouse,clientX:520}); // Transparent pixel inside both boxes.
    assert.deepEqual(pixel(2), [65,191,144,112]);
    get('box-targets').events.pointerdown({...mouse,clientX:520});
    await get('box-targets').events.pointerup({...mouse,clientX:520});
    assert.equal(run('targetRequests.length'), requestsBefore);
    get('box-targets').events.pointermove({...mouse,clientX:720});
    assert.deepEqual(pixel(0), [255,32,48,144]);
    get('box-targets').events.pointerleave();
    get('box-targets').children[1].events.focus();
    assert.deepEqual(pixel(2), [255,132,142,144]); // Keyboard preview has the same mask highlight.
    get('box-targets').children[1].events.blur();
    assert.deepEqual(pixel(2), [65,191,144,112]);
    get('box-targets').events.pointerdown(mouse);
    await get('box-targets').events.pointerup(mouse);
    assert.equal(run('JSON.stringify(targetRequests.at(-1))'),
      '["/api/tracking/instance",{"revision":31,"frame_sequence":6,"instance_id":8}]');
    assert.deepEqual(pixel(0), [32,176,128,112]);
    assert.deepEqual(pixel(2), [255,32,48,144]);
    // A missing selected instance must not color background or a different ID.
    run('status.tracking.instance_id = 99; renderTracking();');
    assert.deepEqual(pixel(1), [0,0,0,0]);
    assert.deepEqual(pixel(2), [65,191,144,112]);
    run('status.tracking = {target:null}; maskPointer=null; focusedMaskId=null;');
  }
  get('box-targets').rect = null;
  const css = fs.readFileSync(path.join(__dirname,'../src/spring_turret/static/app.css'),'utf8');
  assert.match(css, /\.box-target\.mask-instance:focus-visible\s*\{[^}]*border: 0;[^}]*outline: none;[^}]*pointer-events: none;/);
  assert.doesNotMatch(css, /mask-instance:hover \.box-label/);
  console.log('validated both mask modes: red target, light-red pixel hover, overlap/background hit tests and keyboard retarget');

  run(`displayedDetection = null; activeModel = 'sam3.1-tracking'; draftInitialized = true;
    detectionStreamOpen = false; streamFrame = null; loadingFrame = null;
    status.detection = {...status.detection, enabled:true, model:'sam3.1-tracking',
      state:'running', revision:32, frame_sequence:8, frame_url:'/32-8.jpg',
      frame_age_ms:100, latency_ms:230, prompts:['light'], categories:[], boxes:[],
      progress_stage:null, mask_overlay:null}; renderDetection(status.detection);`);
  completeFrame();
  assert.equal(get('camera-feed').src, '/32-8.jpg');
  run(`status.detection = {...status.detection, frame_age_ms:20000,
    progress_stage:'compiling_tracker_memory_attention_and_mask'};
    renderDetection(status.detection);`);
  assert.equal(get('camera-feed').src, '/32-8.jpg');
  assert.equal(run('frameDetection.frame_sequence'), 8);
  assert.equal(get('detection-status').hidden, true);
  assert.doesNotMatch(get('detection-status').textContent, /FPS|Waiting for detection/);
  assert.equal(get('fps-value').textContent, '—');
  run(`status.detection = {...status.detection, frame_sequence:9, frame_url:'/32-9.jpg',
    frame_age_ms:100, progress_stage:null}; renderDetection(status.detection);`);
  completeFrame();
  assert.equal(get('camera-feed').src, '/32-9.jpg');
  assert.doesNotMatch(get('detection-status').textContent, /FPS/);
  assert.match(get('fps-value').textContent, /^[\d.—]+$/);
  run(`status.detection = {...status.detection, revision:33, frame_sequence:null, frame_url:null,
    state:'compiling_tracker_text_encoder', progress_stage:'compiling_tracker_text_encoder'};
    renderDetection(status.detection);`);
  assert.equal(get('camera-feed').src, '/32-9.jpg');
  assert.equal(run('frameDetection.frame_sequence'), 9);
  assert.equal(get('mask-overlay').hidden, true);
  assert.doesNotMatch(appSource, /\/stream\.mjpg/);
  assert.equal(get('detection-status').hidden, true);
  run(`status.detection = {...status.detection, state:'error', error:'worker failed'};
    renderDetection(status.detection);`);
  assert.equal(get('detection-status').textContent, 'worker failed');
  assert.equal(get('detection-status').hidden, false);
  console.log('validated temporal recapture quietly preserves paired frames and resumes without flashing');

  run(`status.runtime = {backend_pid:10}; frameBackendPid = 10;
    render({...status, runtime:{backend_pid:11}, detection:{...status.detection,
      state:'running',revision:1,frame_sequence:1,frame_url:'/1-1.jpg',error:null,
      progress_stage:null,frame_age_ms:10}});`);
  assert.equal(run('clickableFrame()'),false); // Old picture cannot retarget a new process.
  assert.equal(run('loadingFrame.detection.revision'),1);
  completeFrame();
  assert.equal(run('frameDetection.revision'),1);
  assert.equal(run('clickableFrame()'),true);
  assert.equal(get('camera-feed').src,'/1-1.jpg');
  run('render({...status, runtime:undefined});');
  assert.equal(run('clickableFrame()'),true);
  assert.equal(run('status.runtime.backend_pid'),11);
  console.log('validated backend restart accepts reset frame numbers and invalidates old click targets');
  const retainedFrame = get('camera-feed'), retainedPixels = [...retainedFrame.raster];
  run('render({...status,camera:{...status.camera,online:false,error:"temporary reconnect"}});');
  assert.equal(get('message').textContent, 'temporary reconnect');
  assert.equal(get('camera-feed'),retainedFrame);
  assert.deepEqual(get('camera-feed').raster,retainedPixels);
  run(`render({...status,camera:{...status.camera,online:true},detection:{...status.detection,
    frame_sequence:2,frame_url:'/1-2.jpg',mask_overlay:{format:'indexed-png',png:'bad-draw'},temporal_tracking:true}});`);
  run('loadingFrame.mask.failDraw = true;'); completeFrame();
  assert.deepEqual(get('camera-feed').raster,retainedPixels);
  assert.match(get('message').textContent,/retaining the last result/);
  console.log('validated persistent canvas, no offline black cover and offscreen draw failure retention');

  // Run two real mount closures in one JS realm, with independent document
  // facades as used by the dashboard's shadow roots. All requests are mocked.
  const devices = {}, streams = new Map(), requests = [], states = {};
  const fixtureState = JSON.parse(run('JSON.stringify(status)'));
  for (const id of ['arc', 'thor']) {
    const map = new Map(), events = {};
    const element = key => {if (!map.has(key)) {
      const node = new Element(); node.ownerMap = map; node.ownerKey = key; map.set(key,node);
    } return map.get(key);};
    element('prompt-row-template').content = {firstElementChild: new Element()};
    element('camera-feed').parentElement = {style: {}};
    const images = [];
    devices[id] = {element, events, images, doc: {getElementById: element,
      createElement: tag => {const node = new Element(); if (tag === 'img') images.push(node); return node;}, querySelectorAll: () => [],
      addEventListener: (type, handler) => {events[type] = handler;}}};
    states[id] = structuredClone(fixtureState);
    states[id].detection = {...states[id].detection, model: 'sam3.1', revision: 1,
      state: 'running', error: null, progress_stage: null,
      frame_sequence: 1, frame_url: '/api/detection/frame/1-1.jpg', prompts: [id], boxes: [], categories: [], classes: null};
    states[id].servo.armed = false; states[id].servo.ready = true;
    for (const axis of Object.values(states[id].servo.axes)) axis.torque = false;
  }
  const dual = vm.createContext({document: {getElementById: () => null}, AbortSignal,
    performance: {now: () => 1000}, window: {confirm: () => false},
    setTimeout() {return 1;}, setInterval() {},
    EventSource: class {constructor(url) {streams.set(url, this);}},
    fetch: async (url, options = {}) => {
      requests.push([url, options]);
      const id = /^\/devices\/(arc|thor)\//.exec(url)?.[1];
      assert.ok(id, `Unscoped request: ${url}`);
      if (url.endsWith('/api/servo/arm')) states[id].servo.armed = true;
      if (url.endsWith('/api/servo/disable')) states[id].servo.armed = false;
      if (url.endsWith('/api/detection/prompts')) states[id].detection.prompts = JSON.parse(options.body).prompts;
      return {ok: true, json: async () => structuredClone(states[id])};
    },
  });
  vm.runInContext(appSource, dual);
  for (const id of ['arc', 'thor']) dual.mountTurret(devices[id].doc, `/devices/${id}`);
  const tick = () => new Promise(resolve => setImmediate(resolve));
  await tick();
  assert.equal(requests.filter(([, options]) => options.method === 'POST').length, 0);
  for (const id of ['arc', 'thor']) {
    assert.equal(devices[id].element('camera-feed').src, undefined);
    devices[id].images.at(-1).events.load();
    assert.equal(devices[id].element('camera-feed').src, `/devices/${id}/api/detection/frame/1-1.jpg`);
    assert.equal(devices[id].element('prompt-rows').children[0].fields.input.value, id);
  }
  devices.arc.element('prompt-rows').children[0].fields.input.value = 'hand';
  devices.arc.element('detection-form').events.submit({preventDefault() {}});
  await tick();
  assert.deepEqual(states.arc.detection.prompts, ['hand']);
  assert.deepEqual(states.thor.detection.prompts, ['thor']);
  devices.arc.element('motor-toggle').events.click(); await tick();
  assert.equal(states.arc.servo.armed, true); assert.equal(states.thor.servo.armed, false);
  assert.equal(devices.arc.element('motor-toggle').attributes['aria-label'], 'Stop motors');
  assert.equal(devices.thor.element('motor-toggle').attributes['aria-label'], 'Start motors');
  devices.arc.events.keydown({key: 'Escape'}); await tick();
  assert.equal(states.arc.servo.armed, false);
  assert.equal(requests.filter(([url, options]) => url.startsWith('/devices/thor/') && options.method === 'POST').length, 0);
  const arcStream = streams.get('/devices/arc/api/detection/events'); arcStream.onopen();
  arcStream.onmessage({data: JSON.stringify({...states.arc.detection, frame_sequence: 2, frame_url: '/api/detection/frame/1-2.jpg', jpeg: 'YXJj'})});
  assert.equal(devices.arc.element('camera-feed').src, '/devices/arc/api/detection/frame/1-1.jpg');
  devices.arc.images.at(-1).events.load();
  assert.equal(devices.arc.element('camera-feed').src, 'data:image/jpeg;base64,YXJj');
  assert.equal(devices.thor.element('camera-feed').src, '/devices/thor/api/detection/frame/1-1.jpg');
  const thorStream = streams.get('/devices/thor/api/detection/events'); thorStream.onopen();
  for (const [id, stream] of [['arc', arcStream], ['thor', thorStream]]) {
    stream.onmessage({data: JSON.stringify({...states[id].detection, revision:2,
      model:'sam3.1-tracking', state:id === 'arc' ? 'compiling_tracker_memory_update' : 'compiling',
      progress:{phase:'preparing',stage:id === 'arc' ? 'compiling_tracker_memory_update' : 'compiling'},
      frame_sequence:null,frame_url:null})});
  }
  assert.equal(devices.arc.element('detection-status').hidden, true);
  assert.equal(devices.arc.element('detection-status').textContent, devices.thor.element('detection-status').textContent);
  for (const [id, stream] of [['arc', arcStream], ['thor', thorStream]]) {
    stream.onmessage({data: JSON.stringify({...states[id].detection, revision:2,
      model:'sam3.1-tracking', state:'running',progress:{phase:'running',stage:'running'},
      frame_sequence:3,frame_age_ms:5,frame_url:'/api/detection/frame/2-3.jpg',
      latency_ms:220,timing:{model_ms:null,tracking_ms:218},pipeline_timing:{cycle_ms:240},mask_overflow:{}})});
    assert.equal(devices[id].element('detection-status').textContent, '');
    assert.equal(devices[id].element('detection-status').hidden, true);
  }
  console.log('validated quiet Arc/Thor preparation and hidden running summaries');
  console.log('validated side-by-side panel isolation: prompts, streams, image URLs, Start/Stop and focused Escape');
})().catch(error => { console.error(error); process.exitCode = 1; });
