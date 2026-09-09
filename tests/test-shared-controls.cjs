'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.events = {};
    this.value = ''; this.attributes = {}; this.dataset = {};
    this.classList = {toggle() {}, add() {}};
    this.style = {setProperty(name, value) {this[name] = value;}};
    this.naturalWidth = 1280; this.naturalHeight = 720;
  }
  get options() { return this.children; }
  getContext() { return this.context ||= {drawImage: () => {}}; }
  append(child) { child.parent = this; this.children.push(child); }
  replaceChildren() { this.children = []; }
  addEventListener(name, handler) { this.events[name] = handler; }
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) { return this.attributes[name] ?? null; }
  toggleAttribute(name, value) { if (value) this.attributes[name] = ''; else delete this.attributes[name]; }
  querySelector(selector) { return this.fields?.[selector]; }
  remove() { this.parent.children = this.parent.children.filter(c => c !== this); }
  replaceWith(element) {
    element.parentElement = this.parentElement;
    element.ownerMap = this.ownerMap; element.ownerKey = this.ownerKey;
    if (this.ownerMap) this.ownerMap.set(this.ownerKey,element);
  }
  focus() {}
  cloneNode() {
    const row = new Element();
    row.fields = Object.fromEntries(['.detection-prompt', '.target-prompt', '.remove-prompt'].map(s => [s, new Element()]));
    row.fields['.detection-prompt'].tagName = 'INPUT';
    row.fields['.detection-prompt'].replaceWith = replacement => {row.fields['.detection-prompt'] = replacement;};
    return row;
  }
}
const ids = new Map();
const get = id => {if (!ids.has(id)) ids.set(id, new Element()); return ids.get(id);};
const document = {getElementById: get, createElement: tag => new Element(tag)};
get('prompt-row-template').content = {firstElementChild: new Element()};
for (const id of ['sam3.1', 'sam3.1-mask', 'sam3.1-tracking', 'sam3.1-mask']) {
  const option = new Element('option'); option.value = id; get('detection-model').append(option);
}
const ctx = vm.createContext({document});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../src/spring_turret/static/shared-controls.js'), 'utf8'), ctx);
const devices = [{id:'arc', label:'Arc'}, {id:'thor', label:'Thor'}];
const controller = ctx.mountSharedControls(document, devices);
const states = {}, clients = {}, calls = [], failures = new Map(), blocks = new Map(), routeFailures = new Map();
for (const {id} of devices) {
  states[id] = {detection:{enabled:true, model:'sam3.1-tracking', prompts:['hand'], max_prompts:8,
    models:['sam3.1','sam3.1-tracking','sam3.1-mask'].map(id => ({id,available:true})),
    state:'running', frame_age_ms:10, latency_ms:100, classes:null, colors:['#55e8ce'],
    categories:[{prompt:'hand',count:id==='arc'?2:5}]}, tracking:{target:null, instance_id:null}, servo:{armed:true}};
  clients[id] = {
    setSharedBusy(value) { this.busy = value; },
    async command(route, body) {
      calls.push({id,route,body:structuredClone(body)});
      if (blocks.has(id)) await blocks.get(id);
      if (failures.has(id)) throw new Error(failures.get(id));
      if (routeFailures.has(`${id}:${route}`)) throw new Error(routeFailures.get(`${id}:${route}`));
      const state = states[id];
      if (route === '/api/detection/model') {
        state.detection.model = body.model;
        state.detection.classes = null;
        state.detection.prompts = [];
      } else if (route === '/api/detection/prompts') state.detection.prompts = [...body.prompts];
      else if (route === '/api/tracking/target') state.tracking = {target:body.target,instance_id:null};
      else if (route === '/api/servo/arm') state.servo.armed = true;
      else if (route === '/api/servo/gains') Object.assign(state.servo.axes[body.axis].position_gains,{p:body.p,d:body.d});
      else if (route === '/api/servo/gains/reset') Object.assign(state.servo.axes[body.axis].position_gains,state.servo.axes[body.axis].gain_baseline);
      else throw new Error(`Unexpected mutation ${route}`);
      controller.update(id, structuredClone(state));
      return structuredClone(state);
    },
  };
  controller.attach(id, clients[id]);
}
const rows = () => get('prompt-rows').children;
const input = i => rows()[i].querySelector('.detection-prompt');
const values = () => rows().map(row => row.querySelector('.detection-prompt').value);
const submit = () => get('detection-form').events.submit({preventDefault() {}});
const changeModel = model => {get('detection-model').value = model; return get('detection-model').events.change();};
const tick = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  controller.update('thor', structuredClone(states.thor));
  assert.equal(rows().length,0); // Network response order cannot choose the draft.
  controller.update('arc', structuredClone(states.arc));
  assert.deepEqual(values(),['hand']);
  assert.equal(calls.length,0); // No live configuration/motion on page load.
  const menu = get('detection-model'), menuWrites = [];
  for (const [node, properties] of [[menu,['value','disabled']],...menu.options.map(o=>[o,['disabled','title']])]) {
    for (const property of properties) {
      let value = node[property];
      Object.defineProperty(node,property,{get:()=>value,set:next=>{menuWrites.push(property);value=next;},configurable:true});
    }
  }
  // Native keyboard navigation may preview a value before the change event.
  menu.value = 'sam3.1-mask'; menuWrites.length = 0;
  for (let i=0;i<100;i++) controller.update(i%2?'arc':'thor',structuredClone(states[i%2?'arc':'thor']));
  assert.deepEqual(menuWrites,[]); // Includes same-value setters which rebuild Firefox's popup.
  assert.equal(menu.value,'sam3.1-mask');
  assert.equal(calls.length,0);
  menu.value = 'sam3.1-tracking'; // Native Escape/cancel restores its prior selection.
  console.log('validated shared model popup remains untouched by alternating backend frames');
  input(0).value='cup'; input(0).events.input();
  controller.update('thor', structuredClone(states.thor));
  assert.deepEqual(values(),['cup']); // Polls cannot overwrite a shared draft.
  assert.equal(calls.length,0);
  await submit();
  assert.deepEqual(states.arc.detection.prompts,['cup']);
  assert.deepEqual(states.thor.detection.prompts,['cup']);
  assert.equal(calls.length,2);
  assert.ok(calls.every(c=>c.route==='/api/detection/prompts'));
  assert.ok(states.arc.servo.armed && states.thor.servo.armed);
  assert.equal(get('shared-message').hidden,true);
  console.log('validated single draft, prompt controls, explicit two-device submission and no motor writes');

  get('add-prompt').events.click(); input(1).value='bottle'; input(1).events.input();
  await submit();
  assert.deepEqual(states.arc.detection.prompts,['cup','bottle']);
  assert.deepEqual(states.thor.detection.prompts,['cup','bottle']);
  await rows()[0].querySelector('.target-prompt').events.click();
  assert.equal(states.arc.tracking.target,'cup'); assert.equal(states.thor.tracking.target,'cup');
  await rows()[0].querySelector('.target-prompt').events.click();
  assert.equal(states.arc.tracking.target,null); assert.equal(states.thor.tracking.target,null);
  states.arc.tracking={target:'bottle',instance_id:123};
  controller.update('arc',structuredClone(states.arc));
  await rows()[1].querySelector('.target-prompt').events.click();
  assert.equal(states.arc.tracking.instance_id,null);
  assert.equal(states.thor.tracking.target,'bottle');
  assert.ok(calls.every(c=>!c.route.includes('/servo/')));

  await changeModel('sam3.1-mask');
  for (const id of ['arc','thor']) {
    assert.equal(states[id].detection.model,'sam3.1-mask');
    assert.deepEqual(states[id].detection.prompts,['cup','bottle']);
  }
  assert.equal(input(0).tagName,'INPUT'); // Free-text input, not a fixed-class select.
  await changeModel('sam3.1');
  for(const id of ['arc','thor']) {
    assert.equal(states[id].detection.model,'sam3.1');
    assert.deepEqual(states[id].detection.prompts,['cup','bottle']);
    const own=calls.filter(c=>c.id===id);
    assert.equal(own.at(-2).route,'/api/detection/model');
    assert.equal(own.at(-1).route,'/api/detection/prompts');
  }
  await changeModel('sam3.1-mask');
  assert.notEqual(input(0).tagName,'SELECT');
  get('prompt-rows').children[1].fields['.remove-prompt'].events.click();
  input(0).value='person'; input(0).events.input();
  let prevented=false;
  states.arc.servo.armed = states.thor.servo.armed = false;
  const beforeEnter = calls.length;
  await get('detection-form').events.keydown({key:'Enter',target:input(0),preventDefault(){prevented=true;}});
  assert.equal(prevented,true);
  assert.deepEqual(states.arc.detection.prompts,['person']);
  assert.deepEqual(states.thor.detection.prompts,['person']);
  assert.equal(states.arc.servo.armed, true);
  assert.equal(states.thor.servo.armed, true);
  for (const id of ['arc','thor']) assert.deepEqual(calls.slice(beforeEnter).filter(c=>c.id===id).map(c=>c.route),
    ['/api/detection/prompts','/api/tracking/target','/api/servo/arm']);
  assert.equal(states.arc.tracking.target,'person');
  assert.equal(states.thor.tracking.target,'person');
  get('add-prompt').events.click(); input(1).value='cup'; input(1).events.input();
  const beforeRetarget=calls.length;
  await get('detection-form').events.keydown({key:'Enter',target:input(1),preventDefault(){}});
  for (const id of ['arc','thor']) {
    assert.equal(states[id].tracking.target,'cup');
    assert.deepEqual(calls.slice(beforeRetarget).filter(c=>c.id===id).map(c=>c.route),
      ['/api/detection/prompts','/api/tracking/target']); // Already running: no duplicate Start.
  }
  input(0).value='CUP'; input(0).events.input();
  await get('detection-form').events.keydown({key:'Enter',target:input(0),preventDefault(){}});
  assert.deepEqual(states.arc.detection.prompts,['cup']);
  assert.equal(states.arc.tracking.target,'cup'); // Canonical applied spelling, not a toggle off.
  input(0).value=' '; input(0).events.input();
  const beforeBlank=calls.length;
  await get('detection-form').events.keydown({key:'Enter',target:input(0),preventDefault(){}});
  assert.equal(calls.length,beforeBlank);
  assert.match(get('shared-message').textContent,/Enter an object/);
  input(0).value='person'; input(0).events.input();
  for (const extra of [{repeat:true},{isComposing:true}])
    await get('detection-form').events.keydown({key:'Enter',target:input(0),preventDefault(){},...extra});
  assert.equal(calls.length,beforeBlank);
  states.arc.servo.armed=states.thor.servo.armed=false;
  routeFailures.set('thor:/api/tracking/target','Target rejected');
  const beforeTargetFailure=calls.length;
  await get('detection-form').events.keydown({key:'Enter',target:input(0),preventDefault(){}});
  assert.equal(states.arc.servo.armed,true);
  assert.equal(states.thor.servo.armed,false);
  assert.deepEqual(calls.slice(beforeTargetFailure).filter(c=>c.id==='thor').map(c=>c.route),
    ['/api/detection/prompts','/api/tracking/target']);
  assert.match(get('shared-message').textContent,/Thor: Target rejected/);
  routeFailures.clear();
  await get('detection-form').events.keydown({key:'Enter',target:input(0),preventDefault(){}});
  await changeModel('sam3.1');
  assert.deepEqual(values(),['cup','bottle']);
  await changeModel('sam3.1-mask');
  assert.deepEqual(values(),['person']); // Per-model shared draft survives switches.
  console.log('validated shared target class, ordered model/prompt changes, free-text SAM prompts and Enter');

  failures.set('thor','Backend unavailable; outcome unconfirmed');
  input(0).value='cup'; input(0).events.input();
  const beforeFailure=calls.length;
  await submit();
  assert.equal(calls.length,beforeFailure+2);
  assert.deepEqual(states.arc.detection.prompts,['cup']);
  assert.deepEqual(states.thor.detection.prompts,['person']);
  assert.match(get('shared-message').textContent,/Thor: Backend unavailable/);
  controller.offline('thor'); controller.update('thor',structuredClone(states.thor));
  await tick();
  assert.equal(calls.length,beforeFailure+2); // Reconnect never retries or rolls back Arc.
  assert.deepEqual(values(),['cup']);
  failures.clear(); await submit();
  assert.deepEqual(states.thor.detection.prompts,['cup']);
  assert.equal(get('shared-message').hidden,true);

  let release; blocks.set('thor',new Promise(resolve=>{release=resolve;}));
  const pending=submit(); await tick();
  assert.equal(get('detection-model').disabled,true);
  assert.equal(get('update-prompts').disabled,true);
  assert.equal(clients.arc.busy,true); assert.equal(clients.thor.busy,true);
  const inFlight=calls.length;
  await submit();
  assert.equal(calls.length,inFlight);
  release(); await pending; blocks.clear();
  assert.equal(clients.arc.busy,false); assert.equal(clients.thor.busy,false);

  states.thor.detection.models.find(m=>m.id==='sam3.1').available=false;
  controller.update('thor',structuredClone(states.thor));
  const option=get('detection-model').options.find(o=>o.value==='sam3.1');
  assert.equal(option.disabled,true); assert.match(option.title,/Thor/);
  const beforeUnavailable=calls.length;
  await changeModel('sam3.1');
  assert.equal(calls.length,beforeUnavailable);
  assert.equal(get('detection-model').value,'sam3.1-mask');
  console.log('validated partial failures, no automatic retries, in-flight exclusion and common model availability');

  const gainStart = calls.length;
  for (const id of ['arc','thor']) {
    states[id].servo={online:true,armed:true,axes:Object.fromEntries(['x','y'].map((axis,i)=>[axis,{
      position_gains:{p:id==='arc'?400+i*100:600+i*100,i:0,d:i*10},gain_baseline:{p:400,d:0}}]))};
    controller.update(id,structuredClone(states[id]));
  }
  assert.equal(calls.length,gainStart); // Mixed initial values never auto-align.
  assert.equal(get('shared-p-value').textContent,'Mixed');
  assert.equal(get('shared-d-value').textContent,'Mixed');
  get('shared-p-gain').value='800'; get('shared-p-gain').events.input();
  controller.update('thor',structuredClone(states.thor));
  assert.equal(get('shared-p-gain').value,'800');
  assert.equal(calls.length,gainStart); // One commit on release, no drag requests.
  await get('shared-p-gain').events.change();
  assert.equal(calls.length,gainStart+4);
  assert.ok(calls.slice(gainStart).every(c=>c.route==='/api/servo/gains'));
  for(const id of ['arc','thor']) {
    assert.deepEqual(calls.slice(gainStart).filter(c=>c.id===id).map(c=>c.body),[
      {axis:'x',p:800,d:0},{axis:'y',p:800,d:10}]);
    assert.equal(states[id].servo.armed,true);
  }
  assert.equal(get('shared-p-value').textContent,'800');
  assert.equal(get('shared-d-value').textContent,'Mixed');
  get('shared-d-gain').value='100'; get('shared-d-gain').events.input();
  await get('shared-d-gain').events.change();
  assert.ok(calls.slice(-4).every(c=>c.body.p===800 && c.body.d===100));
  assert.equal(get('shared-d-value').textContent,'100');
  await get('shared-gains-reset').events.click();
  assert.ok(calls.slice(-4).every(c=>c.route==='/api/servo/gains/reset'));
  assert.equal(get('shared-p-value').textContent,'400');
  assert.equal(get('shared-d-value').textContent,'0');

  controller.offline('thor');
  const offlineCount=calls.length;
  assert.equal(get('shared-p-gain').disabled,true);
  await get('shared-gains-reset').events.click();
  assert.equal(calls.length,offlineCount);
  controller.update('thor',structuredClone(states.thor));
  failures.set('thor','Disconnected');
  get('shared-p-gain').value='900';
  await get('shared-p-gain').events.change();
  assert.equal(calls.length,offlineCount+4); // Both axes attempted, no retries.
  assert.match(get('shared-gains-message').textContent,/Thor X: Disconnected.*Thor Y: Disconnected/);
  assert.equal(get('shared-p-value').textContent,'Mixed');
  failures.clear(); controller.update('thor',structuredClone(states.thor));
  assert.equal(calls.length,offlineCount+4);
  assert.ok(calls.slice(gainStart).every(c=>!c.route.includes('/arm') && !c.route.includes('/disable')));

  let releaseGains; blocks.set('thor',new Promise(resolve=>{releaseGains=resolve;}));
  const pendingGain=get('shared-gains-reset').events.click(); await tick();
  assert.equal(get('shared-p-gain').disabled,true);
  assert.equal(clients.arc.busy,false); // No blocking of panel Start/Stop/retarget.
  const gainBusyCount=calls.length;
  await get('shared-gains-reset').events.click();
  assert.equal(calls.length,gainBusyCount);
  releaseGains(); await pendingGain; blocks.clear();
  console.log('validated shared X/Y P/D across both devices, mixed values, reset, partial failures and no auto-arm');

  // The actual per-device client runs without model/form/template elements
  // when mounted in shared-control mode. Local motor and instance commands
  // remain namespaced; no new model/prompt/servo writes occur during mounting.
  const panelRequests=[], panels={}, streams=new Map();
  const panelCtx=vm.createContext({document:{getElementById:()=>null},AbortSignal,
    performance:{now:()=>1000},window:{confirm:()=>false},setTimeout(){},setInterval(){},
    EventSource:class{constructor(url){streams.set(url,this);}},
    fetch:async(url,options={})=>{
      panelRequests.push({url,options});
      const id=url.split('/')[2], state=panels[id].state;
      if(url.endsWith('/api/servo/disable')) state.servo.armed=false;
      if(url.endsWith('/api/tracking/instance')) state.tracking.instance_id=JSON.parse(options.body).instance_id;
      return {ok:true,json:async()=>structuredClone(state)};
    }});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/spring_turret/static/app.js'),'utf8'),panelCtx);
  for(const id of ['arc','thor']) {
    const map=new Map(), events={};
    const missing=new Set(['detection-model','prompt-rows','add-prompt','update-prompts','detection-form','prompt-row-template']);
    const element=key=>{if(missing.has(key))return null; if(!map.has(key)) {
      const node=new Element(); node.ownerMap=map; node.ownerKey=key; map.set(key,node);
    }return map.get(key);};
    element('camera-feed').parentElement={style:{}};
    const state=structuredClone(states[id]);
    Object.assign(state.detection,{revision:1,frame_sequence:1,frame_url:'/api/detection/frame/1-1.jpg',
      boxes:[{instance_id:7,prompt:'cup',score:.9,xyxy:[.1,.1,.3,.3]}]});
    state.camera={online:true,width:1280,height:720};
    state.servo={online:true,ready:true,armed:true,axes:{x:{degrees:0,torque:true},y:{degrees:0,torque:true}}};
    const images=[];
    panels[id]={element,events,state,images};
    panels[id].client=panelCtx.mountTurret({getElementById:element,createElement:tag=>{
      const node=new Element(tag); if(tag==='img')images.push(node); return node;
    },
      addEventListener:(name,handler)=>{events[name]=handler;}},`/devices/${id}`,{sharedControls:true});
  }
  await tick();
  assert.equal(panelRequests.filter(r=>r.options.method==='POST').length,0);
  for(const id of ['arc','thor']) for(const axis of ['x','y'])
    assert.equal(panels[id].element(`${axis}-gains`).hidden,true);
  panels.arc.images.at(-1).events.load();
  panels.arc.client.setSharedBusy(true);
  assert.equal(panels.arc.element('box-targets').children[0].disabled,true);
  panels.arc.client.setSharedBusy(false);
  await panels.arc.element('box-targets').children[0].events.click({detail:0});
  assert.equal(panels.arc.state.tracking.instance_id,7);
  assert.notEqual(panels.thor.state.tracking.instance_id,7);
  panels.arc.events.keydown({key:'Escape'}); await tick();
  assert.equal(panels.arc.state.servo.armed,false); assert.equal(panels.thor.state.servo.armed,true);
  assert.ok(panelRequests.filter(r=>r.options.method==='POST').every(r=>r.url.startsWith('/devices/arc/')));
  console.log('validated control-free panels with independent streams, instance retargeting and motors');
})().catch(error=>{console.error(error);process.exitCode=1;});
