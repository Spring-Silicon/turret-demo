"use strict";

// On page startup, align each device with the displayed model/prompts once.
// Matching devices keep their model/session; startup never changes motor or pause intent.
function mountSharedControls(document, devices) {
  const rows = document.getElementById('prompt-rows');
  const modelSelector = document.getElementById('detection-model');
  const pauseButton = document.getElementById('inference-toggle');
  const add = document.getElementById('add-prompt');
  const submit = document.getElementById('update-prompts');
  const message = document.getElementById('shared-message');
  const states = new Map(), clients = new Map(), offline = new Set(), drafts = new Map();
  let model = null, initialized = false, busy = false, error = '', dirty = false;
  let renderedModel = null;
  let appliedSelection = null, startupScheduled = false;
  const startupPending = new Set(devices.map(device => device.id));
  const readRows = () => [...rows.children].map(row => row.querySelector('.detection-prompt').value.trim());
  const normalized = values => [...new Map(values.filter(Boolean).map(value => [value.toLowerCase(), value])).values()];
  const detections = () => devices.map(device => states.get(device.id)?.detection);
  const connected = () => devices.filter(device => states.has(device.id) && !offline.has(device.id));
  const limit = () => Math.min(...detections().map(d => d?.max_prompts || 8));
  // Both devices expose the same profiles; hardware adapters stay server-side.
  const availableOn = (id, device) => {
    const d = states.get(device.id)?.detection;
    return d?.enabled && d.models?.some(m => m.id === id && m.available);
  };
  const available = id => connected().length > 0 && connected().every(device => availableOn(id, device));
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  function makeRow(value = '') {
    const row = document.getElementById('prompt-row-template').content.firstElementChild.cloneNode(true);
    const input = row.querySelector('.detection-prompt');
    input.value = value;
    input.addEventListener('input', () => { dirty = true; error = ''; render(); });
    row.querySelector('.remove-prompt').addEventListener('click', () => {
      if (busy) return;
      if (rows.children.length === 1) input.value = ''; else row.remove();
      dirty = true; error = ''; render();
      rows.children[0].querySelector('.detection-prompt').focus();
    });
    rows.append(row);
    return input;
  }

  function setRows(values) {
    rows.replaceChildren();
    (values.length ? values : ['']).forEach(makeRow);
  }

  function render() {
    if (!initialized && states.size) {
      // Deterministic initial draft: prefer Arc/config order, not reply order.
      const first = states.get(devices[0].id) || (offline.has(devices[0].id) && states.values().next().value);
      if (first) {
        model = first.detection.model;
        initialized = true; setRows(first.detection.prompts || []);
        appliedSelection = {model, prompts: normalized(readRows())};
      }
    }
    // Polls never overwrite a user's draft or repeatedly reset model sessions.
    const ready = initialized && connected().some(d => states.get(d.id)?.detection?.enabled);
    const disabled = !ready || busy;
    const paused = connected().some(d => states.get(d.id)?.detection?.paused === true);
    pauseButton.disabled = disabled || !connected().every(d => typeof states.get(d.id)?.detection?.paused === 'boolean');
    pauseButton.textContent = paused ? 'Resume' : 'Pause';
    pauseButton.setAttribute('aria-label', paused ? 'Resume inference' : 'Pause inference');
    pauseButton.setAttribute('aria-pressed', String(paused));
    if (modelSelector.disabled !== disabled) modelSelector.disabled = disabled;
    const selection = model || 'sam3.1';
    // Firefox rebuilds its native popup when select/option state is rewritten,
    // even to the same value. Leave in-progress hover/keyboard selection alone
    // until the committed model actually changes, not on each device frame.
    if (renderedModel !== selection) {
      if (modelSelector.value !== selection) modelSelector.value = selection;
      renderedModel = selection;
    }
    for (const option of modelSelector.options) {
      const disabled = !available(option.value);
      if (option.disabled !== disabled) option.disabled = disabled;
      const missing = connected().filter(d => !availableOn(option.value, d));
      const title = missing.length ? `Unavailable on ${missing.map(d => d.label).join(', ')}`
        : '';
      if (option.title !== title) option.title = title;
    }
    add.disabled = !ready || busy || rows.children.length >= limit();
    submit.disabled = !ready || busy || !available(model);
    submit.textContent = busy ? 'Applying…' : 'Update prompts';
    [...rows.children].forEach((row, index) => {
      const input = row.querySelector('.detection-prompt');
      input.disabled = !ready || busy;
      input.setAttribute('aria-label', `Object ${index + 1} to find`);
      row.querySelector('.remove-prompt').disabled = !ready || busy;
      row.style.setProperty('--prompt-color', detections().find(d => d?.model === model)?.colors?.[index] || '#55e8ce');
    });
    const mismatched = initialized && devices.filter(device => {
      const d = states.get(device.id)?.detection;
      return d && !startupPending.has(device.id) && (d.model !== model || !same(d.prompts, normalized(readRows())));
    });
    const notices = [];
    const unavailable = devices.filter(d => offline.has(d.id) || !states.has(d.id));
    if (unavailable.length) notices.push(connected().length
      ? `${unavailable.map(d => d.label).join(', ')} offline; controls apply to connected devices`
      : 'No devices connected; waiting for reconnection');
    if (!busy && mismatched?.length) notices.push(dirty ? 'Unapplied changes' : 'Device settings differ; Update prompts applies this selection to both');
    message.textContent = error || notices.join(' · ');
    message.hidden = !message.textContent;
    message.classList.toggle('error', Boolean(error));
  }

  function scheduleStartupSync() {
    if (!initialized || busy || startupScheduled || !startupPending.size) return;
    startupScheduled = true;
    Promise.resolve().then(async () => {
      startupScheduled = false;
      if (busy || !appliedSelection) return;
      const selection = appliedSelection;
      const ready = connected().filter(device => startupPending.has(device.id) && clients.has(device.id));
      // Claim before any async command/update callback to prevent duplicate resets.
      ready.forEach(device => startupPending.delete(device.id));
      const changed = ready.filter(device => {
        const d = states.get(device.id).detection;
        return d.model !== selection.model || !same(d.prompts, selection.prompts);
      });
      if (!changed.length) { if (ready.length) render(); return; }
      await fanOut(async (client, device) => {
        if (!availableOn(selection.model, device)) throw new Error('Selected model is unavailable');
        let current = states.get(device.id);
        if (current.detection.model !== selection.model)
          current = await client.command('/api/detection/model', {model: selection.model});
        if (!same(current.detection.prompts, selection.prompts))
          current = await client.command('/api/detection/prompts', {prompts: selection.prompts});
        states.set(device.id, current);
      }, changed);
    });
  }

  async function fanOut(action, scope = devices) {
    if (busy) return false;
    busy = true; error = '';
    clients.forEach(client => client.setSharedBusy(true)); render();
    try {
      const results = await Promise.allSettled(scope.map(async device => {
        if (offline.has(device.id) || !states.has(device.id)) throw new Error('Offline; skipped');
        const client = clients.get(device.id);
        if (!client) throw new Error('Device is not ready');
        return action(client, device);
      }));
      const failures = results.flatMap((r, i) => r.status === 'rejected' ? [`${scope[i].label}: ${r.reason.message || r.reason}`] : []);
      error = failures.join(' · ');
      // Successful peers are not rolled back, and ambiguous failures are never
      // retried automatically. The next explicit Update can align both again.
      return failures.length === 0;
    } finally {
      busy = false; clients.forEach(client => client.setSharedBusy(false)); render();
      scheduleStartupSync();
    }
  }

  async function applySelection(startAfter = false, targetPrompt) {
    if (busy || !initialized || !available(model)) return;
    if (targetPrompt === '') { error = 'Enter an object to target'; render(); return; }
    const selectedModel = model, prompts = normalized(readRows());
    appliedSelection = {model: selectedModel, prompts};
    drafts.set(model, prompts);
    const ok = await fanOut(async (client, device) => {
      if (states.get(device.id)?.detection?.model !== selectedModel)
        await client.command('/api/detection/model', {model: selectedModel});
      let result = await client.command('/api/detection/prompts', {prompts});
      states.set(device.id, result); offline.delete(device.id);
      if (targetPrompt !== undefined) {
        const target = result.detection.prompts.find(prompt => prompt.toLowerCase() === targetPrompt.toLowerCase());
        if (!target) throw new Error('The target prompt was not applied');
        result = await client.command('/api/tracking/target', {target});
        states.set(device.id, result);
      }
      if (startAfter && !(result.servo?.run_requested ?? result.servo?.armed)) {
        const started = await client.command('/api/servo/arm', {});
        states.set(device.id, started);
      }
    });
    if (ok) { dirty = false; setRows(states.get(devices[0].id).detection.prompts); }
    render();
  }

  pauseButton.addEventListener('click', () => {
    if (pauseButton.disabled || busy) return;
    const paused = !connected().some(d => states.get(d.id)?.detection?.paused === true);
    return fanOut(async (client, device) => {
      if (states.get(device.id)?.detection?.paused !== paused)
        states.set(device.id, await client.command('/api/detection/pause', {paused}));
    });
  });

  modelSelector.addEventListener('change', async () => {
    if (busy || !initialized) return;
    const next = modelSelector.value;
    if (!available(next)) { renderedModel = null; render(); return; }
    drafts.set(model, readRows());
    const oldPrompts = readRows();
    model = next; dirty = true;
    const selected = drafts.get(next) || oldPrompts;
    appliedSelection = {model: next, prompts: normalized(selected)};
    setRows(selected);
    await fanOut(async (client, device) => {
      const result = await client.command('/api/detection/model', {model: next});
      states.set(device.id, result); offline.delete(device.id);
      const applied = await client.command('/api/detection/prompts', {prompts: normalized(selected)});
      states.set(device.id, applied);
    });
    dirty = Boolean(error); render();
  });
  document.getElementById('detection-form').addEventListener('submit', event => {
    event.preventDefault(); return applySelection();
  });
  document.getElementById('detection-form').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.repeat && !event.isComposing && ['INPUT', 'SELECT'].includes(event.target.tagName)) {
      event.preventDefault();
      return applySelection(true, event.target.tagName === 'INPUT' ? event.target.value.trim() : undefined);
    }
  });
  document.addEventListener?.('keydown', event => {
    const element = event.composedPath?.()[0] || event.target;
    const cycle = {ArrowLeft: ['arc', -1], ArrowRight: ['arc', 1],
      ArrowUp: ['thor', -1], ArrowDown: ['thor', 1]}[event.key];
    if (cycle) {
      if (busy || event.defaultPrevented || event.repeat || event.isComposing
          || event.metaKey || event.ctrlKey || event.altKey || element?.isContentEditable
          || ['INPUT', 'SELECT', 'TEXTAREA'].includes(element?.tagName)) return;
      const [id, direction] = cycle;
      if (offline.has(id) || !clients.has(id)) return;
      event.preventDefault();
      return clients.get(id).cycleTarget(direction);
    }
    if (event.key !== 'Enter' || event.defaultPrevented || event.repeat || event.isComposing
        || ['INPUT', 'SELECT', 'BUTTON'].includes(event.target?.tagName)) return;
    // Enter handled inside a device panel is already prevented there. At the
    // shared page level, explicitly start both without reapplying any prompts.
    event.preventDefault();
    return fanOut(async (client, device) => {
      const servo = states.get(device.id)?.servo;
      if (!(servo?.run_requested ?? servo?.armed))
        states.set(device.id, await client.command('/api/servo/arm', {}));
    });
  });
  add.addEventListener('click', () => {
    if (add.disabled) return;
    dirty = true; error = ''; const input = makeRow(); render(); input.focus();
  });
  render();
  return {
    attach(id, client) { clients.set(id, client); scheduleStartupSync(); },
    update(id, state) { states.set(id, state); offline.delete(id); render(); scheduleStartupSync(); },
    offline(id) { offline.add(id); render(); scheduleStartupSync(); },
  };
}
