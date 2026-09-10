"use strict";

// Only explicit user actions fan out. Loading/reconnecting this UI never
// changes a model, resets temporal state, or starts/stops either motor.
function mountSharedControls(document, devices) {
  const rows = document.getElementById('prompt-rows');
  const modelSelector = document.getElementById('detection-model');
  const add = document.getElementById('add-prompt');
  const submit = document.getElementById('update-prompts');
  const message = document.getElementById('shared-message');
  const states = new Map(), clients = new Map(), offline = new Set(), drafts = new Map();
  let model = null, initialized = false, busy = false, error = '', dirty = false;
  let renderedModel = null;
  const readRows = () => [...rows.children].map(row => row.querySelector('.detection-prompt').value.trim());
  const normalized = values => [...new Map(values.filter(Boolean).map(value => [value.toLowerCase(), value])).values()];
  const detections = () => devices.map(device => states.get(device.id)?.detection);
  const limit = () => Math.min(...detections().map(d => d?.max_prompts || 8));
  // v18 is an Intel implementation of the temporal profile. Thor keeps its
  // existing CUDA tracking implementation, not the Intel-only worker ID.
  const effectiveModel = (id, device) => id === 'sam3.1-v18' && device.id === 'thor' ? 'sam3.1-tracking' : id;
  const availableOn = (id, device) => {
    const d = states.get(device.id)?.detection;
    return d?.enabled && d.models?.some(m => m.id === effectiveModel(id, device) && m.available);
  };
  const available = id => devices.every(device => availableOn(id, device));
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
    row.querySelector('.target-prompt').addEventListener('click', () => {
      if (busy || row.querySelector('.target-prompt').disabled) return;
      const prompt = input.value.trim();
      const stop = devices.every(d => {
        const tracking = states.get(d.id)?.tracking;
        return tracking?.target === prompt && tracking.instance_id == null;
      });
      return fanOut(client => client.command('/api/tracking/target', {target: stop ? null : prompt}));
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
      }
    }
    // Once initialized, polls only update availability. Never overwrite
    // a user's draft with another device's result or auto-resubmit on mismatch.
    const ready = initialized && detections().some(d => d?.enabled);
    const disabled = !ready || busy;
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
      const missing = devices.filter(d => !availableOn(option.value, d));
      const title = missing.length ? `Unavailable on ${missing.map(d => d.label).join(', ')}`
        : option.value === 'sam3.1-v18' ? 'Arc: Israel v18 tracking; Thor: SAM 3.1 Tracking' : '';
      if (option.title !== title) option.title = title;
    }
    add.disabled = !ready || busy || rows.children.length >= limit();
    submit.disabled = !ready || busy || !available(model);
    submit.textContent = busy ? 'Applying…' : 'Update prompts';
    const seen = new Set();
    [...rows.children].forEach((row, index) => {
      const input = row.querySelector('.detection-prompt'), prompt = input.value.trim();
      input.disabled = !ready || busy;
      input.setAttribute('aria-label', `Object ${index + 1} to find`);
      row.querySelector('.remove-prompt').disabled = !ready || busy;
      const applied = prompt && !seen.has(prompt) && devices.every(device => {
        const d = states.get(device.id)?.detection;
        return !offline.has(device.id) && d?.model === effectiveModel(model, device) && d.prompts.includes(prompt);
      });
      seen.add(prompt);
      const selected = applied && devices.every(d => states.get(d.id)?.tracking?.target === prompt);
      const nearest = selected && devices.some(d => states.get(d.id)?.tracking?.instance_id != null);
      const target = row.querySelector('.target-prompt');
      const label = nearest ? `Track nearest ${prompt} on both` : selected ? `Stop tracking ${prompt} on both` : `Track ${prompt || 'object'} on both`;
      target.disabled = !ready || busy || !applied;
      target.setAttribute('aria-pressed', String(Boolean(selected)));
      target.setAttribute('aria-label', label);
      target.title = applied ? label : 'Apply this prompt to both devices first';
      row.style.setProperty('--prompt-color', detections().find(d => d?.model === model)?.colors?.[index] || '#55e8ce');
    });
    const mismatched = initialized && devices.filter(device => {
      const d = states.get(device.id)?.detection;
      return d && (d.model !== effectiveModel(model, device) || !same(d.prompts, normalized(readRows())));
    });
    const notices = [];
    if (!busy && mismatched?.length) notices.push(dirty ? 'Unapplied changes' : 'Device settings differ; Update prompts applies this selection to both');
    message.textContent = error || notices.join(' · ');
    message.hidden = !message.textContent;
    message.classList.toggle('error', Boolean(error));
  }

  async function fanOut(action) {
    if (busy) return false;
    busy = true; error = '';
    clients.forEach(client => client.setSharedBusy(true)); render();
    try {
      const results = await Promise.allSettled(devices.map(async device => {
        const client = clients.get(device.id);
        if (!client) throw new Error('Device is not ready');
        return action(client, device);
      }));
      const failures = results.flatMap((r, i) => r.status === 'rejected' ? [`${devices[i].label}: ${r.reason.message || r.reason}`] : []);
      error = failures.join(' · ');
      // Successful peers are not rolled back, and ambiguous failures are never
      // retried automatically. The next explicit Update can align both again.
      return failures.length === 0;
    } finally {
      busy = false; clients.forEach(client => client.setSharedBusy(false)); render();
    }
  }

  async function applySelection(startAfter = false, targetPrompt) {
    if (busy || !initialized || !available(model)) return;
    if (targetPrompt === '') { error = 'Enter an object to target'; render(); return; }
    const selectedModel = model, prompts = normalized(readRows());
    drafts.set(model, prompts);
    const ok = await fanOut(async (client, device) => {
      const deviceModel = effectiveModel(selectedModel, device);
      if (states.get(device.id)?.detection?.model !== deviceModel)
        await client.command('/api/detection/model', {model: deviceModel});
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

  modelSelector.addEventListener('change', async () => {
    if (busy || !initialized) return;
    const next = modelSelector.value;
    if (!available(next)) { renderedModel = null; render(); return; }
    drafts.set(model, readRows());
    const oldPrompts = readRows();
    model = next; dirty = true;
    const selected = drafts.get(next) || oldPrompts;
    setRows(selected);
    await fanOut(async (client, device) => {
      const result = await client.command('/api/detection/model', {model: effectiveModel(next, device)});
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
    attach(id, client) { clients.set(id, client); },
    update(id, state) { states.set(id, state); offline.delete(id); render(); },
    offline(id) { offline.add(id); render(); },
  };
}
