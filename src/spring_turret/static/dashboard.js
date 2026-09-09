"use strict";

async function loadDashboard() {
  const [configResponse, templateResponse] = await Promise.all([
    fetch('/frontend-config', {cache: 'no-store'}),
    fetch('/panel.html', {cache: 'no-store'}),
  ]);
  if (!configResponse.ok || !templateResponse.ok) throw new Error('Unable to load device panels');
  const {devices} = await configResponse.json();
  const template = new DOMParser().parseFromString(await templateResponse.text(), 'text/html');
  template.querySelectorAll('script').forEach(script => script.remove());
  const sharedHost = document.getElementById('shared-controls');
  for (const selector of ['.model-control', '#detection-form', '#prompt-row-template']) {
    sharedHost.insertBefore(template.querySelector(selector).cloneNode(true), document.getElementById('shared-message'));
    template.querySelector(selector).remove();
  }
  const shared = mountSharedControls(document, devices);
  for (const device of devices) {
    const host = document.createElement('article');
    host.className = 'device-panel';
    host.setAttribute('aria-label', `${device.label} demo`);
    host.dataset.device = device.id;
    const root = host.attachShadow({mode: 'open'});
    for (const href of ['/app.css', '/panel.css']) {
      const link = document.createElement('link');
      link.rel = 'stylesheet'; link.href = href; root.append(link);
    }
    root.append(...[...template.body.children].map(child => child.cloneNode(true)));
    root.querySelector('h1').textContent = device.label;
    document.getElementById('devices').append(host);
    // Scope frames and motor/instance control to this panel. Only class-level
    // inference controls are shared; a clicked native ID belongs to one device.
    // Escape acts only on the panel containing keyboard focus.
    const client = mountTurret({
      getElementById: id => root.getElementById(id),
      createElement: tag => document.createElement(tag),
      querySelectorAll: selector => root.querySelectorAll(selector),
      addEventListener: (...args) => root.addEventListener(...args),
    }, device.prefix, {sharedControls: true,
      onStatus: state => shared.update(device.id, state),
      onOffline: () => shared.offline(device.id),
    });
    shared.attach(device.id, client);
  }
}

loadDashboard().catch(error => {
  const message = document.getElementById('dashboard-error');
  message.textContent = error.message;
  message.hidden = false;
});
