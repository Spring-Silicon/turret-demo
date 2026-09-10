#!/usr/bin/env node
// Local UI host. No camera, model, calibration or servo objects live here.
import http from 'node:http';
import https from 'node:https';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import {networkInterfaces} from 'node:os';
import {isIP} from 'node:net';

const STATIC = new URL('../src/spring_turret/static/', import.meta.url);
const GET_PATHS = new Set(['/api/status', '/api/detection/status',
  '/api/detection/events', '/stream.mjpg']);
const POST_PATHS = new Set(['/api/detection/model', '/api/detection/prompts',
  '/api/detection/pause',
  '/api/detection/prompt', '/api/tracking/target', '/api/tracking/instance',
  '/api/servo/arm', '/api/servo/disable', '/api/servo/recalibrate',
  '/api/servo/recover-gains',
  '/api/servo/keepalive', '/api/servo/position', '/api/servo/gains', '/api/servo/gains/reset']);
const ASSETS = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/kiosk-ready.js', ['kiosk-ready.js', 'application/javascript; charset=utf-8']],
  ['/app.js', ['app.js', 'application/javascript; charset=utf-8']],
  ['/app.css', ['app.css', 'text/css; charset=utf-8']],
]);
const SECURITY = {
  'Cache-Control': 'no-store',
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'Content-Security-Policy': "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'",
};

export function resolveWiredInterface({interfaceName, interfaceMac}, interfaces = networkInterfaces()) {
  if (interfaceMac) return Object.keys(interfaces).find(name =>
    (!interfaceName || name === interfaceName) && interfaces[name].some(info => info.mac?.toLowerCase() === interfaceMac.toLowerCase()));
  return interfaceName;
}

export function wiredLinkReady(device) {
  if (!device.interfaceName && !device.interfaceMac) return true;
  const interfaces = networkInterfaces(), interfaceName = resolveWiredInterface(device, interfaces);
  if (!interfaceName) return false;
  try {
    return readFileSync(`/sys/class/net/${interfaceName}/carrier`, 'utf8').trim() === '1' &&
      (interfaces[interfaceName] || []).some(info => info.address === device.localAddress);
  } catch { return false; }
}

export function frontendOptions(values) {
  const {backend, arc, thor, name, 'thor-interface':interfaceName,
    'thor-mac':interfaceMac, 'thor-local-address':localAddress} = values;
  if (backend && (arc || thor)) throw new Error('Choose --backend or --arc/--thor');
  if (!backend && !arc && !thor) throw new Error('Choose --arc, --thor, or both');
  if ((interfaceName || interfaceMac || localAddress) && !thor) throw new Error('Wired options require --thor');
  const thorDevice = {url:thor, label:'NVIDIA Jetson Thor', interfaceName, interfaceMac, localAddress};
  if (arc && thor) return {name:name || 'Spring Silicon + NVIDIA Jetson Thor',
    backends:{arc:{url:arc,label:'Spring Silicon'},thor:thorDevice}};
  // A single named device uses the original standalone UI and unprefixed API.
  return {backend:backend || arc || thor, name:name || (arc ? 'Spring Silicon' : thor ? 'NVIDIA Jetson Thor' : 'Turret Demo'),
          loadingLabel: thor ? 'Thor is Loading' : 'Spring is Coming',
          ...(thor && (interfaceName || interfaceMac || localAddress) ? {singleDevice:thorDevice} : {})};
}

export function createFrontend({backend, backends, name = 'Turret Demo', timeoutMs = 15000,
                                isLinkReady = wiredLinkReady, singleDevice, loadingLabel = "Spring is Coming"}) {
  if ((!backend && !backends) || (backend && backends)) throw new Error('Choose one backend or a device map');
  const unified = Boolean(backends);
  const entries = unified ? Object.entries(backends) : [['default', {...singleDevice, url: backend, label: name}]];
  if (!entries.length || entries.length > 8) throw new Error('Between one and eight devices required');
  const devices = new Map(entries.map(([id, {url, label, localAddress, interfaceName, interfaceMac}]) => {
    if (!/^[a-z][a-z0-9-]*$/.test(id)) throw new Error('Invalid device ID');
    const target = new URL(url);
    if (!['http:', 'https:'].includes(target.protocol) || target.username ||
        target.password || target.pathname !== '/' || target.search || target.hash) {
      throw new Error('Backend must be an HTTP(S) origin without credentials or a path');
    }
    if ((interfaceName || interfaceMac || localAddress) && (!isIP(localAddress || '') || !isIP(target.hostname) ||
        !(interfaceName || interfaceMac) ||
        (interfaceName && !/^[a-zA-Z0-9_.:-]{1,15}$/.test(interfaceName)) ||
        (interfaceMac && !/^([0-9a-f]{2}:){5}[0-9a-f]{2}$/i.test(interfaceMac)))) {
      throw new Error('Wired backend requires an interface and literal source/destination IPs');
    }
    const transport = target.protocol === 'https:' ? https : http;
    return [id, {id, label: label || id, prefix: unified ? `/devices/${id}` : '', target, transport,
      localAddress, interfaceName, interfaceMac,
      agents: [false, true].map(() => new transport.Agent({keepAlive: true, maxSockets: 16, maxFreeSockets: 4})),
      activeReads: 0, activeCommands: 0}];
  }));
  const title = name.replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const assets = new Map([...ASSETS].map(([path, [file, type]]) => {
    const raw = readFileSync(new URL(file, STATIC));
    const body = file === 'index.html' ? Buffer.from(raw.toString().replace('<title>Turret Demo</title>', `<title>${title}</title>`).replace('Spring is Coming', loadingLabel === 'Thor is Loading' ? loadingLabel : 'Spring is Coming')) : raw;
    return [path, {body, type}];
  }));
  if (unified) {
    assets.set('/panel.html', {body: readFileSync(new URL('index.html', STATIC)), type: 'text/html; charset=utf-8'});
    const dashboard = {body: readFileSync(new URL('dashboard.html', STATIC)), type: 'text/html; charset=utf-8'};
    assets.set('/', dashboard); assets.set('/index.html', dashboard);
    for (const file of ['dashboard.js', 'dashboard.css', 'panel.css', 'shared-controls.js']) {
      assets.set('/' + file, {body: readFileSync(new URL(file, STATIC)),
        type: file.endsWith('.js') ? 'application/javascript; charset=utf-8' : 'text/css; charset=utf-8'});
    }
    assets.set('/frontend-config', {body: Buffer.from(JSON.stringify({devices: [...devices.values()].map(({id, label, prefix}) => ({id, label, prefix}))})), type: 'application/json'});
  }
  const jsonError = (res, code, error) => {
    if (res.destroyed) return;
    if (res.headersSent) { res.destroy(); return; }
    const body = JSON.stringify({error});
    res.writeHead(code, {...SECURITY, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body)});
    res.end(body);
  };
  const kioskReady = new Map();
  const server = http.createServer({headersTimeout: 10000, requestTimeout: 15000}, async (req, res) => {
    // Bind loopback AND check Host/Origin: arbitrary websites cannot issue local
    // motor commands through DNS rebinding or cross-origin form submissions.
    const port = server.address().port;
    if (![ `127.0.0.1:${port}`, `localhost:${port}` ].includes(req.headers.host)) {
      return jsonError(res, 403, 'Local host required');
    }
    if ((req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) ||
        req.headers['sec-fetch-site'] === 'cross-site') {
      return jsonError(res, 403, 'Same-origin request required');
    }
    if (!req.url.startsWith('/') || req.url.startsWith('//') || req.url.includes('\\')) {
      return jsonError(res, 400, 'Invalid request path');
    }
    let path = req.url.split('?')[0];
    const readyToken = /^\/kiosk-ready\/([a-f0-9]{32})$/.exec(path)?.[1];
    if (readyToken && ['GET', 'POST'].includes(req.method)) {
      const now = Date.now();
      for (const [token, expires] of kioskReady) if (expires <= now) kioskReady.delete(token);
      if (req.method === 'POST') {
        req.resume();
        if (kioskReady.size >= 16) kioskReady.delete(kioskReady.keys().next().value);
        kioskReady.set(readyToken, now + 120000);
      }
      const body = JSON.stringify({ready: kioskReady.has(readyToken)});
      res.writeHead(200, {...SECURITY, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body)});
      return res.end(body);
    }
    if (assets.has(path) && ['GET', 'HEAD'].includes(req.method)) {
      const {body, type} = assets.get(path);
      res.writeHead(200, {...SECURITY, 'Content-Type': type, 'Content-Length': body.length});
      return res.end(req.method === 'HEAD' ? undefined : body);
    }
    let device = devices.get('default'), upstreamPath = req.url;
    if (unified) {
      const match = /^\/devices\/([a-z][a-z0-9-]*)(\/.*)$/.exec(path);
      device = match && devices.get(match[1]);
      if (!device) return jsonError(res, 404, 'Unknown device');
      path = match[2];
      upstreamPath = req.url.slice(device.prefix.length);
    }
    const {target, transport, agents} = device;
    // A direct-cable deployment never changes to a tailnet/Wi-Fi URL. Refuse
    // connections without the cable's address/carrier, and bind the source IP
    // on every socket (including commands and long-lived video/SSE streams).
    if (!isLinkReady(device)) return jsonError(res, 502, 'Wired backend link unavailable');
    const command = req.method === 'POST';
    if (!(command ? POST_PATHS.has(path) : req.method === 'GET' &&
        (GET_PATHS.has(path) || /^\/api\/detection\/frame\/\d+-\d+\.jpg$/.test(path)))) {
      return jsonError(res, 404, 'Not found');
    }
    // Reserve capacity for commands even if viewers hold streaming connections.
    if (command ? device.activeCommands >= 8 : device.activeReads >= 24) {
      return jsonError(res, 503, 'Frontend busy');
    }
    if (command) device.activeCommands++; else device.activeReads++;
    res.once('close', () => { if (command) device.activeCommands--; else device.activeReads--; });
    req.setTimeout(10000, () => req.destroy());
    let body = Buffer.alloc(0);
    if (command) {
      if (Number(req.headers['content-length'] || 0) > 4096) {
        return jsonError(res, 413, 'Command exceeds 4096 bytes');
      }
      try {
        for await (const chunk of req) {
          if (body.length + chunk.length > 4096) return jsonError(res, 413, 'Command exceeds 4096 bytes');
          body = Buffer.concat([body, chunk]);
        }
      } catch { return jsonError(res, 400, 'Incomplete command'); }
    }
    // Keep upstream origin fixed: neither client headers, path nor query can
    // choose a destination. Never retry commands after ambiguous failures.
    const upstream = transport.request(target, {method: req.method, path: upstreamPath,
      localAddress: device.localAddress,
      agent: agents[Number(command)], headers: command ? {
        'Content-Type': 'application/json', 'Content-Length': body.length,
      } : {Accept: req.headers.accept || '*/*'}}, reply => {
      clearTimeout(headerTimer);
      const headers = {...SECURITY, 'X-Accel-Buffering': 'no'};
      for (const key of ['content-type', 'content-length']) {
        if (reply.headers[key] !== undefined) headers[key] = reply.headers[key];
      }
      res.writeHead(reply.statusCode, headers);
      res.flushHeaders();
      // Streaming pipe bounds buffering, preserves paired JPEG+metadata bytes,
      // and does not wait for the next complete event/frame before forwarding.
      reply.on('error', () => res.destroy());
      reply.pipe(res);
      res.once('close', () => reply.destroy());
    });
    const unavailable = () => jsonError(res, 502, command
      ? 'Backend unavailable; command outcome unconfirmed. Check status before retrying.'
      : 'Backend unavailable');
    const headerTimer = setTimeout(() => upstream.destroy(new Error('Backend response timed out')), timeoutMs);
    headerTimer.unref();
    upstream.setTimeout(timeoutMs, () => upstream.destroy(new Error('Backend connection stalled')));
    upstream.once('error', unavailable);
    upstream.once('close', () => clearTimeout(headerTimer));
    res.once('close', () => { clearTimeout(headerTimer); upstream.destroy(); });
    upstream.end(command ? body : undefined);
  });
  server.maxConnections = 64 * devices.size;
  server.on('close', () => devices.forEach(device => device.agents.forEach(agent => agent.destroy())));
  return server;
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  const {values} = parseArgs({options: {backend: {type: 'string'}, arc: {type: 'string'}, thor: {type: 'string'}, port: {type: 'string', default: '8080'},
    'thor-interface': {type: 'string'}, 'thor-local-address': {type: 'string'},
    'thor-mac': {type: 'string'},
    name: {type: 'string'}, help: {type: 'boolean'}}});
  if (values.help) {
    console.log(`Usage: node ${fileURLToPath(import.meta.url)} (--backend http://HOST:8080 | --arc http://ARC:8080 | --thor http://THOR:8080 | --arc http://ARC:8080 --thor http://THOR:8080) [--port 8080] [--name TITLE] [--thor-interface enp8s0 --thor-local-address 192.168.249.1]`);
  } else {
    const port = Number(values.port);
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('A valid port is required');
    const options = frontendOptions(values);
    const server = createFrontend(options);
    server.listen(port, '127.0.0.1', () => console.log(`${options.name}: http://127.0.0.1:${port}/`));
    const stop = () => { server.close(); server.closeAllConnections(); };
    process.once('SIGTERM', stop);
    process.once('SIGINT', stop);
  }
}
