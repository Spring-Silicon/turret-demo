import assert from 'node:assert/strict';
import http from 'node:http';
import {once} from 'node:events';
import {test} from 'node:test';
import {createFrontend, frontendOptions} from '../scripts/frontend.mjs';

test('named Arc, Thor and combined modes share the same launcher', async t => {
  for (const name of ['arc','thor']) {
    const requests=[];
    const backend=http.createServer((req,res)=>{requests.push(req.url);res.end('{}');});
    const origin=await listen(backend);
    const options=frontendOptions({[name]:origin});
    assert.equal(options.backend,origin);
    assert.equal(options.name,name === 'arc' ? 'Spring Silicon' : 'NVIDIA Jetson Thor');
    const frontend=createFrontend(options), local=await listen(frontend);
    t.after(async()=>{await stop(frontend);await stop(backend);});
    const html=await (await fetch(local)).text();
    assert.match(html,/id="motor-toggle"/);
    assert.doesNotMatch(html,/id="devices"/);
    assert.equal(requests.length,0);
    await fetch(local+'/api/status');
    assert.deepEqual(requests,['/api/status']);
  }
  const both=frontendOptions({arc:'http://arc:8080',thor:'http://thor:8080'});
  assert.deepEqual(Object.keys(both.backends),['arc','thor']);
  assert.equal(both.backends.arc.label,'Spring Silicon');
  assert.equal(both.backends.thor.label,'NVIDIA Jetson Thor');
  const combined=createFrontend(both), combinedLocal=await listen(combined);
  t.after(()=>stop(combined));
  const labels=await (await fetch(combinedLocal+'/frontend-config')).json();
  assert.deepEqual(labels.devices.map(({id,label})=>({id,label})),[
    {id:'arc',label:'Spring Silicon'}, {id:'thor',label:'NVIDIA Jetson Thor'},
  ]);
  const combinedHtml=await (await fetch(combinedLocal)).text();
  assert.match(combinedHtml,/<title>Spring Silicon \+ NVIDIA Jetson Thor · Turret Demo<\/title>/);
  assert.equal(both.backend,undefined);
  assert.equal(frontendOptions({backend:'http://host:8080'}).name,'Turret Demo');
  for (const invalid of [{},{backend:'http://host',thor:'http://thor'},
    {arc:'http://arc','thor-interface':'eth0'}]) assert.throws(()=>frontendOptions(invalid));
  const wired=frontendOptions({thor:'http://192.168.249.2:8080',
    'thor-interface':'enp8s0','thor-local-address':'192.168.249.1'});
  assert.equal(wired.singleDevice.localAddress,'192.168.249.1');
});

async function listen(server) {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return `http://127.0.0.1:${server.address().port}`;
}
async function stop(server) {
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
async function fixture(t, handler, options = {}) {
  const backend = http.createServer(handler);
  const origin = await listen(backend);
  const frontend = createFrontend({backend: origin, ...options});
  const local = await listen(frontend);
  t.after(async () => { await stop(frontend); await stop(backend); });
  return {local, backend, frontend};
}

test('serves local assets without requesting upstream UI, even when offline', async t => {
  let calls = 0;
  const {local, backend} = await fixture(t, (_, res) => { calls++; res.end('REMOTE UI'); }, {name: 'Arc <local>'});
  const html = await (await fetch(local)).text();
  assert.match(html, /<title>Arc &lt;local&gt;<\/title>/);
  assert.match(html, /id="motor-toggle"/);
  assert.match(await (await fetch(`${local}/app.js`)).text(), /EventSource/);
  assert.match(await (await fetch(`${local}/app.css`)).text(), /\.feed/);
  assert.equal((await fetch(`${local}/app.css`, {method: 'HEAD'})).status, 200);
  assert.equal(calls, 0);
  await stop(backend);
  assert.equal((await fetch(local)).status, 200);
  const bad = await fetch(`${local}/api/status`);
  assert.equal(bad.status, 502);
  assert.equal((await bad.json()).error, 'Backend unavailable');
});

test('preserves API queries, command body, errors and exact image bytes', async t => {
  const received = [];
  const {local} = await fixture(t, async (req, res) => {
    let body = ''; for await (const chunk of req) body += chunk;
    received.push({url: req.url, body, method: req.method});
    if (req.url.endsWith('.jpg')) {res.setHeader('Content-Type', 'image/jpeg'); return res.end(Buffer.from([255,216,0,1,255,217]));}
    res.writeHead(req.method === 'POST' ? 409 : 200, {'Content-Type': 'application/json'});
    res.end(JSON.stringify({error: 'motors stopped', frame_url: '/api/detection/frame/2-3.jpg'}));
  });
  assert.equal((await fetch(`${local}/api/detection/status?revision=2&sequence=3`)).status, 200);
  const response = await fetch(`${local}/api/servo/position`, {method: 'POST', body: '{"axis":"x","degrees":12}', headers: {Origin: local}});
  assert.equal(response.status, 409);
  assert.equal((await response.json()).frame_url, '/api/detection/frame/2-3.jpg');
  assert.equal(received[0].url, '/api/detection/status?revision=2&sequence=3');
  assert.equal(received[1].body, '{"axis":"x","degrees":12}');
  assert.deepEqual(Buffer.from(await (await fetch(`${local}/api/detection/frame/2-3.jpg`)).arrayBuffer()), Buffer.from([255,216,0,1,255,217]));
});

test('SSE and MJPEG stream immediately; disconnect releases upstream', async t => {
  for (const [path, type, first] of [
    ['/api/detection/events', 'text/event-stream', 'data: {"frame_sequence":9,"jpeg":"abc"}\n\n'],
    ['/stream.mjpg', 'multipart/x-mixed-replace; boundary=frame', '--frame\r\nContent-Type: image/jpeg\r\n\r\nJPEG'],
  ]) {
    let closed;
    const disconnected = new Promise(resolve => { closed = resolve; });
    const {local} = await fixture(t, (_, res) => {
      res.writeHead(200, {'Content-Type': type});
      res.write(first); // Deliberately never finish the response.
      res.once('close', closed);
    });
    const abort = new AbortController();
    const response = await fetch(local + path, {signal: abort.signal});
    assert.equal(response.headers.get('content-type'), type);
    assert.equal(new TextDecoder().decode((await response.body.getReader().read()).value), first);
    abort.abort();
    await Promise.race([disconnected, new Promise((_, reject) => {
      const timer = setTimeout(() => reject(new Error('stream leaked upstream')), 1500); timer.unref();
    })]);
  }
});

test('commands are never retried after a lost reply; header stalls time out', async t => {
  let calls = 0;
  const {local} = await fixture(t, (req, _) => {calls++; if (req.method === 'POST') req.socket.destroy();}, {timeoutMs: 100});
  const response = await fetch(`${local}/api/servo/disable`, {method: 'POST', body: '{}'});
  assert.equal(response.status, 502);
  assert.match((await response.json()).error, /outcome unconfirmed/);
  assert.equal(calls, 1);
  assert.equal((await fetch(`${local}/api/status`)).status, 502);
  assert.equal(calls, 2);
});

test('rejects cross-origin motor writes, oversized commands and unknown routes', async t => {
  let calls = 0;
  const {local} = await fixture(t, (_, res) => {calls++; res.end('{}');});
  for (const headers of [{Origin: 'https://foreign.example'}, {'Sec-Fetch-Site': 'cross-site'}]) {
    assert.equal((await fetch(`${local}/api/servo/arm`, {method: 'POST', headers})).status, 403);
  }
  assert.equal((await fetch(`${local}/api/servo/arm`, {method: 'POST', body: 'x'.repeat(4097)})).status, 413);
  for (const path of ['/etc/passwd', '/api/unknown', '//foreign.example/api/status', '/api/detection/frame/../../secret.jpg']) {
    assert.notEqual((await fetch(local + path)).status, 200);
  }
  const status = await new Promise(resolve => {
    http.get(local + '/api/status', {headers: {Host: 'foreign.example'}}, response => {response.resume(); resolve(response.statusCode);});
  });
  assert.equal(status, 403);
  assert.equal(calls, 0);
});

test('rejects unsafe or ambiguous upstream configuration', () => {
  for (const backend of ['file:///tmp', 'http://user:pass@host', 'http://host/api', 'http://host/?url=other', 'http://host/#fragment']) {
    assert.throws(() => createFrontend({backend}), /origin/);
  }
  for (const options of [{interfaceName:'enp8s0'}, {localAddress:'192.168.249.1'},
                        {interfaceName:'../eth', localAddress:'192.168.249.1'}]) {
    assert.throws(() => createFrontend({backends:{thor:{url:'http://192.168.249.2:8080',...options}}}), /Wired backend/);
  }
  assert.throws(() => createFrontend({backends:{thor:{url:'http://agxthor-4:8080',
    interfaceName:'enp8s0',localAddress:'192.168.249.1'}}}), /literal/);
});

test('wired backend binds its source and fails closed without affecting local Arc', async t => {
  let calls = 0, linkUp = true;
  const backend = http.createServer((req,res) => {
    calls++; assert.equal(req.socket.remoteAddress,'127.0.0.1'); res.end('{}');
  });
  const origin = await listen(backend);
  const frontend = createFrontend({backends:{arc:{url:origin},
    thor:{url:origin,interfaceName:'ethernet-test',localAddress:'127.0.0.1'}},
    isLinkReady: device => !device.interfaceName || linkUp});
  const local = await listen(frontend);
  t.after(async () => {await stop(frontend); await stop(backend);});
  assert.equal((await fetch(local+'/devices/thor/api/status')).status,200);
  linkUp = false;
  assert.equal((await fetch(local+'/devices/thor/api/status')).status,502);
  assert.equal((await fetch(local+'/devices/thor/api/servo/arm',{method:'POST',body:'{}'})).status,502);
  assert.equal(calls,1);
  assert.equal((await fetch(local+'/devices/arc/api/status')).status,200);
  assert.equal(calls,2);
});

test('unified page keeps status, commands and streams pinned to separate backends', async t => {
  const requests = {arc: [], thor: []}, backends = {}, servers = [];
  for (const id of ['arc', 'thor']) {
    const server = http.createServer(async (req, res) => {
      let body = ''; for await (const chunk of req) body += chunk;
      requests[id].push({path: req.url, method: req.method, body});
      if (req.url === '/api/detection/events') {
        res.writeHead(200, {'Content-Type': 'text/event-stream'});
        return res.end(`data: {"device":"${id}","jpeg":"${id}"}\n\n`);
      }
      res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({device: id}));
    });
    backends[id] = {url: await listen(server), label: id === 'arc' ? 'Arc' : 'Thor'};
    servers.push(server);
  }
  const frontend = createFrontend({backends});
  const local = await listen(frontend);
  t.after(async () => {await stop(frontend); await Promise.all(servers.map(stop));});
  const html = await (await fetch(local)).text();
  assert.match(html, /id="devices"/); assert.doesNotMatch(html, /iframe/);
  const config = await (await fetch(local + '/frontend-config')).json();
  assert.deepEqual(config.devices.map(d => d.prefix), ['/devices/arc', '/devices/thor']);
  assert.match(html, /id="shared-controls"/);
  for (const file of ['/panel.html', '/dashboard.js', '/dashboard.css', '/panel.css', '/shared-controls.js']) {
    assert.equal((await fetch(local + file)).status, 200);
  }
  assert.equal(requests.arc.length + requests.thor.length, 0);
  for (const id of ['arc', 'thor']) {
    assert.equal((await (await fetch(`${local}/devices/${id}/api/status`)).json()).device, id);
    await fetch(`${local}/devices/${id}/api/servo/position`, {method: 'POST', body: JSON.stringify({axis: 'x', degrees: id === 'arc' ? 10 : -10})});
    const event = await (await fetch(`${local}/devices/${id}/api/detection/events`)).text();
    assert.match(event, new RegExp(`"device":"${id}"`));
    assert.equal(requests[id][1].path, '/api/servo/position');
    assert.equal(JSON.parse(requests[id][1].body).degrees, id === 'arc' ? 10 : -10);
  }
  assert.equal((await fetch(local + '/api/servo/arm', {method: 'POST'})).status, 404);
  assert.equal((await fetch(local + '/devices/unknown/api/status')).status, 404);
  assert.equal(requests.arc.length, 3); assert.equal(requests.thor.length, 3);
  await stop(servers[0]);
  assert.equal((await fetch(local + '/devices/arc/api/status')).status, 502);
  assert.equal((await (await fetch(local + '/devices/thor/api/status')).json()).device, 'thor');
});
