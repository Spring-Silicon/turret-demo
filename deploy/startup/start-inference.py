#!/usr/bin/env python3
"""Seed the unattended demo once; never change models or send motor commands."""
import json
import time
from urllib.request import Request, urlopen


def start_inference(open_url=urlopen, sleep=time.sleep, attempts=60):
    url = 'http://127.0.0.1:8080'
    for attempt in range(attempts):
        try:
            with open_url(url + '/api/status', timeout=2) as response:
                state = json.load(response)['detection']
            break
        except (OSError, ValueError, KeyError):
            if attempt == attempts - 1:
                raise RuntimeError('Backend did not become ready for startup inference')
            sleep(1)
    if state.get('prompts'):
        print('Existing prompts preserved; no startup command needed', flush=True)
        return
    if not state.get('enabled'):
        raise RuntimeError('Inference is disabled; startup will not override it')
    request = Request(url + '/api/detection/prompts',
                      data=json.dumps({'prompts': ['person']}).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    # Readiness GETs may retry, but never replay a command after an ambiguous
    # failure. This oneshot has no restart policy and cannot restart the backend.
    with open_url(request, timeout=10) as response:
        response.read()
    print('Startup inference requested for person; model and motor state unchanged', flush=True)


if __name__ == '__main__':
    start_inference()
