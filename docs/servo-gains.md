# Servo P/D configuration

Both installed turrets use these saved gains (September 9, 2026):

| Axis | P | D |
| --- | ---: | ---: |
| X | 400 | 400 |
| Y | 500 | 250 |

The combined and individual pages do not expose gain sliders, number inputs,
or gain-reset buttons. Manual angle sliders, Start/Stop, zero calibration,
models and prompts are unchanged. Loading/reconnecting the viewer never writes
gains; the backend restores its saved values independently of the browser.
The administrative gain API remains available, including validated persistence
and reset to the original baseline. Removing the UI does not change that baseline.

These are the XL330 internal **position-controller raw register values**, not
the camera tracking controller. P is 1–16383 (retaining the application's positive
P requirement), D is 0–16383. I, feedforward, motor goals, zeros and angle limits
are unchanged. Very high gains can cause oscillation; start with small changes.
The [ROBOTIS control table](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#position-pid-gain80-82-84-feedforward-1st2nd-gains88-90)
defines P at address 84 and D at 80; these are RAM, writable while torque is on.

Both implementations use the same command boundary:

```
POST /api/servo/gains        {"axis":"x","p":400,"d":0}
POST /api/servo/gains/reset  {"axis":"x"}
```

`servo.axes.x.position_gains` in status contains acknowledged P/I/D readback;
`gain_baseline` contains the P/D reset values. Y uses the same fields. Hardware
gains are read on connection, not on every feedback tick or browser poll.
Updates share the existing serial-bus lock, verify readback, and roll back on
write or storage failure. Unconfirmed rollback enters normal hardware recovery.

Overrides are atomically saved beside the configured calibration file, replacing
its `.json` suffix with `.gains.json`. Back up this file with the corresponding
turret's calibration; its axis IDs/directions must match. No weights or OS image
changes are needed. Without `calibration_file`, overrides last for the process
only. Reconnection/restart followed by Start reapplies saved overrides. Reset
uses installed `servo.axes.<axis>.position_gains` P/D when configured; otherwise
the initially read hardware values, saved with the first explicit adjustment.
Reset does not promote a tuned value to the baseline or rewrite mechanical zeros.

Deploy `servo.py`, `servo_gains.py`, `server.py`, and `interfaces.py` to both
backend packages; deploy `static/{index.html,app.js,app.css}` and
`scripts/frontend.mjs` to the local viewer. Restart the affected services once
and preserve the current model/prompts/target and operator run intent. There is
no gain change needed to validate deployment: GET status reports actual gains.
