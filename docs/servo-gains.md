# Servo P/D controls

The combined page has **one P slider, one D slider and one Reset P/D button**
shared by X/Y on both turrets. Changing P applies P to all four servos while
preserving each one's D, and vice versa. Reset restores each axis's saved baseline.
Different readbacks display **Mixed**, not whichever device replied last. All
devices must be connected to adjust the shared controls; partial failures name
the device/axis and are not automatically retried. Model and Start/Stop controls
are unaffected. Standalone device pages retain independent X/Y tuning controls.

A slider commits on release or keyboard
change. Opening, polling, or reconnecting the page never writes gains. Stop stays
available during a gain update; changing gains never arms a stopped turret.

Shared sliders default to **P 1–2000 / D 0–2000**, step 1, with visible endpoints.
Type exact values into the adjacent number fields and press Enter or leave the
field to apply. Typing alone does not send commands; Enter here never starts
motors. Numbers accept the full register range below. The slider expands in
500-unit increments when a typed or existing gain exceeds 2000, up to 16383.
Blank, fractional and out-of-range inputs are rejected without motor writes.

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
