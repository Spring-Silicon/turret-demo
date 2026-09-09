#!/usr/bin/python3
"""Reveal Plymouth branding only on the Arc driver's native framebuffer."""
from pathlib import Path
import subprocess
import time

end = time.monotonic() + 15
while time.monotonic() < end:
    for fb in Path('/sys/class/graphics').glob('fb[0-9]*'):
        try:
            native = fb.joinpath('name').read_text().strip() == 'xedrmfb'
            size = tuple(map(int, fb.joinpath('virtual_size').read_text().strip().split(',')))
        except (OSError, ValueError):
            continue
        if native and size[0] >= 1920 and size[1] >= 1080:
            subprocess.run(['/usr/bin/plymouth', 'display-message', '--text=spring-native-display-ready'], check=False)
            raise SystemExit(0)
    time.sleep(.1)
# A missing/changed monitor must not prevent login or remote recovery.
