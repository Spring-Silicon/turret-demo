#!/usr/bin/python3
"""Keep the native brand cover visible until Firefox paints the local demo."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time
import urllib.request
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gdk, GLib, Gtk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8081/')
    parser.add_argument('--profile', type=Path, default=Path.home() / 'snap/firefox/common/spring-turret-kiosk-profile')
    args = parser.parse_args()
    if not args.profile.is_dir():
        raise RuntimeError('Dedicated Firefox profile is missing')
    token = secrets.token_hex(16)
    url = args.url.rstrip('/') + '/?kiosk=' + token
    ready_url = args.url.rstrip('/') + '/kiosk-ready/' + token
    cover = Gtk.Window(type=Gtk.WindowType.POPUP)
    cover.set_title('Spring startup cover')
    cover.set_accept_focus(False)
    cover.set_decorated(False)
    screen = Gdk.Display.get_default()
    monitor = screen.get_primary_monitor() or screen.get_monitor(0)
    geometry = monitor.get_geometry()
    cover.move(geometry.x, geometry.y)
    cover.set_default_size(geometry.width, geometry.height)
    css = Gtk.CssProvider()
    css.load_from_data(b'window { background-color: #000; }')
    cover.get_style_context().add_provider(css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    logo = Gtk.Image.new_from_file('/usr/share/plymouth/themes/spring-silicon/spring-silicon-plymouth.png')
    logo.set_halign(Gtk.Align.CENTER)
    logo.set_valign(Gtk.Align.CENTER)
    cover.add(logo)
    cover.show_all()
    cover.get_window().set_cursor(Gdk.Cursor.new_for_display(screen, Gdk.CursorType.BLANK_CURSOR))
    began = time.monotonic()
    process = None
    launched = None
    revealed = False
    last_attempt = 0

    def stop(*_):
        if process is not None and process.poll() is None:
            process.terminate()
        Gtk.main_quit()

    def tick():
        nonlocal process, launched, revealed, last_attempt
        now = time.monotonic()
        if process is None:
            # The logo is already painted while the local frontend starts.
            if now - last_attempt < 1:
                return True
            last_attempt = now
            try:
                with urllib.request.urlopen(args.url, timeout=.5) as response:
                    if response.status != 200:
                        return True
            except OSError:
                if now - began > 90:
                    print('Local frontend did not become ready', flush=True)
                    Gtk.main_quit()
                return True
            environment = {**os.environ, 'MOZ_ENABLE_WAYLAND': '0', 'GDK_BACKEND': 'x11', 'DISABLE_WAYLAND': '1'}
            process = subprocess.Popen(['/usr/bin/firefox', '--no-remote', '--profile', str(args.profile), '--kiosk', url], env=environment)
            launched = now
        if process.poll() is not None:
            Gtk.main_quit()
            return False
        if not revealed:
            cover.get_window().raise_()
            ready = False
            try:
                with urllib.request.urlopen(ready_url, timeout=.5) as response:
                    ready = json.load(response).get('ready') is True
            except (OSError, ValueError):
                pass
            if ready:
                cover.hide()
                revealed = True
                print('Local demo painted; startup cover released', flush=True)
            elif now - launched > 90:
                # A failed browser load must not strand a successful session
                # behind a permanent cover. Restart this viewer via systemd.
                print('Browser did not paint the local demo; restarting viewer', flush=True)
                stop()
                return False
        return True

    for sig in (signal.SIGTERM, signal.SIGINT):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, stop)
    GLib.timeout_add(200, tick)
    Gtk.main()
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    return 0 if revealed and process is not None and process.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
