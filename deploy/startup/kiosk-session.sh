#!/bin/sh
# GDM's dedicated X11 session: black root, a minimal window manager, and the demo.
# No GNOME shell, desktop, dock, welcome screen or generic autostart programs.
set -eu
export MOZ_ENABLE_WAYLAND=0 GDK_BACKEND=x11 DISABLE_WAYLAND=1
export XDG_CURRENT_DESKTOP=SpringKiosk XDG_SESSION_DESKTOP=spring-turret
/usr/bin/xsetroot -solid black
/usr/bin/xset s off
/usr/bin/xset -dpms
/usr/bin/dbus-update-activation-environment --systemd DISPLAY XAUTHORITY XDG_SESSION_TYPE XDG_CURRENT_DESKTOP XDG_SESSION_DESKTOP
/usr/bin/openbox --config-file /etc/spring-turret-kiosk/openbox.xml &
wm_pid=$!
cleanup() {
    /usr/bin/systemctl --user stop spring-turret-kiosk.service || true
    kill "$wm_pid" 2>/dev/null || true
}
trap cleanup EXIT HUP INT TERM
/usr/bin/systemctl --user restart spring-turret-kiosk.service
wait "$wm_pid"
