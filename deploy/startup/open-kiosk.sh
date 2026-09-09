#!/bin/sh
set -eu
url=http://127.0.0.1:8081/
profile="$HOME/snap/firefox/common/spring-turret-kiosk-profile"
test -d "$profile"
# Only wait for the local page. Either backend may still be booting/compiling;
# normal read-only UI reconnects recover without sending any motor commands.
attempt=0
until /usr/bin/curl --fail --silent --max-time 2 "$url" >/dev/null; do
    attempt=$((attempt + 1))
    test "$attempt" -lt 60 || exit 1
    sleep 1
done
exec /usr/bin/firefox --no-remote --profile "$profile" --kiosk "$url"
