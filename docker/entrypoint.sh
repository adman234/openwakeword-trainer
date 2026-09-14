#!/bin/sh
set -e

mkdir -p "$DATA_DIR/certs" "$DATA_DIR/projects" "$DATA_DIR/downloads" "$DATA_DIR/background" "$DATA_DIR/cache" "$OUTPUT_DIR"

# Browsers only allow the microphone on HTTPS pages, so generate a self-signed certificate once.
# CERT_HOSTS can add IPs or hostnames (comma separated). Delete /data/certs to regenerate.
if [ ! -s "$DATA_DIR/certs/cert.pem" ] || [ ! -s "$DATA_DIR/certs/key.pem" ]; then
  SAN="DNS:localhost,IP:127.0.0.1"
  for host in $(echo "${CERT_HOSTS:-}" | tr ',' ' '); do
    if echo "$host" | grep -Eq '^[0-9.]+$|:'; then SAN="$SAN,IP:$host"; else SAN="$SAN,DNS:$host"; fi
  done
  echo "Creating self-signed HTTPS certificate ($SAN)"
  openssl req -x509 -nodes -newkey rsa:2048 -days 3650 -subj "/CN=openwakeword-trainer" \
    -addext "subjectAltName=$SAN" \
    -keyout "$DATA_DIR/certs/key.pem" -out "$DATA_DIR/certs/cert.pem" >/dev/null 2>&1
fi

# On Unraid, run as nobody:users (PUID=99, PGID=100) so files in appdata stay manageable from shares.
if [ -n "${PUID:-}" ] && [ -n "${PGID:-}" ] && [ "$(id -u)" = "0" ] && [ "$PUID" != "0" ]; then
  export HOME="$DATA_DIR/cache/home" USER=wakeword
  mkdir -p "$HOME"
  find "$DATA_DIR" "$OUTPUT_DIR" \( ! -user "$PUID" -o ! -group "$PGID" \) -exec chown "$PUID:$PGID" {} + 2>/dev/null || true
  exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups python -m owwt
fi

exec python -m owwt
