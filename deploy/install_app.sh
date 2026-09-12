#!/usr/bin/env bash
set -euo pipefail

if [[ $(id -u) != 0 ]]; then
  echo 'Run with sudo.' >&2
  exit 1
fi
readonly APP_DIR=/opt/tron-vanity
test -f "$APP_DIR/deploy/requirements-l4.txt"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv jq tmux
# Re-running preserves the venv and pip download cache.
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/python" -m pip install -r "$APP_DIR/deploy/requirements-l4.txt"
"$APP_DIR/.venv/bin/python" -m pip check
install -m 0755 "$APP_DIR/deploy/secure_vanity_run.py" "$APP_DIR/secure_vanity_run.py"
install -m 0755 "$APP_DIR/deploy/tron-vanity" /usr/local/bin/tron-vanity
echo TRON_VANITY_APP_INSTALLED
