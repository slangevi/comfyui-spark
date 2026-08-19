#!/usr/bin/env bash
# Asserts the running service serves /system_stats backed by a CUDA device.
# Also used by other scripts as a readiness gate.
set -euo pipefail
cd "$(dirname "$0")/.."

# Honour a non-default port set in .env, not just one exported in the shell.
# shellcheck disable=SC1091
[ -f .env ] && . ./.env

PORT="${COMFYUI_PORT:-8188}"
URL="http://127.0.0.1:${PORT}/system_stats"
TIMEOUT="${VERIFY_HTTP_TIMEOUT:-240}"

echo "==> verify-http: waiting up to ${TIMEOUT}s for ${URL}"
deadline=$((SECONDS + TIMEOUT))
until curl -fsS "$URL" -o /dev/null 2>/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "FAIL: no successful response from $URL within ${TIMEOUT}s" >&2
        docker compose logs --tail=40 comfyui >&2 || true
        exit 1
    fi
    sleep 3
done

resp="$(mktemp)"
trap 'rm -f "$resp"' EXIT
curl -fsS "$URL" -o "$resp"

python3 - "$resp" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
system = data["system"]
devices = data["devices"]
print("    comfyui   ", system["comfyui_version"])
print("    pytorch   ", system["pytorch_version"])
if not devices:
    sys.exit("FAIL: /system_stats reported no devices at all")
dev = devices[0]
print("    device    ", dev["name"], "(type=%s)" % dev["type"])
if dev["type"] != "cuda":
    sys.exit("FAIL: primary device type is %r, not 'cuda' — ComfyUI is on the CPU" % dev["type"])
if "GB10" not in dev["name"]:
    sys.exit("FAIL: expected a GB10, got %r" % dev["name"])
PY

echo "==> verify-http: PASS"
