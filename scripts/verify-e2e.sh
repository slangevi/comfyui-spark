#!/usr/bin/env bash
# Asserts a real generation completes end to end. Skips cleanly, and exits 0,
# when no checkpoint is installed (spec §8).
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f .env ] && . ./.env

PORT="${COMFYUI_PORT:-8188}"
BASE="http://127.0.0.1:${PORT}"
TIMEOUT="${VERIFY_E2E_TIMEOUT:-300}"

echo "==> verify-e2e: end-to-end generation"

./scripts/verify-http.sh >/dev/null

# Ask ComfyUI itself what checkpoints it can see, rather than guessing from
# the host filesystem — this also proves --base-directory resolved correctly.
ckpt="$(curl -fsS "${BASE}/object_info/CheckpointLoaderSimple" \
    | python3 -c '
import json, sys
info = json.load(sys.stdin)["CheckpointLoaderSimple"]
names = info["input"]["required"]["ckpt_name"][0]
print(names[0] if names else "")
')"

if [ -z "$ckpt" ]; then
    echo "    SKIP: no checkpoint installed."
    echo "    Install one, then re-run:"
    echo "      make fetch-model URL=<safetensors-url> DEST=checkpoints"
    echo "==> verify-e2e: SKIPPED"
    exit 0
fi
echo "    using checkpoint: $ckpt"

payload="$(mktemp)"; result="$(mktemp)"
trap 'rm -f "$payload" "$result"' EXIT

python3 - "$ckpt" "$payload" <<'PY'
import json, sys
ckpt, out = sys.argv[1], sys.argv[2]
graph = json.load(open("workflows/minimal-txt2img.json"))
graph["4"]["inputs"]["ckpt_name"] = ckpt
json.dump({"prompt": graph, "client_id": "verify-e2e"}, open(out, "w"))
PY

prompt_id="$(curl -fsS -X POST "${BASE}/prompt" -H 'Content-Type: application/json' \
    --data-binary "@${payload}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["prompt_id"])')"
echo "    queued prompt $prompt_id"

deadline=$((SECONDS + TIMEOUT))
while true; do
    curl -fsS "${BASE}/history/${prompt_id}" -o "$result"
    status="$(python3 - "$result" "$prompt_id" <<'PY'
import json, sys
hist = json.load(open(sys.argv[1])).get(sys.argv[2])
if not hist:
    print("pending"); raise SystemExit
print(hist.get("status", {}).get("status_str", "pending"))
PY
)"
    case "$status" in
        success) break ;;
        error)
            echo "FAIL: ComfyUI reported an execution error" >&2
            python3 -m json.tool "$result" >&2
            exit 1 ;;
    esac
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "FAIL: generation did not finish within ${TIMEOUT}s" >&2
        docker compose logs --tail=40 comfyui >&2 || true
        exit 1
    fi
    sleep 3
done

# The graph must have produced a real image on disk, not merely reported success.
images="$(python3 - "$result" "$prompt_id" <<'PY'
import json, sys
hist = json.load(open(sys.argv[1]))[sys.argv[2]]
names = [img["filename"]
         for node in hist.get("outputs", {}).values()
         for img in node.get("images", [])]
print("\n".join(names))
PY
)"
[ -n "$images" ] || { echo "FAIL: run succeeded but produced no images" >&2; exit 1; }

while read -r name; do
    [ -n "$name" ] || continue
    # </dev/null so exec does not consume the here-string driving this loop.
    docker compose exec -T comfyui test -s "/data/output/$name" </dev/null \
        || { echo "FAIL: $name is missing or empty in /data/output" >&2; exit 1; }
    echo "    wrote /data/output/$name"
done <<< "$images"

echo "==> verify-e2e: PASS"
