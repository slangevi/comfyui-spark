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

payload="$(mktemp)"; result="$(mktemp)"
trap 'rm -f "$payload" "$result"' EXIT

# Prefer the library's z-image-turbo workflow: its Qwen3 text encoder takes
# the Triton path (torch._native kernels compiled against a C toolchain on
# first use) that SD 1.5's CLIP never does, so this catches an image missing
# gcc/python3-dev. Fall back to SD 1.5 when the library's models aren't
# installed, so a fresh deployment still verifies. Both decisions come from
# what ComfyUI itself can see (/models/<folder>), which also proves
# --base-directory resolved correctly.
z_ready="$(curl -fsS "${BASE}/models/diffusion_models" "${BASE}/models/text_encoders" 2>/dev/null \
    | python3 -c '
import json, sys
seen = set()
for line in sys.stdin.read().split("]")[:-1]:
    seen.update(json.loads(line + "]"))
need = {"z_image_turbo_nvfp4.safetensors", "qwen_3_4b_fp8_mixed.safetensors"}
print("yes" if need <= seen and __import__("os").path.exists("workflows/z-image-turbo.params.json") else "no")
')"

if [ "$z_ready" = "yes" ]; then
    echo "    using workflow: z-image-turbo (Qwen3 text encoder — exercises the Triton path)"
    python3 - "$payload" <<'PY'
import json, sys
out = sys.argv[1]
graph = json.load(open("workflows/z-image-turbo.json"))
params = json.load(open("workflows/z-image-turbo.params.json"))["params"]
def setp(name, value):
    node, _, key = params[name]["path"].partition(".inputs.")
    graph[node]["inputs"][key] = value
setp("prompt", "a red apple on a wooden table, soft window light")
setp("seed", 20260922)
setp("prefix", "verify-e2e")
json.dump({"prompt": graph, "client_id": "verify-e2e"}, open(out, "w"))
PY
else
    ckpt="$(curl -fsS "${BASE}/object_info/CheckpointLoaderSimple" \
        | python3 -c '
import json, sys
info = json.load(sys.stdin)["CheckpointLoaderSimple"]
names = info["input"]["required"]["ckpt_name"][0]
print(names[0] if names else "")
')"
    if [ -z "$ckpt" ]; then
        echo "    SKIP: neither the z-image-turbo models nor any checkpoint is installed."
        echo "    Fetch the library, or install one checkpoint, then re-run:"
        echo "      HF_TOKEN=\$(cat ~/.config/hf/token) ./scripts/fetch-library.sh"
        echo "      make fetch-model URL=<safetensors-url> DEST=checkpoints"
        echo "==> verify-e2e: SKIPPED"
        exit 0
    fi
    echo "    using workflow: minimal-txt2img with checkpoint $ckpt (library models not installed)"
    python3 - "$ckpt" "$payload" <<'PY'
import json, sys
ckpt, out = sys.argv[1], sys.argv[2]
graph = json.load(open("workflows/minimal-txt2img.json"))
graph["4"]["inputs"]["ckpt_name"] = ckpt
json.dump({"prompt": graph, "client_id": "verify-e2e"}, open(out, "w"))
PY
fi

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
