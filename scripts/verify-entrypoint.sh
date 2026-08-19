#!/usr/bin/env bash
# Asserts the six startup behaviors of spec §7.6, without needing compose.
set -euo pipefail
IMAGE="${IMAGE:-comfyui-spark:latest}"
TMP="$(mktemp -d)"
trap 'chmod -R u+w "$TMP" 2>/dev/null || true; rm -rf "$TMP"' EXIT

echo "==> verify-entrypoint: startup contract"

# Case 1 — a cold start seeds the /data tree and builds the overlay venv.
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
for p in models/checkpoints models/loras models/text_encoders models/controlnet \
         input output temp user custom_nodes/ComfyUI-Manager venv/bin/python; do
    [ -e "$TMP/$p" ] || { echo "FAIL: cold start did not create $p" >&2; exit 1; }
done
echo "    cold start seeded /data and created the overlay venv"

# Case 2 — the overlay venv really does inherit the baked torch.
docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
    -c 'import torch; assert torch.cuda.is_available(); print("    overlay venv sees torch", torch.__version__)'

# Case 3 — a second start is idempotent and must not re-clone the Manager.
MARKER="$TMP/custom_nodes/ComfyUI-Manager/.idempotency-marker"
touch "$MARKER"
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
[ -e "$MARKER" ] || { echo "FAIL: second start re-cloned ComfyUI-Manager" >&2; exit 1; }
echo "    second start left the existing Manager checkout alone"

# Case 4 — no GPU and no override must refuse to start.
# NVIDIA_VISIBLE_DEVICES=void is required: the host's default runtime is
# nvidia, so omitting --gpus would still expose the GPU.
if docker run --rm -e NVIDIA_VISIBLE_DEVICES=void -e COMFYUI_SKIP_LAUNCH=1 \
        -v "$TMP:/data" "$IMAGE" >/dev/null 2>&1; then
    echo "FAIL: started with no GPU instead of refusing" >&2; exit 1
fi
echo "    refused to start with no GPU visible"

# Case 5 — the documented override works.
docker run --rm -e NVIDIA_VISIBLE_DEVICES=void -e COMFYUI_ALLOW_CPU=1 \
    -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
echo "    COMFYUI_ALLOW_CPU=1 bypasses the GPU guard"

# Case 6 — an unwritable /data fails fast rather than half-starting.
if docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 \
        -v "$TMP:/data:ro" "$IMAGE" >/dev/null 2>&1; then
    echo "FAIL: accepted a read-only /data" >&2; exit 1
fi
echo "    refused to start with an unwritable /data"

echo "==> verify-entrypoint: PASS"
