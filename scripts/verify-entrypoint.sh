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

overlay_pth() { find "$TMP/venv" -name '_baked_venv.pth'; }
overlay_sentinel() { find "$TMP/venv" -name '.overlay-venv-built'; }

# Case 7 — sentinel absent + broken import: the venv never finished its
# first build, so it must be rebuilt automatically.
SENTINEL="$(overlay_sentinel)"
[ -n "$SENTINEL" ] || { echo "FAIL: expected an existing .overlay-venv-built sentinel from Case 1" >&2; exit 1; }
PTH="$(overlay_pth)"
[ -n "$PTH" ] || { echo "FAIL: expected an existing _baked_venv.pth from Case 1" >&2; exit 1; }
rm -f "$SENTINEL" "$PTH"
if docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
        -c 'import torch' >/dev/null 2>&1; then
    echo "FAIL: overlay venv still imported torch after removing its .pth — test setup is invalid" >&2; exit 1
fi
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
    -c 'import torch; print("    sentinel-absent + broken venv: rebuilt automatically, torch", torch.__version__)'
[ -n "$(overlay_sentinel)" ] || { echo "FAIL: rebuild did not (re)write the completion sentinel" >&2; exit 1; }

# Case 8 — sentinel present + broken import: a previously-working venv must
# NOT be silently wiped. It must refuse to start, and the user's own
# overlay-installed package must still be importable afterward.
#
# The installed file's path is resolved through the interpreter itself
# (six.__file__), not a bare `find -name six.py`: pip vendors its own copies
# of six under pip/_vendor/, so a name-based find matches multiple files and
# would make the survival check meaningless.
docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/pip "$IMAGE" \
    install --no-cache-dir six >/dev/null
SIX_CONTAINER_PATH="$(docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
    -c 'import six; print(six.__file__)')"
[ -n "$SIX_CONTAINER_PATH" ] || { echo "FAIL: six did not install into the overlay venv — test setup is invalid" >&2; exit 1; }
SIX_FILE="$TMP${SIX_CONTAINER_PATH#/data}"
[ -e "$SIX_FILE" ] || { echo "FAIL: could not map installed six.py ($SIX_CONTAINER_PATH) onto the host bind mount ($SIX_FILE) — test setup is invalid" >&2; exit 1; }
PTH="$(overlay_pth)"
[ -n "$PTH" ] || { echo "FAIL: expected an existing _baked_venv.pth before breaking the venv again" >&2; exit 1; }
rm -f "$PTH"
if docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
        -c 'import torch' >/dev/null 2>&1; then
    echo "FAIL: overlay venv still imported torch after removing its .pth — test setup is invalid" >&2; exit 1
fi
if docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null 2>&1; then
    echo "FAIL: a previously-working venv was silently rebuilt instead of refusing to start" >&2; exit 1
fi
[ -e "$SIX_FILE" ] || { echo "FAIL: the user's overlay package (six) was deleted by a refused start" >&2; exit 1; }
[ -n "$(overlay_sentinel)" ] || { echo "FAIL: the sentinel itself was deleted by a refused start" >&2; exit 1; }
echo "    sentinel-present + broken venv: refused to start, overlay package survived on disk"

# Case 9 — rebuilding the venv while a Manager checkout already exists must
# reinstall Manager's requirements, not just leave a Manager-free venv.
# `toml` is one of Manager's requirements.txt entries and is absent from the
# baked /opt/venv, so its presence afterward can only come from a reinstall.
rm -f "$(overlay_sentinel)"
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
    -c 'import toml; print("    venv rebuilt with an existing Manager checkout: its requirements were reinstalled")'

# Case 10 — `make reset-venv` (spec §9: `rm -rf /data/venv` wholesale) must
# recover cleanly. This deletes the sentinel along with everything else, so
# the next start sees a completely absent venv (not merely a broken one) —
# distinct from Case 7's "leftover files from an interrupted build" shape.
# It must build fresh, succeed, and reinstall Manager's requirements again
# since the checkout is still on disk.
rm -rf "$TMP/venv"
[ ! -e "$TMP/venv" ] || { echo "FAIL: test setup could not remove \$TMP/venv" >&2; exit 1; }
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
    -c 'import torch, toml; print("    make reset-venv equivalent (whole-directory delete): rebuilt cleanly, Manager reqs reinstalled")'
[ -n "$(overlay_sentinel)" ] || { echo "FAIL: reset-venv recovery did not write a fresh completion sentinel" >&2; exit 1; }

echo "==> verify-entrypoint: PASS"
