#!/usr/bin/env bash
# Prepares /data and launches ComfyUI. Implements spec §7.6.
set -euo pipefail

DATA_DIR="${COMFYUI_DATA_DIR:-/data}"
COMFYUI_HOME="${COMFYUI_HOME:-/opt/comfyui}"
MANAGER_REPO="https://github.com/Comfy-Org/ComfyUI-Manager.git"
# main, pinned 2026-08-18. Governs the first clone only; after that the
# user updates Manager from the UI and we leave the checkout alone.
MANAGER_REF="${MANAGER_REF:-d5992a117ee98663b6e93fb040c1e43ee3b25d68}"

log() { echo "entrypoint: $*"; }
die() { echo "entrypoint: FATAL: $*" >&2; exit 1; }

# 1. /data must exist and be writable by this user.
[ -d "$DATA_DIR" ] || die "$DATA_DIR does not exist — is the bind mount configured?"
if ! touch "$DATA_DIR/.write-test" 2>/dev/null; then
    die "$DATA_DIR is not writable by uid $(id -u):$(id -g) — check ownership of COMFYUI_DATA_PATH on the host, and the PUID/PGID build args"
fi
rm -f "$DATA_DIR/.write-test"

# 2. Refuse to run on the CPU by accident. The worst failure mode is not a
#    crash, it is a silent CPU render discovered forty minutes in.
if [ "${COMFYUI_ALLOW_CPU:-0}" = "1" ]; then
    log "COMFYUI_ALLOW_CPU=1 — skipping the GPU check"
else
    if ! /opt/venv/bin/python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        die "no CUDA device visible. Check the nvidia runtime and the deploy.resources.reservations.devices block in docker-compose.yml. Set COMFYUI_ALLOW_CPU=1 to run on the CPU anyway."
    fi
    log "GPU: $(/opt/venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))')"
fi

# 3. Seed the tree ComfyUI's folder_paths.py expects, so a cold start shows
#    the full set of model folders in the UI.
mkdir -p "$DATA_DIR"/input "$DATA_DIR"/output "$DATA_DIR"/temp \
         "$DATA_DIR"/user "$DATA_DIR"/custom_nodes
for d in checkpoints configs loras vae text_encoders clip diffusion_models unet \
         clip_vision style_models embeddings diffusers vae_approx controlnet \
         t2i_adapter gligen upscale_models latent_upscale_models hypernetworks \
         photomaker classifiers; do
    mkdir -p "$DATA_DIR/models/$d"
done

# 4. The overlay venv — where ComfyUI-Manager's pip installs land so they
#    survive restarts (spec §7.2), inheriting baked torch so no multi-gigabyte
#    download happens at boot.
#
#    `venv --system-site-packages` is not enough here: /opt/venv is itself a
#    venv, and since Python 3.11 a nested venv's --system-site-packages
#    resolves against the *real* base interpreter (sys._base_executable),
#    not the immediate parent venv — so it would see the OS's dist-packages,
#    never /opt/venv's torch. Point it at /opt/venv explicitly with a .pth
#    file instead; site.py appends it to sys.path after the overlay's own
#    site-packages, so a node that installs a newer version of a baked
#    package still shadows it (the precedence spec §7.2 requires).
#
#    "ready" means the interpreter exists AND can actually import torch —
#    not merely that `venv` finished — because those are two separate steps
#    (create the venv, then write the .pth) and a container killed between
#    them (OOM, host reboot, `docker stop` mid-first-boot) would otherwise
#    leave an executable-but-broken bin/python that every later start would
#    treat as done. Detect that half-built state and rebuild rather than
#    hand off to an interpreter that can't import torch.
overlay_venv_ready() {
    [ -x "$DATA_DIR/venv/bin/python" ] \
        && "$DATA_DIR/venv/bin/python" -c 'import torch' >/dev/null 2>&1
}

if ! overlay_venv_ready; then
    if [ -e "$DATA_DIR/venv" ]; then
        log "overlay venv at $DATA_DIR/venv exists but can't import torch — an earlier start was likely interrupted before it finished; rebuilding"
        rm -rf "$DATA_DIR/venv"
    else
        log "creating the overlay venv at $DATA_DIR/venv"
    fi
    /opt/venv/bin/python -m venv "$DATA_DIR/venv"
    BAKED_SITE="$(/opt/venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
    OVERLAY_SITE="$("$DATA_DIR/venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    echo "$BAKED_SITE" > "$OVERLAY_SITE/_baked_venv.pth"
    overlay_venv_ready || die "overlay venv at $DATA_DIR/venv still can't import torch after rebuilding it — check that /opt/venv itself has torch installed"
fi

# 5. ComfyUI-Manager, first start only.
if [ ! -d "$DATA_DIR/custom_nodes/ComfyUI-Manager" ]; then
    log "cloning ComfyUI-Manager @ ${MANAGER_REF}"
    git clone --filter=blob:none "$MANAGER_REPO" "$DATA_DIR/custom_nodes/ComfyUI-Manager"
    # advice.detachedHead=false: this prints straight to container stdout on
    # every fresh deployment (docker compose logs), not just in a terminal.
    git -c advice.detachedHead=false -C "$DATA_DIR/custom_nodes/ComfyUI-Manager" checkout "$MANAGER_REF"
    if [ -f "$DATA_DIR/custom_nodes/ComfyUI-Manager/requirements.txt" ]; then
        "$DATA_DIR/venv/bin/pip" install --no-cache-dir \
            -r "$DATA_DIR/custom_nodes/ComfyUI-Manager/requirements.txt"
    fi
fi

# Testing seam used by scripts/verify-entrypoint.sh: do the preparation, then
# stop instead of handing off to a server that would never exit.
if [ "${COMFYUI_SKIP_LAUNCH:-0}" = "1" ]; then
    log "COMFYUI_SKIP_LAUNCH=1 — preparation complete, not launching"
    exit 0
fi

# 6. Hand off. COMFYUI_ARGS is deliberately unquoted so it word-splits into
#    separate flags.
log "starting ComfyUI"
# shellcheck disable=SC2086
exec "$DATA_DIR/venv/bin/python" "$COMFYUI_HOME/main.py" \
    --listen 0.0.0.0 \
    --port 8188 \
    --base-directory "$DATA_DIR" \
    --disable-auto-launch \
    ${COMFYUI_ARGS:-}
