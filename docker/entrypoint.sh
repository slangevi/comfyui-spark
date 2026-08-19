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
#
#    This checks the BAKED venv, which is only half the story: ComfyUI runs on
#    the overlay venv (step 8), and the overlay can lose the GPU on its own by
#    shadowing baked torch. Checking here anyway is still worth it — it is the
#    cheapest, earliest signal, it runs before the multi-minute venv build, and
#    it cleanly separates "the container has no GPU at all" (this check) from
#    "the container has a GPU but the interpreter that renders cannot see it"
#    (step 7), which need completely different remedies.
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
#    A broken `import torch` is ambiguous on its own: it could mean the
#    initial build never finished (safe to rebuild — nothing of the user's
#    was ever on it), or it could mean a *working* venv, with the user's
#    overlay-installed packages on it, broke later (a custom node's pip
#    install pulling an incompatible numpy/torch, an OOM mid-upgrade,
#    corruption) — in which case deleting it destroys the one thing this
#    task exists to persist. VENV_SENTINEL disambiguates: it is written only
#    once, right after a build has been verified to actually import torch.
#    Its absence means "never finished" (safe to rebuild). Its presence
#    alongside a broken import means "something broke a working venv" (must
#    not auto-wipe — fail fast and point at the manual escape hatch instead).
VENV_SENTINEL="$DATA_DIR/venv/.overlay-venv-built"
VENV_REBUILT=0

# Three distinguishable states, because they need three different remedies:
#   ok       torch imports and reaches a CUDA device
#   no-cuda  torch imports but sees no GPU — with the baked venv already
#            proven fine in step 2, that means the overlay is shadowing it
#            with a CPU-only build. Rebuilding is NOT the automatic answer:
#            an overlay that renders on the CPU still holds the user's
#            packages.
#   broken   torch does not import at all (a half-built venv, a deleted .pth,
#            a Python minor-version bump under the overlay's feet)
overlay_torch_state() {
    if [ ! -x "$DATA_DIR/venv/bin/python" ]; then
        echo broken
        return 0
    fi
    local rc=0
    "$DATA_DIR/venv/bin/python" - >/dev/null 2>&1 <<'PY' || rc=$?
import sys
try:
    import torch
except Exception:
    sys.exit(2)
sys.exit(0 if torch.cuda.is_available() else 1)
PY
    case "$rc" in
        0) echo ok ;;
        1) echo no-cuda ;;
        *) echo broken ;;
    esac
}

# COMFYUI_ALLOW_CPU=1 is a deliberate "I know there is no GPU here" — it must
# not turn a perfectly good overlay venv into a rebuild (or, worse, into a
# refusal to start), so in that mode a working `import torch` is all the
# readiness this venv needs.
overlay_state_ready() {
    case "$1" in
        ok)      return 0 ;;
        no-cuda) [ "${COMFYUI_ALLOW_CPU:-0}" = "1" ] ;;
        *)       return 1 ;;
    esac
}

OVERLAY_STATE="$(overlay_torch_state)"
if overlay_state_ready "$OVERLAY_STATE"; then
    # Migration path: a venv built by an older entrypoint (before this
    # sentinel existed) is working right now — stamp it so a future break is
    # correctly recognized as "this used to work", not misread as "never
    # finished".
    [ -e "$VENV_SENTINEL" ] || touch "$VENV_SENTINEL"
else
    if [ -e "$VENV_SENTINEL" ] && [ "$OVERLAY_STATE" = "no-cuda" ]; then
        die "the overlay venv at $DATA_DIR/venv imports torch but cannot see a GPU, while the baked venv at /opt/venv can. Something installed into the overlay is shadowing baked torch with a CPU-only build — most likely ComfyUI-Manager or a custom node's requirements.txt. Find it with '$DATA_DIR/venv/bin/pip list --local | grep -i torch', then remove it ('$DATA_DIR/venv/bin/pip uninstall torch torchvision torchaudio') so the baked cu130 build shows through again. Refusing to delete the overlay automatically — your installed custom-node packages are on it. 'make down && make reset-venv && make up' rebuilds it from scratch (this permanently deletes every package installed there). Set COMFYUI_ALLOW_CPU=1 to run on the CPU anyway."
    fi
    if [ -e "$VENV_SENTINEL" ]; then
        die "the overlay venv at $DATA_DIR/venv previously worked but can no longer import torch. Refusing to delete it automatically — your installed custom-node packages are still on disk and may be recoverable. Inspect $DATA_DIR/venv by hand, or run 'make reset-venv' to rebuild it from scratch (this permanently deletes every package installed there)."
    fi
    if [ -e "$DATA_DIR/venv" ]; then
        log "overlay venv at $DATA_DIR/venv never finished building (no completion marker) — rebuilding"
        rm -rf "$DATA_DIR/venv"
    else
        log "creating the overlay venv at $DATA_DIR/venv"
    fi
    /opt/venv/bin/python -m venv "$DATA_DIR/venv"
    BAKED_SITE="$(/opt/venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
    OVERLAY_SITE="$("$DATA_DIR/venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    echo "$BAKED_SITE" > "$OVERLAY_SITE/_baked_venv.pth"
    overlay_state_ready "$(overlay_torch_state)" \
        || die "overlay venv at $DATA_DIR/venv still can't reach torch and the GPU after rebuilding it — check that /opt/venv itself has torch installed"
    touch "$VENV_SENTINEL"
    VENV_REBUILT=1
fi

# 5. ComfyUI-Manager.
#
#    The install is three steps that must all happen — clone, checkout the
#    pin, install requirements into the overlay venv — and a container that
#    dies partway through (a user running `make down` during the documented
#    "few minutes" of a first start) must not leave a state that later starts
#    mistake for "done". Guarding on the directory alone did exactly that: a
#    kill between clone and checkout left Manager on whatever `main` happened
#    to be, silently defeating the supply-chain pin, and a kill during the
#    requirements install left a Manager checkout with none of its
#    dependencies. Both looked complete to the next start.
#
#    So, mirroring the overlay venv above: clone into a scratch directory
#    OUTSIDE custom_nodes (ComfyUI imports every directory it finds in
#    custom_nodes, so a half-cloned one must never be visible there), check
#    out the pin, and only then rename it into place — rename(2) on the same
#    filesystem is atomic, so custom_nodes/ComfyUI-Manager either does not
#    exist or is a complete checkout at the pinned ref. A .manager-ready
#    sentinel, written last, covers the remaining window: the requirements
#    install, which happens after the rename because it is the slow part and
#    is safely repeatable.
MANAGER_DIR="$DATA_DIR/custom_nodes/ComfyUI-Manager"
MANAGER_TMP="$DATA_DIR/.manager-clone.tmp"
MANAGER_SENTINEL="$MANAGER_DIR/.manager-ready"

# A checkout is complete iff git can resolve HEAD (an interrupted clone leaves
# HEAD pointing at an unborn branch) and the working tree was written out.
manager_checkout_complete() {
    [ -f "$MANAGER_DIR/requirements.txt" ] \
        && git -C "$MANAGER_DIR" rev-parse --verify HEAD >/dev/null 2>&1
}

if [ -e "$MANAGER_TMP" ]; then
    log "removing a leftover Manager clone directory from an interrupted start"
    rm -rf "$MANAGER_TMP"
fi

MANAGER_NEEDS_REQS="$VENV_REBUILT"
if [ ! -e "$MANAGER_SENTINEL" ]; then
    if [ -d "$MANAGER_DIR" ] && ! manager_checkout_complete; then
        # Only ever deletes a checkout git itself cannot resolve — a complete
        # one (including one the user has since updated from the UI) is left
        # alone, sentinel or not.
        log "the ComfyUI-Manager checkout at $MANAGER_DIR is incomplete (git cannot resolve HEAD, or requirements.txt is missing) — an earlier clone was interrupted; redoing it"
        rm -rf "$MANAGER_DIR"
    fi
    if [ ! -d "$MANAGER_DIR" ]; then
        log "cloning ComfyUI-Manager @ ${MANAGER_REF}"
        git clone --filter=blob:none "$MANAGER_REPO" "$MANAGER_TMP"
        # advice.detachedHead=false: this prints straight to container stdout on
        # every fresh deployment (docker compose logs), not just in a terminal.
        git -c advice.detachedHead=false -C "$MANAGER_TMP" checkout "$MANAGER_REF"
        mv "$MANAGER_TMP" "$MANAGER_DIR"
    fi
    MANAGER_NEEDS_REQS=1
fi
if [ "$MANAGER_NEEDS_REQS" = "1" ] && [ -f "$MANAGER_DIR/requirements.txt" ]; then
    log "installing ComfyUI-Manager's requirements into the overlay venv"
    "$DATA_DIR/venv/bin/pip" install --no-cache-dir -r "$MANAGER_DIR/requirements.txt"
fi
# Reached only if the pip install above exited 0 (set -e), so the sentinel
# means "clone, checkout and requirements all completed".
if [ -d "$MANAGER_DIR" ]; then
    touch "$MANAGER_SENTINEL"
fi

# 6. Keep ComfyUI-Manager on pip rather than uv.
#
#    Manager defaults use_uv=True on Linux whenever `uv` is importable — and
#    `uv` is in Manager's own requirements.txt, installed above. That default
#    is actively wrong for this two-layer environment: uv does not process
#    .pth files, so `uv pip list` inside the overlay sees only the overlay's
#    own 27 packages, not the ~100 baked ones. Manager decides what a custom
#    node still needs from that list, so under uv it re-installs torch, numpy,
#    transformers, pillow, safetensors and friends from PyPI into the overlay,
#    where they shadow the baked, GPU-correct copies. pip reads the .pth file,
#    sees all 129, and correctly skips what is already satisfied.
#
#    Written on every start, not just at clone time: the flag is a property of
#    this environment, not a user preference, and existing deployments already
#    have use_uv=True on disk from before this was understood. The edit is
#    line-scoped so every other setting — including keys and sections this
#    file knows nothing about — is carried through untouched.
seed_manager_pip_mode() {
    local cfg="$1"
    /opt/venv/bin/python - "$cfg" <<'PY'
import os, sys, tempfile

path = sys.argv[1]
lines = []
if os.path.exists(path):
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

out, section, done, changed = [], None, False, False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        if section == "default" and not done:
            # Leaving [default] without having seen use_uv — add it.
            out.append("use_uv = False\n")
            done = changed = True
        section = stripped[1:-1].strip().lower()
    elif section == "default" and stripped.lower().replace(" ", "").startswith("use_uv="):
        if stripped.split("=", 1)[1].strip().lower() != "false":
            line = "use_uv = False\n"
            changed = True
        done = True
    out.append(line)

if section == "default" and not done:
    if out and not out[-1].endswith("\n"):
        out.append("\n")
    out.append("use_uv = False\n")
    done = changed = True
if not done:
    # No [default] section at all (or no file): Manager's read_config() raises
    # on a missing section and falls back to its uv-on-Linux default, so the
    # section has to exist for the flag to be read.
    if out and not out[-1].endswith("\n"):
        out.append("\n")
    out.append("[default]\nuse_uv = False\n")
    changed = True

if changed:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory or ".", prefix=".config.ini.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.writelines(out)
    os.replace(tmp, path)
    print("changed")
PY
}

# Manager moved its state directory in the ComfyUI release that added
# folder_paths.get_system_user_directory(); it picks the new path only when
# that function exists. Repair whichever config files are actually on disk,
# and if there are none, create the one this image's ComfyUI implies.
MANAGER_CFG_NEW="$DATA_DIR/user/__manager/config.ini"
MANAGER_CFG_OLD="$DATA_DIR/user/default/ComfyUI-Manager/config.ini"
MANAGER_CFGS=()
# `[ -f x ] && arr+=(...)` would abort the script under `set -e` whenever the
# file is absent — the AND-list's own exit status is the failing test's.
if [ -f "$MANAGER_CFG_NEW" ]; then MANAGER_CFGS+=("$MANAGER_CFG_NEW"); fi
if [ -f "$MANAGER_CFG_OLD" ]; then MANAGER_CFGS+=("$MANAGER_CFG_OLD"); fi
if [ "${#MANAGER_CFGS[@]}" -eq 0 ]; then
    if grep -q 'def get_system_user_directory' "$COMFYUI_HOME/folder_paths.py" 2>/dev/null; then
        MANAGER_CFGS=("$MANAGER_CFG_NEW")
    else
        MANAGER_CFGS=("$MANAGER_CFG_OLD")
    fi
fi
for cfg in "${MANAGER_CFGS[@]}"; do
    if [ -n "$(seed_manager_pip_mode "$cfg")" ]; then
        log "set use_uv = False in $cfg (uv cannot see the baked venv; see spec §7.2)"
    fi
done

# 7. The assertion that matters: the interpreter that actually renders.
#
#    Step 2 proved /opt/venv reaches the GPU. ComfyUI runs on
#    $DATA_DIR/venv/bin/python, whose site-packages comes FIRST on sys.path —
#    so anything installed into the overlay (by Manager, by a custom node's
#    requirements.txt, by hand) can replace baked torch with a build that has
#    no CUDA, and step 2 would still pass. Everything above this line has just
#    finished writing to that overlay, so this is the last honest moment to
#    check.
if [ "${COMFYUI_ALLOW_CPU:-0}" != "1" ]; then
    if ! "$DATA_DIR/venv/bin/python" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        die "the overlay venv ($DATA_DIR/venv/bin/python) — the interpreter that runs ComfyUI — cannot reach the GPU, even though the baked venv can. The overlay comes first on sys.path, so the likely cause is a package installed there shadowing baked torch with a CPU-only build (ComfyUI-Manager, or a custom node's requirements.txt naming torch). Recover with: '$DATA_DIR/venv/bin/pip list --local | grep -i torch' to see what the overlay owns, then '$DATA_DIR/venv/bin/pip uninstall torch torchvision torchaudio' to let the baked cu130 build show through again; or 'make down && make reset-venv && make up' to rebuild the overlay from scratch (this permanently deletes every package installed there). Set COMFYUI_ALLOW_CPU=1 to run on the CPU anyway."
    fi
    log "overlay GPU: $("$DATA_DIR/venv/bin/python" -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.__version__)')"
fi

# Testing seam used by scripts/verify-entrypoint.sh: do the preparation, then
# stop instead of handing off to a server that would never exit.
if [ "${COMFYUI_SKIP_LAUNCH:-0}" = "1" ]; then
    log "COMFYUI_SKIP_LAUNCH=1 — preparation complete, not launching"
    exit 0
fi

# 8. Hand off. COMFYUI_ARGS is deliberately unquoted so it word-splits into
#    separate flags.
log "starting ComfyUI"
# shellcheck disable=SC2086
exec "$DATA_DIR/venv/bin/python" "$COMFYUI_HOME/main.py" \
    --listen 0.0.0.0 \
    --port 8188 \
    --base-directory "$DATA_DIR" \
    --disable-auto-launch \
    ${COMFYUI_ARGS:-}
