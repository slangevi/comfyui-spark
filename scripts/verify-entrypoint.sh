#!/usr/bin/env bash
# Asserts the startup contract of spec §7.6 against a throwaway /data, without
# needing compose: the documented behaviors themselves (cases 1-6), plus the
# recovery and refusal paths that only exist because they were got wrong once
# (cases 7-17 — interrupted builds, interrupted clones, an overlay venv that
# has lost the GPU, a stale .pth form, and the real --cpu launch).
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-comfyui-spark:latest}"
TMP="$(mktemp -d)"
trap 'chmod -R u+w "$TMP" 2>/dev/null || true; rm -rf "$TMP"' EXIT

# The pin the entrypoint is supposed to check out. Read from the entrypoint
# itself so this test cannot drift away from it.
MANAGER_REF="$(sed -n 's/^MANAGER_REF="\${MANAGER_REF:-\([0-9a-f]*\)}"$/\1/p' docker/entrypoint.sh)"
[ -n "$MANAGER_REF" ] || { echo "FAIL: could not read MANAGER_REF out of docker/entrypoint.sh" >&2; exit 1; }

echo "==> verify-entrypoint: startup contract"

# Case 1 — a cold start seeds the /data tree and builds the overlay venv.
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
for p in models/checkpoints models/loras models/text_encoders models/controlnet \
         input output temp user custom_nodes/ComfyUI-Manager venv/bin/python; do
    [ -e "$TMP/$p" ] || { echo "FAIL: cold start did not create $p" >&2; exit 1; }
done
echo "    cold start seeded /data and created the overlay venv"

# Case 2 — the overlay venv really does inherit the baked torch, INCLUDING
# packages that only work through the baked layer's own .pth files: distutils
# exists on Python 3.12 only via setuptools' distutils-precedence.pth, which a
# plain-path _baked_venv.pth would never execute.
docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
    -c 'import torch, distutils; assert torch.cuda.is_available(); print("    overlay venv sees torch", torch.__version__, "and distutils resolves")'

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

# Case 11 — an interrupted ComfyUI-Manager install must self-heal, in all
# three states a kill can leave behind. The realistic trigger is mundane: a
# user gets impatient during the documented "few minutes" of a first start and
# runs `make down`.
MANAGER="$TMP/custom_nodes/ComfyUI-Manager"
CLONE_TMP="$TMP/.manager-clone.tmp"

# 11a — killed mid-clone. Nothing partial may appear under custom_nodes
# (ComfyUI imports every directory it finds there), and the pin must not be
# left to whatever `main` happened to be.
rm -rf "$MANAGER" "$CLONE_TMP"
CID="$(docker run -d --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE")"
interrupted=0
for _ in $(seq 1 400); do
    if [ -e "$CLONE_TMP/.git" ]; then interrupted=1; break; fi
    docker ps -q --no-trunc | grep -q "$CID" || break
    sleep 0.25
done
docker rm -f "$CID" >/dev/null 2>&1 || true
[ "$interrupted" = "1" ] || { echo "FAIL: could not catch the Manager clone in progress — the test never interrupted anything" >&2; exit 1; }
[ ! -e "$MANAGER" ] || { echo "FAIL: a clone interrupted in flight left a partial checkout in custom_nodes/" >&2; exit 1; }
[ -e "$CLONE_TMP" ] || { echo "FAIL: expected the interrupted clone's scratch directory to still be on disk" >&2; exit 1; }
echo "    clone killed in flight: nothing partial published into custom_nodes/"

# 11b — the same genuine partial clone, but sitting where a pre-fix entrypoint
# (which cloned straight into custom_nodes/) would have left it. The next start
# must recognise it as incomplete, redo it at the pinned ref, and clear the
# leftover scratch directory in the same pass.
cp -a "$CLONE_TMP" "$MANAGER"
if git -C "$MANAGER" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "FAIL: the 'partial' checkout has a resolvable HEAD — test setup is invalid" >&2; exit 1
fi
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
[ ! -e "$CLONE_TMP" ] || { echo "FAIL: the leftover clone scratch directory was not cleaned up" >&2; exit 1; }
[ -e "$MANAGER/.manager-ready" ] || { echo "FAIL: recovery did not write the Manager completion sentinel" >&2; exit 1; }
HEAD_SHA="$(git -C "$MANAGER" rev-parse HEAD)"
[ "$HEAD_SHA" = "$MANAGER_REF" ] || { echo "FAIL: Manager is at $HEAD_SHA, not the pinned $MANAGER_REF" >&2; exit 1; }
echo "    interrupted clone self-healed to the pinned ref, scratch directory cleaned"

# 11c — killed after the checkout but during `pip install -r requirements.txt`:
# the checkout is complete, so it must NOT be re-cloned (a user may have
# updated Manager from the UI since), but its requirements must be reinstalled
# rather than skipped forever. `toml` is a Manager requirement absent from the
# baked venv, so its return can only come from a reinstall.
touch "$MANAGER/.reclone-canary"
rm -f "$MANAGER/.manager-ready"
docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/pip "$IMAGE" \
    uninstall -y -q toml >/dev/null
if docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
        -c 'import toml' >/dev/null 2>&1; then
    echo "FAIL: toml is still importable after uninstalling it — test setup is invalid" >&2; exit 1
fi
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
[ -e "$MANAGER/.reclone-canary" ] || { echo "FAIL: a complete Manager checkout was deleted and re-cloned" >&2; exit 1; }
[ -e "$MANAGER/.manager-ready" ] || { echo "FAIL: the completion sentinel was not restored" >&2; exit 1; }
docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
    -c 'import toml' >/dev/null \
    || { echo "FAIL: Manager's requirements were not reinstalled after an interrupted install" >&2; exit 1; }
echo "    interrupted requirements install: reinstalled without re-cloning"

# Case 11d — the entrypoint keeps ComfyUI-Manager on pip, not uv. uv ignores
# .pth files, so under uv Manager cannot see the baked layer at all and
# reinstalls ~100 baked packages (torch included) into the overlay, where they
# shadow the GPU-correct copies.
MANAGER_CFG="$TMP/user/__manager/config.ini"
grep -qx 'use_uv = False' "$MANAGER_CFG" \
    || { echo "FAIL: expected 'use_uv = False' in $MANAGER_CFG, got:" >&2; cat "$MANAGER_CFG" >&2; exit 1; }
# ...and repairs an existing config that has it wrong, without touching the
# rest of the file.
printf '[default]\nuse_uv = True\nshare_option = all\n' > "$MANAGER_CFG"
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
grep -qx 'use_uv = False' "$MANAGER_CFG" \
    || { echo "FAIL: an existing use_uv = True was not repaired" >&2; cat "$MANAGER_CFG" >&2; exit 1; }
grep -qx 'share_option = all' "$MANAGER_CFG" \
    || { echo "FAIL: repairing use_uv clobbered another setting in $MANAGER_CFG" >&2; cat "$MANAGER_CFG" >&2; exit 1; }
echo "    Manager config seeded with use_uv = False, other settings preserved"

# Cases 12 and 13 — the overlay venv is the interpreter that actually renders,
# and it can lose the GPU on its own: its site-packages comes FIRST on
# sys.path, so anything installed there shadows the baked, GPU-correct torch.
# Both cases fake that shadowing with a stub `torch` module rather than
# downloading a real CPU-only build — the point under test is the entrypoint's
# reaction to an overlay torch that reports no CUDA, and a stub reports it
# exactly.
OVERLAY_SITE="$(dirname "$(overlay_pth)")"
STUB="$OVERLAY_SITE/torch.py"

# Case 12 — an overlay torch that never sees a GPU must stop the start, and
# must NOT be "fixed" by wiping the overlay: the user's packages are on it.
cat > "$STUB" <<'PY'
class cuda:
    @staticmethod
    def is_available(): return False
__version__ = "0.0.0+stub-cpu-only"
PY
set +e
out="$(docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" 2>&1)"
rc=$?
set -e
[ "$rc" -ne 0 ] || { echo "FAIL: started with an overlay torch that cannot see the GPU" >&2; echo "$out" >&2; exit 1; }
grep -q 'imports torch but cannot see a GPU' <<<"$out" \
    || { echo "FAIL: the readiness check did not reject an overlay venv without CUDA:" >&2; echo "$out" >&2; exit 1; }
grep -qi 'shadow' <<<"$out" || { echo "FAIL: the refusal does not name overlay shadowing as the cause:" >&2; echo "$out" >&2; exit 1; }
[ -e "$STUB" ] || { echo "FAIL: the refused start deleted overlay packages" >&2; exit 1; }
[ -n "$(overlay_sentinel)" ] || { echo "FAIL: the refused start deleted the venv sentinel" >&2; exit 1; }
echo "    overlay torch without CUDA: refused to start, overlay left intact"

# Case 13 — the same failure, but appearing only AFTER the readiness check has
# passed: the shape of a Manager or custom-node install that replaces torch
# midway through a start. This stub reports CUDA available on its first import
# and not afterwards, so the readiness check accepts the venv and only the
# pre-launch assertion can catch it.
cat > "$STUB" <<'PY'
import os
_flag = os.path.join(os.path.dirname(__file__), "_stub_import_seen")
_first = not os.path.exists(_flag)
if _first:
    open(_flag, "w").close()
class cuda:
    _ok = _first
    @staticmethod
    def is_available(): return cuda._ok
    @staticmethod
    def get_device_name(i): return "STUB"
    @staticmethod
    def get_device_capability(i): return (0, 0)
__version__ = "0.0.0+stub-flaky"
PY
set +e
out="$(docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" 2>&1)"
rc=$?
set -e
[ "$rc" -ne 0 ] || { echo "FAIL: an overlay venv that lost the GPU after the readiness check still started" >&2; echo "$out" >&2; exit 1; }
grep -q 'the interpreter that runs ComfyUI' <<<"$out" \
    || { echo "FAIL: the pre-launch assertion did not fire (some earlier check did):" >&2; echo "$out" >&2; exit 1; }
echo "    overlay venv losing the GPU mid-start: caught before launch"

# ...and with the stub gone, a normal start works again — so cases 12 and 13
# failed because of the stub, not because the venv was left damaged.
rm -f "$STUB" "$OVERLAY_SITE/_stub_import_seen"
rm -rf "$OVERLAY_SITE/__pycache__"
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
echo "    with the stub removed, the same /data starts cleanly again"

# Case 14 — a no-cuda overlay with NO sentinel must be refused, not wiped: a
# pre-sentinel-era venv that is CPU-shadowed still holds the user's packages,
# and the GPU guard has already proven the device works, so "no cuda" can
# never mean "unfinished build".
cat > "$STUB" <<'PY'
class cuda:
    @staticmethod
    def is_available(): return False
__version__ = "0.0.0+stub-cpu-only"
PY
rm -f "$(overlay_sentinel)"
set +e
out="$(docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" 2>&1)"
rc=$?
set -e
[ "$rc" -ne 0 ] || { echo "FAIL: a sentinel-less no-cuda overlay started instead of refusing" >&2; echo "$out" >&2; exit 1; }
grep -q 'imports torch but cannot see a GPU' <<<"$out" \
    || { echo "FAIL: a sentinel-less no-cuda overlay was not refused with the shadowing message:" >&2; echo "$out" >&2; exit 1; }
[ -e "$STUB" ] || { echo "FAIL: a sentinel-less no-cuda overlay was WIPED — user packages destroyed" >&2; exit 1; }
rm -f "$STUB"; rm -rf "$OVERLAY_SITE/__pycache__"
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
echo "    no-cuda overlay without a sentinel: refused, never wiped"

# Case 15 — a plain-path _baked_venv.pth (the pre-import-form shape older
# deployments have on disk) is healed on the next healthy start.
BAKED_SITE_PATH="$(docker run --rm --entrypoint /opt/venv/bin/python "$IMAGE" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$BAKED_SITE_PATH" > "$(overlay_pth)"
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
grep -q '^import site' "$(overlay_pth)" \
    || { echo "FAIL: a plain-path .pth was not rewritten to the import form" >&2; exit 1; }
docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
    -c 'import distutils' >/dev/null \
    || { echo "FAIL: distutils still unimportable after the .pth heal" >&2; exit 1; }
echo "    plain-path .pth from an older deployment: healed to the import form"

# Case 16 — an interrupted Manager-requirements install must be redone: the
# absence of .manager-ready forces the reinstall on the next start.
rm -f "$TMP/custom_nodes/ComfyUI-Manager/.manager-ready"
out="$(docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" 2>&1)"
grep -q "installing ComfyUI-Manager's requirements" <<<"$out" \
    || { echo "FAIL: a missing .manager-ready did not trigger a requirements reinstall:" >&2; echo "$out" >&2; exit 1; }
[ -e "$TMP/custom_nodes/ComfyUI-Manager/.manager-ready" ] \
    || { echo "FAIL: .manager-ready was not restored after the reinstall" >&2; exit 1; }
echo "    missing .manager-ready: requirements reinstalled, sentinel restored"

# Case 17 — a config.ini the use_uv rewriter cannot process must be fatal: a
# silent skip would leave use_uv=True and quietly disable the uv guard.
CFG="$TMP/user/__manager/config.ini"
mkdir -p "$(dirname "$CFG")"
cp "$CFG" "$CFG.bak" 2>/dev/null || true
printf '[default]\nuse_uv = True\n\xff\xfe broken' > "$CFG"
set +e
out="$(docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" 2>&1)"
rc=$?
set -e
[ "$rc" -ne 0 ] || { echo "FAIL: an unprocessable Manager config was silently skipped" >&2; echo "$out" >&2; exit 1; }
grep -q 'could not enforce use_uv' <<<"$out" \
    || { echo "FAIL: the refusal does not name the use_uv rewrite:" >&2; echo "$out" >&2; exit 1; }
if [ -f "$CFG.bak" ]; then mv "$CFG.bak" "$CFG"; else rm -f "$CFG"; fi
echo "    unprocessable Manager config: fatal, not silently skipped"

# Case 18 — COMFYUI_ALLOW_CPU=1 must produce a service that actually BOOTS on
# a GPU-less host, not just pass the guards: without --cpu the pinned ComfyUI
# executes torch.cuda.current_device() unguarded at import and crash-loops.
# This is the one case that exercises the real launch (no COMFYUI_SKIP_LAUNCH).
CPU_NAME="verify-entrypoint-cpu-$$"
docker run -d --name "$CPU_NAME" -e NVIDIA_VISIBLE_DEVICES=void -e COMFYUI_ALLOW_CPU=1 \
    -v "$TMP:/data" "$IMAGE" >/dev/null
cpu_ok=0
for _ in $(seq 1 60); do
    if docker logs "$CPU_NAME" 2>&1 | grep -q 'To see the GUI go to'; then cpu_ok=1; break; fi
    if [ -z "$(docker ps -q -f name="^${CPU_NAME}$")" ]; then break; fi
    sleep 3
done
logs="$(docker logs "$CPU_NAME" 2>&1 | tail -15)"
docker rm -f "$CPU_NAME" >/dev/null 2>&1 || true
[ "$cpu_ok" = "1" ] || { echo "FAIL: COMFYUI_ALLOW_CPU=1 did not produce a booted CPU service:" >&2; echo "$logs" >&2; exit 1; }
echo "    COMFYUI_ALLOW_CPU=1 boots a real CPU service (--cpu appended)"

echo "==> verify-entrypoint: PASS"
