#!/usr/bin/env bash
# Asserts that packages installed into the overlay venv survive the
# container being torn down and rebuilt from the image — the entire reason
# for the two-layer environment (spec §7.2, §8).
#
# The challenge is `docker compose up -d --force-recreate`, not
# `docker compose restart`. A plain restart stops and starts the SAME
# container object; it never discards the container's writable layer, so it
# cannot distinguish an overlay install (on the /data bind mount) from a
# baked-venv install (in the image layer) — both would "survive" a restart
# regardless of whether the two-layer design works at all. Force-recreate
# actually discards that writable layer, which is what a real redeploy
# (`make down && make up`, an image rebuild, a host reboot after
# `docker compose down`) does.
set -euo pipefail
cd "$(dirname "$0")/.."

# six is tiny, pure-Python, and NOT a ComfyUI dependency, so finding it later
# can only be explained by the overlay venv having persisted.
PKG=six

echo "==> verify-persistence: overlay venv survives a container recreate"

cleanup() {
    docker compose exec -T comfyui /data/venv/bin/pip uninstall -y -q "$PKG" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Guard: if it is already present, the test would pass vacuously.
if docker compose exec -T comfyui /data/venv/bin/python -c "import $PKG" >/dev/null 2>&1; then
    echo "    $PKG was already installed; removing so the test means something"
    docker compose exec -T comfyui /data/venv/bin/pip uninstall -y -q "$PKG"
fi

docker compose exec -T comfyui /data/venv/bin/pip install --no-cache-dir -q "$PKG"
echo "    installed $PKG into the overlay venv"

# It must live in the OVERLAY layer, not the baked one. `pip list --local`
# filters by path against sys.prefix, so it excludes packages reached only
# via the _baked_venv.pth file too.
if ! docker compose exec -T comfyui /data/venv/bin/pip list --local 2>/dev/null \
        | awk '{print tolower($1)}' | grep -qx "$PKG"; then
    echo "FAIL: $PKG is not in the overlay layer — 'pip list --local' does not list it." >&2
    echo "      It was installed somewhere that a recreate will discard." >&2
    exit 1
fi
echo "    $PKG is in the overlay layer, not the baked layer"

docker compose up -d --force-recreate comfyui >/dev/null
echo "    force-recreated the container"

./scripts/verify-http.sh >/dev/null
echo "    service came back up"

if ! docker compose exec -T comfyui /data/venv/bin/python -c "import $PKG" >/dev/null 2>&1; then
    echo "FAIL: $PKG did not survive the recreate — the overlay venv is not persisted." >&2
    echo "      Check that COMFYUI_DATA_PATH is bind-mounted at /data and that" >&2
    echo "      the entrypoint creates /data/venv rather than a venv inside the image." >&2
    exit 1
fi
echo "    $PKG survived the recreate"

echo "==> verify-persistence: PASS"
