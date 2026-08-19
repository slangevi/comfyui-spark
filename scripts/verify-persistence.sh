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
#
# NOTE: this test force-recreates the running container. Any in-flight
# generation is interrupted.
set -euo pipefail
cd "$(dirname "$0")/.."

# The probe package has to be one no real dependency chain can drag in.
# `six` was used here first and was the wrong choice: it is one of the most
# common transitive dependencies in Python, so the moment a custom node pulled
# it in, a routine `make verify` would uninstall a package that node needed.
# `pyjokes` is pure-Python, dependency-free, tiny, and nothing in the
# diffusion/ML stack depends on it.
PKG=pyjokes

echo "==> verify-persistence: overlay venv survives a container recreate"

# Overlay-local, not merely importable. `pip list --local` filters by path
# against sys.prefix, so it excludes packages reached only through the
# _baked_venv.pth file — which is exactly the distinction this test exists to
# make. Testing `import` instead would be satisfied by a baked-layer copy and
# the test would pass without proving anything.
overlay_has() {
    docker compose exec -T comfyui /data/venv/bin/pip list --local 2>/dev/null \
        | awk '{print tolower($1)}' | grep -qx "$1"
}

INSTALLED_BY_TEST=0
cleanup() {
    [ "$INSTALLED_BY_TEST" = "1" ] || return 0
    if ! docker compose exec -T comfyui true >/dev/null 2>&1; then
        echo "    NOTE: cleanup could not reach the comfyui container, so $PKG is" >&2
        echo "          still installed in the overlay venv. Remove it with:" >&2
        echo "          docker compose exec comfyui /data/venv/bin/pip uninstall -y $PKG" >&2
        return 0
    fi
    if overlay_has "$PKG"; then
        docker compose exec -T comfyui /data/venv/bin/pip uninstall -y -q "$PKG" >/dev/null 2>&1 \
            || echo "    NOTE: could not uninstall the $PKG probe from the overlay venv; remove it by hand." >&2
    fi
}
trap cleanup EXIT

# If it is already in the overlay layer, leave it exactly where it is: this
# script must never remove a package it did not install (that was the old
# behaviour, and it silently deleted `six` from a live deployment on every
# run). An already-present copy does not weaken the assertion either — what
# makes the test meaningful is `pip list --local` proving the package is on
# the bind mount, before and after, not who put it there.
if overlay_has "$PKG"; then
    echo "    $PKG is already in the overlay layer; leaving it in place"
else
    docker compose exec -T comfyui /data/venv/bin/pip install --no-cache-dir -q "$PKG"
    INSTALLED_BY_TEST=1
    echo "    installed $PKG into the overlay venv"
fi

if ! overlay_has "$PKG"; then
    echo "FAIL: $PKG is not in the overlay layer — 'pip list --local' does not list it." >&2
    echo "      It was installed somewhere that a recreate will discard." >&2
    exit 1
fi
echo "    $PKG is in the overlay layer, not the baked layer"

docker compose up -d --force-recreate comfyui >/dev/null
echo "    force-recreated the container"

./scripts/verify-http.sh >/dev/null
echo "    service came back up"

# Both halves, per spec §8: importable AND still overlay-local. The import
# alone would be satisfied by a baked-layer copy of the same package, which is
# precisely the vacuous pass this test has to rule out.
if ! docker compose exec -T comfyui /data/venv/bin/python -c "import $PKG" >/dev/null 2>&1; then
    echo "FAIL: $PKG did not survive the recreate — the overlay venv is not persisted." >&2
    echo "      Check that COMFYUI_DATA_PATH is bind-mounted at /data and that" >&2
    echo "      the entrypoint creates /data/venv rather than a venv inside the image." >&2
    exit 1
fi
if ! overlay_has "$PKG"; then
    echo "FAIL: $PKG is importable after the recreate but is no longer in the" >&2
    echo "      overlay layer — it is being satisfied by the baked venv, so this" >&2
    echo "      proves nothing about persistence." >&2
    exit 1
fi
echo "    $PKG survived the recreate, still in the overlay layer"

echo "==> verify-persistence: PASS"
