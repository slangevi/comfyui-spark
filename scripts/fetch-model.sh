#!/usr/bin/env bash
# Downloads a model file into the right /data/models subdirectory.
# Usage: ./scripts/fetch-model.sh <URL> <SUBDIR>
#   e.g. ./scripts/fetch-model.sh https://example.com/model.safetensors checkpoints
set -euo pipefail
cd "$(dirname "$0")/.."

URL="${1:-}"
SUBDIR="${2:-}"

if [ -z "$URL" ] || [ -z "$SUBDIR" ]; then
    cat >&2 <<'USAGE'
usage: fetch-model.sh <URL> <SUBDIR>

SUBDIR is a folder under /data/models, e.g.:
  checkpoints  loras  vae  text_encoders  diffusion_models  controlnet
  clip_vision  upscale_models  embeddings

Set HF_TOKEN in .env first for gated HuggingFace repos.
USAGE
    exit 2
fi

# shellcheck disable=SC1091
[ -f .env ] && . ./.env
DATA_PATH="${COMFYUI_DATA_PATH:-/home/scott/LLMs/comfyui}"
DEST_DIR="${DATA_PATH}/models/${SUBDIR}"
FILENAME="$(basename "${URL%%\?*}")"
DEST="${DEST_DIR}/${FILENAME}"

mkdir -p "$DEST_DIR"

if [ -s "$DEST" ]; then
    echo "already present: $DEST"
    exit 0
fi

echo "fetching $FILENAME"
echo "     to  $DEST_DIR"

# The token is attached on an exact host match, never a substring match:
# *huggingface.co* would also match huggingface.co.evil.example (or the
# string anywhere in a path or query) and hand the bearer token to a host
# the user never intended. curl already strips the header on a cross-host
# redirect, so the CDN hop stays tokenless either way.
host="$(printf '%s' "$URL" | sed -E 's#^[a-zA-Z][a-zA-Z0-9+.-]*://([^/?@]*@)?([^/:?]+).*#\2#')"
auth=()
case "$host" in
    huggingface.co|*.huggingface.co)
        if [ -n "${HF_TOKEN:-}" ]; then
            auth=(-H "Authorization: Bearer ${HF_TOKEN}")
            echo "     using HF_TOKEN"
        fi
        ;;
esac

# --continue-at - resumes a partial download; multi-GB files over a flaky
# link should not restart from zero. But only resume OUR OWN partial: a
# leftover .part from a different URL with the same basename would be
# silently spliced into corrupt weights. The .part.url marker records which
# download a .part belongs to; on mismatch, start over.
if [ -f "${DEST}.part" ]; then
    if [ ! -f "${DEST}.part.url" ] || [ "$(cat "${DEST}.part.url")" != "$URL" ]; then
        echo "     discarding a leftover partial from a different download"
        rm -f "${DEST}.part" "${DEST}.part.url"
    fi
fi
printf '%s' "$URL" > "${DEST}.part.url"
curl -fL --progress-bar --continue-at - "${auth[@]}" -o "${DEST}.part" "$URL"
mv "${DEST}.part" "$DEST"
rm -f "${DEST}.part.url"

echo "done: $DEST ($(du -h "$DEST" | cut -f1))"
echo "ComfyUI picks it up on the next refresh; no restart needed."
