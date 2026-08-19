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

auth=()
if [ -n "${HF_TOKEN:-}" ] && [[ "$URL" == *huggingface.co* ]]; then
    auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    echo "     using HF_TOKEN"
fi

# --continue-at - resumes a partial download; multi-GB files over a flaky
# link should not restart from zero.
curl -fL --progress-bar --continue-at - "${auth[@]}" -o "${DEST}.part" "$URL"
mv "${DEST}.part" "$DEST"

echo "done: $DEST ($(du -h "$DEST" | cut -f1))"
echo "ComfyUI picks it up on the next refresh; no restart needed."
