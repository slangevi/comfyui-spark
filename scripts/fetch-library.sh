#!/usr/bin/env bash
# scripts/fetch-library.sh — fetch every model the workflows/ library needs.
#
# Idempotent: fetch-model.sh skips files that are already present, so this is
# safe to re-run after adding a workflow or after an interrupted download.
# Gated repos (LTX-2.5) need HF_TOKEN in the environment:
#   HF_TOKEN=$(cat ~/.config/hf/token) ./scripts/fetch-library.sh
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() { ./scripts/fetch-model.sh "$1" "$2"; }

# ── z-image-turbo (Comfy-Org/z_image_turbo, Apache 2.0) ─────────────────────
# Default is the nvfp4 transformer + fp8-mixed encoder: measured 1.37x faster
# than bf16 on a GB10 with a 44% smaller working set. The bf16 pair is kept
# for side-by-side quality checks (`--set model=z_image_turbo_bf16.safetensors`).
Z=https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files
fetch "$Z/diffusion_models/z_image_turbo_nvfp4.safetensors"    diffusion_models
fetch "$Z/diffusion_models/z_image_turbo_bf16.safetensors"     diffusion_models
fetch "$Z/text_encoders/qwen_3_4b_fp8_mixed.safetensors"       text_encoders
fetch "$Z/text_encoders/qwen_3_4b.safetensors"                 text_encoders
fetch "$Z/vae/ae.safetensors"                                  vae

# ── qwen-image-2.1 t2i + edit (Comfy-Org/Qwen-Image-2.1, Apache 2.0) ───────
# The int8_convrot files are what the bundled 2.1 templates reference.
Q=https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main
fetch "$Q/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"  diffusion_models
fetch "$Q/text_encoders/qwen3vl_8b_int8_convrot.safetensors"         text_encoders
fetch "$Q/vae/qwen_image_2.1_vae_bf16.safetensors"                   vae

# ── ltx-2.5 t2v (Lightricks/LTX-2.5 — GATED, accept the license first) ─────
# Distilled + int8: video diffusion is compute-bound on the GB10, so lower
# precision buys download size, not speed; the 8-step distill is what buys speed.
L=https://huggingface.co/Lightricks/LTX-2.5/resolve/main
fetch "$L/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" diffusion_models
fetch "$L/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"        text_encoders
fetch "$L/vae/ltx-2.5-video-vae-bf16.safetensors"                                           vae
fetch "$L/vae/ltx-2.5-audio-vae-bf16.safetensors"                                           vae
fetch "$L/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"    latent_upscale_models
# Optional prompt enhancer the T2V template wires in (Comfy-Org/gemma-4).
fetch https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e2b_it_int8_convrot.safetensors text_encoders

echo "fetch-library: done"
