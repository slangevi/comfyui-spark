#!/usr/bin/env bash
# Asserts that torch reaches the GB10 (spec §2.1, §8) in BOTH of the two
# Python environments — because they can disagree:
#
#   /opt/venv   the baked layer. All a freshly built image has, and the only
#               thing that can be checked before any overlay exists.
#   /data/venv  the overlay, and the interpreter that actually runs ComfyUI.
#               Its site-packages comes first on sys.path, so a CPU-only torch
#               installed there (ComfyUI-Manager, a custom node's
#               requirements.txt) shadows the baked cu130 build and renders on
#               the CPU while the baked check above still passes.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f .env ] && . ./.env

IMAGE="${IMAGE:-comfyui-spark:latest}"
DATA_PATH="${COMFYUI_DATA_PATH:-/home/scott/LLMs/comfyui}"

# One assertion, run against whichever interpreter is being interrogated.
ASSERT='
import sys, torch
problems = []
print("    torch     ", torch.__version__)
print("    loaded from", torch.__file__)
if not torch.cuda.is_available():
    problems.append("torch.cuda.is_available() is False")
else:
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("    device    ", name)
    print("    capability", cap)
    print("    arch_list ", torch.cuda.get_arch_list())
    if "GB10" not in name:
        problems.append("expected a GB10, got %r" % name)
    if cap != (12, 1):
        problems.append("expected capability (12, 1), got %r" % (cap,))
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    if not torch.isfinite((a @ a).float()).all():
        problems.append("bf16 matmul produced non-finite values")
    else:
        print("    bf16 matmul on device: finite")
for p in problems:
    print("FAIL:", p, file=sys.stderr)
sys.exit(1 if problems else 0)
'

echo "==> verify-gpu: baked venv (/opt/venv) in ${IMAGE}"
docker run --rm --gpus all --entrypoint /opt/venv/bin/python "$IMAGE" -c "$ASSERT"

echo "==> verify-gpu: overlay venv (/data/venv) — the interpreter that runs ComfyUI"
if [ -n "$(docker compose ps -q --status running comfyui 2>/dev/null)" ]; then
    # Preferred: interrogate the live container, so this fails if the running
    # service has lost the GPU, not merely if a fresh one would.
    docker compose exec -T comfyui /data/venv/bin/python -c "$ASSERT"
elif [ -x "$DATA_PATH/venv/bin/python" ]; then
    echo "    (service is not running; checking the overlay venv on disk instead)"
    docker run --rm --gpus all -v "$DATA_PATH:/data" \
        --entrypoint /data/venv/bin/python "$IMAGE" -c "$ASSERT"
else
    echo "    SKIP: no overlay venv at $DATA_PATH/venv yet — it is built on the"
    echo "          first start. Re-run after 'make up' to check the interpreter"
    echo "          that actually renders."
fi

echo "==> verify-gpu: PASS"
