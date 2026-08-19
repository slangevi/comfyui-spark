#!/usr/bin/env bash
# Asserts the built image can actually reach the GB10 GPU (spec §2.1, §8).
set -euo pipefail
IMAGE="${IMAGE:-comfyui-spark:latest}"

echo "==> verify-gpu: torch reaches the GB10 in ${IMAGE}"

docker run --rm --gpus all --entrypoint /opt/venv/bin/python "$IMAGE" -c '
import sys, torch
problems = []
if not torch.cuda.is_available():
    problems.append("torch.cuda.is_available() is False")
else:
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("    torch     ", torch.__version__)
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

echo "==> verify-gpu: PASS"
