# ComfyUI Docker Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package ComfyUI as a GPU-accelerated Docker service on this DGX Spark, reachable at `http://127.0.0.1:8188`, where Manager-installed custom nodes survive restarts and rebuilds.

**Architecture:** A single image bakes Python 3.12, PyTorch `cu130`, and ComfyUI v0.33.1. At startup an entrypoint seeds the bind-mounted `/data` tree, creates an *overlay* venv there with `--system-site-packages`, and launches ComfyUI from it with `--base-directory /data`. ComfyUI-Manager therefore installs nodes and their pip dependencies onto the bind mount, while torch stays baked in the image.

**Tech Stack:** Docker 29.2 / Compose v5, `nvidia/cuda:13.1.0-runtime-ubuntu24.04` (arm64), Python 3.12, PyTorch 2.13.0+cu130, ComfyUI v0.33.1, ComfyUI-Manager, Bash, Make.

**Spec:** `docs/superpowers/specs/2026-08-18-comfyui-docker-design.md`

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec; do not substitute "equivalent" ones.

- **Platform:** `aarch64` only. GPU is NVIDIA GB10, compute capability `(12, 1)`. `sm_121` is absent from torch's arch list *by design* — `sm_120` cubins run on it (spec §2.1). Never "fix" this by hunting for an sm_121 build.
- **Base image:** `nvidia/cuda:13.1.0-runtime-ubuntu24.04@sha256:88bc2ff57b4a4cbb3dc900cf492203958b24ec7148695992cf0ce8e5cdebd606` — the `runtime` variant, no compiler toolchain.
- **PyTorch:** `torch==2.13.0+cu130` from `--index-url https://download.pytorch.org/whl/cu130`.
- **ComfyUI:** repo `https://github.com/Comfy-Org/ComfyUI.git`, ref `72865f4f27eaf5396f8f36370e0a2be3a9a090ee` (v0.33.1). The old `comfyanonymous/ComfyUI` path 301-redirects; use the new one.
- **ComfyUI-Manager:** repo `https://github.com/Comfy-Org/ComfyUI-Manager.git`, ref `d5992a117ee98663b6e93fb040c1e43ee3b25d68` (`main`, 2026-08-18). Its release tags are abandoned — do not "upgrade" to tag 4.2.2.
- **Data root:** host `/home/scott/LLMs/comfyui` → container `/data`.
- **Port:** published as `127.0.0.1:8188` only. Never `0.0.0.0` — ComfyUI has no authentication.
- **Image name:** `comfyui-spark:latest`. Container name: `comfyui`.
- **Pin comments:** every pinned upstream ref carries a `pinned YYYY-MM-DD` comment, matching `sparkyard/llama-cpp/llama-cpp.Dockerfile`.
- **The default Docker runtime on this host is `nvidia`**, and the CUDA base image sets `NVIDIA_VISIBLE_DEVICES=all`. A container therefore sees the GPU *even without* `--gpus all`. To genuinely hide the GPU in a test, pass `-e NVIDIA_VISIBLE_DEVICES=void`.
- **Working directory:** all paths are relative to `/home/scott/Development/ai/comfyui`.

## File Structure

| File | Responsibility |
|---|---|
| `Dockerfile` | Builds the baked image: OS packages, container user, `/opt/venv` (torch + ComfyUI reqs), ComfyUI source |
| `docker/entrypoint.sh` | Runtime preparation: assert `/data` + GPU, seed the tree, build the overlay venv, clone Manager, exec ComfyUI |
| `docker-compose.yml` | Service wiring: GPU reservation, GB10 ulimits, bind mount, port, healthcheck |
| `.env.example` | Documents every configuration variable; copied to the gitignored `.env` |
| `Makefile` | The user-facing interface; thin wrappers over `docker compose` and `scripts/` |
| `scripts/verify-gpu.sh` | Asserts the *image* reaches the GB10 |
| `scripts/verify-entrypoint.sh` | Asserts the six §7.6 entrypoint behaviors, without compose |
| `scripts/verify-http.sh` | Asserts the *running service* serves `/system_stats` with a CUDA device |
| `scripts/verify-persistence.sh` | Asserts overlay-venv installs survive a restart |
| `scripts/verify-e2e.sh` | Asserts a real image generation completes; skips cleanly with no checkpoint |
| `scripts/fetch-model.sh` | Downloads weights into the correct `/data/models` subdirectory |
| `workflows/minimal-txt2img.json` | The API-format workflow `verify-e2e.sh` submits |
| `README.md` | Quickstart, the two-layer venv explanation, troubleshooting |

**Note on a spec amendment:** the spec's §8 names four verification scripts. This plan adds a fifth, `verify-entrypoint.sh`, because §7.6 defines six distinct entrypoint behaviors that no other script covers. Task 2 updates §8 and §7.8 of the spec to match.

---

### Task 1: GPU-capable image

Produces the image and proves it can reach the GPU. Nothing runs ComfyUI yet.

**Files:**
- Create: `.gitignore`
- Create: `Dockerfile`
- Test: `scripts/verify-gpu.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: image `comfyui-spark:latest` with interpreter `/opt/venv/bin/python`, ComfyUI source at `/opt/comfyui`, container user `comfy` at uid/gid from build args `PUID`/`PGID`. Later tasks invoke `scripts/verify-gpu.sh` with optional env `IMAGE` (default `comfyui-spark:latest`).

- [ ] **Step 1: Create `.gitignore`**

```gitignore
.env
*.pyc
__pycache__/
.DS_Store
```

- [ ] **Step 2: Write the failing test `scripts/verify-gpu.sh`**

```bash
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
```

Then `chmod +x scripts/verify-gpu.sh`.

- [ ] **Step 3: Run it to confirm it fails**

Run: `./scripts/verify-gpu.sh`
Expected: FAIL — `Unable to find image 'comfyui-spark:latest' locally` and a non-zero exit. The test must fail for the *right* reason: no image exists yet.

- [ ] **Step 4: Write the `Dockerfile` with the torch companions unpinned**

`torchvision`/`torchaudio` are intentionally unpinned in this step; Step 6 replaces them with the versions the resolver actually selects (spec §7.1 build step 4).

```dockerfile
# syntax=docker/dockerfile:1

# ComfyUI for DGX Spark — GB10 / aarch64 / sm_121.
# Base pinned by arm64 manifest digest 2026-08-18; re-verify on the next bump.
FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04@sha256:88bc2ff57b4a4cbb3dc900cf492203958b24ec7148695992cf0ce8e5cdebd606

ARG PUID=1000
ARG PGID=1000
# ComfyUI v0.33.1, pinned 2026-08-18. Bump via `make update-comfyui`.
ARG COMFYUI_REF=72865f4f27eaf5396f8f36370e0a2be3a9a090ee

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# The CUDA base image ships no Python at all.
# libgl1 + libglib2.0-0: OpenCV-based custom nodes. ffmpeg: video nodes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        git curl ca-certificates \
        libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 already owns uid/gid 1000 as "ubuntu". Rename it rather than
# collide, so files written to the bind mount are owned by the host user.
RUN set -eux; \
    if getent group "${PGID}" >/dev/null; then \
        groupmod -n comfy "$(getent group "${PGID}" | cut -d: -f1)"; \
    else \
        groupadd -g "${PGID}" comfy; \
    fi; \
    if getent passwd "${PUID}" >/dev/null; then \
        usermod -l comfy -d /home/comfy -m -g "${PGID}" "$(getent passwd "${PUID}" | cut -d: -f1)"; \
    else \
        useradd -u "${PUID}" -g "${PGID}" -m -d /home/comfy comfy; \
    fi

# The baked venv. The overlay venv on /data inherits from it (spec §7.2).
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# torch is fixed by the spec §2.1 probe. Companions pinned in Step 6.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0+cu130 torchvision torchaudio

# ComfyUI lives in the image, never on the bind mount, so an upgrade is a
# rebuild and can never leave a half-updated working tree on disk.
RUN git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git /opt/comfyui \
 && git -C /opt/comfyui checkout "${COMFYUI_REF}" \
 && pip install --no-cache-dir -r /opt/comfyui/requirements.txt \
 && chown -R "${PUID}:${PGID}" /opt/comfyui

USER comfy
WORKDIR /opt/comfyui
EXPOSE 8188
```

- [ ] **Step 5: Build the image**

Run: `docker build -t comfyui-spark:latest .`
Expected: success. Watch for any line reading `Building wheel for ...` — every dependency should install from a prebuilt `aarch64` wheel (spec §7.1). If something builds from source, stop and report it rather than adding a toolchain.

- [ ] **Step 6: Pin the resolved torch companion versions**

Capture what actually resolved:

```bash
docker run --rm --entrypoint /opt/venv/bin/pip comfyui-spark:latest \
  freeze | grep -iE '^torch(vision|audio)=='
```

Copy the two exact strings into the `Dockerfile`, replacing the bare
`torchvision torchaudio` in the pip line so it reads (substituting the real
versions you just captured, keeping the `+cu130` local-version suffix):

```dockerfile
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0+cu130 \
        torchvision==<captured> \
        torchaudio==<captured>
```

- [ ] **Step 7: Rebuild with the pins and run the test**

Run: `docker build -t comfyui-spark:latest . && ./scripts/verify-gpu.sh`
Expected: PASS, printing device `NVIDIA GB10`, capability `(12, 1)`, an `arch_list` containing `sm_120` but **not** `sm_121`, and `bf16 matmul on device: finite`.

- [ ] **Step 8: Commit**

```bash
git add .gitignore Dockerfile scripts/verify-gpu.sh
git commit -m "feat: GPU-capable ComfyUI base image for GB10

Bakes torch cu130 and ComfyUI v0.33.1 into nvidia/cuda:13.1.0-runtime.
verify-gpu.sh asserts capability (12,1) and a finite bf16 matmul, so a
silent CPU fallback fails the build rather than surfacing mid-render."
```

---

### Task 2: Entrypoint

Implements all six startup behaviors from spec §7.6 and proves each one.

**Files:**
- Create: `docker/entrypoint.sh`
- Modify: `Dockerfile` (append `COPY` + `ENTRYPOINT`, before the `USER comfy` line)
- Modify: `docs/superpowers/specs/2026-08-18-comfyui-docker-design.md` (§8 and §7.8)
- Test: `scripts/verify-entrypoint.sh`

**Interfaces:**
- Consumes: image `comfyui-spark:latest` and `/opt/venv/bin/python` from Task 1.
- Produces: `/usr/local/bin/entrypoint.sh` as the image `ENTRYPOINT`. Reads env `COMFYUI_ARGS`, `COMFYUI_ALLOW_CPU`, `COMFYUI_SKIP_LAUNCH`, `COMFYUI_DATA_DIR` (default `/data`). Creates `/data/venv/bin/python` — the interpreter every later task execs.

- [ ] **Step 1: Write the failing test `scripts/verify-entrypoint.sh`**

```bash
#!/usr/bin/env bash
# Asserts the six startup behaviors of spec §7.6, without needing compose.
set -euo pipefail
IMAGE="${IMAGE:-comfyui-spark:latest}"
TMP="$(mktemp -d)"
trap 'chmod -R u+w "$TMP" 2>/dev/null || true; rm -rf "$TMP"' EXIT

echo "==> verify-entrypoint: startup contract"

# Case 1 — a cold start seeds the /data tree and builds the overlay venv.
docker run --rm --gpus all -e COMFYUI_SKIP_LAUNCH=1 -v "$TMP:/data" "$IMAGE" >/dev/null
for p in models/checkpoints models/loras models/text_encoders models/controlnet \
         input output temp user custom_nodes/ComfyUI-Manager venv/bin/python; do
    [ -e "$TMP/$p" ] || { echo "FAIL: cold start did not create $p" >&2; exit 1; }
done
echo "    cold start seeded /data and created the overlay venv"

# Case 2 — the overlay venv really does inherit the baked torch.
docker run --rm --gpus all -v "$TMP:/data" --entrypoint /data/venv/bin/python "$IMAGE" \
    -c 'import torch; assert torch.cuda.is_available(); print("    overlay venv sees torch", torch.__version__)'

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

echo "==> verify-entrypoint: PASS"
```

Then `chmod +x scripts/verify-entrypoint.sh`.

- [ ] **Step 2: Run it to confirm it fails**

Run: `./scripts/verify-entrypoint.sh`
Expected: FAIL on Case 1 — the image has no `ENTRYPOINT` yet, so `docker run` exits without creating anything and the `models/checkpoints` check trips.

- [ ] **Step 3: Write `docker/entrypoint.sh`**

```bash
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
#    survive restarts (spec §7.2). --system-site-packages inherits the baked
#    torch, so no multi-gigabyte download happens at boot.
if [ ! -x "$DATA_DIR/venv/bin/python" ]; then
    log "creating the overlay venv at $DATA_DIR/venv"
    /opt/venv/bin/python -m venv --system-site-packages "$DATA_DIR/venv"
fi

# 5. ComfyUI-Manager, first start only.
if [ ! -d "$DATA_DIR/custom_nodes/ComfyUI-Manager" ]; then
    log "cloning ComfyUI-Manager @ ${MANAGER_REF}"
    git clone --filter=blob:none "$MANAGER_REPO" "$DATA_DIR/custom_nodes/ComfyUI-Manager"
    git -C "$DATA_DIR/custom_nodes/ComfyUI-Manager" checkout "$MANAGER_REF"
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
```

- [ ] **Step 4: Wire the entrypoint into the `Dockerfile`**

Insert these two lines immediately **before** the `USER comfy` line:

```dockerfile
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
```

and add this line immediately **after** `EXPOSE 8188`:

```dockerfile
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```

- [ ] **Step 5: Rebuild and run the test**

Run: `docker build -t comfyui-spark:latest . && ./scripts/verify-entrypoint.sh`
Expected: PASS on all six cases. Case 1 is slow the first time — it clones the Manager and installs its requirements.

- [ ] **Step 6: Update the spec to list the fifth script**

In `docs/superpowers/specs/2026-08-18-comfyui-docker-design.md`:

1. In the §8 table, add this row directly after the `verify-gpu.sh` row:

```markdown
| `verify-entrypoint.sh` | The six §7.6 startup behaviors: `/data` seeded, overlay venv created, Manager cloned once and not re-cloned, GPU guard trips with no device, `COMFYUI_ALLOW_CPU=1` bypasses it, unwritable `/data` refused |
```

2. In the sentence below that table, change `runs 1–3 and attempts 4` to
   `runs 1–4 and attempts 5`.

3. In the §7.8 table, change the `verify` row's action to:
   `Runs verify-gpu, verify-entrypoint, verify-http, verify-persistence, then attempts verify-e2e`.

- [ ] **Step 7: Commit**

```bash
git add docker/entrypoint.sh scripts/verify-entrypoint.sh Dockerfile \
        docs/superpowers/specs/2026-08-18-comfyui-docker-design.md
git commit -m "feat: entrypoint that prepares /data and refuses CPU fallback

Seeds the folder_paths tree, builds the --system-site-packages overlay venv,
and clones ComfyUI-Manager once. Refuses to start when no CUDA device is
visible, since a silent CPU render is worse than a crash.

Adds verify-entrypoint.sh and amends spec §8/§7.8 to list it."
```

---

### Task 3: Compose service

Turns the image into a running, reachable service.

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `Makefile`
- Test: `scripts/verify-http.sh`

**Interfaces:**
- Consumes: image and entrypoint from Tasks 1–2.
- Produces: compose service `comfyui` (container `comfyui`), reachable at `http://127.0.0.1:${COMFYUI_PORT}`. Later tasks use `docker compose exec -T comfyui …` and call `scripts/verify-http.sh` as a readiness gate.

- [ ] **Step 1: Write the failing test `scripts/verify-http.sh`**

Doubles as the readiness gate other scripts reuse.

```bash
#!/usr/bin/env bash
# Asserts the running service serves /system_stats backed by a CUDA device.
# Also used by other scripts as a readiness gate.
set -euo pipefail
cd "$(dirname "$0")/.."

# Honour a non-default port set in .env, not just one exported in the shell.
# shellcheck disable=SC1091
[ -f .env ] && . ./.env

PORT="${COMFYUI_PORT:-8188}"
URL="http://127.0.0.1:${PORT}/system_stats"
TIMEOUT="${VERIFY_HTTP_TIMEOUT:-240}"

echo "==> verify-http: waiting up to ${TIMEOUT}s for ${URL}"
deadline=$((SECONDS + TIMEOUT))
until curl -fsS "$URL" -o /dev/null 2>/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "FAIL: no successful response from $URL within ${TIMEOUT}s" >&2
        docker compose logs --tail=40 comfyui >&2 || true
        exit 1
    fi
    sleep 3
done

resp="$(mktemp)"
trap 'rm -f "$resp"' EXIT
curl -fsS "$URL" -o "$resp"

python3 - "$resp" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
system = data["system"]
devices = data["devices"]
print("    comfyui   ", system["comfyui_version"])
print("    pytorch   ", system["pytorch_version"])
if not devices:
    sys.exit("FAIL: /system_stats reported no devices at all")
dev = devices[0]
print("    device    ", dev["name"], "(type=%s)" % dev["type"])
if dev["type"] != "cuda":
    sys.exit("FAIL: primary device type is %r, not 'cuda' — ComfyUI is on the CPU" % dev["type"])
if "GB10" not in dev["name"]:
    sys.exit("FAIL: expected a GB10, got %r" % dev["name"])
PY

echo "==> verify-http: PASS"
```

Then `chmod +x scripts/verify-http.sh`.

- [ ] **Step 2: Run it to confirm it fails**

Run: `./scripts/verify-http.sh`
Expected: FAIL after the timeout — nothing is listening on 8188. To keep this quick, run it as `VERIFY_HTTP_TIMEOUT=10 ./scripts/verify-http.sh`.

- [ ] **Step 3: Write `.env.example`**

```bash
# Copy to .env and adjust. .env is gitignored.

# Host directory bind-mounted at /data — models, outputs, custom nodes,
# the overlay venv. Expect this to grow to hundreds of GB.
COMFYUI_DATA_PATH=/home/scott/LLMs/comfyui

# Host port. Always published on 127.0.0.1 only: ComfyUI has no
# authentication, so anything that can reach it can read and write files
# and execute custom-node code.
COMFYUI_PORT=8188

# Appended to the ComfyUI command line.
#   --highvram                     keep models resident; 121 GB unified memory
#                                  makes eviction pointless
#   --use-pytorch-cross-attention  xformers has no aarch64/sm_121 wheels
# Add --disable-all-custom-nodes here to boot past a node that breaks startup.
COMFYUI_ARGS=--highvram --use-pytorch-cross-attention

# Container user. Keep these matching your host user so files written to
# COMFYUI_DATA_PATH stay editable without sudo. Check with `id -u; id -g`.
PUID=1000
PGID=1000

# Optional. Only read by scripts/fetch-model.sh, for gated HuggingFace repos.
HF_TOKEN=

# Set to 1 to let ComfyUI start without a GPU. Only for debugging.
# COMFYUI_ALLOW_CPU=1
```

- [ ] **Step 4: Write `docker-compose.yml`**

```yaml
# ComfyUI on DGX Spark (GB10). See docs/superpowers/specs/ for the design.
services:
  comfyui:
    build:
      context: .
      args:
        PUID: "${PUID:-1000}"
        PGID: "${PGID:-1000}"
    image: comfyui-spark:latest
    container_name: comfyui
    restart: unless-stopped

    # These four settings are carried over from the working GB10
    # configuration in sparkyard/docker-compose.yml.
    ipc: host                       # shared memory with the GPU driver
    security_opt:
      - seccomp:unconfined          # low-level CUDA calls
    ulimits:
      memlock: -1                   # unified memory needs unlimited locked pages
      stack: 67108864

    # 127.0.0.1 only — ComfyUI has no authentication.
    ports:
      - "127.0.0.1:${COMFYUI_PORT:-8188}:8188"

    volumes:
      - "${COMFYUI_DATA_PATH}:/data"

    environment:
      - COMFYUI_ARGS=${COMFYUI_ARGS:---highvram --use-pytorch-cross-attention}
      - COMFYUI_ALLOW_CPU=${COMFYUI_ALLOW_CPU:-0}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - HF_TOKEN=${HF_TOKEN:-}

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8188/system_stats"]
      interval: 30s
      timeout: 10s
      retries: 3
      # Generous: a cold first start clones the Manager and builds the venv.
      start_period: 120s
```

- [ ] **Step 5: Write the `Makefile`**

`verify-e2e` needs no `|| true` guard: it exits 0 when it skips for want of a
checkpoint, and non-zero only on a genuine failure — see Task 5.

```makefile
.DEFAULT_GOAL := help
SHELL := /bin/bash

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-16s %s\n", $$1, $$2}'

build: ## Build the image
	docker compose build

up: ## Start the service
	docker compose up -d

down: ## Stop the service
	docker compose down

logs: ## Follow the logs
	docker compose logs -f comfyui

shell: ## Interactive shell in the running container
	docker compose exec comfyui bash

verify: ## Run the full verification suite
	./scripts/verify-gpu.sh
	./scripts/verify-entrypoint.sh
	./scripts/verify-http.sh
	./scripts/verify-persistence.sh
	./scripts/verify-e2e.sh

reset-venv: ## Delete the overlay venv; recreated on next start
	@echo "Removing $${COMFYUI_DATA_PATH:-/home/scott/LLMs/comfyui}/venv"
	rm -rf "$${COMFYUI_DATA_PATH:-/home/scott/LLMs/comfyui}/venv"

update-comfyui: ## Print the newest upstream tag and SHA to pin
	@tag=$$(git ls-remote --tags --refs https://github.com/Comfy-Org/ComfyUI.git \
	        | sed 's#.*refs/tags/##' | sort -V | tail -1); \
	sha=$$(git ls-remote https://github.com/Comfy-Org/ComfyUI.git "refs/tags/$$tag" | cut -f1); \
	echo "newest tag: $$tag"; \
	echo "sha:        $$sha"; \
	echo "Paste into the Dockerfile ARG COMFYUI_REF, update the pinned date, then: make build verify"

fetch-model: ## Download a model: make fetch-model URL=... DEST=checkpoints
	./scripts/fetch-model.sh "$(URL)" "$(DEST)"

.PHONY: help build up down logs shell verify reset-venv update-comfyui fetch-model
```

Note the `sort -V` in `update-comfyui`: a plain lexicographic sort ranks
`v0.9.2` above `v0.33.1`, which is exactly how this project nearly got pinned
to a seven-month-old release.

- [ ] **Step 6: Create `.env` and the data directory, then start**

```bash
cp .env.example .env
mkdir -p /home/scott/LLMs/comfyui
make up
```

- [ ] **Step 7: Run the test**

Run: `./scripts/verify-http.sh`
Expected: PASS, printing the ComfyUI version, the PyTorch version, and a device line reading `NVIDIA GB10` with `type=cuda`.

- [ ] **Step 8: Confirm the port binding is localhost-only**

Run: `docker compose ps --format '{{.Name}}\t{{.Ports}}'`
Expected: the mapping reads `127.0.0.1:8188->8188/tcp`. If it shows `0.0.0.0`, stop — the service is exposed to the LAN without authentication.

- [ ] **Step 9: Commit**

```bash
git add docker-compose.yml .env.example Makefile scripts/verify-http.sh
git commit -m "feat: compose service, env surface, and Makefile

Publishes ComfyUI on 127.0.0.1:8188 with the GB10 ipc/ulimits/seccomp
settings from sparkyard. verify-http.sh asserts the primary device is CUDA,
so a service that came up on the CPU fails rather than looking healthy."
```

---

### Task 4: Persistence guarantee

The load-bearing test. The two-layer venv exists solely to make this pass, so it must be shown to fail without the overlay before it is trusted passing with it.

> **Superseded during implementation:** every `docker compose restart` in this task (Steps 1 and 2 below) is wrong and was replaced by `docker compose up -d --force-recreate`. A restart reuses the same container object and never discards its writable layer, so a package installed into the baked `/opt/venv` "survives" it too and the test proves nothing. See spec §8. The probe package also changed from `six` to `pyjokes`, and the assertion after the recreate now re-checks `pip list --local`, not only `import`. This section is left as written for the record; the shipped script is the authority.

**Files:**
- Test: `scripts/verify-persistence.sh`

**Interfaces:**
- Consumes: the running compose service from Task 3, the overlay interpreter `/data/venv/bin/python`, and `scripts/verify-http.sh` as a readiness gate.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test `scripts/verify-persistence.sh`**

```bash
#!/usr/bin/env bash
# Asserts that packages installed into the overlay venv survive a restart —
# the entire reason for the two-layer environment (spec §7.2, §8).
set -euo pipefail
cd "$(dirname "$0")/.."

# six is tiny, pure-Python, and NOT a ComfyUI dependency, so finding it later
# can only be explained by the overlay venv having persisted.
PKG=six

echo "==> verify-persistence: overlay venv survives a restart"

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

# It must live in the OVERLAY layer, not the baked one. --local excludes
# packages inherited via --system-site-packages.
if ! docker compose exec -T comfyui /data/venv/bin/pip list --local 2>/dev/null \
        | awk '{print tolower($1)}' | grep -qx "$PKG"; then
    echo "FAIL: $PKG is not in the overlay layer — 'pip list --local' does not list it." >&2
    echo "      It was installed somewhere that a restart will discard." >&2
    exit 1
fi
echo "    $PKG is in the overlay layer, not the baked layer"

docker compose restart comfyui >/dev/null
echo "    restarted the container"

./scripts/verify-http.sh >/dev/null
echo "    service came back up"

if ! docker compose exec -T comfyui /data/venv/bin/python -c "import $PKG" >/dev/null 2>&1; then
    echo "FAIL: $PKG did not survive the restart — the overlay venv is not persisted." >&2
    echo "      Check that COMFYUI_DATA_PATH is bind-mounted at /data and that" >&2
    echo "      the entrypoint creates /data/venv rather than a venv inside the image." >&2
    exit 1
fi
echo "    $PKG survived the restart"

echo "==> verify-persistence: PASS"
```

Then `chmod +x scripts/verify-persistence.sh`.

- [ ] **Step 2: Prove the test detects the failure it exists to catch**

Do not skip this. A persistence test that cannot fail is worthless.

Temporarily install into the *baked* venv — which lives in the container layer and is discarded on restart — instead of the overlay:

```bash
docker compose exec -T --user root comfyui /opt/venv/bin/pip install --no-cache-dir -q six
docker compose exec -T comfyui /opt/venv/bin/python -c 'import six; print("present before restart")'
docker compose restart comfyui && ./scripts/verify-http.sh >/dev/null
docker compose exec -T comfyui /opt/venv/bin/python -c 'import six' \
  && echo "UNEXPECTED: baked-venv install survived" \
  || echo "as expected: the baked-venv install did NOT survive the restart"
```

Expected: `as expected: the baked-venv install did NOT survive the restart`.
This is the failure mode `verify-persistence.sh` guards against. If instead it
survived, the container was recreated rather than restarted, or `/opt/venv` is
on a volume — investigate before continuing, because the test's meaning depends
on this distinction.

- [ ] **Step 3: Run the real test**

Run: `./scripts/verify-persistence.sh`
Expected: PASS — `six` installs into the overlay, appears in `pip list --local`, and is still importable after the restart.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify-persistence.sh
git commit -m "test: prove overlay venv installs survive a restart

Uses six precisely because it is not a ComfyUI dependency, so its presence
after a restart can only be explained by the overlay venv persisting.
Asserts pip list --local too, catching an install that silently landed in
the baked layer where a restart would discard it."
```

---

### Task 5: End-to-end generation, model fetching, and documentation

Closes the loop: a real render, a way to get weights onto disk, and the docs that make the two-layer venv survivable for a future reader.

**Files:**
- Create: `workflows/minimal-txt2img.json`
- Create: `scripts/fetch-model.sh`
- Create: `scripts/verify-e2e.sh`
- Create: `README.md`

**Interfaces:**
- Consumes: the running service from Task 3; `scripts/verify-http.sh` as a readiness gate.
- Produces: `scripts/fetch-model.sh <URL> <DEST_SUBDIR>`, invoked by `make fetch-model URL=… DEST=…`.

- [ ] **Step 1: Write `workflows/minimal-txt2img.json`**

API-format graph. `verify-e2e.sh` substitutes the real filename into
`ckpt_name`, which is why the placeholder string is distinctive.

```json
{
  "3": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 42,
      "steps": 8,
      "cfg": 7.0,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0,
      "model": ["4", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    }
  },
  "4": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": { "ckpt_name": "__CKPT_NAME__" }
  },
  "5": {
    "class_type": "EmptyLatentImage",
    "inputs": { "width": 512, "height": 512, "batch_size": 1 }
  },
  "6": {
    "class_type": "CLIPTextEncode",
    "inputs": { "text": "a red apple on a wooden table", "clip": ["4", 1] }
  },
  "7": {
    "class_type": "CLIPTextEncode",
    "inputs": { "text": "", "clip": ["4", 1] }
  },
  "8": {
    "class_type": "VAEDecode",
    "inputs": { "samples": ["3", 0], "vae": ["4", 2] }
  },
  "9": {
    "class_type": "SaveImage",
    "inputs": { "filename_prefix": "verify-e2e", "images": ["8", 0] }
  }
}
```

- [ ] **Step 2: Write the failing test `scripts/verify-e2e.sh`**

```bash
#!/usr/bin/env bash
# Asserts a real generation completes end to end. Skips cleanly, and exits 0,
# when no checkpoint is installed (spec §8).
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f .env ] && . ./.env

PORT="${COMFYUI_PORT:-8188}"
BASE="http://127.0.0.1:${PORT}"
TIMEOUT="${VERIFY_E2E_TIMEOUT:-300}"

echo "==> verify-e2e: end-to-end generation"

./scripts/verify-http.sh >/dev/null

# Ask ComfyUI itself what checkpoints it can see, rather than guessing from
# the host filesystem — this also proves --base-directory resolved correctly.
ckpt="$(curl -fsS "${BASE}/object_info/CheckpointLoaderSimple" \
    | python3 -c '
import json, sys
info = json.load(sys.stdin)["CheckpointLoaderSimple"]
names = info["input"]["required"]["ckpt_name"][0]
print(names[0] if names else "")
')"

if [ -z "$ckpt" ]; then
    echo "    SKIP: no checkpoint installed."
    echo "    Install one, then re-run:"
    echo "      make fetch-model URL=<safetensors-url> DEST=checkpoints"
    echo "==> verify-e2e: SKIPPED"
    exit 0
fi
echo "    using checkpoint: $ckpt"

payload="$(mktemp)"; result="$(mktemp)"
trap 'rm -f "$payload" "$result"' EXIT

python3 - "$ckpt" "$payload" <<'PY'
import json, sys
ckpt, out = sys.argv[1], sys.argv[2]
graph = json.load(open("workflows/minimal-txt2img.json"))
graph["4"]["inputs"]["ckpt_name"] = ckpt
json.dump({"prompt": graph, "client_id": "verify-e2e"}, open(out, "w"))
PY

prompt_id="$(curl -fsS -X POST "${BASE}/prompt" -H 'Content-Type: application/json' \
    --data-binary "@${payload}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["prompt_id"])')"
echo "    queued prompt $prompt_id"

deadline=$((SECONDS + TIMEOUT))
while true; do
    curl -fsS "${BASE}/history/${prompt_id}" -o "$result"
    status="$(python3 - "$result" "$prompt_id" <<'PY'
import json, sys
hist = json.load(open(sys.argv[1])).get(sys.argv[2])
if not hist:
    print("pending"); raise SystemExit
print(hist.get("status", {}).get("status_str", "pending"))
PY
)"
    case "$status" in
        success) break ;;
        error)
            echo "FAIL: ComfyUI reported an execution error" >&2
            python3 -m json.tool "$result" >&2
            exit 1 ;;
    esac
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "FAIL: generation did not finish within ${TIMEOUT}s" >&2
        docker compose logs --tail=40 comfyui >&2 || true
        exit 1
    fi
    sleep 3
done

# The graph must have produced a real image on disk, not merely reported success.
images="$(python3 - "$result" "$prompt_id" <<'PY'
import json, sys
hist = json.load(open(sys.argv[1]))[sys.argv[2]]
names = [img["filename"]
         for node in hist.get("outputs", {}).values()
         for img in node.get("images", [])]
print("\n".join(names))
PY
)"
[ -n "$images" ] || { echo "FAIL: run succeeded but produced no images" >&2; exit 1; }

while read -r name; do
    [ -n "$name" ] || continue
    # </dev/null so exec does not consume the here-string driving this loop.
    docker compose exec -T comfyui test -s "/data/output/$name" </dev/null \
        || { echo "FAIL: $name is missing or empty in /data/output" >&2; exit 1; }
    echo "    wrote /data/output/$name"
done <<< "$images"

echo "==> verify-e2e: PASS"
```

Then `chmod +x scripts/verify-e2e.sh`.

- [ ] **Step 3: Run it to confirm it skips rather than passes vacuously**

Run: `./scripts/verify-e2e.sh`
Expected, with no models installed: `SKIP: no checkpoint installed.` and exit 0.
It must reach the skip via the `/object_info` query — if it errors before that,
`--base-directory` is not resolving and that is a real bug.

- [ ] **Step 4: Write `scripts/fetch-model.sh`**

```bash
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
```

Then `chmod +x scripts/fetch-model.sh`.

- [ ] **Step 5: Verify `fetch-model.sh` argument handling**

Run: `./scripts/fetch-model.sh`
Expected: the usage block on stderr, exit code 2. Confirm with `echo $?`.

- [ ] **Step 6: Write `README.md`**

````markdown
# ComfyUI on DGX Spark

GPU-accelerated ComfyUI for this machine's GB10 (Grace-Blackwell, `aarch64`,
compute capability 12.1). Off-the-shelf ComfyUI images are `x86_64`-only, so
this builds its own on `nvidia/cuda:13.1.0-runtime` with PyTorch `cu130`
`aarch64` wheels.

Design: `docs/superpowers/specs/2026-08-18-comfyui-docker-design.md`

## Quickstart

```bash
cp .env.example .env          # adjust COMFYUI_DATA_PATH if needed
mkdir -p /home/scott/LLMs/comfyui
make build
make up
make verify
```

Then open <http://127.0.0.1:8188>.

The first start takes a few minutes: it creates the overlay venv and clones
ComfyUI-Manager. Later starts are quick.

## Getting models

Nothing is downloaded automatically.

```bash
make fetch-model URL=https://.../sd_xl_base_1.0.safetensors DEST=checkpoints
```

`DEST` is any folder under `/data/models` — `checkpoints`, `loras`, `vae`,
`text_encoders`, `diffusion_models`, `controlnet`, `upscale_models`, …
Set `HF_TOKEN` in `.env` for gated HuggingFace repos. Files land in
`$COMFYUI_DATA_PATH/models/$DEST` and ComfyUI sees them on the next refresh —
no restart needed.

## Commands

Run `make` on its own for the full list. The common ones: `build`, `up`,
`down`, `logs`, `shell`, `verify`.

## How this is put together

- **ComfyUI itself lives in the image**, not on disk. Upgrading is a rebuild,
  so there is never a half-updated working tree to debug.
- **Your data lives on the bind mount** at `$COMFYUI_DATA_PATH` → `/data`:
  models, outputs, inputs, custom nodes, settings.
- **Two Python environments**, which is the one genuinely surprising part:

  | Path | Contents | Survives a restart? |
  |---|---|---|
  | `/opt/venv` | torch + ComfyUI's own requirements, baked into the image | No — rebuilt with the image |
  | `/data/venv` | anything ComfyUI-Manager installs | Yes — it is on the bind mount |

  `/data/venv` is created with `--system-site-packages`, so it *inherits*
  everything from `/opt/venv` without copying it. ComfyUI runs from
  `/data/venv`, which is why nodes you install from the UI — and their pip
  dependencies — are still there tomorrow.

## Troubleshooting

**Which layer is a package in?**

```bash
make shell
/data/venv/bin/pip list --local   # the overlay only — Manager's installs
/data/venv/bin/pip list           # both layers
```

If a package appears in the second but not the first, it is baked into the
image and a rebuild governs its version.

**A custom node breaks startup.** Add `--disable-all-custom-nodes` to
`COMFYUI_ARGS` in `.env`, run `make down && make up`, remove the offending
directory from `$COMFYUI_DATA_PATH/custom_nodes/`, then take the flag out.

**The overlay venv is wedged.** `make down && make reset-venv && make up`.
This discards every Manager-installed package — the nodes themselves stay in
`custom_nodes/`, so reinstall their requirements from the Manager UI.

**It seems slow — is it actually on the GPU?** `./scripts/verify-gpu.sh`, or
check `curl -s localhost:8188/system_stats | python3 -m json.tool`. The
entrypoint refuses to start without a GPU, so a CPU fallback should be
impossible; if you hit one, that is a bug worth reporting.

**`sm_121` is missing from `torch.cuda.get_arch_list()`.** Expected, not a
fault. `sm_120` cubins run on the GB10's 12.1 capability. `verify-gpu.sh`
confirms real kernels execute.

## Security

Published on `127.0.0.1:8188` only. ComfyUI has no authentication and custom
nodes execute arbitrary Python, so anything that can reach the port has
effective code execution. Reach it from another machine with an SSH tunnel:

```bash
ssh -L 8188:127.0.0.1:8188 scott@<this-host>
```
````

- [ ] **Step 7: Add `workflows/` to the spec's file layout**

The spec's §6 layout predates this directory. In
`docs/superpowers/specs/2026-08-18-comfyui-docker-design.md`, add this line to
the §6 tree, directly after the `scripts/` block:

```
├── workflows/
│   └── minimal-txt2img.json  # the graph verify-e2e.sh submits
```

- [ ] **Step 8: Run the full suite**

Run: `make verify`
Expected: `verify-gpu`, `verify-entrypoint`, `verify-http`, and
`verify-persistence` all PASS; `verify-e2e` SKIPS with the "no checkpoint"
message and exit 0, so `make verify` succeeds overall.

- [ ] **Step 9: Commit**

```bash
git add workflows/ scripts/fetch-model.sh scripts/verify-e2e.sh README.md
git commit -m "feat: end-to-end verification, model fetcher, and README

verify-e2e asks ComfyUI which checkpoints it can see rather than reading the
host filesystem, so it also proves --base-directory resolved. It skips with
exit 0 when no model is installed, and asserts the output file is non-empty
rather than trusting the success status.

README documents the two-layer venv, which is the one part of this setup a
future reader will not guess."
```

---

## Optional follow-up: a real render

Not a task — no code, and it depends on a multi-gigabyte download the user may
not want. Offer it after Task 5, do not assume it.

With a checkpoint installed, `./scripts/verify-e2e.sh` stops skipping and
performs an actual generation. That is the only step that exercises the full
path — sampler, VAE, and disk write — under real GPU load.

```bash
make fetch-model URL=<sdxl-base-safetensors-url> DEST=checkpoints
./scripts/verify-e2e.sh
```
