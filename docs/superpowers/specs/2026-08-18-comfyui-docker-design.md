# ComfyUI on DGX Spark — Docker Service Design

**Date:** 2026-08-18
**Status:** Approved for planning
**Author:** scott@kingsroad.io (with Claude)

## 1. Purpose

Package ComfyUI as a Docker service that uses this machine's GB10 GPU, running
independently of the existing `sparkyard` LLM stack.

## 2. Machine context (verified 2026-08-18)

| Property | Value |
|---|---|
| GPU | NVIDIA GB10 (Grace-Blackwell), compute capability 12.1 |
| Architecture | `aarch64` |
| Driver / CUDA | 580.173.02 / CUDA 13.0 |
| Memory | 121 GB unified |
| Disk | 437 GB free on single NVMe (`/`) |
| Docker | 29.2.1, Compose v5.0.2 |
| Container toolkit | 1.20.0, `nvidia` is the **default** runtime |
| OS | Ubuntu 24.04.4 LTS |

This combination is why no off-the-shelf ComfyUI image works: nearly all are
`x86_64`-only. The image is built locally.

### 2.1 PyTorch viability probe

Run before committing to the design, in `nvidia/cuda:13.1.0-runtime-ubuntu24.04`
on this host:

```
torch 2.13.0+cu130   cuda_built 13.0
arch_list  ['sm_80','sm_90','sm_100','sm_110','sm_120']
device     NVIDIA GB10   capability (12, 1)   available True
bf16 2048x2048 matmul on GPU -> finite  True
```

**`sm_121` is absent from the arch list, and this is fine.** `sm_120` cubins are
binary-compatible upward within the 12.x family, and the bf16 matmul confirms
real GPU kernels executed rather than a silent CPU fallback. Stock PyTorch
`cu130` aarch64 wheels are therefore sufficient; no custom torch build is needed.

This is the single most important assumption in the design, so
`verify-gpu.sh` (§8) re-asserts it on every deployment.

## 3. Goals

1. `docker compose up -d` yields a working GPU-accelerated ComfyUI at
   `http://127.0.0.1:8188`.
2. Custom nodes installed through ComfyUI-Manager — **and their pip
   dependencies** — survive container restarts and image rebuilds.
3. Model weights and generated outputs live on the host filesystem, readable
   and writable by `scott` without `docker cp` or `sudo`.
4. Reproducible: every upstream input this project names is pinned by digest,
   tag or SHA, with a dated comment, matching the convention in
   `sparkyard/llama-cpp/llama-cpp.Dockerfile` — the base image (manifest
   digest), the torch trio, the ComfyUI ref and the ComfyUI-Manager ref.
   Transitive pip dependencies are **not** pinned: `pip install -r
   requirements.txt` re-resolves them on every build, so two builds of the
   same Dockerfile can differ in those. A constraints/lock file would close
   that gap and is deliberately deferred (§11).
5. A GPU misconfiguration fails loudly at startup rather than silently
   degrading to CPU — asserted on the interpreter that actually renders
   (`/data/venv`), not only on the baked one (§7.6 steps 2 and 7).

## 4. Non-goals (v1)

- SageAttention / xformers / custom fused-attention kernels
- Automatic model downloading on first run
- Authentication, TLS, or LAN exposure
- Multi-GPU or multi-instance scheduling
- Integration with the `sparkyard` compose stack or its `sparkyard` CLI

## 5. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Project scope | Standalone in `~/Development/ai/comfyui` | Independent lifecycle; rebuild ComfyUI without restarting the LLM stack |
| Data root | `/home/scott/LLMs/comfyui` | Reuses the existing AI-model root (192 GB: `ollama/`, `vllm/`) |
| Custom nodes | ComfyUI-Manager + persistent overlay venv | Install from the UI; deps persist |
| Network | `127.0.0.1:8188` only | ComfyUI has no auth; matches llama.cpp/llama-swap/ollama binding |
| Starter model | None | Fast first build; `fetch-model.sh` helper instead |
| ComfyUI version | `Comfy-Org/ComfyUI` v0.33.1 (`72865f4f`) | Latest stable, 2026-08-13 |

### 5.1 Rejected alternatives

- **Everything in the persistent venv** (thin image, torch installed at first
  boot): 5–10 min first start, ~3 GB download, and the image stops pinning what
  actually runs — "rebuild the image" would no longer mean anything.
- **NGC PyTorch base** (`nvcr.io/nvidia/pytorch`, arm64): ~20 GB image, and
  NVIDIA's patched torch conflicts with ComfyUI's pinned deps. Revisit only if
  stock SDPA proves too slow.
- **Named Docker volumes**: dropping in a `.safetensors` or retrieving an output
  PNG would require `docker cp` or root-owned paths.
- **Symlinking `/opt/comfyui/{models,custom_nodes,...}` into `/data`**: obsoleted
  by `--base-directory` (§7.3), which does the same job with one flag and cannot
  be clobbered by a rebuild.

## 6. Repository layout

```
~/Development/ai/comfyui/
├── Dockerfile
├── docker-compose.yml
├── .env.example              # committed; .env is gitignored
├── .gitignore
├── Makefile
├── README.md
├── docker/
│   └── entrypoint.sh
├── scripts/
│   ├── verify-gpu.sh
│   ├── verify-entrypoint.sh
│   ├── verify-http.sh
│   ├── verify-persistence.sh
│   ├── verify-e2e.sh
│   └── fetch-model.sh
├── workflows/
│   └── minimal-txt2img.json  # the graph verify-e2e.sh submits
└── docs/superpowers/specs/2026-08-18-comfyui-docker-design.md
```

## 7. Architecture

### 7.1 Image

Base: `nvidia/cuda:13.1.0-runtime-ubuntu24.04@sha256:88bc2ff57b4a4cbb3dc900cf492203958b24ec7148695992cf0ce8e5cdebd606`
(arm64 manifest digest, pinned 2026-08-18).

The `runtime` (not `devel`) variant suffices: every ComfyUI dependency that
contains compiled code publishes a cp312-compatible `aarch64` wheel — verified
for `comfy-kitchen` 0.2.31, `comfy-aimdo` 0.4.13, `blake3` 1.0.9, `av` 18.1.0,
`safetensors` 0.8.0, and `tokenizers` 0.23.1. Nothing builds from source, so no
compiler toolchain is installed.

Build steps:

1. `apt-get install` python3.12, `python3-venv`, `git`, `curl`,
   `ca-certificates`, `libgl1`, `libglib2.0-0`, `ffmpeg`.
   (`libgl1`/`libglib2.0-0` for OpenCV-based nodes; `ffmpeg` for video nodes.)
2. Create a non-root user with UID/GID from build args `PUID`/`PGID`
   (default `1000:1000`) so bind-mounted files are owned by `scott`.
3. Create the **baked venv** at `/opt/venv`.
4. `pip install --index-url https://download.pytorch.org/whl/cu130` the torch
   trio into `/opt/venv`, all three explicitly pinned. `torch==2.13.0+cu130` is
   fixed by the §2.1 probe. The companion versions are **resolved during
   implementation** by installing against that torch and recording what the
   resolver selects — they are not guessed here. The cu130 aarch64 index
   currently tops out at `torchvision` 0.28.0 and `torchaudio` 2.11.0, so those
   are the expected answers, but the Dockerfile must record the versions that
   actually resolved, not these.
5. `git clone` ComfyUI into `/opt/comfyui`, checkout `72865f4f` (v0.33.1,
   pinned 2026-08-18), and `pip install -r requirements.txt` into `/opt/venv`.
6. `COPY docker/entrypoint.sh`; set as `ENTRYPOINT`.

ComfyUI application code lives in the **image**, not on the bind mount. Upgrading
ComfyUI is a `docker compose build`, and cannot leave a half-updated working tree
on disk.

### 7.2 Two-layer Python environment

The core requirement — Manager-installed nodes and their pip dependencies must
persist — is met with a layered venv rather than by persisting the whole
interpreter:

- `/opt/venv` — **baked** into the image: torch and ComfyUI's `requirements.txt`.
- `/data/venv` — **on the bind mount**, created by the entrypoint on first run:

  ```
  /opt/venv/bin/python -m venv /data/venv
  echo "$(/opt/venv/bin/python -c 'import site; print(site.getsitepackages()[0])')" \
      > "$(/data/venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/_baked_venv.pth"
  ```

  A plain `venv --system-site-packages /data/venv` was tried first and does
  **not** work here: `/opt/venv` is itself a venv, and since Python 3.11 a
  nested venv's `--system-site-packages` resolves against the real base
  interpreter (`sys._base_executable`), not the immediate parent venv — so it
  would see the OS's `dist-packages`, never `/opt/venv`'s torch. A `.pth` file
  naming `/opt/venv`'s site-packages achieves the same inheritance directly.

ComfyUI is launched with `/data/venv/bin/python`. Consequences:

- The `.pth` file means the overlay inherits baked torch; a ~3 GB
  download never happens at boot. `site.py` appends the `.pth` target to
  `sys.path` after the overlay's own site-packages, so precedence is preserved
  (next bullet).
- ComfyUI-Manager shells out to `sys.executable`, which **is** `/data/venv`, so
  every package it installs lands on the bind mount and survives restarts.
- A node requiring a newer version of a baked package installs it into the
  overlay, where it shadows the baked copy. This is the desired precedence.
- Rebuilding the image to bump torch does **not** wipe installed nodes; the
  overlay picks up the new baked torch automatically unless deliberately shadowed.

The trade-off accepted here: debugging a version conflict means checking two
`site-packages` layers. `make shell` and the README document
`pip list --local` (overlay only) versus `pip list` (both) as the diagnostic.

**The layering only works for tools that read `.pth` files.** pip does; `uv`
does not. Measured in the running container through Manager's own
`manager_util.get_installed_packages()`: in pip mode it finds 129 packages,
including `torch`, `numpy`, `transformers`, `pillow` and `safetensors`; in uv
mode it finds 27, and of those five it finds none. ComfyUI-Manager
enables `uv` by default on Linux whenever it can import it — and `uv` is in
Manager's own `requirements.txt` — so by default Manager believes ~100 baked
packages are missing and reinstalls them (torch included) from PyPI into the
overlay, where they shadow the GPU-correct baked copies. The entrypoint
therefore forces `use_uv = False` in Manager's `config.ini` (§7.6 step 6). This
is a property of the environment, not a user preference, so it is re-asserted
on every start rather than seeded once at clone time; the edit is line-scoped
and leaves every other key and section in that file untouched.

The same shadowing can arrive by other routes — any custom node whose
`requirements.txt` names `torch`, `numpy`, `transformers`, `pillow` or
`safetensors` — so detection cannot rely on the `uv` fix alone. `PIP_EXTRA_INDEX_URL`
and `UV_EXTRA_INDEX_URL` point at the cu130 index (§7.4) so that a reinstall
that does happen at least resolves to the right build, and §7.6 step 7 asserts
the outcome before ComfyUI is launched.

### 7.3 Persistence

`/data` is a bind mount of `${COMFYUI_DATA_PATH}` (default
`/home/scott/LLMs/comfyui`). ComfyUI is started with `--base-directory /data`,
which — per `folder_paths.py` at v0.33.1 — roots `models/`, `custom_nodes/`,
`input/`, `output/`, `temp/`, and `user/` under it.

The entrypoint `mkdir -p`s the model subdirectories that `folder_paths.py`
enumerates, so the Manager and the UI see the expected tree on a cold start:

```
checkpoints  configs  loras  vae  text_encoders  clip  diffusion_models  unet
clip_vision  style_models  embeddings  diffusers  vae_approx  controlnet
t2i_adapter  gligen  upscale_models  latent_upscale_models  hypernetworks
photomaker  classifiers
```

ComfyUI-Manager is installed into `/data/custom_nodes/ComfyUI-Manager` on first
start **only if absent**, pinned to `d5992a11` (`main`, 2026-08-18). Manager's
own release tags are abandoned — the newest, `4.2.2`, is from 2026-06-14 and
`main` is 911 commits ahead — so a SHA on `main` is the reproducible choice. The
pin governs the initial clone only; thereafter the user updates Manager from the
UI and the entrypoint leaves it alone.

The install is three steps that must all complete — clone, checkout the pin,
install requirements into the overlay venv — and a container killed partway
through must not leave a state the next start mistakes for "done": a kill
between clone and checkout would silently defeat the pin, and a kill during the
requirements install would leave Manager with none of its dependencies. So it
is made crash-safe the same way the overlay venv is. Clone into a scratch
directory outside `custom_nodes/` (ComfyUI imports every directory it finds
there), checkout, then `mv` into place — an atomic rename on one filesystem, so
the published checkout is either absent or complete and pinned. Write a
`.manager-ready` sentinel only after the requirements install returns 0. A
start that finds no sentinel redoes the requirements install, and re-clones
only if git cannot resolve `HEAD` in the existing checkout — a complete
checkout, including one the user has updated from the UI, is never deleted.

### 7.4 Service definition

```yaml
services:
  comfyui:
    build:
      context: .
      args: {PUID: "${PUID:-1000}", PGID: "${PGID:-1000}"}
    image: comfyui-spark:latest
    container_name: comfyui
    restart: unless-stopped
    ipc: host                      # shared memory with the GPU driver
    security_opt: [seccomp:unconfined]
    ulimits:
      memlock: -1                  # unified memory needs unlimited locked pages
      stack: 67108864
    ports: ["127.0.0.1:${COMFYUI_PORT:-8188}:8188"]
    volumes: ["${COMFYUI_DATA_PATH:-/home/scott/LLMs/comfyui}:/data"]
    environment:
      - COMFYUI_ARGS=${COMFYUI_ARGS:---highvram --use-pytorch-cross-attention}
      - COMFYUI_ALLOW_CPU=${COMFYUI_ALLOW_CPU:-0}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - HF_TOKEN=${HF_TOKEN:-}
      - PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130
      - UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8188/system_stats"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
```

The `ipc`, `security_opt`, and `ulimits` settings are carried over from the
working GB10 configuration in `sparkyard/docker-compose.yml`. Two of them
widen the blast radius of the custom-node code this service exists to run:
`seccomp:unconfined` removes the default syscall filter, and `ipc: host` shares
the host IPC namespace with the other GPU containers on this machine. They are
kept because the GPU path is unreliable without them; the README's security
section states the trade-off rather than leaving it in a compose comment.

`COMFYUI_DATA_PATH` carries its default here as well as in the Makefile and
`fetch-model.sh`, so `make up` without a `.env` starts correctly instead of
failing on an empty bind-mount spec.

`HF_TOKEN` is injected into the container deliberately (Manager's model
downloader reads it), which also exposes it to every custom node — see §7.7.

Port 8188 is free: the host currently listens on 22, 53, 631, 3000, 11000,
11434, 14000, 19000, and 28080.

### 7.5 GB10 tuning defaults

- `--highvram` — with 121 GB of unified memory, evicting models between runs
  costs time and saves nothing.
- `--use-pytorch-cross-attention` — xformers publishes no `aarch64`/sm_121
  wheels; torch 2.13 SDPA is the supported path.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces fragmentation
  across differently-shaped diffusion workloads.

All are overridable via `COMFYUI_ARGS` in `.env` without rebuilding.

### 7.6 Entrypoint responsibilities

In order, failing fast with a distinct message at each step:

1. Assert `/data` exists and is writable → else exit 1.
2. Assert `/opt/venv`'s torch sees a CUDA device, unless `COMFYUI_ALLOW_CPU=1`
   → else exit 1.
3. `mkdir -p` the `/data` tree from §7.3.
4. Create `/data/venv` if absent, linked to `/opt/venv`'s site-packages via
   `.pth` file (§7.2). Readiness of an existing overlay means "torch imports
   **and** reaches the GPU"; a `no-cuda` overlay is never auto-rebuilt (the
   user's packages are on it), it is refused with the shadowing message. Under
   `COMFYUI_ALLOW_CPU=1` a working `import torch` is the whole readiness test,
   so a deliberate CPU start neither rebuilds nor refuses.
5. Install pinned ComfyUI-Manager into `/data/custom_nodes/` if absent, and
   (re)install its requirements whenever either the overlay venv or the
   Manager install did not previously complete (§7.3).
6. Force `use_uv = False` in Manager's `config.ini`, creating the file if
   absent and leaving every other setting in it alone (§7.2).
7. Assert `/data/venv`'s torch sees a CUDA device, unless
   `COMFYUI_ALLOW_CPU=1` → else exit 1, naming overlay shadowing as the likely
   cause and giving the two recoveries (uninstall the overlay copy, or
   `make reset-venv`).
8. `exec /data/venv/bin/python /opt/comfyui/main.py --listen 0.0.0.0 --port 8188
   --base-directory /data --disable-auto-launch ${COMFYUI_ARGS}`

Step 2 exists because the worst failure mode is not a crash — it is ComfyUI
silently running on CPU and the user discovering it forty minutes into a render.

Step 7 exists because step 2 alone does not deliver that guarantee: it
interrogates `/opt/venv`, and ComfyUI runs on `/data/venv`, whose site-packages
comes first on `sys.path`. Everything between the two steps writes to that
overlay. Step 2 is kept anyway — it is the cheapest, earliest signal, it runs
before the multi-minute venv build, and "no GPU in this container at all" and
"the interpreter that renders cannot see the GPU" need different remedies.

`COMFYUI_SKIP_LAUNCH=1` stops after step 7 instead of exec'ing; it is the seam
`verify-entrypoint.sh` uses to test the startup contract without a server that
never exits.

### 7.7 Configuration surface

`.env.example` is committed with every variable documented; `.env` is gitignored.

| Variable | Default | Purpose |
|---|---|---|
| `COMFYUI_DATA_PATH` | `/home/scott/LLMs/comfyui` | Host path bind-mounted at `/data` |
| `COMFYUI_PORT` | `8188` | Host port, bound to `127.0.0.1` only |
| `COMFYUI_ARGS` | `--highvram --use-pytorch-cross-attention` | Appended to the `main.py` command line |
| `PUID` / `PGID` | `1000` / `1000` | Container user, so `/data` files are owned by `scott` |
| `HF_TOKEN` | empty | Optional, for gated repos. Read by `fetch-model.sh` on the host **and** injected into the container, where Manager's downloader and `huggingface-hub` use it — so every custom node can read it. Use a read-only, minimally-scoped token |
| `COMFYUI_ALLOW_CPU` | unset | Set to `1` to bypass both of the entrypoint's GPU assertions |
| `COMFYUI_SKIP_LAUNCH` | unset | Set to `1` to run the entrypoint's preparation and stop before launching ComfyUI. Testing seam for `verify-entrypoint.sh`; not exposed in `docker-compose.yml` |

### 7.8 Makefile targets

The Makefile is the documented interface; every target is a thin wrapper over
`docker compose` so nothing is hidden.

| Target | Action |
|---|---|
| `build` | `docker compose build` |
| `up` / `down` | Start / stop the service |
| `logs` | `docker compose logs -f comfyui` |
| `shell` | Interactive shell in the running container |
| `verify` | Runs verify-gpu, verify-entrypoint, verify-http, verify-persistence, then attempts verify-e2e |
| `reset-venv` | Deletes `/data/venv`; recreated on next start (§9) |
| `update-comfyui` | Prints the newest upstream tag and SHA to paste into the Dockerfile (§10) |
| `fetch-model` | Wrapper over `scripts/fetch-model.sh` |

## 8. Verification

Written before the implementation, per TDD. Each is a standalone script;
`make verify` runs 1–4 and attempts 5.

| Script | Asserts |
|---|---|
| `verify-gpu.sh` | The same assertion against **both** interpreters — `torch.cuda.is_available()`, device name contains `GB10`, capability `(12, 1)`, a bf16 matmul returns finite values: on `/opt/venv` in the image (the only check a freshly built image can make), and on `/data/venv` in the **running** container, which is the one that renders. Skips the second only when no overlay venv exists yet |
| `verify-entrypoint.sh` | The §7.6 startup contract against a throwaway `/data`, in 13 cases: the six documented behaviors (`/data` seeded, overlay venv created, Manager installed once and not re-installed, GPU guard trips with no device, `COMFYUI_ALLOW_CPU=1` bypasses it, unwritable `/data` refused), plus the recovery paths — interrupted venv build rebuilt, previously-working venv refused rather than wiped, Manager requirements reinstalled after a venv rebuild, `reset-venv` recovery, a clone killed in flight self-healing to the pinned ref, an interrupted requirements install reinstalled without re-cloning, `use_uv = False` seeded and repaired, and an overlay venv that cannot reach the GPU refused both before and after the readiness check |
| `verify-http.sh` | `GET /system_stats` returns 200 and lists a CUDA device |
| `verify-persistence.sh` | `pip install pyjokes` (tiny, pure-Python, dependency-free, and nothing in the ML stack can drag it in) into `/data/venv`, `docker compose up -d --force-recreate`, then assert it is still importable **and** still reported by `pip list --local`. Never uninstalls a package it did not install: the probe is skipped if already present, and the cleanup trap reports rather than silently no-ops when it cannot reach the container |
| `verify-e2e.sh` | `POST /prompt` with a minimal workflow, poll `/history`, assert a PNG appears in `/data/output`. Skips with an explicit message when no checkpoint is installed |

`verify-persistence.sh` is the one that would have caught the naive design, so it
must fail against a container built without the overlay venv before it passes
with one. It challenges with `--force-recreate`, not `docker compose restart`:
a plain restart reuses the same container and never discards its writable
layer, so a package installed straight into the baked `/opt/venv` would
"survive" a restart too — only recreating the container from the image
actually exercises the bind-mount-vs-image-layer distinction this test exists
to prove.

## 9. Failure modes

| Failure | Detection | Behavior |
|---|---|---|
| No GPU visible to container | Entrypoint step 2 | Exit 1 with remediation message; `COMFYUI_ALLOW_CPU=1` to override |
| `/data` not writable | Entrypoint step 1 | Exit 1 naming the path and expected UID |
| Overlay venv corrupted | Entrypoint step 4 | Refuses to start if it previously worked (the user's packages are on it); `make reset-venv` deletes `/data/venv`, recreated on next start |
| Overlay torch shadows baked torch with a CPU-only build | Entrypoint steps 4 and 7; `verify-gpu.sh` against the running container | Exit 1 naming the shadowing cause; uninstall the overlay copy, or `make reset-venv` |
| Manager clone or requirements install interrupted | Entrypoint step 5 (`.manager-ready` absent) | Redone on the next start: scratch clone discarded, requirements reinstalled; a complete checkout is never deleted |
| Custom node breaks startup | Container restart loop | README documents `--disable-all-custom-nodes` via `COMFYUI_ARGS` |
| Web UI unreachable | Healthcheck | Container marked unhealthy after 3 × 30 s past a 120 s grace period |

## 10. Risks

- **Upstream pace.** ComfyUI ships roughly weekly. The pin will age;
  `make update-comfyui` re-resolves the newest tag and prints the SHA to paste
  into the Dockerfile. Deliberately manual, so a rebuild is never a surprise upgrade.
- **`sm_121` compatibility rests on upward cubin compatibility**, not on an
  explicit NVIDIA statement for this pairing. `verify-gpu.sh` converts this from
  an assumption into a per-deployment assertion.
- **Manager can install a node that breaks startup.** Mitigated by the documented
  `--disable-all-custom-nodes` escape hatch (§9).
- **Disk.** 437 GB free, and checkpoints run 7–25 GB each. Not enforced by the
  design; the README flags it.

## 11. Future work

Explicitly deferred, in rough priority order: a pip constraints/lock file for
the transitive dependencies of ComfyUI's and Manager's `requirements.txt`, so
that a rebuild is reproducible all the way down (goal 4); SageAttention or
Flash-Attention kernels built for sm_121 if SDPA proves slow; a
`fetch-model.sh` catalogue with checksums; optional Caddy reverse proxy with basic auth for LAN access; and
folding the service into `sparkyard` if the two stacks come to share models.
