# ComfyUI on DGX Spark

GPU-accelerated ComfyUI for this machine's GB10 (Grace-Blackwell, `aarch64`,
compute capability 12.1). Off-the-shelf ComfyUI images are `x86_64`-only, so
this builds its own on `nvidia/cuda:13.1.0-runtime` with PyTorch `cu130`
`aarch64` wheels.

Design: `docs/superpowers/specs/2026-08-18-comfyui-docker-design.md`

## Quickstart

```bash
cp .env.example .env                 # adjust COMFYUI_DATA_PATH if needed
mkdir -p /home/scott/LLMs/comfyui    # ...or whatever you set it to
make build
make up
make verify
```

Then open <http://127.0.0.1:8188>.

What to expect the first time:

- `make build` downloads roughly 5.5 GB (the CUDA base image plus ~3 GB of
  torch wheels) and takes on the order of ten minutes. The finished image
  occupies about 15 GB unpacked — `docker images` reports both numbers.
- The first `make up` takes a few minutes before the UI answers: it creates
  the overlay venv and clones ComfyUI-Manager. Later starts are quick. Let it
  finish — interrupting it is handled (the entrypoint redoes anything it
  finds half-done), but you will just wait again next time.
- `make verify` **force-recreates the running container** as part of the
  persistence test, which interrupts any generation in flight. Run it when
  the machine is idle. It also reaches PyPI twice (a test install, and the
  entrypoint cold-start cases), so it can fail on a transient network error
  that has nothing to do with this code — re-run it before suspecting the
  build.
- Changing `PUID`/`PGID` in `.env` requires `make build` — they are build
  args, and `make up` never rebuilds.

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

With a checkpoint installed, `./scripts/verify-e2e.sh` stops skipping and
runs a real generation (sampler, VAE, and disk write) to prove the whole path
works under real GPU load:

```bash
make fetch-model URL=<sdxl-base-safetensors-url> DEST=checkpoints
./scripts/verify-e2e.sh
```

## Workflow parameter manifests

A workflow may ship a sibling `<name>.params.json` naming the inputs that are
meant to be varied:

```json
{
  "description": "Minimal SD1.5 text-to-image",
  "tags": ["txt2img", "default"],
  "params": {
    "prompt": { "path": "6.inputs.text", "type": "string", "required": true },
    "seed":   { "path": "3.inputs.seed", "type": "int", "default": "random" }
  }
}
```

Nothing in this repo reads these files — they are for callers driving the API.
sandbox-env mounts this directory into agent sandboxes read-only, and its
`comfy` helper uses the manifest to turn `--prompt "..."` into the right node
input, with `tags` selecting which workflow `comfy txt2img` and `comfy video`
reach for. Types are injected as real JSON types: `int` matters, because
ComfyUI rejects `"42"` where it wants `42`.

## Workflow library

`workflows/` holds API-format graphs plus their manifests. `sandbox-env` mounts
this directory into agent sandboxes read-only; each `<name>.json` is runnable
with `comfy run <name>`, and `tags` decide which one `comfy txt2img`,
`comfy edit` (via `run`) and `comfy video` reach for. Exactly one manifest per
tag may carry `default` — `make verify` enforces that, because two defaults
make the helper silently pick whichever sorts first.

| Workflow | Tags | Model | Warm | Reload | Notes |
|---|---|---|---|---|---|
| `z-image-turbo` | `txt2img`, **default** | Z-Image Turbo 6B — NVFP4 transformer + FP8-mixed encoder | ~6 s | ~60 s | 8 steps, cfg 1. `--set model=z_image_turbo_bf16.safetensors --set encoder=qwen_3_4b.safetensors` swaps to bf16 (~9 s warm, ~2x the reload) |
| `qwen-image-2.1-t2i` | `txt2img` | Qwen-Image 2.1 7B, int8 | ~23 s | ~120 s | Best text rendering; size from `aspect` + `megapixels`, or literal `width`/`height` |
| `qwen-image-2.1-edit` | `edit`, **default** | Qwen-Image 2.1 7B, int8 | ~30 s | ~120 s | Single image: `comfy upload FILE`, then `--set image=FILE --set prompt="... <image1> ..."` |
| `ltx-2.5-t2v` | `video`, **default** | LTX-2.5 22B distilled, int8 + Gemma-4 12B | ~78 s | ~220 s | 5 s @ 24 fps, 1280x704, **with audio**. `duration`, `fps`, `aspect`, `megapixels`, `enhance` |
| `minimal-txt2img` | `txt2img` | SD 1.5 | ~2 s | ~15 s | The original smoke-test workflow |

Timings are from a GB10 at the manifests' defaults. **Warm** is a second
run of the same workflow with a fresh seed; **reload** is the first run after
a *different* model family has executed — ComfyUI reloads each family from
disk on every switch, even with `--highvram`, so agents should batch
generations by model. A re-run with an *identical* seed returns in ~2 s
because ComfyUI's node cache short-circuits it; that is not a generation.

Fetch everything the library needs (idempotent, ~90 GB on disk):

```bash
HF_TOKEN=$(cat ~/.config/hf/token) ./scripts/fetch-library.sh   # LTX-2.5 is gated
```

Then `make verify`. Its first stage, `scripts/verify-workflows.sh`, checks
every manifest path against its graph, that each referenced model file is
present, and that no tag has two defaults — with no container running.

### Precision on the GB10

Image DiTs are memory-bandwidth-bound on this box, so smaller weights are
faster as well as smaller: measured on Z-Image Turbo, the NVFP4 transformer
is ~1.4-1.5x faster warm than bf16 and halves the reload time, with no
visible quality difference at the same seed — while every FP8 variant is
*slower* than bf16 (comfy-kitchen's CUDA backend here has a native NVFP4
matmul but no native FP8 one). Video is compute-bound, so precision only buys
download size; the 8-step distill is what buys speed. Hence NVFP4/int8 files
throughout, with Z-Image's bf16 pair kept for comparisons.

### Why the image needs a C compiler

torch 2.13 implements some ops (`torch._native`, e.g. `bmm_outer_product`)
as Triton kernels, and Triton compiles a small C launcher against the Python
headers on first use. Without `gcc` + `python3-dev` every modern text encoder
(Qwen3, Qwen3-VL, Gemma) fails at `CLIPTextEncode` with "Failed to find C
compiler". SD 1.5's CLIP never takes that path, which is why `verify-e2e.sh`
alone would not have caught it.

## Commands

Run `make` on its own for the full list. The common ones: `build`, `up`,
`down`, `logs`, `shell`, `verify`.

## How this is put together

- **ComfyUI itself lives in the image**, not on disk. Upgrading is a rebuild,
  so there is never a half-updated working tree to debug.
- **Your data lives on the bind mount** at `$COMFYUI_DATA_PATH` → `/data`:
  models, outputs, inputs, custom nodes, settings, and the overlay venv below.
- **Two Python environments**, which is the one genuinely surprising part:

  | Path | Contents | Survives a restart? |
  |---|---|---|
  | `/opt/venv` | torch + ComfyUI's own requirements, baked into the image | No — rebuilt with the image |
  | `/data/venv` | anything ComfyUI-Manager installs | Yes — it is on the bind mount |

  `/data/venv` is a plain venv, **not** one created with
  `--system-site-packages`. That flag was tried first and doesn't work here:
  `/opt/venv` is itself a venv, and since Python 3.11 a nested venv's
  `--system-site-packages` resolves against the real base OS interpreter, not
  its immediate parent venv — so it would see the system's `dist-packages`
  and never `/opt/venv`'s torch. Instead, the entrypoint drops a `.pth` file
  named `_baked_venv.pth` into `/data/venv`'s site-packages, containing the
  path to `/opt/venv`'s site-packages. Python's `site` module appends that
  path to `sys.path` after the overlay's own site-packages, so:

  - the overlay *inherits* baked torch without copying it — no multi-gigabyte
    re-download at first boot;
  - a node that installs a newer version of a baked package still shadows it,
    because the overlay's own site-packages comes first on `sys.path`.

  ComfyUI runs from `/data/venv/bin/python`, so nodes you install from the
  UI — and their pip dependencies — are still there tomorrow, while a fresh
  `docker compose build` only ever touches `/opt/venv`.

  One consequence is worth knowing about: **ComfyUI-Manager is pinned to pip,
  not `uv`**. Manager prefers `uv` on Linux whenever it can import it, and it
  can — `uv` is in Manager's own `requirements.txt`. But `uv` does not read
  `.pth` files, so inside `/data/venv` it sees only the overlay's own 27
  packages, not the ~100 baked ones. Manager decides what a node still
  needs from that list, so under `uv` it would reinstall torch, numpy,
  transformers and friends from PyPI into the overlay, where they shadow the
  GPU-correct baked copies. The entrypoint therefore writes
  `use_uv = False` into `user/__manager/config.ini` on every start (leaving
  every other setting in that file alone). Installs are slower and correct
  rather than fast and wrong.

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

**Upgrading an old deployment onto a broken venv.** The entrypoint tells a
fresh, never-finished venv apart from a working one that later broke by
writing a completion sentinel the first time a build finishes, and only
auto-rebuilds the former. A venv built by an *older* entrypoint, from before
that sentinel existed, has no such file — so if it is *already* broken the
first time a sentinel-aware entrypoint runs against it, it reads as "never
finished" and gets rebuilt, silently discarding whatever was on it. This
can't happen on a fresh install like this one; it only affects a long-lived
deployment upgraded across entrypoint versions while its overlay venv was
already unhealthy. Back up `$COMFYUI_DATA_PATH/venv` first if that's you.

**It seems slow — is it actually on the GPU?** `./scripts/verify-gpu.sh`, or
check `curl -s localhost:8188/system_stats | python3 -m json.tool`. The
entrypoint checks both Python layers before it launches — the baked venv and
the overlay venv that actually renders — and refuses to start if either
cannot reach the GPU, so a silent CPU fallback at startup should be
impossible.

The remaining exposure is a change made *while the service is up*: a custom
node whose `requirements.txt` names `torch` could ask for a reinstall into
`/data/venv`, where it would shadow the baked build. Three layers stand in
the way. First, prevention: a constraints file baked into the image
(`PIP_CONSTRAINT`/`UV_CONSTRAINT`) forces any in-container install of the
torch trio to the exact `cu130` builds — a wrong-build shadow now fails the
install loudly instead of succeeding quietly. Second, the entrypoint's
guards re-check on every **container start** — `make down && make up`, a
host reboot, a force-recreate (a bare `docker compose up -d` on a running
container is a no-op and re-runs nothing). Manager's own in-UI "Restart"
button and its deferred-install restart bypass those guards (`os.execv`
replaces `main.py` without re-running the entrypoint) — which is where the
third layer comes in: the healthcheck parses `/system_stats` and requires
the primary device to be CUDA, so a container that somehow ends up rendering
on CPU flips to `unhealthy` in `docker ps` within about 90 seconds, however
it got there. After installing a node, `./scripts/verify-gpu.sh` remains the
definitive check (it interrogates the live container's overlay interpreter).
To fix a shadowed overlay:

```bash
docker compose exec comfyui /data/venv/bin/pip list --local | grep -i torch
docker compose exec comfyui /data/venv/bin/pip uninstall torch torchvision torchaudio
make down && make up
```

Uninstalling the overlay copy lets the baked `cu130` build show through
again. If that is not enough, `make down && make reset-venv && make up`
rebuilds the overlay from scratch (and discards every Manager-installed
package). `make verify` runs `verify-gpu.sh` against the live container, so a
routine verification catches this too.

**The overlay venv refuses to start after an image rebuild.** If the base
image's Python minor version changed (3.12 → 3.13), `/data/venv`'s paths no
longer match the interpreter that created it and `import torch` breaks. The
entrypoint refuses to start rather than delete it. The recovery is
`make down && make reset-venv && make up`; the custom nodes themselves are
untouched, so reinstall their requirements from the Manager UI afterwards.

**`sm_121` is missing from `torch.cuda.get_arch_list()`.** Expected, not a
fault. PyTorch's cu130 build ships cubins up to `sm_120`; the GB10's compute
capability 12.1 (`sm_121`) runs `sm_120` cubins directly — newer GPU
architectures in the same major family are binary-compatible with cubins
built for earlier minor versions in that family. `verify-gpu.sh` doesn't just
check the arch list, it launches a real bf16 matmul on the device and checks
the result is finite, so it confirms actual kernels execute rather than
trusting the (absent) `sm_121` entry.

## Security

Published on `127.0.0.1:8188` only. ComfyUI has no authentication and custom
nodes execute arbitrary Python, so anything that can reach the port has
effective code execution. Reach it from another machine with an SSH tunnel:

```bash
ssh -L 8188:127.0.0.1:8188 scott@<this-host>
```

Three consequences of that "custom nodes execute arbitrary Python" line are
worth stating plainly, because they are all deliberate trade-offs:

- **`HF_TOKEN` is injected into the container** (`docker-compose.yml`), not
  only used by `scripts/fetch-model.sh` on the host — Manager's model
  downloader needs it for gated repos, and `huggingface-hub` picks it up from
  the environment automatically. That means **every custom node you install
  can read it and send it anywhere**. Use a read-only token, scoped to the
  fewest repositories that work, and rotate it if you install nodes you have
  not read.
- **`seccomp:unconfined`** (`docker-compose.yml`) is carried over from the
  working GB10 configuration in `sparkyard`: the CUDA userspace driver makes
  ioctl and memory calls that Docker's default seccomp profile blocks on this
  platform. It is kept because the GPU does not work reliably without it, but
  it materially weakens the kernel-attack-surface reduction that normally
  stands between a hostile custom node and a container escape.
- **`ipc: host`** (`docker-compose.yml`, same origin) puts the container in
  the host IPC namespace so CUDA can share memory with the driver. That
  namespace is shared with the other GPU containers on this machine, so a
  hostile node is not isolated from their shared-memory segments either.

Net: treat installing a custom node as running unreviewed code as your user,
not as running it in a sandbox. `--disable-all-custom-nodes` in
`COMFYUI_ARGS` is the off switch.
