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

With a checkpoint installed, `./scripts/verify-e2e.sh` stops skipping and
runs a real generation (sampler, VAE, and disk write) to prove the whole path
works under real GPU load:

```bash
make fetch-model URL=<sdxl-base-safetensors-url> DEST=checkpoints
./scripts/verify-e2e.sh
```

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
entrypoint refuses to start without a GPU, so a CPU fallback should be
impossible; if you hit one, that is a bug worth reporting.

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
