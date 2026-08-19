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

# The baked venv. The overlay venv on /data inherits from it (spec §7.2).
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# torch is fixed by the spec §2.1 probe. Companions pinned 2026-08-18 to the
# versions the resolver actually selected for torch==2.13.0+cu130.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0+cu130 \
        torchvision==0.28.0+cu130 \
        torchaudio==2.11.0+cu130

# Runtime constraint file: any pip (or uv) invocation inside the container
# that names the torch trio — a custom node's requirements.txt, a Manager
# "fix" — resolves to exactly these GPU-correct builds or fails loudly,
# instead of silently shadowing the baked torch with a CPU wheel from PyPI.
# Keep in lockstep with the pip install above.
RUN printf 'torch==2.13.0+cu130\ntorchvision==0.28.0+cu130\ntorchaudio==2.11.0+cu130\n' \
        > /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt \
    UV_CONSTRAINT=/opt/constraints.txt

# Ubuntu 24.04 already owns uid/gid 1000 as "ubuntu". Rename it rather than
# collide, so files written to the bind mount are owned by the host user.
# This layer sits BELOW the ~3GB torch layer deliberately: changing PUID/PGID
# (a build arg — see .env.example) then only rebuilds from here down.
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

# ComfyUI lives in the image, never on the bind mount, so an upgrade is a
# rebuild and can never leave a half-updated working tree on disk.
RUN git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git /opt/comfyui \
 && git -C /opt/comfyui checkout "${COMFYUI_REF}" \
 && pip install --no-cache-dir -r /opt/comfyui/requirements.txt \
 && chown -R "${PUID}:${PGID}" /opt/comfyui

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Runtime resolution order: the overlay first. This covers `docker compose
# exec` / `make shell` sessions, which take the image ENV rather than the
# entrypoint's exports — without it, bare `pip` in a shell resolves to the
# root-owned /opt/venv and fails with PermissionError. /data/venv does not
# exist at build time; PATH lookup skips missing entries, so build-stage
# `pip` still resolves to /opt/venv. Placed after the last build-stage pip
# use so VIRTUAL_ENV cannot confuse any build tooling.
ENV VIRTUAL_ENV=/data/venv \
    PATH="/data/venv/bin:${PATH}"

USER comfy
WORKDIR /opt/comfyui
EXPOSE 8188
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
