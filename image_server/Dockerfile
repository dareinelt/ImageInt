# The ImageInt image server: Qwen-Image-2.1 on CPU.
#
# This container owns the image model. It serves the OpenAI Images API over
# HTTP so the gateway can treat it like any other OpenAI-compatible endpoint,
# but the model itself runs through diffusers + transformers -- not vLLM.
# vLLM-Omni publishes GPU images only and therefore cannot satisfy the CPU-only
# requirement of this project.
#
# Two build targets, both valid:
#
#   cpu    (default) torch from the CPU wheel index, no CUDA libraries at all.
#          This is the reference deployment and what compose builds.
#   cuda   torch with CUDA wheels, for a host with an NVIDIA GPU. A preparation
#          only -- the CPU path stays the supported one.
#
# Build:  docker build -t imageint-image:local ./image_server
#         docker build -t imageint-image:cuda --target cuda ./image_server
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    IMAGEINT_IMAGE_HOST=0.0.0.0 \
    IMAGEINT_IMAGE_PORT=8000

WORKDIR /app

# libgomp is what torch's CPU kernels link against, libglib/libgl is the usual
# Pillow companion, curl is kept for interactive debugging only (the health
# check itself is a Python one-liner and needs no extra binary).
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      libgomp1 libglib2.0-0 libgl1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# The weights land in the HF cache on first start and must survive a container
# recreation, so the volume is declared and the directory pre-owned.
RUN useradd --uid 10002 --create-home --shell /usr/sbin/nologin imageint \
 && mkdir -p /models/hf \
 && chown -R imageint:imageint /app /models

# ── CUDA target (preparation only) ───────────────────────────────────────────
FROM base AS cuda

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY --chown=imageint:imageint . /app/image_server

USER imageint
EXPOSE 8000
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=5 \
    CMD python -c "import http.client,sys; c=http.client.HTTPConnection('127.0.0.1',8000,timeout=8); c.request('GET','/health'); sys.exit(0 if c.getresponse().status in (200,503) else 1)"

CMD ["python", "-m", "image_server"]

# ── CPU target (reference, default) ──────────────────────────────────────────
FROM base AS cpu

# torch first, from the CPU-only index, so pip never resolves a CUDA wheel that
# would add roughly 2 GB of unused libraries to the image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY --chown=imageint:imageint . /app/image_server

USER imageint
EXPOSE 8000
VOLUME ["/models"]

# Loading the weights takes minutes and /health answers 503 for exactly that
# window. Accepting 503 keeps Docker from restarting a container that is simply
# still loading; readiness is what /health reports to the gateway.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=5 \
    CMD python -c "import http.client,sys; c=http.client.HTTPConnection('127.0.0.1',8000,timeout=8); c.request('GET','/health'); sys.exit(0 if c.getresponse().status in (200,503) else 1)"

CMD ["python", "-m", "image_server"]
