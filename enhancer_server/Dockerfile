# The ImageInt prompt-enhancer server: Qwen-Image-2.1-PE-T2I on CPU.
#
# This container owns the prompt enhancer. It serves the OpenAI
# chat-completions API over HTTP so the gateway can treat it like any other
# OpenAI-compatible endpoint, but the model itself runs through transformers --
# not vLLM. vLLM's CPU backend cannot initialise this checkpoint: Qwen3.5-VL
# ships a nested text config without an explicit ``architectures`` entry, which
# the CPU backend rejects during engine startup. ``trust_remote_code`` on
# transformers loads the checkpoint's own modelling code instead.
#
# Two build targets, both valid:
#
#   cpu    (default) torch from the CPU wheel index, no CUDA libraries at all.
#          This is the reference deployment and what compose builds.
#   cuda   torch with CUDA wheels, for a host with an NVIDIA GPU. A preparation
#          only -- the CPU path stays the supported one.
#
# Build:  docker build -t imageint-enhancer:local ./enhancer_server
#         docker build -t imageint-enhancer:cuda --target cuda ./enhancer_server
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    IMAGEINT_PE_HOST=0.0.0.0 \
    IMAGEINT_PE_PORT=8000

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

COPY --chown=imageint:imageint . /app/enhancer_server

USER imageint
EXPOSE 8000
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=5 \
    CMD python -c "import http.client,sys; c=http.client.HTTPConnection('127.0.0.1',8000,timeout=8); c.request('GET','/health'); sys.exit(0 if c.getresponse().status in (200,503) else 1)"

CMD ["python", "-m", "enhancer_server"]

# ── CPU target (reference, default) ──────────────────────────────────────────
FROM base AS cpu

# torch first, from the CPU-only index, so pip never resolves a CUDA wheel that
# would add roughly 2 GB of unused libraries to the image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY --chown=imageint:imageint . /app/enhancer_server

USER imageint
EXPOSE 8000
VOLUME ["/models"]

# Loading the weights takes minutes and /health answers 503 for exactly that
# window. Accepting 503 keeps Docker from restarting a container that is simply
# still loading; readiness is what /health reports to the gateway.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=5 \
    CMD python -c "import http.client,sys; c=http.HTTPConnection('127.0.0.1',8000,timeout=8); c.request('GET','/health'); sys.exit(0 if c.getresponse().status in (200,503) else 1)"

CMD ["python", "-m", "enhancer_server"]
