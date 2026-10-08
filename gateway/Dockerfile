# The ImageInt gateway: the only container that is published.
#
# It is tiny on purpose -- the models run in their own images -- so this is a
# plain Python image with an unprivileged user and a health check. The image
# contains no build tools and no model weights; `docker compose up` pulls the
# weights into the two model containers instead.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    IMAGEINT_STORAGE_DIR=/data/images

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Generated images are written here; the volume is declared so the directory
# keeps the right owner even when a bind mount is added later.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin imageint \
 && mkdir -p /data/images \
 && chown -R imageint:imageint /app /data
VOLUME ["/data"]

USER imageint

EXPOSE 8080

# Liveness only: /health never contacts a model server, so a slow first model
# download does not make the gateway look unhealthy. Readiness lives in
# /v1/ready and is what a client should poll.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
