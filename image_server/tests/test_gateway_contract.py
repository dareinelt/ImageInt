"""Wire compatibility with the gateway.

The image server replaced the vLLM-Omni container, and the whole point of the
replacement is that the gateway did *not* have to change. These tests take the
gateway's own request builder and response parsers and run them against a live
image-server response, so a drift in either direction fails here rather than in
production.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from image_server.app import create_app
from image_server.config import Settings

ROOT = Path(__file__).resolve().parents[2]
GATEWAY = ROOT / "gateway"

# The gateway package is called `app`, the image server's is `image_server`, so
# both can be imported side by side.
if str(GATEWAY) not in sys.path:
    sys.path.insert(0, str(GATEWAY))

vllm = pytest.importorskip("app.vllm")
config = pytest.importorskip("app.config")


def test_the_gateway_payload_is_answered_in_the_shape_the_gateway_reads():
    settings = Settings(engine="stub", preload=True)
    image_settings = config.ImageSettings()

    app = create_app(settings)
    with TestClient(app) as client:
        app.state.engine.ensure_loaded()
        payload = vllm.build_image_payload(
            image_settings, "Ein Leuchtturm im Sturm", 2400, 1792
        )
        response = client.post("/v1/images/generations", json=payload)
        assert response.status_code == 200
        decoded = response.json()

    # The gateway's own parser has to find the picture and its metadata.
    image, content_type = vllm._extract_image(decoded)
    assert image is not None and image.startswith(b"\x89PNG")
    assert content_type == "image/png"

    meta = vllm._extract_meta(decoded)
    assert (meta["width"], meta["height"]) == (2400, 1792)
    assert meta["model"] == payload["model"]
    assert meta["seed"] is not None


def test_the_documented_canvas_survives_the_round_trip_unchanged():
    """2400x1792 is not a multiple of 64, and the gateway sends it verbatim."""

    from PIL import Image
    import io

    app = create_app(Settings(engine="stub", preload=True))
    with TestClient(app) as client:
        app.state.engine.ensure_loaded()
        body = client.post(
            "/v1/images/generations",
            json={"prompt": "Ein Bergsee", "size": "2400x1792"},
        ).json()

    assert body["size"] == "2400x1792"
    with Image.open(io.BytesIO(vllm._extract_image(body)[0])) as image:
        assert image.size == (2400, 1792)
