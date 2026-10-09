"""Entry point: ``python -m image_server``."""

from __future__ import annotations

import logging
import os

from .app import create_app
from .config import load_settings


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=os.environ.get("IMAGEINT_IMAGE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # uvicorn is imported lazily so importing the package for tests stays cheap.
    import uvicorn

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=os.environ.get("IMAGEINT_IMAGE_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
