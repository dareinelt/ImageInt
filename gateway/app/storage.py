"""Where generated pictures are kept between the render and the download.

A finished image is written to disk once, so a chat client can fetch it more
than once (a preview, then the full view) without paying for another CPU
generation. Files are named after the job id, which is a hex UUID, and the name
is re-validated on every access: nothing that arrives over HTTP may become part
of a path.

Writes are atomic -- temporary file plus ``os.replace`` -- so a reader never
observes a half-written PNG, and a crashed container leaves no truncated image
that would later be served as if it were complete.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("imageint.storage")

#: Job ids are hex UUIDs. Anything else never reaches the filesystem.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

#: Content type to file suffix. The diffusion model emits PNG; the others are
#: accepted so a differently configured server does not produce mislabelled
#: files.
_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
_SUFFIX_TO_TYPE = {suffix: kind for kind, suffix in _SUFFIX.items()}


class ImageStore:
    """The image directory, with job-id addressed reads and writes."""

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)

    def ensure(self) -> None:
        """Create the directory if it does not exist yet."""

        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str) -> Optional[Path]:
        """Existing file of one job, or ``None``."""

        if not _ID_RE.match(job_id or ""):
            return None
        for suffix in _SUFFIX.values():
            candidate = self.directory / f"{job_id}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def write(self, job_id: str, data: bytes, content_type: str = "image/png") -> Optional[Path]:
        """Store one image atomically and return its path."""

        if not _ID_RE.match(job_id or ""):
            return None
        if not data:
            return None

        suffix = _SUFFIX.get((content_type or "").split(";")[0].strip(), ".png")
        self.ensure()
        target = self.directory / f"{job_id}{suffix}"
        # A unique temporary name: two jobs writing at once must not share one.
        temporary = self.directory / f".{job_id}.{os.getpid()}.part"
        temporary.write_bytes(data)
        os.replace(temporary, target)
        return target

    def read(self, job_id: str) -> Optional[bytes]:
        """The stored image bytes, or ``None`` when there is no file."""

        path = self.path(job_id)
        if path is None:
            return None
        try:
            return path.read_bytes()
        except OSError as exc:
            log.warning("Gespeichertes Bild %s ist nicht lesbar: %s", job_id, exc)
            return None

    def content_type(self, job_id: str) -> str:
        path = self.path(job_id)
        if path is None:
            return "application/octet-stream"
        return _SUFFIX_TO_TYPE.get(path.suffix, "image/png")

    def size(self, job_id: str) -> int:
        path = self.path(job_id)
        if path is None:
            return 0
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def delete(self, job_id: str) -> bool:
        path = self.path(job_id)
        if path is None:
            return False
        try:
            path.unlink()
        except OSError as exc:
            log.warning("Gespeichertes Bild %s ist nicht löschbar: %s", job_id, exc)
            return False
        return True

    def cleanup(self, keep: set) -> int:
        """Delete every stored image whose job id is not in ``keep``.

        Called after the job registry pruned itself, so a restart that lost the
        registry does not leave the volume to grow forever. Leftovers of an
        interrupted write are removed as well.
        """

        if not self.directory.is_dir():
            return 0

        removed = 0
        for entry in self.directory.iterdir():
            if not entry.is_file():
                continue
            if entry.name.endswith(".part") or entry.name.startswith("."):
                try:
                    entry.unlink()
                    removed += 1
                except OSError:
                    pass
                continue
            if entry.stem in keep:
                continue
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
        if removed:
            log.info("Aufgeräumt: %d nicht mehr benötigte Bilddatei(en) gelöscht.", removed)
        return removed

    def total_bytes(self) -> int:
        if not self.directory.is_dir():
            return 0
        total = 0
        for entry in self.directory.iterdir():
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
        return total
