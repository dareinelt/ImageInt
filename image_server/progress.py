"""Byte-level download progress for the HuggingFace cache.

``from_pretrained`` pulls tens of gigabytes and reports nothing until it is
done, so the health endpoint has nothing to show while a model loads. This
module watches the cache from a daemon thread and turns it into the ``progress``
object the gateway forwards to ``/v1/health``.

Two sources are combined:

``bytes_total``
    Asked from the Hub once, through ``HfApi().model_info(..., files_metadata=True)``.
    When that call fails -- offline host, private repo, old ``huggingface_hub`` --
    the tracker keeps working and reports only ``bytes_downloaded``.

``bytes_downloaded``
    Polled from ``<cache>/models--<org>--<name>/blobs/``. Only ``blobs`` is
    scanned: ``snapshots`` holds symlinks to the very same files, so counting
    both would double the sum. ``.incomplete`` files are counted, because their
    bytes are already on disk.

Every method swallows its own errors: a broken cache directory must never turn a
working model load into a failure, and the polling thread never blocks the
loading thread.

This module exists twice, in ``image_server/`` and ``enhancer_server/``: the two
services are built from separate Docker build contexts and cannot share code.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# ``huggingface_hub`` arrives with diffusers/transformers but is not a declared
# dependency of this service. A stripped-down environment must still be able to
# load a model; it just cannot know the total download size.
try:  # pragma: no cover - exercised by the real containers
    from huggingface_hub import HfApi as _HfApi
except Exception:  # noqa: BLE001 - any import problem is tolerated
    _HfApi = None

try:  # pragma: no cover - exercised by the real containers
    from huggingface_hub.constants import HF_HUB_CACHE as _HF_HUB_CACHE
except Exception:  # noqa: BLE001 - fall back to the documented location
    _HF_HUB_CACHE = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )


def default_cache_dir() -> str:
    """The cache root ``huggingface_hub`` uses when none is configured."""

    return _HF_HUB_CACHE


def _field(value: Any, name: str) -> Any:
    """Read a field from an object or a dict (the tests use both)."""

    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


class DownloadProgress:
    """Tracks the bytes of one repository as they land in the HF cache."""

    #: Number of polls the transfer rate (and therefore the ETA) averages over.
    RATE_WINDOW = 8

    def __init__(
        self,
        repo_id: str,
        *,
        cache_dir: str = "",
        revision: str = "",
        token: Optional[str] = None,
        interval: float = 1.0,
        api: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.token = (
            token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or None
        )
        self.interval = max(0.1, float(interval))
        #: Injected in tests; ``None`` means "build an ``HfApi`` lazily".
        self._api = api
        self._cache_dir = cache_dir
        self._clock = clock

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False
        self._finished = False

        self._bytes_downloaded = 0
        self._bytes_total: Optional[int] = None
        self._files_downloaded: Optional[int] = None
        self._files_total: Optional[int] = None
        self._current_file: Optional[str] = None
        self._eta_seconds: Optional[int] = None
        self._rate_bytes_per_second: Optional[float] = None
        self._percent: Optional[float] = None
        self._samples: deque = deque(maxlen=self.RATE_WINDOW)
        self._blob_names: dict[str, str] = {}

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        """Start polling. Idempotent, and a no-op once the load has finished."""

        with self._lock:
            if self._started or self._finished:
                return
            self._started = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="imageint-hf-progress", daemon=True
            )
            thread = self._thread
        thread.start()

    def stop(self) -> None:
        """Stop polling. Never joins, so a failing load is not held up."""

        with self._lock:
            self._stopped = True
        self._stop_event.set()

    def finish(self) -> None:
        """Stop polling and report the download as complete (100 %)."""

        with self._lock:
            self._finished = True
            self._stopped = True
        self._stop_event.set()

    # -- the polling thread ------------------------------------------------ #

    def _run(self) -> None:
        # Sample before the metadata call, so the byte count is already there
        # when the Hub cannot be reached at all.
        self._guard(self._sample)
        self._guard(self._read_model_info)
        self._guard(self._sample)
        while not self._stop_event.is_set():
            self._guard(self._sample)
            self._stop_event.wait(self.interval)

    @staticmethod
    def _guard(action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - progress is best effort
            log.debug("download progress update failed: %s", exc)

    # -- total size -------------------------------------------------------- #

    def _read_model_info(self) -> None:
        """Ask the Hub how large the repository is. Runs once per load."""

        api = self._api
        if api is None:
            if _HfApi is None:
                return
            api = _HfApi(token=self.token) if self.token else _HfApi()

        kwargs: dict[str, Any] = {"files_metadata": True}
        if self.revision:
            kwargs["revision"] = self.revision
        info = api.model_info(self.repo_id, **kwargs)

        total = 0
        files = 0
        names: dict[str, str] = {}
        for sibling in getattr(info, "siblings", None) or []:
            filename = _field(sibling, "rfilename")
            blob_id = _field(sibling, "blob_id")
            if filename and blob_id:
                # LFS blobs are stored under their sha256, plain files under
                # their git blob id -- both are what ``blob_id`` carries, so the
                # cache file name can be mapped back to a readable file name.
                names[str(blob_id)] = str(filename)

            size = _field(sibling, "size")
            if size is None:
                continue
            try:
                size = int(size)
            except (TypeError, ValueError):
                continue
            if size < 0:
                continue
            total += size
            files += 1

        with self._lock:
            self._blob_names = names
            self._bytes_total = total if total > 0 else None
            self._files_total = files or None
            self._recompute_locked()

    # -- cache scanning ---------------------------------------------------- #

    def _blobs_dir(self) -> Path:
        root = self._cache_dir or _HF_HUB_CACHE
        return Path(root) / ("models--" + self.repo_id.replace("/", "--")) / "blobs"

    def _scan(self) -> tuple[int, int, Optional[str]]:
        """Return ``(bytes_on_disk, complete_files, most_recent_blob)``."""

        total = 0
        complete = 0
        newest_name: Optional[str] = None
        newest_mtime = -1.0
        try:
            entries = list(os.scandir(self._blobs_dir()))
        except OSError:
            return 0, 0, None

        for entry in entries:
            try:
                if entry.is_symlink() or not entry.is_file():
                    continue
                stat = entry.stat()
            except OSError:
                continue

            name = entry.name
            if name.endswith(".lock"):
                continue

            total += stat.st_size
            if not name.endswith(".incomplete"):
                complete += 1
            if stat.st_mtime >= newest_mtime:
                newest_mtime = stat.st_mtime
                newest_name = name

        return total, complete, newest_name

    def _sample(self) -> None:
        """Read the cache once and refresh every derived value."""

        downloaded, files_downloaded, newest = self._scan()
        now = self._clock()

        with self._lock:
            self._bytes_downloaded = downloaded
            self._files_downloaded = files_downloaded
            if self._files_total is not None:
                self._files_downloaded = min(files_downloaded, self._files_total)

            if newest:
                stem = (
                    newest[: -len(".incomplete")]
                    if newest.endswith(".incomplete")
                    else newest
                )
                self._current_file = self._blob_names.get(stem, stem)

            self._update_rate_locked(now, downloaded)
            self._recompute_locked()

    # -- derived values ---------------------------------------------------- #

    def _update_rate_locked(self, now: float, downloaded: int) -> None:
        if self._samples and downloaded < self._samples[-1][1]:
            # A blob vanished (cache cleanup); start the window over instead of
            # reporting a negative rate.
            self._samples.clear()
        self._samples.append((now, downloaded))

        self._rate_bytes_per_second = None
        if len(self._samples) < 2:
            return
        first_at, first_bytes = self._samples[0]
        span = now - first_at
        if span <= 0:
            return

        # The rate is reported on its own, so it stays available when the total
        # size is unknown (offline host, private repo) and the ETA is not.
        rate = max(0.0, (downloaded - first_bytes) / span)
        self._rate_bytes_per_second = round(rate, 1)

        total = self._bytes_total
        if total is None:
            self._eta_seconds = None
            return
        remaining = max(0, total - downloaded)
        if remaining == 0:
            self._eta_seconds = 0
            return
        self._eta_seconds = int(round(remaining / rate)) if rate > 0 else None

    def _recompute_locked(self) -> None:
        total = self._bytes_total
        if total and total > 0:
            self._percent = round(
                min(100.0, self._bytes_downloaded * 100.0 / total), 1
            )
        else:
            self._percent = None

    # -- reporting --------------------------------------------------------- #

    def snapshot(self) -> dict:
        """The ``progress`` object of the health payload.

        Empty while nothing is downloading -- idle, stopped after a failure, or
        a cache that already holds everything -- so the gateway keeps showing an
        indeterminate bar in those cases.
        """

        with self._lock:
            if self._finished:
                data: dict[str, Any] = {"known": True, "percent": 100.0}
                if self._bytes_total is not None:
                    data["bytes_downloaded"] = self._bytes_total
                    data["bytes_total"] = self._bytes_total
                if self._files_total is not None:
                    data["files_downloaded"] = self._files_total
                    data["files_total"] = self._files_total
                data["eta_seconds"] = 0
                data["bytes_per_second"] = 0.0
                return data

            if not self._started or self._stopped:
                return {}

            known = self._bytes_total is not None
            if not known and self._bytes_downloaded <= 0:
                return {}

            data = {"known": known}
            if known and self._percent is not None:
                data["percent"] = self._percent
            data["bytes_downloaded"] = self._bytes_downloaded
            if self._bytes_total is not None:
                data["bytes_total"] = self._bytes_total
            if self._files_downloaded is not None:
                data["files_downloaded"] = self._files_downloaded
            if self._files_total is not None:
                data["files_total"] = self._files_total
            if self._current_file:
                data["current_file"] = self._current_file
            if self._eta_seconds is not None:
                data["eta_seconds"] = self._eta_seconds
            if self._rate_bytes_per_second is not None:
                data["bytes_per_second"] = self._rate_bytes_per_second
            return data
