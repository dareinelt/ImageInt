"""Download progress of the model servers in the gateway log.

``docker compose logs -f gateway`` is the first place an administrator looks
during a cold start. The model servers already report how far their weight
download has come through ``/health``; this module turns that into one log line
per component every ``IMAGEINT_PROGRESS_LOG_SECONDS`` seconds (20 by default).

The lines are deliberately cheap and quiet:

* progress is logged only while a component is actually loading, so a ready
  server does not repeat its final ``100 %`` forever;
* a component without progress data produces a single state line, not one per
  tick;
* a probe that fails is swallowed -- logging must never disturb the service.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping, Optional

from .health import (
    COMPONENTS,
    ERROR,
    LOADING,
    READY,
    STARTING,
    TRANSIENT,
    UNREACHABLE,
    HealthMonitor,
)

log = logging.getLogger("imageint.progress")


def _number(value: float, digits: int) -> str:
    """Format a number the way the German log lines spell it."""

    return f"{value:.{digits}f}".replace(".", ",")


def format_bytes(value: int) -> str:
    """Human-readable byte count, in the unit that keeps it short."""

    value = max(0, int(value))
    if value >= 1024**3:
        return f"{_number(value / 1024**3, 1)} GB"
    if value >= 1024**2:
        return f"{_number(value / 1024**2, 1)} MB"
    if value >= 1024:
        return f"{_number(value / 1024, 0)} kB"
    return f"{value} B"


def format_duration(seconds: int) -> str:
    """Remaining time as ``2 min 5 s``, ``1 h 3 min`` or ``45 s``."""

    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {rest} s" if rest else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def describe_download(label: str, progress: Mapping[str, Any]) -> str:
    """One log line from a ``progress`` object of the health payload."""

    parts: list[str] = []

    percent = progress.get("percent")
    if isinstance(percent, (int, float)) and not isinstance(percent, bool):
        parts.append(f"{_number(float(percent), 1)} %")

    downloaded = progress.get("bytes_downloaded")
    total = progress.get("bytes_total")
    if isinstance(downloaded, (int, float)) and not isinstance(downloaded, bool):
        if isinstance(total, (int, float)) and not isinstance(total, bool) and total > 0:
            parts.append(f"{format_bytes(downloaded)} von {format_bytes(total)}")
        else:
            parts.append(f"{format_bytes(downloaded)} geladen")

    files_downloaded = progress.get("files_downloaded")
    files_total = progress.get("files_total")
    if (
        isinstance(files_downloaded, (int, float))
        and not isinstance(files_downloaded, bool)
        and isinstance(files_total, (int, float))
        and not isinstance(files_total, bool)
        and files_total > 0
    ):
        parts.append(f"Datei {int(files_downloaded)}/{int(files_total)}")

    current = progress.get("current_file")
    if isinstance(current, str) and current:
        parts.append(current)

    eta = progress.get("eta_seconds")
    if (
        isinstance(eta, (int, float))
        and not isinstance(eta, bool)
        and isinstance(percent, (int, float))
        and not isinstance(percent, bool)
        and percent < 100
    ):
        parts.append(f"noch {format_duration(eta)}")

    if not parts:  # pragma: no cover - vllm._clean_progress drops empty objects
        return f"{label}: Download läuft."
    return f"{label}: Download {', '.join(parts)}"


class ProgressReporter:
    """Turns health probes into log lines, remembering the last state."""

    def __init__(self, health: HealthMonitor) -> None:
        self.health = health
        #: Last reported state per component: one of the health states, or
        #: ``"progress"`` while a download line was written.
        self._states: dict[str, str] = {}

    async def tick(self) -> None:
        """Probe both components and log whatever changed or is downloading."""

        for component in COMPONENTS:
            item = await self.health.probe(component, force=True)
            self.report(component, item)

    def report(self, component: str, item: Mapping[str, Any]) -> None:
        label = str(item.get("label") or component)
        state = str(item.get("state") or "")
        previous = self._states.get(component)

        # A ready server keeps reporting the finished download forever. That is
        # useful for the admin UI and noise in a log, so the transition is the
        # only thing written here.
        if state == READY:
            self._states[component] = READY
            if previous in ("progress", LOADING, STARTING):
                log.info("%s: bereit, Download abgeschlossen.", label)
            return

        progress = item.get("progress") or {}
        if progress and state in TRANSIENT:
            percent = progress.get("percent")
            # ``percent == 100`` means the weights are complete and the server
            # is still building the model; repeating it every tick would drown
            # the interesting lines.
            if (
                isinstance(percent, (int, float))
                and not isinstance(percent, bool)
                and percent >= 100
            ):
                # Weights complete, the server is building the model now.
                self._states[component] = "progress"
                return
            log.info("%s", describe_download(label, progress))
            self._states[component] = "progress"
            return

        self._states[component] = state

        if state == LOADING:
            if previous != LOADING:
                log.info("%s: lädt, noch keine Download-Angaben.", label)
        elif state == STARTING:
            if previous != STARTING:
                log.info("%s: noch nicht erreichbar, gilt als startend.", label)
        elif state in (ERROR, UNREACHABLE):
            if previous != state:
                message = str(item.get("message") or "").strip()
                log.warning("%s: %s%s", label, state, f" ({message})" if message else "")
        # UNCONFIGURED stays silent: a disabled component is not a download.


async def run(health: HealthMonitor, interval: int) -> None:
    """Log download progress every ``interval`` seconds until cancelled."""

    reporter = ProgressReporter(health)
    while True:
        try:
            await reporter.tick()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - logging must never stop the service
            log.debug("Fortschrittsprotokoll fehlgeschlagen: %s", exc)
        await asyncio.sleep(interval)


def task(health: HealthMonitor, interval: int) -> Optional[asyncio.Task]:
    """Start the reporter as a background task, or ``None`` when disabled."""

    if interval <= 0:
        return None
    return asyncio.create_task(run(health, interval), name="imageint-progress-log")
