"""Tests for the periodic download-progress lines in the gateway log."""

from __future__ import annotations

import asyncio
import logging

import pytest

from app import config, progresslog

PROGRESS = {
    "known": True,
    "percent": 42.5,
    "bytes_downloaded": 123456789,
    "bytes_total": 296352743424,
    "files_downloaded": 3,
    "files_total": 12,
    "current_file": "diffusion_pytorch_model-00002-of-00004.safetensors",
    "eta_seconds": 125,
}

LABELS = {"enhancer": "Prompt-Enhancer", "image": "Bildmodell"}


def component(name: str, state: str, progress: dict | None = None, message: str = "") -> dict:
    item = {
        "component": name,
        "label": LABELS[name],
        "state": state,
        "ready": state == "ready",
        "transient": state in ("loading", "starting"),
        "message": message,
    }
    if progress is not None:
        item["progress"] = progress
    return item


class StubHealth:
    """Stands in for ``HealthMonitor``; returns the payloads a test set up."""

    def __init__(self, payloads: dict) -> None:
        self.payloads = payloads
        self.probes: list = []

    async def probe(self, component: str, *, force: bool = False) -> dict:
        self.probes.append((component, force))
        return self.payloads[component]


def lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "imageint.progress"]


def ready_both() -> dict:
    return {
        "enhancer": component("enhancer", "ready"),
        "image": component("image", "ready"),
    }


# -- formatting ------------------------------------------------------------- #


def test_formats_bytes_in_a_readable_unit():
    assert progresslog.format_bytes(512) == "512 B"
    assert progresslog.format_bytes(2048) == "2 kB"
    assert progresslog.format_bytes(5 * 1024**2) == "5,0 MB"
    assert progresslog.format_bytes(296352743424) == "276,0 GB"


def test_formats_durations():
    assert progresslog.format_duration(45) == "45 s"
    assert progresslog.format_duration(125) == "2 min 5 s"
    assert progresslog.format_duration(120) == "2 min"
    assert progresslog.format_duration(3780) == "1 h 3 min"


def test_describes_a_download_with_all_fields():
    line = progresslog.describe_download("Bildmodell", PROGRESS)
    assert line == (
        "Bildmodell: Download 42,5 %, 117,7 MB von 276,0 GB, Datei 3/12, "
        "diffusion_pytorch_model-00002-of-00004.safetensors, noch 2 min 5 s"
    )


def test_describes_an_unknown_total_with_bytes_only():
    line = progresslog.describe_download("Bildmodell", {"known": False, "bytes_downloaded": 5242880})
    assert line == "Bildmodell: Download 5,0 MB geladen"


# -- reporting -------------------------------------------------------------- #


async def test_reports_progress_on_every_tick(caplog):
    payloads = ready_both()
    payloads["image"] = component("image", "loading", PROGRESS)
    health = StubHealth(payloads)
    reporter = progresslog.ProgressReporter(health)
    caplog.set_level(logging.INFO, logger="imageint.progress")

    await reporter.tick()
    await reporter.tick()

    download_lines = [line for line in lines(caplog) if "Download" in line]
    assert len(download_lines) == 2
    assert "42,5 %" in download_lines[0]
    assert "noch 2 min 5 s" in download_lines[0]
    # Force bypasses the health cache, so the 20 s cadence is really observed.
    assert all(force for _, force in health.probes)


async def test_ready_component_is_reported_once(caplog):
    payloads = ready_both()
    payloads["image"] = component("image", "loading", PROGRESS)
    reporter = progresslog.ProgressReporter(StubHealth(payloads))
    caplog.set_level(logging.INFO, logger="imageint.progress")

    await reporter.tick()
    payloads["image"] = component("image", "ready", {"known": True, "percent": 100.0})
    await reporter.tick()
    await reporter.tick()

    messages = lines(caplog)
    assert messages[-1] == "Bildmodell: bereit, Download abgeschlossen."
    assert sum("bereit" in message for message in messages) == 1


async def test_finished_percentage_is_not_repeated_while_loading(caplog):
    payloads = ready_both()
    payloads["image"] = component("image", "loading", {"known": True, "percent": 100.0})
    reporter = progresslog.ProgressReporter(StubHealth(payloads))
    caplog.set_level(logging.INFO, logger="imageint.progress")

    await reporter.tick()
    await reporter.tick()

    assert not any("Download" in line for line in lines(caplog))


async def test_loading_without_progress_is_reported_once(caplog):
    payloads = ready_both()
    payloads["image"] = component("image", "loading")
    reporter = progresslog.ProgressReporter(StubHealth(payloads))
    caplog.set_level(logging.INFO, logger="imageint.progress")

    await reporter.tick()
    await reporter.tick()

    messages = lines(caplog)
    assert messages == ["Bildmodell: lädt, noch keine Download-Angaben."]


async def test_unreachable_component_is_reported_once(caplog):
    payloads = ready_both()
    payloads["enhancer"] = component(
        "enhancer", "unreachable", message="Modellserver nicht erreichbar: Test"
    )
    reporter = progresslog.ProgressReporter(StubHealth(payloads))
    caplog.set_level(logging.WARNING, logger="imageint.progress")

    await reporter.tick()
    await reporter.tick()

    messages = lines(caplog)
    assert messages == [
        "Prompt-Enhancer: unreachable (Modellserver nicht erreichbar: Test)"
    ]


async def test_run_probes_until_cancelled():
    health = StubHealth(ready_both())
    task = asyncio.create_task(progresslog.run(health, 0))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sum(1 for name, _ in health.probes if name == "image") >= 2


def test_task_is_disabled_by_zero():
    assert progresslog.task(StubHealth(ready_both()), 0) is None


# -- configuration ---------------------------------------------------------- #


def test_progress_log_interval_defaults_to_20_seconds(monkeypatch):
    monkeypatch.delenv("IMAGEINT_PROGRESS_LOG_SECONDS", raising=False)
    assert config.load_settings().progress_log_seconds == 20


def test_progress_log_interval_can_be_disabled(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PROGRESS_LOG_SECONDS", "0")
    assert config.load_settings().progress_log_seconds == 0
