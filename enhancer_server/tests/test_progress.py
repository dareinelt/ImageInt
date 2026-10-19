"""The download progress tracker: cache scanning, totals and the payload.

Nothing here touches the network or a real HuggingFace download -- the Hub API
is faked and the cache is a temporary directory -- so these tests pin down the
arithmetic and the ``progress`` contract without a 33 GB fixture.
"""

from __future__ import annotations

import types

import pytest

from enhancer_server.progress import DownloadProgress, default_cache_dir

REPO = "Qwen/Qwen-Image-2.1"
FOLDER = "models--Qwen--Qwen-Image-2.1"


class FakeApi:
    """Stand-in for ``huggingface_hub.HfApi``."""

    def __init__(self, siblings=None, error: Exception | None = None) -> None:
        self.siblings = siblings or []
        self.error = error
        self.calls: list = []

    def model_info(self, repo_id, **kwargs):
        self.calls.append((repo_id, kwargs))
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(siblings=self.siblings)


def sibling(name: str, size: int | None, blob_id: str | None = None) -> dict:
    return {"rfilename": name, "size": size, "blob_id": blob_id or name}


def blobs_dir(tmp_path) -> object:
    path = tmp_path / FOLDER / "blobs"
    path.mkdir(parents=True)
    return path


def write_blob(directory, name: str, size: int) -> None:
    (directory / name).write_bytes(b"x" * size)


def activate(tracker: DownloadProgress) -> DownloadProgress:
    """Mark the tracker as running; the maths below needs no polling thread."""

    tracker._started = True
    return tracker


def tracker_for(tmp_path, api=None, **kwargs) -> DownloadProgress:
    return activate(
        DownloadProgress(REPO, cache_dir=str(tmp_path), api=api or FakeApi(), **kwargs)
    )


# --------------------------------------------------------------------------- #
# Totals
# --------------------------------------------------------------------------- #
def test_the_total_is_the_sum_of_the_sibling_sizes(tmp_path):
    api = FakeApi([sibling("a.bin", 100), sibling("b.bin", 300)])
    tracker = tracker_for(tmp_path, api)

    tracker._read_model_info()

    assert tracker._bytes_total == 400
    assert tracker._files_total == 2
    assert api.calls[0][0] == REPO
    assert api.calls[0][1]["files_metadata"] is True


def test_the_configured_revision_is_asked_for(tmp_path):
    api = FakeApi([sibling("a.bin", 10)])
    tracker = tracker_for(tmp_path, api, revision="main")

    tracker._read_model_info()

    assert api.calls[0][1]["revision"] == "main"


def test_siblings_without_a_size_still_name_their_file(tmp_path):
    api = FakeApi([{"rfilename": "a.bin", "blob_id": "abc", "size": None}])
    tracker = tracker_for(tmp_path, api)

    tracker._read_model_info()

    assert tracker._bytes_total is None
    assert tracker._blob_names == {"abc": "a.bin"}


def test_a_failing_hub_call_only_costs_the_total(tmp_path):
    directory = blobs_dir(tmp_path)
    write_blob(directory, "abc", 100)
    tracker = tracker_for(tmp_path, FakeApi(error=RuntimeError("offline")))

    tracker._guard(tracker._read_model_info)
    tracker._sample()

    snapshot = tracker.snapshot()
    assert snapshot["known"] is False
    assert snapshot["bytes_downloaded"] == 100
    assert "bytes_total" not in snapshot
    assert "percent" not in snapshot


# --------------------------------------------------------------------------- #
# Scanning the cache
# --------------------------------------------------------------------------- #
def test_the_percentage_is_the_share_of_the_total_on_disk(tmp_path):
    directory = blobs_dir(tmp_path)
    write_blob(directory, "abc", 100)
    tracker = tracker_for(tmp_path, FakeApi([sibling("a.bin", 400)]))

    tracker._read_model_info()
    tracker._sample()

    snapshot = tracker.snapshot()
    assert snapshot["known"] is True
    assert snapshot["percent"] == 25.0
    assert snapshot["bytes_downloaded"] == 100
    assert snapshot["bytes_total"] == 400
    assert snapshot["files_downloaded"] == 1
    assert snapshot["files_total"] == 1


def test_an_incomplete_file_counts_with_its_bytes(tmp_path):
    directory = blobs_dir(tmp_path)
    write_blob(directory, "abc.incomplete", 50)
    tracker = tracker_for(tmp_path, FakeApi([sibling("a.bin", 100)]))

    tracker._read_model_info()
    tracker._sample()

    snapshot = tracker.snapshot()
    assert snapshot["bytes_downloaded"] == 50
    assert snapshot["files_downloaded"] == 0
    assert snapshot["percent"] == 50.0


def test_the_lock_file_is_ignored(tmp_path):
    directory = blobs_dir(tmp_path)
    write_blob(directory, "abc.lock", 1000)
    tracker = tracker_for(tmp_path, FakeApi([sibling("a.bin", 10)]))

    tracker._read_model_info()
    tracker._sample()

    assert tracker.snapshot()["bytes_downloaded"] == 0


def test_only_blobs_are_counted_not_the_snapshots(tmp_path):
    directory = blobs_dir(tmp_path)
    write_blob(directory, "abc", 100)
    snapshot_dir = tmp_path / FOLDER / "snapshots" / "abcdef"
    snapshot_dir.mkdir(parents=True)
    write_blob(snapshot_dir, "model.safetensors", 900)
    tracker = tracker_for(tmp_path, FakeApi([sibling("model.safetensors", 100)]))

    tracker._read_model_info()
    tracker._sample()

    assert tracker.snapshot()["bytes_downloaded"] == 100


def test_symlinked_blobs_are_not_counted_twice(tmp_path):
    directory = blobs_dir(tmp_path)
    write_blob(directory, "abc", 100)
    try:
        (directory / "link").symlink_to(directory / "abc")
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks sind auf diesem System nicht verfÃ¼gbar.")

    tracker = tracker_for(tmp_path, FakeApi([sibling("a.bin", 100)]))
    tracker._read_model_info()
    tracker._sample()

    assert tracker.snapshot()["bytes_downloaded"] == 100


def test_a_missing_cache_directory_is_not_an_error(tmp_path):
    tracker = tracker_for(tmp_path, FakeApi([sibling("a.bin", 100)]))

    tracker._read_model_info()
    tracker._sample()

    assert tracker.snapshot()["bytes_downloaded"] == 0


def test_the_cache_falls_back_to_the_huggingface_location(monkeypatch, tmp_path):
    monkeypatch.setattr("enhancer_server.progress._HF_HUB_CACHE", str(tmp_path))
    tracker = DownloadProgress(REPO)

    assert tracker._blobs_dir() == tmp_path / FOLDER / "blobs"
    assert default_cache_dir() == str(tmp_path)


# --------------------------------------------------------------------------- #
# Current file and ETA
# --------------------------------------------------------------------------- #
def test_the_current_file_is_the_readable_name_of_the_newest_blob(tmp_path):
    directory = blobs_dir(tmp_path)
    write_blob(directory, "deadbeef", 10)
    api = FakeApi([sibling("model.safetensors", 10, blob_id="deadbeef")])
    tracker = tracker_for(tmp_path, api)

    tracker._read_model_info()
    tracker._sample()

    assert tracker.snapshot()["current_file"] == "model.safetensors"


def test_the_eta_follows_the_measured_rate(tmp_path):
    directory = blobs_dir(tmp_path)
    now = [0.0]
    tracker = tracker_for(
        tmp_path, FakeApi([sibling("a.bin", 400)]), clock=lambda: now[0]
    )
    tracker._read_model_info()

    tracker._sample()
    write_blob(directory, "abc", 100)
    now[0] = 10.0
    tracker._sample()

    # 100 bytes in 10 s, 300 bytes left.
    assert tracker.snapshot()["eta_seconds"] == 30


def test_a_shrinking_cache_does_not_report_a_negative_rate(tmp_path):
    directory = blobs_dir(tmp_path)
    now = [0.0]
    tracker = tracker_for(
        tmp_path, FakeApi([sibling("a.bin", 400)]), clock=lambda: now[0]
    )
    tracker._read_model_info()
    write_blob(directory, "abc", 200)
    tracker._sample()

    (directory / "abc").unlink()
    now[0] = 10.0
    tracker._sample()

    assert tracker.snapshot().get("eta_seconds") is None


# --------------------------------------------------------------------------- #
# The payload contract
# --------------------------------------------------------------------------- #
def test_an_idle_tracker_reports_nothing(tmp_path):
    tracker = DownloadProgress(REPO, cache_dir=str(tmp_path), api=FakeApi())
    assert tracker.snapshot() == {}


def test_a_stopped_tracker_reports_nothing(tmp_path):
    directory = blobs_dir(tmp_path)
    write_blob(directory, "abc", 100)
    tracker = tracker_for(tmp_path, FakeApi([sibling("a.bin", 400)]))
    tracker._read_model_info()
    tracker._sample()

    tracker.stop()

    assert tracker.snapshot() == {}


def test_a_finished_load_reports_one_hundred_percent(tmp_path):
    tracker = tracker_for(tmp_path, FakeApi([sibling("a.bin", 400)]))
    tracker._read_model_info()

    tracker.finish()

    snapshot = tracker.snapshot()
    assert snapshot["known"] is True
    assert snapshot["percent"] == 100.0
    assert snapshot["bytes_total"] == 400
    assert snapshot["eta_seconds"] == 0


def test_the_snapshot_stays_json_serialisable(tmp_path):
    import json

    directory = blobs_dir(tmp_path)
    write_blob(directory, "abc", 100)
    tracker = tracker_for(tmp_path, FakeApi([sibling("a.bin", 400)]))
    tracker._read_model_info()
    tracker._sample()

    assert json.loads(json.dumps(tracker.snapshot()))["percent"] == 25.0

