"""Tests for the job registry: queueing, capacity, retention and shutdown."""

import asyncio
import time

import pytest

from app.config import Settings
from app.errors import ImageIntError
from app.jobs import DONE, ERROR, QUEUED, RUNNING, JobRegistry
from app.storage import ImageStore
from conftest import png_bytes


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "storage_dir": str(tmp_path / "images"),
        "max_concurrent_jobs": 1,
        "max_queued_jobs": 0,
        "max_jobs": 10,
        "job_retention_seconds": 3600,
    }
    values.update(overrides)
    return Settings(**values)


class Controlled:
    """A runner that blocks until the test releases it."""

    def __init__(self) -> None:
        self.started = asyncio.Semaphore(0)
        self.release = asyncio.Event()
        self.seen: list = []

    async def __call__(self, job) -> None:
        self.seen.append(job.id)
        job.stage = "rendering"
        self.started.release()
        await self.release.wait()
        job.status = DONE


@pytest.fixture
async def registry_factory(tmp_path):
    created = []

    def build(runner, **overrides):
        settings = make_settings(tmp_path, **overrides)
        store = ImageStore(settings.storage_dir)
        registry = JobRegistry(settings, store, runner)
        created.append(registry)
        return registry

    yield build
    # Cancel whatever a test left running, so no task outlives its event loop.
    for registry in created:
        await registry.shutdown()


async def test_a_submitted_job_runs_and_finishes(registry_factory, tmp_path):
    async def runner(job):
        job.image_bytes = 4
        job.status = DONE

    registry = registry_factory(runner)
    job = registry.submit("ein rotes Haus")
    assert job.status in (QUEUED, RUNNING)
    assert await registry.wait(job, 5) is True
    assert job.status == DONE
    assert job.done is True
    assert job.finished_at is not None


async def test_wait_returns_immediately_for_a_zero_timeout(registry_factory):
    async def runner(job):
        await asyncio.sleep(5)

    registry = registry_factory(runner)
    job = registry.submit("ein rotes Haus")
    assert await registry.wait(job, 0) is False
    assert job.status in (QUEUED, RUNNING)


async def test_wait_times_out_for_a_slow_job(registry_factory):
    async def runner(job):
        await asyncio.sleep(5)

    registry = registry_factory(runner)
    job = registry.submit("ein rotes Haus")
    assert await registry.wait(job, 0.05) is False


async def test_the_capacity_is_running_plus_queued(registry_factory):
    async def runner(job):
        await asyncio.sleep(5)

    # One slot, no queue: the second request is refused right away.
    registry = registry_factory(runner, max_concurrent_jobs=1, max_queued_jobs=0)
    registry.submit("eins")
    with pytest.raises(ImageIntError) as excinfo:
        registry.submit("zwei")
    assert excinfo.value.code == "busy"
    assert excinfo.value.status_code == 503
    assert excinfo.value.headers["Retry-After"] == "30"


async def test_two_queued_jobs_are_accepted(registry_factory):
    async def runner(job):
        await asyncio.sleep(5)

    registry = registry_factory(runner, max_concurrent_jobs=1, max_queued_jobs=2)
    registry.submit("eins")
    registry.submit("zwei")
    registry.submit("drei")
    with pytest.raises(ImageIntError) as excinfo:
        registry.submit("vier")
    assert excinfo.value.code == "busy"
    assert registry.stats()["jobs"] == 3


async def test_a_finished_job_frees_its_slot(registry_factory):
    async def runner(job):
        job.status = DONE
        job.image_bytes = 1

    registry = registry_factory(runner, max_concurrent_jobs=1, max_queued_jobs=0)
    first = registry.submit("eins")
    assert await registry.wait(first, 5) is True
    second = registry.submit("zwei")
    assert await registry.wait(second, 5) is True


async def test_a_runner_that_raises_becomes_an_error_job(registry_factory):
    async def runner(job):
        raise RuntimeError("Speicher voll")

    registry = registry_factory(runner)
    job = registry.submit("ein rotes Haus")
    assert await registry.wait(job, 5) is True
    assert job.status == ERROR
    assert job.error_code == "internal_error"
    assert "Speicher voll" in job.error_message


async def test_a_runner_that_returns_nothing_becomes_an_error(registry_factory):
    async def runner(job):
        return None

    registry = registry_factory(runner)
    job = registry.submit("ein rotes Haus")
    assert await registry.wait(job, 5) is True
    assert job.status == ERROR
    assert job.error_code == "internal_error"


async def test_a_job_with_bytes_but_no_status_counts_as_done(registry_factory):
    async def runner(job):
        job.image_bytes = 12

    registry = registry_factory(runner)
    job = registry.submit("ein rotes Haus")
    assert await registry.wait(job, 5) is True
    assert job.status == DONE


async def test_lookup_of_an_unknown_job_is_a_404(registry_factory):
    async def runner(job):
        return None

    registry = registry_factory(runner)
    with pytest.raises(ImageIntError) as excinfo:
        registry.get("gibt-es-nicht")
    assert excinfo.value.code == "not_found"
    assert registry.find("gibt-es-nicht") is None


async def test_the_image_is_only_served_when_the_job_is_done(registry_factory, tmp_path):
    payload = png_bytes((8, 8))

    async def runner(job):
        store = registry.store
        store.write(job.id, payload)
        job.image_bytes = len(payload)
        job.status = DONE

    registry = registry_factory(runner)
    job = registry.submit("ein rotes Haus")
    assert await registry.wait(job, 5) is True
    assert registry.image(job.id) == payload
    assert registry.image_content_type(job.id) == "image/png"


async def test_prune_removes_expired_jobs_and_their_files(registry_factory):
    payload = png_bytes((8, 8))

    async def runner(job):
        registry.store.write(job.id, payload)
        job.image_bytes = len(payload)
        job.status = DONE

    registry = registry_factory(runner, job_retention_seconds=60)
    job = registry.submit("ein rotes Haus")
    assert await registry.wait(job, 5) is True
    assert registry.store.path(job.id) is not None

    # Age the job beyond the retention window.
    job.finished_at = time.time() - 3600
    assert registry.prune() == 1
    assert registry.find(job.id) is None
    assert registry.store.path(job.id) is None


async def test_prune_keeps_running_jobs(registry_factory):
    async def runner(job):
        await asyncio.sleep(5)

    registry = registry_factory(runner, job_retention_seconds=1, max_jobs=1)
    job = registry.submit("ein rotes Haus")
    job.created_at = time.time() - 3600
    assert registry.prune() == 0
    assert registry.find(job.id) is not None


async def test_prune_honours_max_jobs(registry_factory):
    async def runner(job):
        job.status = DONE
        job.image_bytes = 1

    registry = registry_factory(runner, max_jobs=2, max_queued_jobs=10)
    jobs = [registry.submit(f"prompt {index}") for index in range(5)]
    for job in jobs:
        await registry.wait(job, 5)

    registry.prune()
    assert len(registry.list(limit=50)) == 2
    # The newest two survive.
    remaining = {job.id for job in registry.list(limit=50)}
    assert remaining == {jobs[-1].id, jobs[-2].id}


async def test_list_returns_the_newest_first(registry_factory):
    async def runner(job):
        job.status = DONE
        job.image_bytes = 1

    registry = registry_factory(runner, max_queued_jobs=10)
    first = registry.submit("eins")
    await registry.wait(first, 5)
    second = registry.submit("zwei")
    await registry.wait(second, 5)
    assert [job.id for job in registry.list()] == [second.id, first.id]


async def test_stats_report_the_queue_and_the_store(registry_factory):
    async def runner(job):
        await asyncio.sleep(5)

    registry = registry_factory(runner, max_concurrent_jobs=1, max_queued_jobs=2)
    registry.submit("eins")
    registry.submit("zwei")
    stats = registry.stats()
    assert stats["jobs"] == 2
    assert stats["queued"] + stats["running"] == 2
    assert stats["max_concurrent_jobs"] == 1
    assert stats["max_queued_jobs"] == 2
    assert stats["stored_bytes"] == 0


async def test_shutdown_cancels_running_jobs(registry_factory):
    started = asyncio.Event()

    async def runner(job):
        started.set()
        await asyncio.sleep(30)

    registry = registry_factory(runner)
    job = registry.submit("ein rotes Haus")
    await asyncio.wait_for(started.wait(), 5)
    await registry.shutdown()
    assert job.status == ERROR
    assert job.error_code == "cancelled"
    assert job.finished_at is not None


async def test_cleanup_store_drops_orphaned_files(registry_factory):
    async def runner(job):
        job.status = DONE
        job.image_bytes = 1

    registry = registry_factory(runner)
    registry.store.write("verwaisterauftrag", png_bytes((8, 8)))
    assert registry.cleanup_store() == 1
    assert registry.store.path("verwaisterauftrag") is None


async def test_public_view_reports_the_image_url_once_done(registry_factory):
    async def runner(job):
        job.status = DONE
        job.image_bytes = 1
        job.width, job.height = 8, 8
        job.enhanced_prompt = "ein rotes Haus, sehr detailliert"

    registry = registry_factory(runner)
    job = registry.submit("ein rotes Haus")
    await registry.wait(job, 5)
    payload = job.public("http://imageint.example/")
    assert payload["image_url"] == f"http://imageint.example/v1/images/{job.id}"
    assert payload["enhanced_prompt"] == "ein rotes Haus, sehr detailliert"
    assert "error" not in payload
