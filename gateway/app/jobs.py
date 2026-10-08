"""The job registry: the async half of the API.

On a CPU host a single 2048x2048 picture takes minutes, and the text model in
LLMInt is waiting for a tool result. A synchronous HTTP request would therefore
either time out or block a chat turn for the whole render, so generation is a
job:

* ``POST /v1/images/generations`` creates one and waits only
  ``IMAGEINT_SYNC_TIMEOUT`` seconds. A fast host answers with the finished image,
  a slow one with **202** and the job id -- which is the normal case here.
* ``GET /v1/jobs/{id}`` reports the state, and ``GET /v1/images/{id}`` serves the
  bytes once it is done.

Requests are *queued* rather than refused while a slot is busy, because refusing
would turn a second chat request into an error for no good reason. The queue is
bounded by ``IMAGEINT_MAX_QUEUED_JOBS``; beyond that the endpoint answers 503
``busy`` with a ``Retry-After``, which is a state the client can act on.

The registry holds no state a restart must preserve: it lives in memory, and
images on disk are cleaned up against it (see :mod:`app.storage`). A job that
was running when the container died is simply gone, which is honest -- its image
does not exist.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .config import Settings
from .errors import busy, not_found
from .storage import ImageStore

log = logging.getLogger("imageint.jobs")

#: Job lifecycle. ``queued`` and ``running`` are the transient states a client
#: polls through; ``done`` and ``error`` are terminal.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"

TERMINAL = (DONE, ERROR)

#: How long a client should wait before polling a busy endpoint again. Long
#: enough not to hammer a CPU-only service, short enough to feel responsive.
BUSY_RETRY_AFTER = 30


@dataclass
class Job:
    """One generation request, from submission to finished image."""

    id: str
    prompt: str
    request: dict = field(default_factory=dict)
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    #: Filled in by the pipeline as it progresses, so a poller can see that
    #: something is happening even before the picture exists.
    stage: str = "queued"
    enhanced_prompt: str = ""
    negative_prompt: str = ""
    wh_ratio: str = ""
    width: int = 0
    height: int = 0
    seed: Optional[int] = None
    parse_ok: Optional[bool] = None
    model: str = ""
    enhancer_model: str = ""
    timings: dict = field(default_factory=dict)

    error_code: str = ""
    error_message: str = ""
    image_bytes: int = 0
    content_type: str = ""

    #: Signalled once the job reaches a terminal state, so the synchronous path
    #: can wait for it and the async path can ignore it.
    _finished: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def done(self) -> bool:
        return self.status in TERMINAL

    def public(self, base_url: str = "") -> dict:
        """The JSON representation a client sees."""

        payload: dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "stage": self.stage,
            "prompt": self.prompt,
            "created_at": round(self.created_at, 3),
            "started_at": round(self.started_at, 3) if self.started_at else None,
            "finished_at": round(self.finished_at, 3) if self.finished_at else None,
            "width": self.width,
            "height": self.height,
            "wh_ratio": self.wh_ratio,
            "seed": self.seed,
            "model": self.model,
            "parse_ok": self.parse_ok,
            "timings": dict(self.timings),
            "image_bytes": self.image_bytes,
        }
        if self.enhanced_prompt:
            payload["enhanced_prompt"] = self.enhanced_prompt
        if self.error_code:
            payload["error"] = self.error_code
            payload["message"] = self.error_message
        if self.status == DONE:
            payload["image_url"] = self.image_url(base_url)
        return payload

    def image_url(self, base_url: str = "") -> str:
        base = (base_url or "").rstrip("/")
        path = f"/v1/images/{self.id}"
        return f"{base}{path}" if base else path


#: The pipeline signature: take a job, fill it in, return the same job. It must
#: not raise for expected upstream failures -- those belong in ``job.status``.
Runner = Callable[[Job], Awaitable[None]]


class JobRegistry:
    """Bounded queue plus concurrency limit for image generation."""

    def __init__(self, settings: Settings, store: ImageStore, runner: Runner) -> None:
        self.settings = settings
        self.store = store
        self.runner = runner
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()
        self._slots = asyncio.Semaphore(max(1, settings.max_concurrent_jobs))

    # -- submission -------------------------------------------------------- #

    def submit(self, prompt: str, request: Optional[dict] = None) -> Job:
        """Queue one generation and return its job immediately.

        Raises :class:`~app.errors.ImageIntError` with 503 ``busy`` when every
        slot -- running plus waiting -- is taken, so the caller gets a clear
        refusal instead of an unbounded backlog. The capacity is
        ``max_concurrent_jobs + max_queued_jobs``, which is why
        ``IMAGEINT_MAX_QUEUED_JOBS=0`` means "reject as soon as the running slots
        are full" rather than "reject everything".
        """

        self.prune()

        capacity = self.settings.max_concurrent_jobs + self.settings.max_queued_jobs
        active = sum(1 for job in self._jobs.values() if job.status not in TERMINAL)
        if active >= capacity:
            raise busy(
                f"Alle {capacity} Bildgenerierungs-Plätze sind belegt. Bitte später erneut versuchen.",
                BUSY_RETRY_AFTER,
            )

        job = Job(id=uuid.uuid4().hex, prompt=prompt, request=dict(request or {}))
        self._jobs[job.id] = job

        task = asyncio.create_task(self._run(job))
        # Keep a reference: a task that is only referenced by the loop's weak
        # set can be garbage collected mid-flight.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        log.info("Bildauftrag %s eingereiht (%d Zeichen Prompt).", job.id, len(prompt))
        return job

    async def _run(self, job: Job) -> None:
        async with self._slots:
            job.status = RUNNING
            job.started_at = time.time()
            try:
                await self.runner(job)
            except asyncio.CancelledError:
                # Shutting down. Nothing can be salvaged for a caller that is
                # gone too, but the job must not stay "running" forever.
                job.status = ERROR
                job.error_code = "cancelled"
                job.error_message = "Der Dienst wurde während der Generierung beendet."
                raise
            except Exception as exc:  # noqa: BLE001 - a job must never kill the loop
                log.exception("Bildauftrag %s ist unerwartet gescheitert.", job.id)
                job.status = ERROR
                job.error_code = "internal_error"
                job.error_message = str(exc) or exc.__class__.__name__
            finally:
                if job.status not in TERMINAL:
                    # The pipeline returned without deciding; treat a job with
                    # bytes on disk as done, anything else as an error.
                    if job.image_bytes:
                        job.status = DONE
                    else:
                        job.status = ERROR
                        job.error_code = job.error_code or "internal_error"
                        job.error_message = job.error_message or "Unbekannter Fehler."
                job.finished_at = time.time()
                job._finished.set()
                log.info(
                    "Bildauftrag %s beendet: %s (%.1f s).",
                    job.id,
                    job.status,
                    (job.finished_at - (job.started_at or job.finished_at)),
                )
                self.prune()

    # -- lookup ------------------------------------------------------------ #

    def get(self, job_id: str) -> Job:
        """One job by id, or a 404 with a German message."""

        job = self._jobs.get(job_id)
        if job is None:
            raise not_found(
                f"Unbekannter Bildauftrag {job_id!r}. Aufträge werden nach "
                f"{self.settings.job_retention_seconds // 3600} Stunden entfernt."
            )
        return job

    def find(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def image(self, job_id: str) -> Optional[bytes]:
        """The finished image of one job, or ``None``."""

        job = self._jobs.get(job_id)
        if job is None or job.status != DONE:
            return None
        return self.store.read(job_id)

    def image_content_type(self, job_id: str) -> str:
        return self.store.content_type(job_id) or "image/png"

    def list(self, limit: int = 50) -> list:
        """Newest jobs first, for the admin view and for tests."""

        jobs = sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)
        return jobs[: max(1, limit)]

    async def wait(self, job: Job, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for a job to finish.

        Returns ``True`` when it reached a terminal state. ``timeout <= 0``
        returns immediately, which is how a client asks for the pure async
        behaviour.
        """

        if job.done:
            return True
        if timeout <= 0:
            return False
        try:
            await asyncio.wait_for(job._finished.wait(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return False
        return True

    # -- housekeeping ------------------------------------------------------ #

    def stats(self) -> dict:
        counts = {QUEUED: 0, RUNNING: 0, DONE: 0, ERROR: 0}
        for job in self._jobs.values():
            counts[job.status] = counts.get(job.status, 0) + 1
        return {
            "jobs": len(self._jobs),
            "queued": counts[QUEUED],
            "running": counts[RUNNING],
            "done": counts[DONE],
            "error": counts[ERROR],
            "max_concurrent_jobs": self.settings.max_concurrent_jobs,
            "max_queued_jobs": self.settings.max_queued_jobs,
            "max_jobs": self.settings.max_jobs,
            "job_retention_seconds": self.settings.job_retention_seconds,
            "stored_bytes": self.store.total_bytes(),
        }

    def prune(self) -> int:
        """Drop finished jobs that are expired or beyond ``IMAGEINT_MAX_JOBS``.

        Running jobs are never dropped, whatever the limits say -- a job in
        flight is the only thing a client is actually waiting for.
        """

        now = time.time()
        removed = 0

        for job_id, job in list(self._jobs.items()):
            if not job.done:
                continue
            if now - (job.finished_at or job.created_at) > self.settings.job_retention_seconds:
                self._jobs.pop(job_id, None)
                self.store.delete(job_id)
                removed += 1

        finished = [job for job in self._jobs.values() if job.done]
        if len(finished) > self.settings.max_jobs:
            finished.sort(key=lambda job: job.finished_at or job.created_at)
            for job in finished[: len(finished) - self.settings.max_jobs]:
                self._jobs.pop(job.id, None)
                self.store.delete(job.id)
                removed += 1

        if removed:
            log.info("%d abgeschlossene Bildaufträge aus dem Verzeichnis entfernt.", removed)
        return removed

    def cleanup_store(self) -> int:
        """Remove files on disk that belong to no known job (e.g. after restart)."""

        return self.store.cleanup(set(self._jobs))

    async def shutdown(self) -> None:
        """Cancel everything still running, so the container stops promptly."""

        pending = [task for task in self._tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def reset(self) -> None:
        """Drop all state; used by tests."""

        self._jobs.clear()
