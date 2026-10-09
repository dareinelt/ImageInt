"""Readiness of the two model servers, and the loading contract built on it.

Both model servers download tens of gigabytes and then build the model in memory.
On the reference host that is a long, one-time wait -- and a client that asks for
a picture during it is not making an error, it is simply early. The states below
say which of the two it is:

``ready``
    ``/health`` answered 200.
``loading``
    The server answered, but with 503: it is up and still building the model.
``starting``
    Nothing is listening yet, and the container has not been up long enough to
    call that a failure. This is the state during the first model download.
``unreachable``
    Nothing is listening and the grace period is over.
``error``
    The server answered with something that is not 200 or 503.
``unconfigured``
    No URL is set for this component.

``loading`` and ``starting`` are **transient**: the request is valid, the answer
is just not available yet, so the API replies 503 ``service_loading`` with a
``Retry-After`` and the client keeps its input. Everything else is a real
failure. This is the distinction that keeps a slow CPU host from looking broken.

Probing is cached for ``IMAGEINT_HEALTH_CACHE_SECONDS`` so a polling client does
not turn every status request into two upstream calls.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from . import host, vllm
from .config import Settings
from .errors import (
    host_unsupported,
    not_configured,
    service_loading,
    upstream_unavailable,
)

log = logging.getLogger("imageint.health")

READY = "ready"
LOADING = "loading"
STARTING = "starting"
UNREACHABLE = "unreachable"
ERROR = "error"
UNCONFIGURED = "unconfigured"

#: States that mean "not yet, try again" rather than "this cannot work".
TRANSIENT = (LOADING, STARTING)

#: Components, in the order they appear in the API responses. The enhancer runs
#: first in every request, so it is listed first here too.
COMPONENTS = ("enhancer", "image")

#: Components whose server implements ``POST /v1/warmup``, and can therefore be
#: configured to load lazily. Only these may be nudged by the readiness probe.
WARMABLE = ("enhancer", "image")

COMPONENT_LABELS = {
    "enhancer": "Prompt-Enhancer",
    "image": "Bildmodell",
}


class HealthMonitor:
    """State machine over the ``/health`` endpoints of both model servers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[str, dict] = {}
        self._cache_at: dict[str, float] = {}
        # When a component was first seen at all. It decides whether "nothing is
        # listening" is a first start or a broken deployment.
        self._first_seen: dict[str, float] = {}
        self._ever_ready: dict[str, bool] = {}

    # -- configuration ----------------------------------------------------- #

    def _target(self, component: str) -> tuple[str, str, str]:
        if component == "enhancer":
            settings = self.settings.enhancer
            return settings.url, settings.token, settings.model
        settings = self.settings.image
        return settings.url, settings.token, settings.model

    # -- probing ----------------------------------------------------------- #

    async def probe(self, component: str, *, force: bool = False) -> dict:
        """Current state of one component, cached for a few seconds."""

        now = time.monotonic()
        self._first_seen.setdefault(component, now)

        if not force:
            cached = self._cache.get(component)
            cached_at = self._cache_at.get(component, 0.0)
            if cached is not None and now - cached_at < self.settings.health_cache_seconds:
                return cached

        url, token, model = self._target(component)
        result = await vllm.health(url, token, model)
        state = self._classify(component, result)

        payload = {
            "component": component,
            "label": COMPONENT_LABELS.get(component, component),
            "state": state,
            "ready": state == READY,
            "transient": state in TRANSIENT,
            "url": result.get("url") or "",
            "model": result.get("model") or "",
            "http": result.get("http") or 0,
            "message": result.get("message") or "",
            "since_seconds": round(now - self._first_seen[component], 1),
        }
        if state == READY:
            self._ever_ready[component] = True

        self._cache[component] = payload
        self._cache_at[component] = now
        return payload

    def _classify(self, component: str, result: dict) -> str:
        """Turn one probe result into one of the states above."""

        if not result.get("configured"):
            return UNCONFIGURED
        if result.get("ok"):
            return READY
        if result.get("loading"):
            return LOADING
        if result.get("http"):
            return ERROR

        # Nothing answered. During a first start that is expected for minutes;
        # after the grace period, or after the component was ready once, it is a
        # failure.
        elapsed = time.monotonic() - self._first_seen.get(component, time.monotonic())
        if self._ever_ready.get(component):
            return UNREACHABLE
        if elapsed < self.settings.starting_grace_seconds:
            return STARTING
        return UNREACHABLE

    async def snapshot(self, *, force: bool = False) -> dict:
        """Both components plus the host verdict, as reported by the API."""

        components = {}
        for component in COMPONENTS:
            components[component] = await self.probe(component, force=force)

        blocked = host.blocking()
        ready = not blocked and all(item["ready"] for item in components.values())
        transient = any(item["transient"] for item in components.values())

        if blocked:
            state = UNREACHABLE
        elif ready:
            state = READY
        elif transient:
            state = LOADING
        else:
            state = UNREACHABLE

        return {
            "ok": ready,
            "ready": ready,
            "state": state,
            "transient": transient,
            "host": host.inspect(),
            "components": components,
        }

    async def require_ready(self, component: str) -> dict:
        """Probe one component and raise the right error when it cannot serve.

        Called before a request is accepted, so a client learns about a loading
        model immediately instead of after a two-minute timeout.
        """

        if host.blocking():
            raise host_unsupported(host.blocking()[0])

        state = await self.probe(component, force=True)
        if state["ready"]:
            return state

        label = state["label"]
        retry_after = self.settings.loading_retry_after

        if state["state"] == UNCONFIGURED:
            raise not_configured(
                f"Für {label} ist keine Modell-URL konfiguriert "
                f"(IMAGEINT_{'PE' if component == 'enhancer' else 'IMAGE'}_URL)."
            )
        if state["state"] in TRANSIENT:
            # A model server that answered 503 is alive and building itself. If
            # it loads lazily it is waiting for a request to start -- and a
            # request is exactly what this caller is about to be refused. Ask it
            # to start loading in the background so the retry after
            # ``retry_after`` seconds can actually succeed, and answer this
            # request with the same "not yet" as before. Both model servers
            # implement ``POST /v1/warmup`` and both can be configured to load
            # lazily, so both get the nudge.
            if component in WARMABLE and state["state"] == LOADING and state["url"]:
                await vllm.warmup(state["url"], self._target(component)[1])
            raise service_loading(
                component,
                f"{label} ist noch nicht bereit ({state['state']}). "
                f"{state['message']} Bitte nach {retry_after} Sekunden erneut versuchen.",
                retry_after,
            )
        raise upstream_unavailable(
            f"{component}_unavailable",
            f"{label} ist nicht erreichbar. {state['message']}",
        )

    def reset(self) -> None:
        """Drop all cached state; used by tests and after a config change."""

        self._cache.clear()
        self._cache_at.clear()
        self._first_seen.clear()
        self._ever_ready.clear()

    def summary(self) -> Optional[str]:
        """One-line summary for the startup log, or ``None`` if not probed yet."""

        if not self._cache:
            return None
        return ", ".join(
            f"{name}={self._cache[name]['state']}" for name in COMPONENTS if name in self._cache
        )
