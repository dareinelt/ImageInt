"""Host sizing: what an image endpoint must provide, and how this one compares.

ImageInt is dimensioned for the smallest machine that is expected to host it:
**eight CPU cores with AVX2 and 32 GB RAM**. That is the reference the thread
defaults, the container budgets and the estimates in the README are based on.

Two CPU features are treated differently on purpose:

* **AVX2 is a hard requirement on bare metal.** The CPU builds of both model
  servers dispatch their kernels through AVX2; without it the service cannot
  produce a picture at all. :func:`blocking` therefore reports it, the startup
  hook logs it as an error and :mod:`app.main` refuses generation requests with
  HTTP 503 ``host_unsupported``.
* **AVX-512 is informational only.** It speeds the prompt-processing path up on
  the hosts that have it, but every supported build also runs without it, so its
  absence is reported as a note and never as a failure.

Missing cores or RAM are neither: an undersized endpoint still runs, it is just
slower than documented, so they are reported as warnings for the administrator.

**Virtual machines must not be blocked.** ``/proc/cpuinfo`` inside a VM is a
report by the hypervisor, not by the silicon, and hypervisors mask or rename CPU
flags all the time: nested virtualisation, a masked CPUID leaf or a hardened
guest configuration can hide ``avx2`` from a CPU that executes it perfectly
well. A false negative there would make the service refuse to run on a machine
that is fine, so in the default ``auto`` mode a missing AVX2 flag *on a detected
virtual machine* is downgraded to a warning and only the runtime decides, by
failing to load the model. On bare metal the flag is trustworthy and stays
blocking.

``IMAGEINT_HOST_CHECK`` overrides that decision:

``auto`` (default)
    Block on missing AVX2 on bare metal, warn on a virtual machine.
``strict``
    Block on missing AVX2 everywhere, VMs included.
``off``
    Never block; report everything as a warning.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("imageint.host")

#: Minimum cores of a supported endpoint. Also the default thread count of the
#: model servers, because they share the machine.
MIN_CORES = 8

#: Minimum RAM of a supported endpoint, in GB. Qwen-Image-2.1 is a 7B diffusion
#: transformer whose text encoder alone needs 17.5 GB in bf16, so 32 GB is the
#: smallest machine that can hold the int8 profile described in the README.
MIN_MEMORY_GB = 32.0

#: AVX2 is required on x86-64. ARM hosts (Apple Silicon, Graviton) use NEON
#: instead and need no check.
REQUIRES_AVX2 = True

#: AVX-512 is a bonus, never a precondition.
REQUIRES_AVX512 = False

#: How strictly the CPU check is enforced; see the module docstring.
HOST_CHECK_MODES = ("auto", "strict", "off")
DEFAULT_HOST_CHECK = "auto"

#: Steady-state footprint of each container in the **int8** profile, in GB. That
#: is the profile which fits the 32 GB reference machine, so these are the
#: numbers an operator has to plan against; they are the int8 column of the
#: table in the README and are reported verbatim by ``GET /v1/health``.
#: In bf16 the two model servers need roughly 33 GB and 19 GB instead.
BUDGET_GB = {
    "gateway": 0.4,
    "enhancer": 12.0,
    "image": 17.0,
}

#: Values above this are "unlimited" rather than a real limit.
_UNLIMITED = 1 << 50

_CGROUP_LIMITS = (
    "/sys/fs/cgroup/memory.max",  # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
)

_X86 = ("x86_64", "amd64", "i386", "i686")

#: DMI vendor/product strings that identify a hypervisor. A guest's CPU flags are
#: whatever the hypervisor chose to expose, so they cannot be trusted as proof
#: that a feature is really missing.
_HYPERVISOR_DMI = (
    "/sys/class/dmi/id/product_name",
    "/sys/class/dmi/id/sys_vendor",
    "/sys/class/dmi/id/bios_vendor",
)
_HYPERVISOR_NAMES = (
    "qemu",
    "kvm",
    "vmware",
    "virtualbox",
    "vbox",
    "xen",
    "bochs",
    "parallels",
    "bhyve",
    "innotek",
    "hyper-v",
    "microsoft corporation",
    "amazon ec2",
    "google compute engine",
    "openstack",
    "nutanix",
    "proxmox",
    "prl hyperv",
    "virtual machine",
    # Firmware and machine names that only ever appear in a guest: QEMU's
    # "Standard PC (Q35 + ICH9, 2009)", its SeaBIOS/OVMF firmware, and the
    # vendor strings of the common cloud images.
    "standard pc",
    "seabios",
    "edk ii",
    "ovmf",
    "kubevirt",
    "ahv",
    "digitalocean",
    "hetzner",
    "red hat",
    "google",
)

#: Markers that identify a container. A container shares the host kernel, but
#: Docker Desktop and Podman machine run inside a VM whose CPU model the outer
#: hypervisor may have masked, so the same caution applies.
_CONTAINER_MARKERS = ("/.dockerenv", "/run/.containerenv")
_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "lxc", "podman", "libpod")


def _host_check_mode() -> str:
    """The configured strictness, defaulting to ``auto`` on any odd value."""

    raw = os.environ.get("IMAGEINT_HOST_CHECK", "").strip().lower()
    return raw if raw in HOST_CHECK_MODES else DEFAULT_HOST_CHECK



def _container_memory_gb() -> Optional[float]:
    """Memory ceiling of this container, if one is set (cgroup v2 then v1)."""

    for path in _CGROUP_LIMITS:
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw == "" or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= _UNLIMITED:
            continue
        return value / (1024**3)
    return None


def _host_memory_gb() -> Optional[float]:
    """Physical memory of the machine the model servers run on.

    Deliberately not the cgroup limit: the gateway container is capped at a few
    hundred megabytes on purpose, while the question here is how much memory the
    two model containers may use.
    """

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    else:
        if pages > 0 and size > 0:
            return pages * size / (1024**3)

    # Docker without lxcfs reports the host total here; a fallback for the few
    # platforms where sysconf has no SC_PHYS_PAGES.
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024**2)
    except (OSError, IndexError, ValueError):
        pass
    return None


def _cpuinfo_flags() -> str:
    """The CPU flag string, padded so a substring search is unambiguous."""

    try:
        raw = Path("/proc/cpuinfo").read_text(errors="ignore")
    except OSError:
        return " "
    return f" {raw} "


def _has_feature(flag: str) -> Optional[bool]:
    """``True``/``False`` on x86-64, ``None`` when the question does not apply."""

    if platform.machine().lower() not in _X86:
        return None
    flags = _cpuinfo_flags()
    if flags.strip() == "":
        return None
    return f" {flag} " in flags


def _has_avx2() -> Optional[bool]:
    return _has_feature("avx2")


def _has_avx512() -> Optional[bool]:
    """AVX-512 via its foundation flag; ``None`` when the check does not apply."""

    return _has_feature("avx512f")


def _container_runtime() -> str:
    """Name of the container runtime, or ``""`` when not containerised."""

    for marker in _CONTAINER_MARKERS:
        if Path(marker).exists():
            return "container"
    try:
        cgroup = Path("/proc/1/cgroup").read_text(errors="ignore").lower()
    except OSError:
        cgroup = ""
    for name in _CGROUP_MARKERS:
        if name in cgroup:
            return name
    return ""


def virtualization() -> str:
    """Best-effort description of the virtualisation layer, ``""`` for bare metal.

    Used only to decide whether the CPU flags may be trusted; the value is also
    reported by :func:`inspect` so an administrator can see what was detected.
    """

    flags = _cpuinfo_flags()
    if " hypervisor " in flags:
        return "virtual machine (hypervisor-Flag)"

    for path in _HYPERVISOR_DMI:
        try:
            value = Path(path).read_text(errors="ignore").strip().lower()
        except OSError:
            continue
        if not value:
            continue
        for name in _HYPERVISOR_NAMES:
            if name in value:
                return f"virtual machine ({value})"

    runtime = _container_runtime()
    if runtime:
        return f"container ({runtime})"

    return ""


def _cpu_flags_are_trustworthy() -> bool:
    """``False`` when a missing flag may be a reporting artefact, not a fact."""

    if _host_check_mode() == "off":
        return False
    if _host_check_mode() == "strict":
        return True
    return virtualization() == ""


def blocking() -> list:
    """Reasons this endpoint cannot serve images at all, in German.

    Only features whose absence makes every request fail belong here. Today that
    is AVX2 on x86-64 and nothing else — and only where the CPU report can be
    trusted, because a hypervisor may hide a flag the CPU actually has.
    """

    found = []
    if _host_check_mode() == "off":
        return found
    if _has_avx2() is False and _cpu_flags_are_trustworthy():
        found.append(
            "Die CPU meldet kein AVX2. Die CPU-Builds von vLLM nutzen AVX2-Kernel, "
            "deshalb kann dieser Endpunkt keine Bilder erzeugen. Bitte auf einer "
            "x86-64-CPU mit AVX2 oder auf einem Host mit Nvidia-GPU (CUDA) betreiben."
        )
    return found


def warnings() -> list:
    """Deviations from the reference machine that slow the endpoint down."""

    found = []

    if _has_avx2() is False and not blocking():
        layer = virtualization() or "unbekannte Umgebung"
        if _host_check_mode() == "off":
            found.append(
                "Die CPU meldet kein AVX2, die Prüfung ist aber per "
                "IMAGEINT_HOST_CHECK=off abgeschaltet. Läuft der Dienst nicht, ist das "
                "die wahrscheinlichste Ursache."
            )
        else:
            found.append(
                f"Die CPU meldet kein AVX2, die Prüfung wird in {layer} aber nicht als "
                "Ausschlusskriterium gewertet: Hypervisoren und Container-Laufzeiten "
                "maskieren CPUID-Flags, die CPU kann AVX2 trotzdem ausführen. Schlägt "
                "der Start der Modellserver mit einem illegalen Befehl fehl, fehlt AVX2 "
                "wirklich. Mit IMAGEINT_HOST_CHECK=strict wird die Prüfung erzwungen."
            )

    cores = os.cpu_count() or 0
    if cores and cores < MIN_CORES:
        found.append(
            f"Nur {cores} CPU-Kerne verfügbar – dimensioniert ist dieser Dienst "
            f"für mindestens {MIN_CORES}."
        )

    memory_gb = _host_memory_gb()
    if memory_gb and memory_gb < MIN_MEMORY_GB:
        found.append(
            f"Nur {memory_gb:.1f} GB RAM verfügbar – dimensioniert ist dieser "
            f"Dienst für mindestens {MIN_MEMORY_GB:.0f} GB."
        )

    return found


def notes() -> list:
    """Observations that are worth reporting but never block a request."""

    found = []
    if _has_avx512() is False:
        found.append(
            "Die CPU meldet kein AVX-512. Das ist keine Voraussetzung – der Dienst "
            "läuft ohne diese Erweiterung –, aber auf Hosts mit AVX-512 ist die "
            "Prompt-Verarbeitung spürbar schneller."
        )
    elif _has_avx512() is True:
        found.append(
            "Die CPU meldet AVX-512 – die Prompt-Verarbeitung profitiert davon."
        )
    return found


def inspect() -> dict:
    """Cores, memory, AVX2 and AVX-512 of this endpoint, plus the reference."""

    cores = os.cpu_count() or 0
    memory_gb = _host_memory_gb()
    container_gb = _container_memory_gb()

    blocked = blocking()
    found = warnings()

    return {
        "cores": cores,
        "memory_gb": round(memory_gb, 1) if memory_gb else None,
        # Informational: the gateway itself is capped on purpose and that cap
        # says nothing about the machine the models run on.
        "container_memory_gb": round(container_gb, 2) if container_gb else None,
        "avx2": _has_avx2(),
        "avx512": _has_avx512(),
        "architecture": platform.machine(),
        "virtualization": virtualization(),
        "host_check": _host_check_mode(),
        "minimum": {
            "cores": MIN_CORES,
            "memory_gb": MIN_MEMORY_GB,
            # AVX2 is required, AVX-512 is not; the API reports both so a client
            # can show the administrator the whole picture.
            "avx2": REQUIRES_AVX2,
            "avx512": REQUIRES_AVX512,
        },
        "budget_gb": dict(BUDGET_GB),
        "supported": not blocked,
        "meets_minimum": not blocked and not found,
        "blocking": blocked,
        "warnings": found,
        "notes": notes(),
    }


def log_sizing() -> None:
    """Log the reference values, any deviation and any hard blocker at startup."""

    info = inspect()
    memory = f"{info['memory_gb']} GB" if info["memory_gb"] else "unbekannt"
    log.info(
        "Dimensionierung: %s Kerne, %s RAM, AVX2 %s, AVX-512 %s. Referenz: %s Kerne, %.0f GB, AVX2.",
        info["cores"],
        memory,
        {True: "ja", False: "nein", None: "entfällt"}[info["avx2"]],
        {True: "ja", False: "nein", None: "entfällt"}[info["avx512"]],
        MIN_CORES,
        MIN_MEMORY_GB,
    )
    if info["virtualization"]:
        log.info(
            "Laufzeitumgebung: %s – CPU-Flags können maskiert sein (Prüfmodus %s).",
            info["virtualization"],
            info["host_check"],
        )
    for warning in info["warnings"]:
        log.warning("Dimensionierung: %s", warning)
    for note in info["notes"]:
        log.info("Dimensionierung: %s", note)
    for blocker in info["blocking"]:
        log.error("Nicht unterstützter Host: %s", blocker)
