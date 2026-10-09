"""Host sizing, with the virtual-machine case as a first-class requirement.

The AVX2 flag is a hard requirement on bare metal, but a hypervisor or a
container runtime may simply not report a flag the CPU has. These tests pin both
halves: a VM must never be blocked, a bare-metal machine without AVX2 must be.
"""

from __future__ import annotations

import os
import platform

import pytest

from app import host


@pytest.fixture
def as_x86(monkeypatch):
    """Pretend to be x86-64 so the AVX checks apply at all."""

    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host.platform, "machine", lambda: "x86_64")


def flags(monkeypatch, value: str) -> None:
    monkeypatch.setattr(host, "_cpuinfo_flags", lambda: f" {value} ")


@pytest.fixture(autouse=True)
def clean_mode(monkeypatch):
    monkeypatch.delenv("IMAGEINT_HOST_CHECK", raising=False)


# --------------------------------------------------------------------------- #
# Feature detection
# --------------------------------------------------------------------------- #
def test_avx2_and_avx512_are_detected(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2 avx avx2 avx512f")
    assert host._has_avx2() is True
    assert host._has_avx512() is True


def test_avx512_absence_is_not_the_same_as_avx2_absence(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2 avx avx2")
    assert host._has_avx2() is True
    assert host._has_avx512() is False


def test_flag_search_does_not_match_substrings(as_x86, monkeypatch):
    # `avx512f` must not be found inside `avx512fp16`, and `avx2` not inside
    # `avx2x`; the padded substring search is what prevents that.
    flags(monkeypatch, "avx512fp16 avx2x")
    assert host._has_avx2() is False
    assert host._has_avx512() is False


def test_non_x86_skips_the_cpu_check(monkeypatch):
    monkeypatch.setattr(host.platform, "machine", lambda: "arm64")
    assert host._has_avx2() is None
    assert host._has_avx512() is None
    assert host.blocking() == []


def test_unreadable_cpuinfo_is_not_a_blocker(as_x86, monkeypatch):
    monkeypatch.setattr(host, "_cpuinfo_flags", lambda: " ")
    assert host._has_avx2() is None
    assert host.blocking() == []


# --------------------------------------------------------------------------- #
# Bare metal: AVX2 is a hard stop
# --------------------------------------------------------------------------- #
def test_bare_metal_without_avx2_is_blocked(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2 avx")
    monkeypatch.setattr(host, "virtualization", lambda: "")

    info = host.inspect()
    assert info["avx2"] is False
    assert info["supported"] is False
    assert info["meets_minimum"] is False
    assert len(info["blocking"]) == 1
    assert "AVX2" in info["blocking"][0]


def test_bare_metal_with_avx2_has_no_blocker(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2 avx avx2")
    monkeypatch.setattr(host, "virtualization", lambda: "")
    assert host.blocking() == []
    assert host.inspect()["supported"] is True


# --------------------------------------------------------------------------- #
# Virtual machines must never be blocked
# --------------------------------------------------------------------------- #
def test_vm_without_avx2_is_not_blocked(as_x86, monkeypatch):
    """A hypervisor that masks AVX2 must not stop the service from running."""

    flags(monkeypatch, "fpu sse2 hypervisor")
    monkeypatch.setattr(host, "virtualization", lambda: "virtual machine (qemu)")

    info = host.inspect()
    assert info["avx2"] is False
    assert info["blocking"] == []
    assert info["supported"] is True
    # Reported, but as something the administrator should know rather than as a
    # reason to refuse service.
    assert any("AVX2" in warning for warning in info["warnings"])


def test_vm_is_recognised_through_the_hypervisor_flag(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2 hypervisor")
    assert "hypervisor" in host.virtualization()


def test_vm_is_recognised_through_dmi(as_x86, monkeypatch, tmp_path):
    flags(monkeypatch, "fpu sse2")
    product = tmp_path / "product_name"
    product.write_text("Standard PC (Q35 + ICH9, 2009)\n")
    monkeypatch.setattr(host, "_HYPERVISOR_DMI", (str(product),))
    assert host.virtualization() != ""


def test_vm_is_recognised_through_the_dmi_vendor(as_x86, monkeypatch, tmp_path):
    flags(monkeypatch, "fpu sse2")
    vendor = tmp_path / "sys_vendor"
    vendor.write_text("QEMU\n")
    monkeypatch.setattr(host, "_HYPERVISOR_DMI", (str(vendor),))
    assert "qemu" in host.virtualization()


def test_bare_metal_dmi_is_not_a_vm(as_x86, monkeypatch, tmp_path):
    flags(monkeypatch, "fpu sse2")
    vendor = tmp_path / "sys_vendor"
    vendor.write_text("ASUSTeK COMPUTER INC.\n")
    monkeypatch.setattr(host, "_HYPERVISOR_DMI", (str(vendor),))
    monkeypatch.setattr(host, "_container_runtime", lambda: "")
    assert host.virtualization() == ""


def test_container_is_recognised(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2")
    monkeypatch.setattr(host, "_HYPERVISOR_DMI", ())
    monkeypatch.setattr(host, "_container_runtime", lambda: "docker")
    assert "container" in host.virtualization()
    # Docker Desktop and Podman machine run inside a VM, so a masked flag here
    # is not proof of anything either.
    assert host._cpu_flags_are_trustworthy() is False


def test_strict_mode_blocks_on_a_vm(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2 hypervisor")
    monkeypatch.setenv("IMAGEINT_HOST_CHECK", "strict")

    info = host.inspect()
    assert info["host_check"] == "strict"
    assert info["supported"] is False
    assert len(info["blocking"]) == 1


def test_off_mode_never_blocks(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2")
    monkeypatch.setenv("IMAGEINT_HOST_CHECK", "off")
    monkeypatch.setattr(host, "virtualization", lambda: "")

    info = host.inspect()
    assert info["blocking"] == []
    assert info["supported"] is True
    assert any("abgeschaltet" in warning for warning in info["warnings"])


def test_unknown_mode_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("IMAGEINT_HOST_CHECK", "vielleicht")
    assert host._host_check_mode() == "auto"


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #
def test_missing_cores_and_memory_are_warnings_not_blockers(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    monkeypatch.setattr(host, "_host_memory_gb", lambda: 4.0)

    info = host.inspect()
    assert info["blocking"] == []
    assert info["supported"] is True
    assert info["meets_minimum"] is False
    assert len(info["warnings"]) >= 2


def test_reference_machine_meets_the_minimum(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2 avx avx2")
    monkeypatch.setattr(host, "virtualization", lambda: "")
    monkeypatch.setattr(os, "cpu_count", lambda: host.MIN_CORES)
    monkeypatch.setattr(host, "_host_memory_gb", lambda: host.MIN_MEMORY_GB)

    info = host.inspect()
    assert info["meets_minimum"] is True
    assert info["blocking"] == []
    assert info["warnings"] == []


def test_avx512_is_reported_in_the_minimum_block(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2 avx avx2")
    minimum = host.inspect()["minimum"]
    assert minimum["cores"] == 8
    assert minimum["memory_gb"] == 32.0
    assert minimum["avx2"] is True
    # Documented as not required, so the API can show exactly that.
    assert minimum["avx512"] is False


def test_avx512_note_is_informational(as_x86, monkeypatch):
    flags(monkeypatch, "fpu sse2 avx avx2")
    assert any("AVX-512" in note for note in host.notes())

    flags(monkeypatch, "fpu sse2 avx avx2 avx512f")
    assert any("AVX-512" in note for note in host.notes())
    assert host.blocking() == []


def test_log_sizing_runs_without_error(caplog):
    host.log_sizing()
