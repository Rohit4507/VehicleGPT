"""Emulation tool tests.

The point of these is that core pinning is *not* environment emulation, and the
tool must say so rather than let a laptop measurement be relabelled as Tier C.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "phase0"))

import emulate  # noqa: E402


class TestCommandBuilding:
    def test_no_cores_leaves_command_untouched(self):
        assert emulate.build_command(["echo", "hi"], None) == ["echo", "hi"]

    def test_pins_to_the_requested_cores(self):
        cpus = emulate.available_cpus()
        cmd = emulate.build_command(["bench"], cores=min(2, len(cpus)))
        assert cmd[0].endswith("taskset")
        assert cmd[1] == "-c"
        assert cmd[2] == ",".join(str(c) for c in cpus[: min(2, len(cpus))])
        assert cmd[-1] == "bench"

    def test_refuses_more_cores_than_exist(self):
        with pytest.raises(RuntimeError, match="only .* are available"):
            emulate.build_command(["x"], cores=len(emulate.available_cpus()) + 4)

    def test_available_cpus_is_non_empty(self):
        assert emulate.available_cpus()


class TestCpuLimiting:
    def test_no_limit_means_no_prefix(self):
        assert emulate.cpu_limit_prefix(None) == []

    def test_raises_when_no_mechanism_exists(self, monkeypatch):
        monkeypatch.setattr(emulate.shutil, "which", lambda _: None)
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        with pytest.raises(RuntimeError, match="cannot cap CPU time"):
            emulate.cpu_limit_prefix(50)

    def test_uses_cpulimit_when_present(self, monkeypatch):
        monkeypatch.setattr(emulate.shutil, "which",
                            lambda name: "/usr/bin/cpulimit" if name == "cpulimit" else None)
        prefix = emulate.cpu_limit_prefix(50)
        assert prefix[0].endswith("cpulimit") and "--limit" in prefix


class TestCorrection:
    def host(self, bw):
        return emulate.HostProfile(
            platform="test", machine="test", cores_available=8, cores_tested=2,
            bandwidth_gbs=bw, scalar_mflops=1000.0, method="test")

    def test_ratio_uses_the_bandwidth_anchor(self):
        host = self.host(emulate.PI5_IMPLIED_GBS)   # exactly Tier C bandwidth
        c = emulate.tier_c_correction(host)
        assert c["bandwidth_ratio_vs_tier_c"] == pytest.approx(1.0, abs=0.01)
        assert c["implied_decode_tps_this_host"] == pytest.approx(
            emulate.PI5_DECODE_TPS, rel=0.02)

    def test_a_faster_host_is_flagged_as_faster(self):
        c = emulate.tier_c_correction(self.host(emulate.PI5_IMPLIED_GBS * 3))
        assert c["bandwidth_ratio_vs_tier_c"] == pytest.approx(3.0, abs=0.02)
        assert "not a Tier C device" in c["warning"]

    def test_correction_is_labelled_derived_not_measured(self):
        c = emulate.tier_c_correction(self.host(20.0))
        assert c["status"] == "[D]"

    def test_unavailable_without_a_measurement(self):
        c = emulate.tier_c_correction(self.host(None))
        assert c["available"] is False and c["status"] == "[T]"

    def test_core_ratio_reported_separately_from_bandwidth(self):
        """Core count and bandwidth are different axes; conflating them is the bug."""
        c = emulate.tier_c_correction(self.host(emulate.PI5_IMPLIED_GBS))
        assert c["core_ratio_vs_tier_c"] == pytest.approx(0.5)
        assert c["bandwidth_ratio_vs_tier_c"] == pytest.approx(1.0, abs=0.01)
