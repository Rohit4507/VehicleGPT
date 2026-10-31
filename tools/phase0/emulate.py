#!/usr/bin/env python3
"""Constrain a process to a Tier-C-like CPU envelope, and say what that misses.

``taskset`` pins a process to N cores. ``cpulimit`` (or a cgroup quota) caps its
CPU share. Together they reproduce two of the three things that make a Cortex-A53
slow: **core count** and **available CPU time**.

They do not reproduce the third, and it is the one that decides LLM decode
speed: **memory bandwidth**. A laptop CPU pinned to two cores still has the
laptop's memory controller. LLM decode is bandwidth-bound, so a two-core laptop
will decode several times faster than a two-core A53 - and if that is reported
as "Tier C", the project repeats the extrapolation error it set out to fix,
in a more convincing costume.

So this tool does two things:

``--calibrate``
    Measures the host's single-thread bandwidth and compute, compares them with
    the Raspberry Pi 5 A76 anchor, and emits the correction factor needed to
    turn a host measurement into a Tier C estimate - labelled ``[D]``, because
    that is what it is.

``-- <command>``
    Runs the command under the constraint, so the *relative* comparison between
    models is honest even though the absolute numbers are not Tier C.

Usage
-----
    # what does this machine have that the target does not?
    python3 tools/phase0/emulate.py --calibrate

    # run a benchmark under a Tier-C-like envelope
    python3 tools/phase0/emulate.py --cores 2 --cpu-percent proportional \\
        -- python3 tools/phase0/bakeoff.py --models ~/models/*.gguf

The correction factor is an estimate and is labelled as one. Its job is to stop
a laptop number being quoted as a device number, not to replace the device.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

# Anchor: Raspberry Pi 5, Cortex-A76 @ 2.4 GHz, 4 threads. Qwen3-0.6B Q4_K_M is
# 414.3 MB on disk and decodes at 25.26 tok/s, which implies 10.47 GB/s of
# effective bandwidth (docs/06 §10).
PI5_DECODE_TPS = 25.26
PI5_MODEL_MB = 414.3
PI5_IMPLIED_GBS = PI5_DECODE_TPS * PI5_MODEL_MB / 1000.0   # ~10.47 GB/s
PI5_CORES = 4

C_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t n = 48u * 1024u * 1024u / sizeof(double);   /* 48 MB, way past L3 */
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; }

    /* Streaming copy: 1 read + 1 write per element. */
    double best = 1e9;
    for (int r = 0; r < 7; r++) {
        double t0 = now();
        memcpy(b, a, n * sizeof(double));
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    double bytes = 2.0 * n * sizeof(double);
    printf("bandwidth_gbs %.3f\n", bytes / best / 1e9);

    /* Single-precision-ish FMA loop: a crude but stable FLOPs proxy. */
    volatile double acc = 0.0;
    best = 1e9;
    for (int r = 0; r < 5; r++) {
        double t0 = now();
        double s = 0.0;
        for (size_t i = 0; i < 20000000u; i++) s += a[i & 4095] * 1.0000001;
        acc = s;
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    printf("scalar_mflops %.3f\n", 2.0 * 20000000u / best / 1e6);
    (void)acc;

    free(a); free(b);
    return 0;
}
"""


@dataclass
class HostProfile:
    platform: str
    machine: str
    cores_available: int
    cores_tested: int
    bandwidth_gbs: float | None
    scalar_mflops: float | None
    method: str


# ---------------------------------------------------------------------------
# Constrained execution
# ---------------------------------------------------------------------------

def available_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:                       # non-Linux
        return list(range(os.cpu_count() or 1))


def build_command(cmd: list[str], cores: int | None) -> list[str]:
    """Wrap ``cmd`` in taskset when a core count is requested."""
    if cores is None:
        return cmd
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset not found; install util-linux")
    cpus = available_cpus()[:cores]
    if len(cpus) < cores:
        raise RuntimeError(
            f"asked for {cores} cores but only {len(cpus)} are available")
    return [taskset, "-c", ",".join(str(c) for c in cpus), *cmd]


def cpu_limit_prefix(cpu_percent: float | None) -> list[str]:
    """Best available mechanism for capping CPU time.

    cpulimit if present (portable, no privileges), else systemd-run with a
    CPUQuota (cgroup v2, present on most modern Linux), else nothing - and the
    caller is told, because silently running unthrottled is worse than not
    running at all.
    """
    if cpu_percent is None:
        return []
    if cpulimit := shutil.which("cpulimit"):
        return [cpulimit, "--limit", f"{cpu_percent:.0f}", "--"]
    if (shutil.which("systemd-run") and os.environ.get("XDG_RUNTIME_DIR")
            and _has_cgroup_cpu()):
        return [shutil.which("systemd-run"), "--user", "--scope", "--quiet",
                "-p", f"CPUQuota={cpu_percent:.0f}%", "--"]
    raise RuntimeError(
        "cannot cap CPU time: neither cpulimit nor a usable cgroup v2 cpu "
        "controller is available. Install cpulimit, or drop --cpu-percent and "
        "rely on --cores alone (which does NOT cap CPU time).")


def _has_cgroup_cpu() -> bool:
    try:
        ctl = Path("/sys/fs/cgroup/cgroup.controllers").read_text().split()
        return "cpu" in ctl
    except OSError:
        return False


def run_constrained(cmd: list[str], cores: int | None,
                    cpu_percent: float | None) -> int:
    full = build_command(cpu_limit_prefix(cpu_percent) + list(cmd), cores)
    print(f"running: {' '.join(full)}", file=sys.stderr)
    return subprocess.call(full)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _compile_probe() -> Path | None:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return None
    out = Path(tempfile.mkdtemp()) / "membench"
    src = out.with_suffix(".c")
    src.write_text(C_SOURCE)
    proc = subprocess.run([cc, "-O2", "-o", str(out), str(src)],
                          capture_output=True, text=True)
    return out if proc.returncode == 0 else None


def measure(cores: int) -> tuple[float | None, float | None, str]:
    """Return (bandwidth GB/s, scalar MFLOP/s, method) on ``cores`` cores."""
    probe = _compile_probe()
    if probe is not None:
        cmd = build_command([str(probe)], cores)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        bw = mf = None
        for line in proc.stdout.splitlines():
            k, _, v = line.partition(" ")
            if k == "bandwidth_gbs":
                bw = float(v)
            elif k == "scalar_mflops":
                mf = float(v)
        return bw, mf, "compiled C probe"
    return None, None, "unavailable (no C compiler)"


def profile(cores: int | None = None) -> HostProfile:
    cpus = available_cpus()
    n = cores or len(cpus)
    bw, mf, method = measure(n)
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return HostProfile(
        platform=platform.platform(),
        machine=cpu or platform.machine(),
        cores_available=len(cpus), cores_tested=n,
        bandwidth_gbs=None if bw is None else round(bw, 3),
        scalar_mflops=None if mf is None else round(mf, 1),
        method=method,
    )


def tier_c_correction(host: HostProfile) -> dict:
    """Estimate how much faster this host is than Tier C, and why.

    Decode is bandwidth-bound: tokens/second scales with bandwidth divided by
    model size. So the ratio of measured host bandwidth to the Pi 5 anchor's
    implied bandwidth is a defensible first-order correction - and it is still
    an estimate, so it is tagged ``[D]``.
    """
    if host.bandwidth_gbs is None:
        return {"available": False, "reason": host.method,
                "status": "[T]",
                "note": "install a C compiler to enable calibration"}

    bw_ratio = host.bandwidth_gbs / PI5_IMPLIED_GBS
    implied_decode = host.bandwidth_gbs * 1000.0 / PI5_MODEL_MB
    # Core count affects prefill (compute-bound) more than decode.
    core_ratio = host.cores_tested / PI5_CORES

    return {
        "available": True,
        "status": "[D]",
        "host_bandwidth_gbs": host.bandwidth_gbs,
        "pi5_implied_bandwidth_gbs": round(PI5_IMPLIED_GBS, 2),
        "bandwidth_ratio_vs_tier_c": round(bw_ratio, 2),
        "core_ratio_vs_tier_c": round(core_ratio, 2),
        "implied_decode_tps_this_host": round(implied_decode, 1),
        "pi5_decode_tps": PI5_DECODE_TPS,
        "to_estimate_tier_c": (
            f"divide a decode rate measured on this host by {bw_ratio:.2f}, "
            f"and a prefill rate by roughly {core_ratio:.2f} x the per-core "
            "ratio. That is still an estimate, not a measurement."),
        "warning": ("core pinning does not reduce memory bandwidth; this host "
                    "is not a Tier C device and a result from it must not be "
                    "reported as one"),
    }


def print_calibration(host: HostProfile, correction: dict) -> None:
    W = 78
    print("=" * W)
    print("Tier C emulation calibration".center(W))
    print("=" * W)
    print(f"\nhost        {host.machine}")
    print(f"platform    {host.platform}")
    print(f"cores       {host.cores_tested} of {host.cores_available} available")
    print(f"measured    {host.method}")
    if correction.get("available"):
        print(f"\nbandwidth   {host.bandwidth_gbs:.2f} GB/s single-stream")
        print(f"scalar      {host.scalar_mflops:.0f} MFLOP/s")
        print(f"\nTier C anchor (RPi 5 A76): {PI5_DECODE_TPS} tok/s on a "
              f"{PI5_MODEL_MB:.0f} MB model")
        print(f"  implied bandwidth         {correction['pi5_implied_bandwidth_gbs']} GB/s")
        print("\nthis host vs Tier C")
        print(f"  bandwidth ratio           {correction['bandwidth_ratio_vs_tier_c']}x")
        print(f"  core ratio               {correction['core_ratio_vs_tier_c']}x")
        print(f"  implied decode (0.6B Q4)  {correction['implied_decode_tps_this_host']} tok/s"
              f"   {correction['status']}")
        print(f"\n  {correction['to_estimate_tier_c']}")
        print(f"\n! {correction['warning']}")
    else:
        print(f"\ncalibration unavailable: {correction['reason']}")
        print("constraint still applies, but the correction factor is unknown")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true",
                    help="measure this host and report the Tier C correction")
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--cpu-percent", type=float, default=None,
                    help="cap CPU time via cpulimit or a cgroup quota")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.calibrate or not args.command:
        host = profile(args.cores)
        correction = tier_c_correction(host)
        if args.json:
            print(json.dumps({"host": asdict(host), "tier_c": correction}, indent=2))
        else:
            print_calibration(host, correction)
        return 0

    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given after --")
    try:
        return run_constrained(cmd, args.cores, args.cpu_percent)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
