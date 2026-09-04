"""
phase_a.py — Phase A (pre-data / baseline state) capture.

Everything here is captured BEFORE any pipeline logic runs: what code is
under test, what environment it's running in, and exactly what synthetic
input will be fed to it. Nothing here is expensive per-cycle except
`pip freeze`-equivalent package listing, which is cached for the life of
the process (installed packages don't change mid-run) rather than shelled
out to on every single cycle — this harness may run for a long time
unattended on a memory-constrained Raspberry Pi alongside the real capture
app, so cheap-per-cycle matters.
"""

from __future__ import annotations

import functools
import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Dict, Optional

import numpy as np


@functools.lru_cache(maxsize=1)
def _installed_packages() -> Dict[str, str]:
    """{name: version} for every installed distribution — the pip-freeze
    equivalent, without shelling out to pip. Cached for the process
    lifetime."""
    packages = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary")
        if name:
            packages[name] = dist.version
    return dict(sorted(packages.items(), key=lambda kv: kv[0].lower()))


def git_commit_hash(short: bool = False) -> Optional[str]:
    """Git commit hash of the code under test. None if not a git checkout
    (e.g. a source tarball on the Pi) — never fatal."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_snapshot() -> Dict:
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _installed_packages(),
    }


def hash_input(data: bytes) -> str:
    """Content hash of exactly what the cycle will feed the pipeline —
    logged instead of the array itself so raw synthetic data never needs to
    be committed or retained to be reproducible (re-run the same seed)."""
    return hashlib.sha256(data).hexdigest()


def capture(scenario_config: Dict, seed: int, input_bytes: bytes,
            cycle_id: str) -> Dict:
    """
    Assemble the full Phase A record for one cycle.

    Parameters
    ----------
    scenario_config : the resolved test_matrix.yaml scenario for this cycle
    seed             : the deterministic seed used to (re)generate the input
    input_bytes      : the exact bytes of the generated synthetic input
    cycle_id         : this cycle's identifier (e.g. "0001")
    """
    return {
        "phase": "A",
        "cycle_id": cycle_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "environment": environment_snapshot(),
        "scenario_config": scenario_config,
        "seed": seed,
        "input_hash_sha256": hash_input(input_bytes),
    }
