#!/usr/bin/env python3
"""
scheduler.py — runs test cycles continuously or on a defined cadence.

Continuous mode round-robins the test matrix indefinitely, sleeping between
cycles. Meant to run in the background on the Pi/dev machine during active
development — start it with
`python -m testing.continuous.scheduler --mode continuous`.

Memory discipline (see the approved plan's "Hardware constraints" section):
this may run unattended for a long time on a Raspberry Pi that also runs
the real capture app, so it checks its own process RSS every cycle via
resource.getrusage and backs off (a longer sleep) once it crosses
`--rss-limit-mb`, rather than free-running as fast as it can regardless of
memory pressure.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

from testing.continuous import run_cycle

DEFAULT_CYCLE_INTERVAL_S = 30.0
DEFAULT_RSS_LIMIT_MB = 512
BACKOFF_MULTIPLIER = 4.0


def _current_rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS — normalize by platform.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024 if sys.platform != "darwin" else raw / (1024 * 1024)


def run_continuous(interval_s: float, rss_limit_mb: float) -> None:
    print(f"scheduler: continuous mode — interval={interval_s}s, "
          f"rss_limit={rss_limit_mb}MB", flush=True)
    while True:
        try:
            summary = run_cycle.run_one_cycle()
            print(f"scheduler: cycle {summary['cycle_id']} "
                  f"{'PASS' if summary['passed'] else 'FAIL'}", flush=True)
        except Exception as e:
            print(f"scheduler: cycle raised {e!r} — continuing", flush=True)

        rss_mb = _current_rss_mb()
        sleep_s = interval_s
        if rss_mb > rss_limit_mb:
            sleep_s = interval_s * BACKOFF_MULTIPLIER
            print(f"scheduler: RSS {rss_mb:.0f}MB over limit "
                  f"{rss_limit_mb:.0f}MB — backing off to {sleep_s:.0f}s", flush=True)
        time.sleep(sleep_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["continuous"], required=True)
    parser.add_argument("--interval", type=float, default=DEFAULT_CYCLE_INTERVAL_S,
                        help="Seconds between cycles in continuous mode.")
    parser.add_argument("--rss-limit-mb", type=float, default=DEFAULT_RSS_LIMIT_MB,
                        help="Back off cycle cadence above this process RSS.")
    args = parser.parse_args()

    run_continuous(args.interval, args.rss_limit_mb)
    return 0  # unreachable — continuous mode runs until killed


if __name__ == "__main__":
    sys.exit(main())
