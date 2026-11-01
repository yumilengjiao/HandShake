"""Explicit non-training, application-local lifecycle/fault soak with real wall-time metrics."""

import argparse
import json
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

from algorithm.bundle import digest, json_bytes
from backend.interaction.public_presentation import SyntheticPublicPresentation
from backend.interaction.public_replay import (
    ReplayClock,
    acknowledge,
    authored_steps,
    synthetic_calibration,
    synthetic_input,
)
from shared.observations import ObservationFrame


def process_sample():
    rss = None
    path = Path("/proc/self/statm")
    if path.exists():
        rss = int(path.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return {"rss_bytes": rss, "python_threads": threading.active_count()}


class Latencies:
    bounds = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

    def __init__(self):
        self.buckets = [0] * (len(self.bounds) + 1)
        self.count = self.total = self.maximum = 0

    def add(self, start):
        ns = time.monotonic_ns() - start
        self.count += 1
        self.total += ns
        self.maximum = max(self.maximum, ns)
        index = next(
            (i for i, bound in enumerate(self.bounds) if ns <= bound * 1_000_000),
            len(self.bounds),
        )
        self.buckets[index] += 1

    def report(self):
        accumulated = 0
        upper = None
        for i, count in enumerate(self.buckets):
            accumulated += count
            if self.count and accumulated >= self.count * 0.95:
                upper = self.bounds[i] if i < len(self.bounds) else None
                break
        return {
            "count": self.count,
            "mean_ms": self.total / self.count / 1e6 if self.count else None,
            "max_ms": self.maximum / 1e6,
            "p95_bucket_upper_ms": upper,
            "bucket_bounds_ms": self.bounds,
            "bucket_counts": self.buckets,
            "scope": "all samples, fixed histogram; not exact percentile",
        }


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def run_soak(
    output,
    *,
    seconds=None,
    cycles=None,
    allow_synthetic=False,
    max_rss_mb=256,
    max_growth_mb=128,
    pace_ms=25,
    code_revision=None,
):
    if allow_synthetic is not True:
        raise ValueError("soak requires explicit --allow-synthetic")
    if (seconds is None) == (cycles is None):
        raise ValueError("choose exactly one duration or cycle limit")
    if seconds is not None and (type(seconds) is not int or not 1 <= seconds <= 14400):
        raise ValueError("duration must be 1..14400 seconds")
    if cycles is not None and (type(cycles) is not int or not 1 <= cycles <= 100000):
        raise ValueError("cycles must be 1..100000")
    if (
        any(type(v) is not int for v in (max_rss_mb, max_growth_mb, pace_ms))
        or not 64 <= max_rss_mb <= 4096
        or not 1 <= max_growth_mb <= 1024
        or not 0 <= pace_ms <= 1000
    ):
        raise ValueError("invalid memory/pacing budget")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    calibration, sha = synthetic_calibration(root)
    baseline = process_sample()
    clock = ReplayClock()
    started = time.monotonic()
    wall_started = datetime.now().astimezone().isoformat()
    counts, samples = Counter(), deque(maxlen=360)
    stop_times, watchdog_times = Latencies(), Latencies()
    host = None
    frame_id = 0
    peak_rss = baseline["rss_bytes"] or 0
    peak_threads = baseline["python_threads"]
    last_progress = started
    status, failure = "completed", None

    def frame(token="NOG", x=0.5, ack=True):
        nonlocal frame_id
        clock.now += 50_000_000
        frame_id += 1
        prediction, value = synthetic_input(
            host.controller.snapshot(),
            timestamp_ns=clock.now,
            frame_id=frame_id,
            model_version=host.engine.model_version,
            token=token,
            x=x,
        )
        event, ticket = host.update(prediction, value, ready=True)
        if ticket is not None:
            result = ticket.future.result(timeout=2)
            require(result.accepted, "expected local command rejected")
            # The backend's result, not a GUI acknowledgement, is the authoritative page transition.
            expected = (
                min(4, result.page_before + 1)
                if event.action == "next_slide"
                else max(1, result.page_before - 1)
            )
            require(
                result.page_after == expected and host.backend.page == expected,
                "wrong page transition",
            )
            counts["application_commands"] += 1
        if ack:
            require(acknowledge(host), "surface acknowledgement rejected")
        return event, ticket

    def snapshot_report():
        return {
            "format_version": "local-safety-soak-v1",
            "status": status,
            "failure": failure,
            "synthetic": True,
            "scope": "application-local procedural logits/geometry, no camera/model training/OS input",
            "wall_started": wall_started,
            "wall_updated": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "requested_seconds": seconds,
            "requested_cycles": cycles,
            "counts": dict(counts),
            "stop_and_release": stop_times.report(),
            "watchdog_observation": watchdog_times.report(),
            "baseline": baseline,
            "peak_rss_bytes": peak_rss,
            "peak_python_threads": peak_threads,
            "resource_peak_scope": "maximum of per-cycle samples, not a continuous OS high-water mark",
            "samples": list(samples),
            "sample_scope": "last 360 process samples; maxima cover all cycle-end samples",
            "max_rss_mb": max_rss_mb,
            "max_growth_mb": max_growth_mb,
            "pace_ms": pace_ms,
            "code_revision": code_revision,
            "implementation_sha256": digest(Path(__file__).read_bytes()),
            "calibration_sha256": sha,
            "live_available": False,
            "external_control_available": False,
            "limitations": [
                "repeated synthetic software scenarios are not recognition accuracy or real-user false-trigger evidence",
                "surface receipts are simulated host acknowledgements, not Flutter rendering evidence",
                "source time is virtual; stop/release and watchdog observation durations are measured with monotonic wall time",
                "hosts rotate every 32 cycles to test cleanup; dedup capacity/continuous lifetime beyond that is not certified",
            ],
        }

    try:
        while (cycles is None or counts["cycles"] < cycles) and (
            seconds is None or time.monotonic() - started < seconds
        ):
            if host is None or counts["cycles"] % 32 == 0:
                if host is not None:
                    host.close()
                    require(host.dispatcher.wait_released(), "rotation release failed")
                host = SyntheticPublicPresentation(
                    calibration=calibration,
                    calibration_sha256=sha,
                    clock=clock,
                    allow_synthetic=True,
                    watchdog=True,
                )
                counts["host_instances"] += 1
            require(acknowledge(host) and host.enable(), "explicit enable failed")
            clock.now += 50_000_000
            frame_id += 1
            p, value = synthetic_input(
                host.controller.snapshot(),
                timestamp_ns=clock.now,
                frame_id=frame_id,
                model_version=host.engine.model_version,
            )
            require(
                host.arm_owner("operator", p, value, ready=True) and acknowledge(host),
                "explicit owner failed",
            )
            committed = 0
            for token, x, _ in authored_steps():
                event, ticket = frame(token, x)
                if ticket is not None:
                    committed += 1
                    old = host.commands[-1]
                    duplicate = old.model_copy(
                        update={"command_id": "duplicate:" + old.command_id}
                    )
                    result = host.dispatcher.submit(duplicate).future.result(timeout=2)
                    require(
                        not result.accepted and result.error_code == "duplicate_event",
                        "dedup failed",
                    )
                    counts["duplicates_rejected"] += 1
            require(
                committed == 3, "held/return movement retriggered or command missed"
            )
            scenario = counts["cycles"] % 3
            if scenario == 1:
                began = time.monotonic_ns()
                clock.now += host.engine.policy.max_age_ns + 1
                deadline = time.monotonic() + 2
                while not host.dispatcher.stopped and time.monotonic() < deadline:
                    time.sleep(0.001)
                require(host.dispatcher.stopped, "independent watchdog did not stop")
                watchdog_times.add(began)
                counts["source_stall_faults"] += 1
            elif scenario == 2:
                clock.now += 50_000_000
                frame_id += 1
                p, value = synthetic_input(
                    host.controller.snapshot(),
                    timestamp_ns=clock.now,
                    frame_id=frame_id,
                    model_version=host.engine.model_version,
                )
                second = value.observations[0].model_copy(update={"track_id": "second"})
                conflict = ObservationFrame(
                    value.info, value.status, value.observations + (second,)
                )
                before = host.backend.page
                require(
                    host.update(p, conflict, ready=True) == (None, None),
                    "conflict emitted an event",
                )
                require(
                    host.dispatcher.stopped and host.backend.page == before,
                    "conflict did not revoke authority",
                )
                counts["ownership_faults"] += 1
            began = time.monotonic_ns()
            host.stop("soak_cycle_stop")
            require(host.dispatcher.wait_released(), "stop release timed out")
            require(
                host.dispatcher.safety_status()["safe_quiescent"],
                "not safely quiescent after stop",
            )
            stop_times.add(began)
            counts["cycles"] += 1
            current = process_sample()
            peak_rss = max(peak_rss, current["rss_bytes"] or 0)
            peak_threads = max(peak_threads, current["python_threads"])
            require(
                current["python_threads"] <= baseline["python_threads"] + 8,
                "Python thread leak/budget exceeded",
            )
            if current["rss_bytes"] is not None:
                require(
                    current["rss_bytes"] <= max_rss_mb * 1024**2,
                    "absolute RSS budget exceeded",
                )
                require(
                    current["rss_bytes"]
                    - (baseline["rss_bytes"] or current["rss_bytes"])
                    <= max_growth_mb * 1024**2,
                    "RSS growth budget exceeded",
                )
            if time.monotonic() - last_progress >= 5:
                samples.append(
                    {"elapsed_seconds": time.monotonic() - started, **current}
                )
                (root / "progress.tmp").write_bytes(
                    json_bytes(snapshot_report() | {"status": "running"})
                )
                os.replace(root / "progress.tmp", root / "progress.json")
                last_progress = time.monotonic()
            if pace_ms:
                time.sleep(pace_ms / 1000)
    except KeyboardInterrupt:
        status, failure = "interrupted", "KeyboardInterrupt"
    except Exception as error:
        status, failure = "failed", type(error).__name__ + ": " + str(error)[:240]
    finally:
        if host is not None:
            host.close()
            if not host.dispatcher.wait_released():
                status, failure = "failed", "final release not confirmed"
    final = snapshot_report()
    final["complete"] = status == "completed"
    raw = json_bytes(final)
    (root / "report.json").write_bytes(raw)
    (root / "receipt.json").write_bytes(
        json_bytes({"status": status, "report_sha256": digest(raw), "synthetic": True})
    )
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seconds", type=int)
    group.add_argument("--cycles", type=int)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--max-rss-mb", type=int, default=256)
    parser.add_argument("--max-growth-mb", type=int, default=128)
    parser.add_argument("--pace-ms", type=int, default=25)
    parser.add_argument("--code-revision")
    args = parser.parse_args(argv)
    try:
        result = run_soak(**vars(args))
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "elapsed_seconds",
                        "counts",
                        "peak_rss_bytes",
                        "peak_python_threads",
                        "failure",
                    )
                },
                indent=2,
            )
        )
        return (
            0 if result["complete"] else 130 if result["status"] == "interrupted" else 2
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"soak rejected: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
