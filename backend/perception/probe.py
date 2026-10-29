"""Bounded, camera-free real SDK probe; native crashes cannot kill the reporter.

The child loads the explicit asset and infers three black RGB frames in VIDEO
mode. This verifies model operation, NOT real-hand accuracy or camera latency.
"""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
from time import monotonic, sleep


_LOG_LIMIT = 1024 * 1024
_LOG_TAIL = 32 * 1024


def linux_memory(pid):
    """Best-effort sampled RSS, not an OS-enforced allocation limit."""
    if sys.platform != "linux":
        return None
    try:
        fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
        return int(fields["VmRSS"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        return None


def cgroup_memory():
    """Current cgroup v2 and ancestors, without user/session path identifiers."""
    if sys.platform != "linux":
        return []
    try:
        entry = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
        root = Path("/sys/fs/cgroup")
        current = (root / entry.lstrip("/")).resolve()
        snapshots = []
        while current.is_relative_to(root):
            row = {"ancestor_level": len(snapshots)}
            for name in ("memory.max", "memory.high", "memory.current", "memory.events"):
                try:
                    row[name] = (current / name).read_text().strip()
                except OSError:
                    pass
            snapshots.append(row)
            current = current.parent
        return snapshots
    except (OSError, StopIteration):
        return []


def _kill_child(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)  # only the newly created child group
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _validate_limits(timeout, rss_limit_mib):
    for value in (timeout, rss_limit_mib):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("probe timeout and RSS limit must be finite and positive")


def run_isolated(command, *, timeout=30., rss_limit_mib=768., env=None):
    """Run one native probe, retaining failures, partial logs and monitor reasons.

    RSS polling is Linux-only and can miss short peaks; timeout works on all
    platforms. No retries, fallback detector, camera access or model download.
    """
    _validate_limits(timeout, rss_limit_mib)
    before = cgroup_memory()
    started = monotonic()
    reason = None
    peak_rss = None
    samples = 0
    returncode = None
    process = None
    launch_error = None
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                       start_new_session=os.name == "posix")
            while process.poll() is None:
                rss = linux_memory(process.pid)
                if rss is not None:
                    samples += 1
                    peak_rss = max(peak_rss or 0, rss)
                if rss is not None and rss > rss_limit_mib * 1024 * 1024:
                    reason = "rss_limit"
                elif monotonic() - started >= timeout:
                    reason = "timeout"
                elif max(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size) > _LOG_LIMIT:
                    reason = "log_limit"
                if reason:
                    break
                sleep(.02)
        except KeyboardInterrupt:
            reason = "cancelled"
        except OSError as error:
            launch_error = f"{type(error).__name__}: {error}"
        finally:
            if process is not None:
                _kill_child(process)
                try:
                    returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    reason = "shutdown_timeout"
        logs = {}
        for name, stream in (("stdout", stdout), ("stderr", stderr)):
            size = os.fstat(stream.fileno()).st_size
            stream.seek(max(0, size - _LOG_TAIL))
            logs[name] = stream.read(_LOG_TAIL).decode("utf-8", errors="replace")
            logs[f"{name}_truncated"] = size > _LOG_TAIL
    signal_name = None
    if returncode is not None and returncode < 0 and os.name == "posix":
        try:
            signal_name = signal.Signals(-returncode).name
        except ValueError:
            signal_name = f"signal_{-returncode}"
    return {
        "status": reason or ("passed" if returncode == 0 else "failed"),
        "returncode": returncode, "signal": signal_name, "monitor_reason": reason,
        "launch_error": launch_error, "elapsed_seconds": monotonic() - started,
        "timeout_seconds": timeout, "rss_limit_mib": rss_limit_mib,
        "rss_monitor_supported": sys.platform == "linux", "rss_samples": samples,
        "sampled_peak_rss_bytes": peak_rss,
        "cgroup_before": before, "cgroup_after": cgroup_memory(), **logs,
    }


def _stage(name, **details):
    print(json.dumps({"probe_stage": name, **details}, allow_nan=False), flush=True)


def _worker(args):
    import faulthandler
    faulthandler.enable()
    _stage("adapter_import")
    from backend.perception.mediapipe_adapter import MediaPipeDetector
    import numpy as np
    _stage("model_load")
    started = monotonic()
    detector = MediaPipeDetector(args.model, expected_sha256=args.model_sha256)
    _stage("model_ready", load_seconds=monotonic() - started, perception=detector.metadata)
    try:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for index in range(3):
            started = monotonic()
            hands = detector.detect(image, index * 34)
            _stage("inference", frame_index=index, timestamp_ms=index * 34,
                   hand_count=len(hands), inference_seconds=monotonic() - started)
    finally:
        _stage("model_close")
        detector.close()
    _stage("complete", frames=3, camera_accessed=False, pixels_saved=False, landmarks_saved=False)
    return 0


def probe(model, *, model_sha256=None, timeout=30., rss_limit_mib=768.):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [sys.executable, "-u", "-m", "backend.perception.probe", "--_worker", "--model", str(model)]
    if model_sha256 is not None:
        command += ["--model-sha256", model_sha256]
    result = run_isolated(command, timeout=timeout, rss_limit_mib=rss_limit_mib, env=env)
    stages = []
    for line in result["stdout"].splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and "probe_stage" in event:
            stages.append(event)
    if result["status"] == "passed" and (not stages or stages[-1].get("probe_stage") != "complete"):
        result["status"] = "failed"
        result["protocol_error"] = "child exited without completing model inference and close"
    versions = {}
    for name in ("mediapipe", "numpy", "opencv-contrib-python", "protobuf", "certifi"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema": "perception-probe-v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": "real_sdk_three_black_RGB_frames_no_camera_no_recording_no_actions",
        "real_hand_validated": False, "python": platform.python_version(),
        "os": platform.system(), "kernel": platform.release(), "architecture": platform.machine(),
        "versions": versions, "thread_environment": {key: env[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "expected_asset_sha256": model_sha256, "stages": stages, **result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256")
    parser.add_argument("--timeout", type=float, default=30.)
    parser.add_argument("--rss-limit-mib", type=float, default=768.)
    parser.add_argument("--report", type=Path, help="exclusive JSON metadata/log output, never camera pixels")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return _worker(args)
    try:
        _validate_limits(args.timeout, args.rss_limit_mib)
        # Reserve evidence output BEFORE starting any native work; refuse overwrite.
        # Even abrupt native child termination leaves the parent able to report.
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") if args.report else tempfile.TemporaryFile(mode="w+") as output:
            result = probe(args.model, model_sha256=args.model_sha256,
                           timeout=args.timeout, rss_limit_mib=args.rss_limit_mib)
            text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            output.write(text)
        print(text, end="")
        return 0 if result["status"] == "passed" else 2
    except (OSError, ValueError) as error:
        print(f"perception probe error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
