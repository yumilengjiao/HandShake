"""Isolated local SDK probe: no camera, no retries, no package/environment changes."""
import argparse
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


def resource_snapshot():
    result = dict(python=platform.python_version(), executable=sys.executable, system=platform.system(),
                  machine=platform.machine(), packages={})
    for name in ("mediapipe", "numpy", "opencv-contrib-python"):
        try:
            result["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result["packages"][name] = None
    for path in (Path("/proc/self/cgroup"), Path("/proc/meminfo")):
        if path.is_file():
            result[path.name] = path.read_text()
    try:
        relative = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
        current = Path("/sys/fs/cgroup") / relative.lstrip("/")
        entries = []
        while current != current.parent and str(current).startswith("/sys/fs/cgroup"):
            entry = {"path": str(current)}
            for name in ("memory.max", "memory.current", "memory.events", "pids.max", "pids.current"):
                resource = current / name
                if resource.is_file():
                    entry[name] = resource.read_text().strip()
            if len(entry) > 1:
                entries.append(entry)
            current = current.parent
        result["cgroups"] = entries
    except (OSError, StopIteration):
        result["cgroups"] = []
    return result


def probe(model_path=None, *, expected_sha256=None, timeout_seconds=20):
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 60:
        raise ValueError("timeout must be in (0,60] seconds")
    report = dict(format_version="perception-probe-v1", before=resource_snapshot(), camera_opened=False,
                  synthetic_image=True, attempts=0, historical_exit137_reproduced=False)
    if model_path is None or not Path(model_path).is_file():
        return report | dict(status="missing_asset", reason="supply a local Hand Landmarker .task file; historical host asset is absent")
    path = Path(model_path).resolve()
    digest = sha256(path.read_bytes()).hexdigest()
    report.update(asset_sha256=digest, asset_bytes=path.stat().st_size)
    if expected_sha256 is not None and digest != expected_sha256:
        return report | dict(status="hash_mismatch", reason="SDK was not loaded")
    child = '''import json, resource, sys
import numpy as np
from backend.perception.mediapipe_adapter import MediaPipeDetector
print("stage:imported", flush=True)
detector = MediaPipeDetector(sys.argv[1], expected_sha256=sys.argv[2])
print("stage:initialized", flush=True)
hands = detector.detect(np.zeros((128,128,3), dtype=np.uint8), 0)
detector.close()
print(json.dumps({"hands": len(hands), "max_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
'''
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    # New process session allows bounded cleanup of a stuck SDK and descendants.
    process = subprocess.Popen([sys.executable, "-c", child, str(path), digest], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
    report["attempts"] = 1
    try:
        out, err = process.communicate(timeout=timeout_seconds)
        status = "passed_synthetic_image" if process.returncode == 0 else "sdk_failed"
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        out, err = process.communicate()
        status = "timeout"
    report.update(status=status, returncode=process.returncode, elapsed_seconds=time.monotonic() - started,
                  stdout=out[-12000:], stderr=err[-12000:], after=resource_snapshot(),
                  historical_exit137_reproduced=process.returncode in (-9, 137) and status != "timeout",
                  limitation="synthetic image SDK probe only; no real-hand/camera/GUI acceptance; SIGKILL alone does not establish OOM")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--timeout-seconds", type=float, default=20)
    parser.add_argument("--report", type=Path, required=True, help="new JSON file, never overwrites")
    args = parser.parse_args(argv)
    try:
        if args.report.exists():
            raise FileExistsError(args.report)
        report = probe(args.model, expected_sha256=args.expected_sha256, timeout_seconds=args.timeout_seconds)
        with args.report.open("x") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(json.dumps({key: report[key] for key in ("status", "attempts", "camera_opened")}))
        return 0 if report["status"] == "passed_synthetic_image" else 2
    except (OSError, ValueError) as error:
        print(f"probe failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
