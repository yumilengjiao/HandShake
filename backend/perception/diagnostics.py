"""Compatibility wrapper for T14's probe; native execution lives in probe.py.

New callers should use ``handshake-perception-probe``. The earlier function and
CLI argument names remain available without maintaining another native worker.
"""
import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

from backend.perception import probe as canonical


def resource_snapshot():
    versions = {}
    for name in ("mediapipe", "numpy", "opencv-contrib-python"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), executable=sys.executable,
                system=platform.system(), machine=platform.machine(), packages=versions,
                cgroups=canonical.cgroup_memory())


def _validate_timeout(timeout_seconds):
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 60:
        raise ValueError("timeout must be in (0,60] seconds")


def probe(model_path=None, *, expected_sha256=None, timeout_seconds=20):
    _validate_timeout(timeout_seconds)
    report = dict(format_version="perception-probe-v1", before=resource_snapshot(), camera_opened=False,
                  synthetic_image=True, attempts=0, historical_exit137_reproduced=False)
    if model_path is None or not Path(model_path).is_file():
        return report | dict(status="missing_asset", reason="supply a local Hand Landmarker .task file")
    path = Path(model_path).resolve()
    digest = sha256(path.read_bytes()).hexdigest()
    report.update(asset_sha256=digest, asset_bytes=path.stat().st_size)
    if expected_sha256 is not None and digest != expected_sha256:
        return report | dict(status="hash_mismatch", reason="SDK was not loaded")
    native = canonical.probe(path, model_sha256=digest, timeout=timeout_seconds)
    succeeded = native["status"] == "passed"
    status = "passed_synthetic_image" if succeeded else "timeout" if native["status"] == "timeout" else "sdk_failed"
    report.update(status=status, attempts=1, returncode=native["returncode"], elapsed_seconds=native["elapsed_seconds"],
                  stdout=native["stdout"], stderr=native["stderr"], after=resource_snapshot(), canonical_report=native,
                  historical_exit137_reproduced=native["returncode"] in (-9, 137) and native.get("monitor_reason") is None,
                  limitation="SDK black-image probe only; no real-hand/camera/GUI acceptance; SIGKILL alone does not establish OOM")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--timeout-seconds", type=float, default=20)
    parser.add_argument("--report", type=Path, required=True, help="new JSON file, never overwrites")
    args = parser.parse_args(argv)
    try:
        _validate_timeout(args.timeout_seconds)
        # Reserve evidence before native work, as the canonical CLI does.
        with args.report.open("x") as output:
            report = probe(args.model, expected_sha256=args.expected_sha256, timeout_seconds=args.timeout_seconds)
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(json.dumps({key: report[key] for key in ("status", "attempts", "camera_opened")}))
        return 0 if report["status"] == "passed_synthetic_image" else 2
    except (OSError, ValueError) as error:
        print(f"probe failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
