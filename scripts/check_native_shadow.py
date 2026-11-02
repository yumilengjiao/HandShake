"""Run the native shadow integration test with disposable UNTRAINED assets; no devices."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _make_fixture(output, kind):
    # Source-checkout test helper, deliberately excluded from the installed core.
    # It assigns analytical weights and runs forward passes; no optimizer/trainer.
    sys.path.insert(0, str(ROOT))
    from tests.integration.model_bridge_fixture import fixture_bundle

    bundle, _, calibration_sha = fixture_bundle(output, kind)
    return bundle.manifest_sha256, calibration_sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flutter", default=os.environ.get("FLUTTER_BIN", "flutter"))
    parser.add_argument("--model-type", choices=("mlp", "tcn"), default="mlp")
    parser.add_argument(
        "--timeout", type=int, default=180, help="test deadline in seconds"
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    flutter = shutil.which(args.flutter)
    if flutter is None:
        parser.error("Flutter not found; put it on PATH or pass --flutter <executable>")
    print(
        "Native shadow check: procedural images/keypoints + UNTRAINED analytical "
        f"{args.model_type.upper()}; no camera, training or application actions.",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="handshake-shadow-check-") as directory:
        output = Path(directory)
        model_sha, calibration_sha = _make_fixture(output, args.model_type)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("HANDSHAKE_")
            and key not in {"PYTHONPATH", "PYTHONHOME"}
        } | {
            # Do not resolve sys.executable: resolving a venv symlink loses its environment.
            "HANDSHAKE_PYTHON": sys.executable,
            "HANDSHAKE_ROOT": str(ROOT),
            "HANDSHAKE_TEST_SOURCE": "1",
            "HANDSHAKE_MODEL_PACKAGE": str(output / "bundle"),
            "HANDSHAKE_MODEL_SHA256": model_sha,
            "HANDSHAKE_CALIBRATION": str(output / "calibration.json"),
            "HANDSHAKE_CALIBRATION_SHA256": calibration_sha,
        }
        try:
            return subprocess.run(
                [
                    flutter,
                    "--no-version-check",
                    "test",
                    "integration_test/shadow_session_test.dart",
                    "-d",
                    "linux",
                ],
                cwd=ROOT / "frontend/native",
                env=environment,
                timeout=args.timeout,
                check=False,
            ).returncode
        except subprocess.TimeoutExpired:
            print("Native shadow check timed out; no success claimed.", file=sys.stderr)
            return 124
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
