"""Launch native preview with an explicit local core; camera access is opt-in in the UI."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv=None, *, project_root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flutter",
        default=os.environ.get("FLUTTER_BIN", "flutter"),
        help="Flutter 3.47.9 executable; alternatively put it on PATH",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help="run an already-built native executable without Flutter",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="developer-only Flutter debug/hot-reload; normal launches use release mode",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="device index; opened only after clicking camera preview",
    )
    parser.add_argument(
        "--perception-model",
        type=Path,
        help="local MediaPipe hand_landmarker.task (optional; not a gesture classifier)",
    )
    parser.add_argument(
        "--model-package", type=Path, help="verified inference bundle (optional)"
    )
    parser.add_argument(
        "--model-sha256", help="trusted checksums.json SHA256 from export receipt"
    )
    parser.add_argument(
        "--calibration", type=Path, help="optional validation calibration sidecar"
    )
    parser.add_argument("--calibration-sha256", help="trusted calibration file SHA256")
    parser.add_argument(
        "--shadow-session",
        action="store_true",
        help="enable native shadow owner UI, NOT application execution",
    )
    parser.add_argument("--allow-synthetic-model", action="store_true")
    args = parser.parse_args(argv)
    if args.debug and args.bundle:
        parser.error("--debug requires a Flutter source launch, not --bundle")
    if args.shadow_session and not args.calibration:
        parser.error("shadow session requires model, perception and calibration assets")
    if args.allow_synthetic_model and not args.shadow_session:
        parser.error("--allow-synthetic-model belongs to --shadow-session")
    if (
        bool(args.calibration) != bool(args.calibration_sha256)
        or args.calibration
        and not args.model_package
    ):
        parser.error("calibration requires both sidecar/SHA256 and a model package")
    if bool(args.model_package) != bool(args.model_sha256):
        parser.error("provide both --model-package and --model-sha256")
    if args.model_package is not None and args.perception_model is None:
        parser.error("model inference requires --perception-model")
    if args.camera < 0:
        parser.error("camera index must be nonnegative")
    root = (
        Path(project_root).resolve()
        if project_root
        else Path(__file__).resolve().parents[1]
    )
    if args.bundle:
        executable = args.bundle.resolve()
        if not executable.is_file():
            parser.error(f"native executable not found: {executable}")
        command = [str(executable)]
    else:
        executable = shutil.which(args.flutter)
        if executable is None:
            parser.error(
                "Flutter not found; pass --flutter /path/to/flutter/bin/flutter"
            )
        # This is the daily app entry, not a development session. Debug painting
        # and VM-service overhead compete with the live 100ms observation budget.
        command = [
            executable, "--no-version-check", "run", "-d", "linux",
            "--debug" if args.debug else "--release",
        ]
    # Never inherit an old model/calibration selection from the launching shell.
    environment = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "HANDSHAKE_PERCEPTION_MODEL",
            "HANDSHAKE_MODEL_PACKAGE",
            "HANDSHAKE_MODEL_SHA256",
            "HANDSHAKE_CALIBRATION",
            "HANDSHAKE_CALIBRATION_SHA256",
            "HANDSHAKE_SHADOW_SESSION",
            "HANDSHAKE_ALLOW_SYNTHETIC_MODEL",
        }
    } | {
        "HANDSHAKE_PYTHON": sys.executable,
        "HANDSHAKE_ROOT": str(root),
        "HANDSHAKE_CAMERA": str(args.camera),
    }
    if args.perception_model is not None:
        environment["HANDSHAKE_PERCEPTION_MODEL"] = str(args.perception_model.resolve())
    if args.model_package is not None:
        environment["HANDSHAKE_MODEL_PACKAGE"] = str(args.model_package.resolve())
        environment["HANDSHAKE_MODEL_SHA256"] = args.model_sha256
    if args.calibration is not None:
        environment["HANDSHAKE_CALIBRATION"] = str(args.calibration.resolve())
        environment["HANDSHAKE_CALIBRATION_SHA256"] = args.calibration_sha256
    if args.shadow_session:
        environment["HANDSHAKE_SHADOW_SESSION"] = "1"
        environment["HANDSHAKE_ALLOW_SYNTHETIC_MODEL"] = (
            "1" if args.allow_synthetic_model else "0"
        )
    try:
        return subprocess.call(command, cwd=root / "frontend/native", env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
