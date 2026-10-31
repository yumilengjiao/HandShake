"""Launch native preview with an explicit local core; camera access is opt-in in the UI."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv=None):
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
    args = parser.parse_args(argv)
    if bool(args.model_package) != bool(args.model_sha256):
        parser.error("provide both --model-package and --model-sha256")
    if args.model_package is not None and args.perception_model is None:
        parser.error("model inference requires --perception-model")
    if args.camera < 0:
        parser.error("camera index must be nonnegative")
    root = Path(__file__).resolve().parents[1]
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
        command = [executable, "--no-version-check", "run", "-d", "linux"]
    environment = os.environ | {
        "HANDSHAKE_PYTHON": sys.executable,
        "HANDSHAKE_ROOT": str(root),
        "HANDSHAKE_CAMERA": str(args.camera),
    }
    if args.perception_model is not None:
        environment["HANDSHAKE_PERCEPTION_MODEL"] = str(args.perception_model.resolve())
    if args.model_package is not None:
        environment["HANDSHAKE_MODEL_PACKAGE"] = str(args.model_package.resolve())
        environment["HANDSHAKE_MODEL_SHA256"] = args.model_sha256
    try:
        return subprocess.call(command, cwd=root / "frontend/native", env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
