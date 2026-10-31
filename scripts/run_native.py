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
    args = parser.parse_args(argv)
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
    try:
        return subprocess.call(command, cwd=root / "frontend/native", env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
