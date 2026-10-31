"""Launch the native prototype with an explicit local Python core (no camera)."""

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
    args = parser.parse_args(argv)
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
    }
    try:
        return subprocess.call(command, cwd=root / "frontend/native", env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
