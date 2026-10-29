"""Thin installed-package entry: python scripts/camera_probe.py [options]."""
import sys
from backend.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["camera_probe", *sys.argv[1:]]))
