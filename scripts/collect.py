"""Thin installed-package entry: python scripts/collect.py [options]."""
import sys
from backend.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["collect", *sys.argv[1:]]))
