"""Thin installed-package entry: python scripts/annotate.py [options]."""
import sys
from backend.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["annotate", *sys.argv[1:]]))
