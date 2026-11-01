"""Thin installed-package entry point; no neural training or application control."""

from algorithm.evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
