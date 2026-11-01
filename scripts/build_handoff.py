"""Explicit offline handoff kit build/inspection; never install or launch."""

from backend.handoff import main

if __name__ == "__main__":
    raise SystemExit(main())
