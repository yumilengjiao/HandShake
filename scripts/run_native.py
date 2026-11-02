"""Legacy launcher; prefer `uv run app`. Implementation lives in the installed backend."""

from backend.native_launcher import main, subprocess  # noqa: F401 - compatibility for launcher tests


if __name__ == "__main__":
    raise SystemExit(main())
