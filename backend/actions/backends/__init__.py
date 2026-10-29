"""Only in-memory dry-run and fault-injectable mock backends are supplied."""
from backend.actions.backends.dry_run import DryRunBackend
from backend.actions.backends.mock import MockBackend

__all__ = ["DryRunBackend", "MockBackend"]
