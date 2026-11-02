"""Small, non-sensitive lifecycle events; never emit images, paths or raw exceptions."""

import json
import sys
import time

PHASES = frozenset({
    "worker_started", "model_loading", "model_ready", "model_failed",
    "perception_loading", "perception_ready", "perception_failed",
    "camera_opening", "camera_ready", "camera_failed", "first_frame",
    "worker_failed", "release_started", "release_complete", "release_failed",
})


def lifecycle(phase, *, error=None):
    if phase not in PHASES:
        raise ValueError("unknown lifecycle phase")
    event = {"phase": phase, "monotonic_ns": str(time.monotonic_ns())}
    if error is not None:
        # No arbitrary exception messages: they can include private asset paths.
        event["error_type"] = type(error).__name__[:80]
    try:
        print("HS_DIAG " + json.dumps(event, separators=(",", ":")), file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass  # Diagnostics must not interfere with stopping/resource release.
