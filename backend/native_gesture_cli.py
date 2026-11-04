"""Private native gesture-preview child. No camera is opened until a UI request."""

import argparse
import json
import os
import sys

from backend.desktop import decode_command, serve
from backend.native_gesture import NativeGestureSession
from shared.native_gesture_protocol import NativeGestureCommand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--perception-model")
    parser.add_argument("--model-package")
    parser.add_argument("--model-sha256")
    parser.add_argument("--calibration")
    parser.add_argument("--calibration-sha256")
    args = parser.parse_args()
    fields = [
        args.model_package,
        args.model_sha256,
        args.calibration,
        args.calibration_sha256,
    ]
    if any(fields) and not all(fields):
        parser.error("gesture preview needs model, calibration and both trusted hashes")
    if args.camera < 0 or all(fields) and not args.perception_model:
        parser.error("valid camera index and perception asset required")
    assets = (
        dict(zip(["package", "sha256", "calibration", "calibration_sha256"], fields))
        if all(fields)
        else None
    )
    session = NativeGestureSession(
        assets=assets, perception_model=args.perception_model, camera=args.camera
    )
    # Native SDKs may print to stdout. Keep them off the protocol descriptor.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0) as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0) as source:
            try:
                serve(
                    source,
                    output,
                    session=session,
                    snapshot_interval_s=0.03,
                    decoder=lambda line: decode_command(line, NativeGestureCommand),
                    encoder=lambda value: (
                        json.dumps(
                            value, ensure_ascii=False, separators=(",", ":")
                        ).encode()
                        + b"\n"
                    ),
                )
            finally:
                session.close()
    # A blocked native driver must not hang shutdown or claim confirmed release.
    os._exit(0)


if __name__ == "__main__":
    main()
