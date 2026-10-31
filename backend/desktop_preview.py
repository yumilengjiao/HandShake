"""Opt-in camera preview endpoint. Binary stdout v2, bounded RAM, no recording/actions."""

from __future__ import annotations

import argparse
import os
import sys
import time

from backend.desktop import PreviewSession, decode_command, serve
from backend.perception.preview import CameraPreviewWorker
from shared.desktop_protocol import DesktopCommand
from shared.preview_protocol import (
    PreviewCommand,
    PreviewHand,
    PreviewPacket,
    PreviewPoint,
    PreviewSnapshot,
)


class CameraPreviewSession(PreviewSession):
    def __init__(self, *, worker_factory=CameraPreviewWorker, clock=time.monotonic_ns):
        super().__init__(clock=clock)
        self.worker_factory = worker_factory
        self.worker = None
        self.source = "synthetic"
        self.camera_frame = None
        self.started_ns = 0
        self.release_started_ns = None

    def _stop(self, reason):
        super()._stop(reason)
        self.camera_frame = None
        if self.worker is not None:
            if self.release_started_ns is None:
                self.release_started_ns = self.clock()
            self.worker.request_stop()

    def accept(self, command):
        with self.lock:
            # Starting is disallowed while any prior worker still owns resources.
            if command.kind in {"start", "start_camera"} and self.worker is not None:
                if self.worker.done.is_set():
                    self.worker = None
                else:
                    # Validate and consume the ID, but never allocate a second worker.
                    blocked = DesktopCommand(
                        version=1,
                        session_id=command.session_id,
                        command_id=command.command_id,
                        epoch=command.epoch,
                        kind="heartbeat",
                    )
                    super().accept(blocked)
                    return
            previous_epoch = self.epoch
            translated = DesktopCommand(
                version=1,
                session_id=command.session_id,
                command_id=command.command_id,
                epoch=command.epoch,
                kind="start" if command.kind == "start_camera" else command.kind,
            )
            super().accept(translated)
            if (
                command.kind in {"start", "start_camera"}
                and self.state == "preview"
                and self.epoch != previous_epoch
            ):
                self.source = (
                    "camera" if command.kind == "start_camera" else "synthetic"
                )
                self.camera_frame = None
                self.started_ns = self.clock()
                self.release_started_ns = None
                if self.source == "camera":
                    self.reason = "camera_starting"
                    try:
                        self.worker = self.worker_factory()
                        self.worker.start()
                    except (RuntimeError, OSError, ValueError):
                        self._stop("camera_unavailable")

    def snapshot(self):
        with self.lock:
            now = self.clock()
            self._expire(now)
            if self.source == "synthetic":
                original = super().snapshot()
                values = original.model_dump(exclude={"landmarks", "version"})
                hands = (
                    ()
                    if not original.landmarks
                    else (
                        PreviewHand(
                            landmarks=tuple(
                                PreviewPoint(x=p.x, y=p.y, z=p.z)
                                for p in original.landmarks
                            ),
                            validity=(True,) * 21,
                            usable=True,
                        ),
                    )
                )
                values.update(
                    version=2,
                    hands=hands,
                    frame_width=640 if original.state == "preview" else 0,
                    frame_height=480 if original.state == "preview" else 0,
                    timestamp_kind="synthetic_generation",
                    perception_status="processed_hands"
                    if hands
                    else "processed_no_hand",
                )
                return PreviewPacket(PreviewSnapshot(**values))

            worker = self.worker
            if worker is not None and self.state == "preview":
                incoming = worker.take()
                if incoming is not None and (
                    self.camera_frame is None
                    or incoming.frame_id > self.camera_frame.frame_id
                ):
                    self.camera_frame = incoming
                if worker.error is not None or worker.done.is_set():
                    self._stop(worker.error or "camera_unavailable")
                elif (
                    self.camera_frame is not None
                    and now - self.camera_frame.timestamp_ns > 750_000_000
                ) or (
                    self.camera_frame is None and now - self.started_ns >= 8_000_000_000
                ):
                    self._stop("input_stalled")
            frame = self.camera_frame if self.state == "preview" else None
            self.sequence += 1
            state = "preview" if frame is not None else "starting"
            if self.state == "stopped":
                state = (
                    "stopping"
                    if worker is not None and not worker.done.is_set()
                    else "stopped"
                )
                if (
                    state == "stopping"
                    and self.release_started_ns is not None
                    and now - self.release_started_ns >= 1_000_000_000
                ):
                    self.closed = True  # isolated process exits; never reports a successful release
            reason = "camera_preview" if frame is not None else self.reason
            return PreviewPacket(
                PreviewSnapshot(
                    session_id=self.session_id,
                    sequence=str(self.sequence),
                    epoch=str(self.epoch),
                    last_command_id=str(self.last_command_id),
                    timestamp_ns=str(now),
                    state=state,
                    reason=reason,
                    source="camera",
                    frame_id=str(frame.frame_id) if frame else "0",
                    frame_timestamp_ns=str(frame.timestamp_ns) if frame else None,
                    frame_width=frame.width if frame else 0,
                    frame_height=frame.height if frame else 0,
                    timestamp_kind="software_read_complete_not_exposure",
                    perception_status=frame.perception_status
                    if frame
                    else "unavailable",
                    capture_hz=frame.capture_hz if frame else None,
                    hands=frame.hands if frame else (),
                ),
                frame.jpeg if frame else b"",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--perception-model", help="optional local MediaPipe asset; not a gesture model"
    )
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("camera index must be nonnegative")
    session = CameraPreviewSession(
        worker_factory=lambda: CameraPreviewWorker(
            device=args.camera, model_path=args.perception_model
        )
    )
    # Keep native SDK stdout logging away from the binary protocol descriptor.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0) as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0) as input_stream:
            serve(
                input_stream,
                output,
                session=session,
                decoder=lambda line: decode_command(line, PreviewCommand),
                encoder=lambda packet: packet.encode(),
            )
    # This isolated preview process owns no persistence or action backend. A native
    # thread may still be blocked; do not run potentially unbounded SDK destructors.
    # The final snapshot remains 'stopping', never a fabricated release ACK.
    os._exit(0)


if __name__ == "__main__":
    main()
