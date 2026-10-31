"""Synthetic-only native UI bridge. Run with ``python -m backend.desktop``.

Private inherited pipes, not a listening service. No camera, model, files or
application actions are opened by this prototype. stdout is exclusively NDJSON.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import uuid

from backend.perception.synthetic import skeleton
from shared.desktop_protocol import DesktopCommand, DesktopSnapshot, StagePoint

MAX_COMMAND_BYTES = 4096
HEARTBEAT_TIMEOUT_NS = 1_500_000_000
PARENT_TIMEOUT_NS = 5_000_000_000


def decode_command(line: bytes) -> DesktopCommand:
    if len(line) > MAX_COMMAND_BYTES or not line.endswith(b"\n"):
        raise ValueError("command exceeds framing limit")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    # Python's JSON parser otherwise silently accepts duplicate object keys.
    return DesktopCommand.model_validate(
        json.loads(line, object_pairs_hook=unique_object)
    )


class PreviewSession:
    """Single locked source of truth; independent of the potentially blocked writer."""

    def __init__(self, *, clock=time.monotonic_ns):
        self.clock = clock
        self.lock = threading.RLock()
        self.session_id = uuid.uuid4().hex
        self.epoch = self.sequence = self.last_command_id = self.frame_id = 0
        self.state = "stopped"
        self.reason = "initial"
        self.last_heartbeat_ns = clock()
        self.closed = False

    def _stop(self, reason):
        self.epoch += 1
        self.state = "stopped"
        self.reason = reason

    def fail(self, reason):
        with self.lock:
            self._stop(reason)
            self.closed = True

    def accept(self, command: DesktopCommand):
        with self.lock:
            if self.closed:
                return
            if command.session_id != self.session_id:
                raise ValueError("wrong session")
            command_id = int(command.command_id)
            if command_id <= self.last_command_id:
                raise ValueError("non-increasing command id")
            self.last_command_id = command_id
            now = self.clock()
            # Expire before checking start: a queued request cannot revive a lease.
            self._expire(now)
            if self.closed:
                return
            if command.kind == "heartbeat":
                self.last_heartbeat_ns = now
            elif command.kind in {"stop", "shutdown"}:
                # A stop is always safe even if its UI epoch is obsolete.
                self._stop("shutdown" if command.kind == "shutdown" else "user_stop")
                self.closed = command.kind == "shutdown"
            elif int(command.epoch) != self.epoch:
                self.reason = "stale_epoch"
            elif self.state != "preview":
                self.epoch += 1
                self.state = "preview"
                self.reason = "synthetic_preview"
                self.last_heartbeat_ns = now

    def _expire(self, now):
        if (
            now - self.last_heartbeat_ns >= HEARTBEAT_TIMEOUT_NS
            and self.state == "preview"
        ):
            self._stop("heartbeat_timeout")
        if now - self.last_heartbeat_ns >= PARENT_TIMEOUT_NS:
            self.closed = True

    def snapshot(self):
        with self.lock:
            now = self.clock()
            self._expire(now)
            self.sequence += 1
            points = ()
            frame_time = None
            if self.state == "preview":
                self.frame_id += 1
                frame_time = str(now)
                # A deterministic source, not inferred gestures or calibrated scores.
                cx = 0.5 + 0.15 * math.sin(self.frame_id / 15)
                if self.frame_id % 60 >= 8:
                    points = tuple(
                        StagePoint(x=x, y=y, z=z) for x, y, z in skeleton(cx, 0.7)
                    )
            return DesktopSnapshot(
                session_id=self.session_id,
                sequence=str(self.sequence),
                epoch=str(self.epoch),
                last_command_id=str(self.last_command_id),
                timestamp_ns=str(now),
                state=self.state,
                reason=self.reason,
                frame_id=str(self.frame_id),
                frame_timestamp_ns=frame_time,
                landmarks=points,
            )


class LatestSnapshot:
    """Capacity one mailbox: slow rendering never grows a frame queue."""

    def __init__(self):
        self.condition = threading.Condition()
        self.value = None
        self.closed = False

    def put(self, value):
        with self.condition:
            self.value = value
            self.condition.notify()

    def take(self):
        with self.condition:
            self.condition.wait_for(lambda: self.value is not None or self.closed)
            value, self.value = self.value, None
            return value

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def serve(input_stream, output_stream):
    session = PreviewSession()
    mailbox = LatestSnapshot()

    def read_commands():
        try:
            while not session.closed:
                line = input_stream.readline(MAX_COMMAND_BYTES + 1)
                if not line:
                    session.fail("input_closed")
                    return
                session.accept(decode_command(line))
        except (ValueError, OSError):
            session.fail("protocol_error")

    def write_snapshots():
        try:
            while (snapshot := mailbox.take()) is not None:
                output_stream.write(snapshot.model_dump_json().encode() + b"\n")
                output_stream.flush()
        except (OSError, ValueError):
            session.fail("input_closed")

    reader = threading.Thread(target=read_commands, daemon=True, name="desktop-input")
    writer = threading.Thread(
        target=write_snapshots, daemon=True, name="desktop-output"
    )
    writer.start()
    mailbox.put(session.snapshot())
    reader.start()
    try:
        while not session.closed:
            time.sleep(0.1)
            mailbox.put(session.snapshot())
    except KeyboardInterrupt:
        session.fail("shutdown")
    finally:
        mailbox.put(session.snapshot())
        mailbox.close()
        # Never wait indefinitely for a stalled consumer or an unclosed input pipe.
        writer.join(timeout=0.2)


def main():
    # Unbuffered binary output avoids a buffered-IO shutdown lock if the UI stalls.
    import os

    with (
        os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0) as input_stream,
        os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0) as output,
    ):
        serve(input_stream, output)


if __name__ == "__main__":
    main()
