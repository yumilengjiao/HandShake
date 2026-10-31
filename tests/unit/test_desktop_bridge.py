import io
import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.desktop import (
    HEARTBEAT_TIMEOUT_NS,
    LatestSnapshot,
    PreviewSession,
    decode_command,
    serve,
)
from shared.desktop_protocol import DesktopCommand, DesktopSnapshot


class Clock:
    now = 0

    def __call__(self):
        return self.now


def command(session, kind, index=1, epoch=None):
    return DesktopCommand(
        version=1,
        session_id=session.session_id,
        command_id=str(index),
        epoch=str(session.epoch if epoch is None else epoch),
        kind=kind,
    )


def test_start_stop_and_epoch_barrier():
    session = PreviewSession(clock=Clock())
    assert session.snapshot().state == "stopped"
    session.accept(command(session, "start"))
    assert session.snapshot().state == "preview"
    session.accept(command(session, "stop", 2, epoch=0))
    assert session.snapshot().state == "stopped"
    session.accept(command(session, "start", 3, epoch=1))
    assert session.snapshot().reason == "stale_epoch"
    assert session.state == "stopped"
    session.accept(command(session, "start", 4))
    assert session.snapshot().state == "preview"


def test_heartbeat_cannot_restart_after_timeout():
    clock = Clock()
    session = PreviewSession(clock=clock)
    session.accept(command(session, "start"))
    old_epoch = session.epoch
    clock.now = HEARTBEAT_TIMEOUT_NS
    session.accept(command(session, "heartbeat", 2))
    assert session.snapshot().reason == "heartbeat_timeout"
    session.accept(command(session, "start", 3, epoch=old_epoch))
    assert session.state == "stopped"


def test_start_queued_before_timeout_cannot_revive_preview():
    clock = Clock()
    session = PreviewSession(clock=clock)
    session.accept(command(session, "start"))
    delayed = command(session, "start", 2)
    clock.now = HEARTBEAT_TIMEOUT_NS
    session.accept(delayed)
    assert session.state == "stopped"


def test_identity_and_replay_rejected():
    session = PreviewSession()
    start = command(session, "start")
    session.accept(start)
    with pytest.raises(ValueError, match="non-increasing"):
        session.accept(start)
    with pytest.raises(ValueError, match="wrong session"):
        session.accept(command(PreviewSession(), "start", 2))


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"version": True},
        {"kind": "enable_live"},
        {"command_id": 1},
        {"epoch": "01"},
        {"command_id": "-1"},
        {"shell": "ignored?"},
    ],
)
def test_strict_commands(change):
    value = command(PreviewSession(), "start").model_dump() | change
    with pytest.raises(ValidationError):
        decode_command(json.dumps(value).encode() + b"\n")


@pytest.mark.parametrize(
    "line", [b"x" * 4097 + b"\n", b"{}", b'{"x":1,"x":2}\n', b"\xff\n"]
)
def test_invalid_framing(line):
    with pytest.raises(ValueError):
        decode_command(line)


def test_skeleton_round_trip_and_no_current_frame_when_stopped():
    session = PreviewSession()
    session.accept(command(session, "start"))
    for _ in range(10):
        snapshot = session.snapshot()
        assert (
            DesktopSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
        )
    assert len(snapshot.landmarks) == 21
    assert snapshot.control_available is False
    session.accept(command(session, "stop", 2))
    assert session.snapshot().landmarks == ()


def test_latest_mailbox_is_bounded():
    mailbox = LatestSnapshot()
    for index in range(10_000):
        mailbox.put(index)
    assert mailbox.take() == 9999
    mailbox.close()
    assert mailbox.take() is None


def test_eof_and_malformed_input_exit():
    for raw in (b"", b"oops\n"):
        output = io.BytesIO()
        serve(io.BytesIO(raw), output)
        last = DesktopSnapshot.model_validate_json(output.getvalue().splitlines()[-1])
        assert last.state == "stopped"
        assert last.reason in {"input_closed", "protocol_error"}


def test_blocked_output_does_not_block_shutdown():
    release = threading.Event()

    class StalledOutput:
        def write(self, _):
            release.wait(3)

        def flush(self):
            pass

    started = time.monotonic()
    try:
        serve(io.BytesIO(b""), StalledOutput())
        assert time.monotonic() - started < 1
    finally:
        release.set()


def test_stdio_process_start_stop_and_exit():
    process = subprocess.Popen(
        [sys.executable, "-m", "backend.desktop"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    def receive(predicate):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            assert select.select([process.stdout], [], [], 4)[0], "bridge timeout"
            value = DesktopSnapshot.model_validate_json(process.stdout.readline())
            if predicate(value):
                return value
        pytest.fail("expected snapshot was not received")

    def send(snapshot, kind, index):
        process.stdin.write(
            DesktopCommand(
                version=1,
                session_id=snapshot.session_id,
                command_id=str(index),
                epoch=snapshot.epoch,
                kind=kind,
            )
            .model_dump_json()
            .encode()
            + b"\n"
        )

    try:
        initial = receive(lambda _: True)
        assert initial.reason == "initial"
        send(initial, "start", 1)
        running = receive(lambda value: value.state == "preview")
        send(running, "stop", 2)
        stopped = receive(lambda value: value.reason == "user_stop")
        assert int(stopped.epoch) > int(running.epoch)
        send(stopped, "shutdown", 3)
        process.wait(timeout=3)
        assert process.returncode == 0, process.stderr.read().decode()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)


def test_python_dart_fixture_agrees():
    session = PreviewSession(clock=lambda: 9_007_199_254_740_999)
    session.session_id = "a" * 32
    fixture = Path("frontend/native/test/fixtures/snapshot.json").read_text()
    assert DesktopSnapshot.model_validate_json(fixture) == session.snapshot()


def test_schema_matches_contract():
    schema = json.loads(Path("shared/desktop_protocol.schema.json").read_text())
    assert schema["command"] == DesktopCommand.model_json_schema()
    assert schema["snapshot"] == DesktopSnapshot.model_json_schema()


@pytest.mark.skipif(os.name != "posix", reason="POSIX pipe backpressure probe")
def test_unread_stdout_does_not_keep_orphan_core_alive():
    # No heartbeat: even an unread output pipe must not retain an orphan child.
    process = subprocess.Popen(
        [sys.executable, "-m", "backend.desktop"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        process.wait(timeout=8)
        assert process.returncode == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)
