"""Manual application-local actions only. No camera, training or OS input."""

import json
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

import pytest
from pydantic import ValidationError

from backend.actions.dispatcher import Dispatcher
from backend.controller import ApplicationController
from backend.desktop import decode_command
from backend.presentation import PresentationSession
from shared.contracts import ActionCommand
from shared.presentation_protocol import PresentationCommand, PresentationSnapshot


class Clock:
    now = 1_000_000_000

    def __call__(self):
        return self.now


def send(session, kind, *, base=None, **changes):
    state = base or session.snapshot()
    command = PresentationCommand(
        **(
            {
                "version": 1,
                "session_id": state.session_id,
                "command_id": str(session.last_command_id + 1),
                "epoch": state.epoch,
                "based_on_sequence": state.sequence,
                "revision": state.revision,
                "kind": kind,
            }
            | changes
        )
    )
    session.accept(command)
    return command


def settled(session):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        snapshot = session.snapshot()
        if not snapshot.busy:
            return snapshot
        time.sleep(0.001)
    pytest.fail("dispatcher did not finish")


def play(session):
    send(session, "render_ack")
    send(session, "play")
    send(session, "render_ack")


@pytest.fixture
def session():
    value = PresentationSession(clock=Clock())
    yield value
    value.close()


def test_manual_script_updates_real_page_state_and_waits_for_render(session):
    assert session.snapshot().state == "stopped"
    send(session, "next")
    assert session.snapshot().page == 1
    play(session)
    assert session.controller.snapshot().owner_track_id is None  # not a fabricated hand
    for kind, page in [("next", 2), ("next", 3), ("previous", 2)]:
        send(session, kind)
        state = settled(session)
        assert state.page == page and state.last_action.outcome == "application_updated"
        assert state.rendered_revision != state.revision
        send(session, "render_ack")
    assert all(
        r.acknowledgement_level == "application" for r in session.dispatcher.results
    )
    send(session, "stop")
    assert session.dispatcher.wait_released()
    assert (
        session.snapshot().release_confirmed and session.snapshot().state == "stopped"
    )


def test_boundaries_are_acknowledged_without_wraparound(session):
    play(session)
    for kind, expected in [("previous", 1), *[("next", n) for n in (2, 3, 4, 4)]]:
        send(session, kind)
        state = settled(session)
        assert state.page == expected
        if kind == "previous" or state.last_action.page_before == 4:
            assert state.last_action.outcome == "boundary_no_change"
        send(session, "render_ack")


def test_stale_snapshot_does_not_restart_ttl(session):
    play(session)
    old = session.snapshot()
    session.clock.now += 500_000_000
    send(session, "next", base=old)
    assert session.snapshot().state == "stopped"
    assert session.snapshot().page == 1
    assert not session.dispatcher.results


def test_healthy_heartbeats_do_not_replace_render_confirmation(session):
    play(session)
    send(session, "next")
    settled(session)
    for _ in range(4):
        session.clock.now += 300_000_000
        send(session, "heartbeat")
    assert session.snapshot().state == "stopped"
    assert session.snapshot().reason in {"render_timeout", "stale_context"}
    assert session.snapshot().page == 2


def test_old_revision_and_repeated_wire_id_are_rejected(session):
    play(session)
    old = session.snapshot()
    command = send(session, "next")
    settled(session)
    send(session, "render_ack", revision=old.revision)
    assert session.snapshot().rendered_revision != session.snapshot().revision
    with pytest.raises(ValueError, match="replayed"):
        session.accept(command)


def test_stop_ignores_expired_nonce_but_new_play_does_not(session):
    play(session)
    old = session.snapshot()
    session.clock.now += 600_000_000
    send(session, "stop", base=old)
    assert session.dispatcher.wait_released()
    send(session, "play", base=old)
    assert session.snapshot().state == "stopped"
    assert not session.manual_grant


def test_local_backend_obeys_dispatcher_dedup_and_late_stop(session):
    play(session)
    context = session.controller.snapshot()

    def command(identifier):
        return ActionCommand(
            command_id=identifier,
            source_event_id="same-manual-event",
            stream_id=context.stream_id,
            context_epoch=context.context_epoch,
            action="next_slide",
            issued_at_ns=session.clock(),
            expires_at_ns=session.clock() + 400_000_000,
            required_capabilities=["presentation"],
            target_context=context.target_context,
        )

    result = session.dispatcher.submit(command("one")).future.result(1)
    assert result.accepted and result.page_after == 2
    send(session, "render_ack")
    duplicate = session.dispatcher.submit(command("two")).future.result(1)
    assert duplicate.error_code == "duplicate_event"
    assert session.backend.page == 2

    entered, release = Event(), Event()
    original = session.backend.execute

    def blocked(command, now):
        entered.set()
        release.wait(2)
        return original(command, now)

    session.backend.execute = blocked
    value = command("three").model_copy(update={"source_event_id": "new"})
    ticket = session.dispatcher.submit(value)
    assert entered.wait(1)
    send(session, "stop")
    release.set()
    assert not ticket.future.result(1).accepted
    assert session.backend.page == 2


def test_play_requires_a_current_surface_receipt(session):
    send(session, "play")
    assert session.snapshot().reason == "surface_not_ready"
    assert not session.controller.snapshot().enabled
    send(session, "render_ack", revision="1")
    assert session.snapshot().rendered_revision is None


def test_heartbeat_loss_stops_independently_of_new_commands(session):
    play(session)
    session.clock.now += 1_600_000_000
    assert session.snapshot().state == "stopped"
    assert not session.manual_grant
    assert session.backend.stop_requested.is_set()


def test_default_dispatcher_still_requires_gesture_owner():
    controller = ApplicationController("legacy")
    controller.enable(time.monotonic_ns())
    dispatcher = Dispatcher(controller)
    now = time.monotonic_ns()
    try:
        command = ActionCommand(
            command_id="c",
            source_event_id="e",
            stream_id="legacy",
            context_epoch=controller.snapshot().context_epoch,
            action="next_slide",
            issued_at_ns=now,
            expires_at_ns=now + 100_000_000,
            target_context="mock-presentation",
            required_capabilities=["presentation"],
        )
        assert (
            dispatcher.submit(command).future.result(1).error_code == "inactive_context"
        )
    finally:
        dispatcher.close()


@pytest.mark.parametrize(
    "change",
    [{"kind": "run_script"}, {"version": True}, {"command_id": 1}, {"extra": "bad"}],
)
def test_strict_private_commands(session, change):
    state = session.snapshot()
    body = {
        "version": 1,
        "session_id": state.session_id,
        "command_id": "1",
        "epoch": "0",
        "based_on_sequence": state.sequence,
        "revision": "0",
        "kind": "play",
    } | change
    with pytest.raises(ValidationError):
        decode_command(json.dumps(body).encode() + b"\n", PresentationCommand)


def test_schema_fixture_and_no_external_capabilities():
    fixture = Path("frontend/native/test/fixtures/presentation.json").read_text()
    value = PresentationSnapshot.model_validate_json(fixture)
    assert value.scope == "application-local" and not value.external_control_available
    schema = json.loads(Path("shared/presentation_protocol.schema.json").read_text())
    assert schema == {
        "command": PresentationCommand.model_json_schema(),
        "snapshot": PresentationSnapshot.model_json_schema(),
    }
    for field in ("external_control_available", "gesture_control_available"):
        for invalid in (True, 0):
            with pytest.raises(ValueError):
                PresentationSnapshot.model_validate(
                    value.model_dump() | {field: invalid}
                )


def test_eof_child_exits_without_camera_or_apps():
    result = subprocess.run(
        [sys.executable, "-m", "backend.presentation_cli"],
        input=b"",
        capture_output=True,
        timeout=4,
        check=True,
    )
    lines = result.stdout.splitlines()
    assert lines
    assert all(
        PresentationSnapshot.model_validate_json(line).state == "stopped"
        for line in lines
    )
