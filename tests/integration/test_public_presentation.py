"""Synthetic public-policy host integration and stop/lease races; no live input."""

import json
import subprocess
import sys
import time
from threading import Event

import pytest

from backend.interaction.public_presentation import SyntheticPublicPresentation
from backend.interaction.public_replay import (
    ReplayClock,
    acknowledge,
    run_replay,
    synthetic_calibration,
    synthetic_input,
)
from shared.contracts import Action, ActionCommand, Capability


@pytest.fixture
def host(tmp_path):
    calibration, sha = synthetic_calibration(tmp_path)
    clock = ReplayClock()
    runtime = SyntheticPublicPresentation(
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
        watchdog=False,
    )
    runtime.fixture_clock = clock
    runtime.fixture_frame = 0
    yield runtime
    runtime.close()
    assert runtime.dispatcher.wait_released()


def start(host):
    assert acknowledge(host) and host.enable()
    p, frame = synthetic_input(
        host.controller.snapshot(),
        timestamp_ns=host.fixture_clock.now,
        frame_id=host.fixture_frame,
        model_version=host.engine.model_version,
    )
    assert host.arm_owner("operator", p, frame, ready=True)
    assert acknowledge(host)


def step(host, token="NOG", x=0.5, *, ack=True):
    host.fixture_clock.now += 50_000_000
    host.fixture_frame += 1
    p, frame = synthetic_input(
        host.controller.snapshot(),
        timestamp_ns=host.fixture_clock.now,
        frame_id=host.fixture_frame,
        model_version=host.engine.model_version,
        token=token,
        x=x,
    )
    value = host.update(p, frame, ready=True)
    if ack:
        assert acknowledge(host)
    return value


def release(host, *, ack=True):
    for _ in range(12):
        assert step(host, ack=ack)[0] is None
    assert host.engine.state == "ACTIVE"


def motion(host, *, ack=True):
    results = [step(host, "TRT", x, ack=ack) for x in (0.5, 0.56, 0.62, 0.68)]
    assert all(event is None for event, _ in results[:-1])
    return results[-1]


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def test_explicit_synthetic_opt_in_only_and_no_camera_or_external_backend(tmp_path):
    calibration, sha = synthetic_calibration(tmp_path)
    with pytest.raises(ValueError, match="opt-in"):
        SyntheticPublicPresentation(
            calibration=calibration, calibration_sha256=sha, clock=ReplayClock()
        )
    real = calibration.model_copy(
        update={
            "source": calibration.source.model_copy(update={"data_synthetic": False})
        }
    )
    with pytest.raises(ValueError, match="synthetic calibration"):
        SyntheticPublicPresentation(
            calibration=real,
            calibration_sha256=sha,
            clock=ReplayClock(),
            allow_synthetic=True,
        )
    with pytest.raises(TypeError):
        SyntheticPublicPresentation(
            calibration=calibration,
            calibration_sha256=sha,
            clock=ReplayClock(),
            allow_synthetic=True,
            backend=object(),
        )
    # Import isolation is checked in a fresh subprocess below, not via test order.


def test_surface_and_explicit_owner_required_before_public_actions(host):
    assert not host.enable()
    assert acknowledge(host) and host.enable()
    release_before_owner = [step(host)[0] for _ in range(5)]
    assert release_before_owner == [None] * 5
    assert host.engine.state == "WAIT_OWNER" and host.backend.page == 1
    host.fixture_clock.now += 50_000_000
    host.fixture_frame += 1
    p, frame = synthetic_input(
        host.controller.snapshot(),
        timestamp_ns=host.fixture_clock.now,
        frame_id=host.fixture_frame,
        model_version=host.engine.model_version,
    )
    assert host.arm_owner("operator", p, frame, ready=True)
    release(host, ack=False)
    event, ticket = motion(host, ack=False)
    assert event is not None and ticket is None and host.backend.page == 1
    assert host.reason == "event_discarded_no_surface_or_authority"
    assert acknowledge(host)
    # A subsequent surface receipt cannot replay the consumed action.
    for _ in range(5):
        assert step(host, "TRT", 0.68)[0] is None
    assert host.backend.page == 1


def test_page_updates_dedup_and_default_legacy_dispatcher_authorization_unchanged(host):
    start(host)
    release(host)
    event, ticket = motion(host)
    assert event is not None and ticket is not None
    result = ticket.future.result(timeout=2)
    assert result.accepted and result.page_before == 1 and result.page_after == 2
    assert host.backend.page == 2
    assert acknowledge(host)
    context = host.controller.snapshot()
    duplicate = ActionCommand(
        command_id="different-command",
        source_event_id=event.event_id,
        stream_id=context.stream_id,
        context_epoch=context.context_epoch,
        timebase="synthetic",
        action=Action.NEXT_SLIDE,
        issued_at_ns=host.fixture_clock.now,
        expires_at_ns=host.fixture_clock.now + 100_000_000,
        required_capabilities=[Capability.PRESENTATION],
        target_context=context.target_context,
    )
    assert (
        host.dispatcher.submit(duplicate).future.result(timeout=2).error_code
        == "duplicate_event"
    )
    assert host.backend.page == 2


def test_old_surface_tokens_and_reenable_during_release_are_rejected(host):
    initial = host.snapshot()
    old = {key: initial[key] for key in ("sequence", "epoch", "revision")}
    start(host)
    assert not host.acknowledge_surface(**old)
    host.stop()
    assert not host.enable()
    assert host.dispatcher.wait_released()
    assert acknowledge(host) and host.enable()
    assert host.engine.state == "WAIT_OWNER"
    assert host.controller.snapshot().owner_track_id is None
    snapshot = host.snapshot()
    host.fixture_clock.now += 500_000_000
    assert not host.acknowledge_surface(
        **{key: snapshot[key] for key in ("sequence", "epoch", "revision")}
    )


def test_no_prediction_timeout_invalidates_owner_without_new_input(host):
    start(host)
    release(host)
    epoch = host.controller.snapshot().context_epoch
    host.fixture_clock.now += host.engine.policy.max_age_ns + 1
    host.tick()
    assert host.dispatcher.stopped and host.controller.snapshot().owner_track_id is None
    assert host.controller.snapshot().context_epoch > epoch
    assert host.backend.stop_requested.is_set()
    assert host.backend.page == 1


def test_independent_watchdog_stops_when_no_predictions_arrive(host):
    calibration, sha = host.engine.calibration, host.engine.calibration_sha256
    clock = ReplayClock()
    runtime = SyntheticPublicPresentation(
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
        watchdog=True,
    )
    runtime.fixture_clock, runtime.fixture_frame = clock, 0
    try:
        start(runtime)
        clock.now += runtime.engine.policy.max_age_ns + 1
        wait_for(lambda: runtime.dispatcher.stopped)
        assert runtime.backend.stop_requested.is_set()
        assert runtime.controller.snapshot().owner_track_id is None

        def broken_clock():
            raise ValueError("synthetic clock failure")

        runtime._raw_clock = broken_clock
        wait_for(runtime._closed.is_set)
        runtime._raw_clock = clock
        assert not runtime.enable()  # fatal watchdog exit requires a NEW host
    finally:
        runtime.close()
        runtime.close()
        runtime.stop()  # repeated lifecycle callbacks must not restart a dead release worker
        assert runtime.dispatcher.wait_released()


def test_healthy_predictions_cannot_replace_surface_response_or_heartbeat(host):
    start(host)
    release(host)
    _, ticket = motion(host, ack=False)
    assert ticket.future.result(timeout=2).accepted
    # No surface acknowledgements for the new revision. Fresh input alone is insufficient.
    for _ in range(22):
        step(host, ack=False)
    assert host.dispatcher.stopped and not host.controller.snapshot().enabled
    assert host.backend.page == 2


def test_heartbeat_can_keep_lease_but_cannot_fake_a_render_receipt(host):
    start(host)
    release(host)
    _, ticket = motion(host, ack=False)
    assert ticket.future.result(timeout=2).accepted
    for _ in range(22):
        step(host, ack=False)
        snap = host.snapshot()
        assert host.heartbeat(
            **{key: snap[key] for key in ("sequence", "epoch", "revision")}
        )
    assert host.dispatcher.stopped
    assert host.backend.page == 2


def test_stop_observable_before_delayed_backend_mutation(host):
    start(host)
    release(host)
    entered, proceed = Event(), Event()
    original = host.backend.execute

    def blocked(command, now):
        entered.set()
        assert proceed.wait(timeout=2)
        return original(command, now)

    host.backend.execute = blocked
    try:
        event, ticket = motion(host)
        assert event is not None and entered.wait(timeout=2)
        captured_epoch = host.controller.snapshot().context_epoch
        host.stop()
        assert host.controller.snapshot().context_epoch > captured_epoch
        assert host.backend.stop_requested.is_set()
        assert not host.dispatcher.safety_status()["safe_quiescent"]
        assert acknowledge(host)
        with pytest.raises(RuntimeError, match="quiescence"):
            host.enable()
        proceed.set()
        result = ticket.future.result(timeout=2)
        assert not result.accepted and result.page_after == 1
        assert host.dispatcher.wait_released()
        assert host.backend.page == 1
    finally:
        proceed.set()


def test_expired_queued_event_never_gets_new_command_ttl(host):
    start(host)
    release(host)
    entered, proceed = Event(), Event()
    original = host.backend.execute

    def blocked(command, now):
        entered.set()
        assert proceed.wait(timeout=2)
        return original(command, now)

    host.backend.execute = blocked
    try:
        _, ticket = motion(host)
        assert entered.wait(timeout=2)
        host.fixture_clock.now += host.engine.policy.max_age_ns + 1
        proceed.set()
        result = ticket.future.result(timeout=2)
        assert not result.accepted and result.page_after == 1
    finally:
        proceed.set()


def test_bad_source_alignment_and_clock_regression_stop_fail_closed(host):
    start(host)
    release(host)
    p, frame = synthetic_input(
        host.controller.snapshot(),
        timestamp_ns=host.fixture_clock.now,
        frame_id=999,
        model_version=host.engine.model_version,
    )
    with pytest.raises(ValueError, match="source mismatch"):
        host.update(p.model_copy(update={"source_frame_id": 998}), frame, ready=True)
    assert host.dispatcher.stopped and host.backend.page == 1
    host.fixture_clock.now = 0
    with pytest.raises(ValueError, match="clock"):
        host.tick()
    assert host.backend.stop_requested.is_set()


def test_complete_replay_authoritative_page_changes_and_independent_event_truth(
    tmp_path,
):
    output = tmp_path / "replay"
    result = run_replay(output, allow_synthetic=True)
    assert result["page_sequence"] == [1, 2, 3, 2]
    report = json.loads((output / "report.json").read_text())
    assert (
        report["synthetic"]
        and not report["live_available"]
        and not report["external_control_available"]
    )
    layers = report["events"]["kinds"]["command"]["layers"]
    assert layers["submitted"]["tp"] == layers["application_local"]["tp"] == 3
    assert layers["application_local"]["fp"] == layers["application_local"]["fn"] == 0
    assert layers["external_injected"]["tp"] == 0
    assert report["events"]["availability"][0]["active_ns"] > 0
    assert report["final"]["safety"]["safe_quiescent"]
    with pytest.raises(FileExistsError):
        run_replay(output, allow_synthetic=True)


def test_replay_is_deterministic_after_removing_run_specific_ids(tmp_path):
    reports = []
    for name in ("first", "second"):
        run_replay(tmp_path / name, allow_synthetic=True)
        reports.append(json.loads((tmp_path / name / "report.json").read_text()))

    def stable(report):
        return {
            "page_sequence": report["page_sequence"],
            "calibration_sha256": report["calibration_sha256"],
            "policy_sha256": report["policy_sha256"],
            "implementation_sha256": report["implementation_sha256"],
            "trace": [
                {
                    **{k: v for k, v in row.items() if k != "event_id"},
                    "event": row["event_id"] is not None,
                }
                for row in report["trace"]
            ],
        }

    assert stable(reports[0]) == stable(reports[1])


def test_cli_needs_opt_in_and_imports_no_camera_or_torch(tmp_path):
    code = "import sys; from backend.interaction.public_replay import main; assert 'torch' not in sys.modules and 'cv2' not in sys.modules and 'mediapipe' not in sys.modules; main()"
    base = [sys.executable, "-c", code, "--output", str(tmp_path / "report")]
    failed = subprocess.run(base, check=False, capture_output=True, text=True)
    assert failed.returncode == 2 and not (tmp_path / "report").exists()
    result = subprocess.run(
        base + ["--allow-synthetic"], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["page_sequence"] == [1, 2, 3, 2]
    optimized = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "backend.interaction.public_replay",
            "--allow-synthetic",
            "--output",
            str(tmp_path / "optimized"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert optimized.returncode == 0, optimized.stderr
    assert json.loads(optimized.stdout)["page_sequence"] == [1, 2, 3, 2]
