"""Pinned replay and failure tests for the actual model/session bridge."""

import json
import threading

import pytest

from algorithm.bundle import digest, json_bytes
from algorithm.replay import ReplayArchive, ReplayFrame, json_bytes_compact
from backend.controller import ApplicationController
from backend.interaction.model_replay import (
    SessionArchive,
    SessionItem,
    main,
    run_session,
)
from backend.interaction.model_session import ModelPresentationSession
from backend.interaction.public_replay import ReplayClock, synthetic_input
from tests.integration.model_bridge_fixture import fixture_bundle
from tests.integration.test_model_session_bridge import Harness


def recording(root, *, execute=True, long_gap=False):
    controller = ApplicationController("bridge-replay", timebase="synthetic")
    now, index = 0, 0
    steps = [
        SessionItem(kind="surface_ack", at_ns=0, epoch=0, revision=0),
        SessionItem(kind="start", at_ns=0, epoch=0),
        SessionItem(kind="surface_ack", at_ns=0, epoch=1, revision=0),
    ]

    def frame(x=0.5, dt=30_000_000, epoch=1, revision=0):
        nonlocal now, index
        now += dt
        index += 1
        _, f = synthetic_input(
            controller.snapshot(),
            timestamp_ns=now,
            frame_id=index,
            model_version="not-used",
            x=x,
        )
        steps.append(
            SessionItem(
                kind="frame",
                at_ns=now,
                frame=ReplayFrame(
                    width=f.info.width,
                    height=f.info.height,
                    transform=f.info.transform,
                    status=f.status,
                    observations=f.observations,
                ),
            )
        )
        steps.append(
            SessionItem(kind="surface_ack", at_ns=now, epoch=epoch, revision=revision)
        )

    for _ in range(32):
        frame()
    steps.append(
        SessionItem(kind="request_owner", at_ns=now, epoch=1, track_id="operator")
    )
    frame(epoch=2)
    if long_gap:
        now += 800_000_000
        steps.append(SessionItem(kind="tick", at_ns=now))
    else:
        for _ in range(32 + 15):
            frame(epoch=2)
        for i, x in enumerate((0.56, 0.62, 0.68, 0.74)):
            frame(x, 50_000_000, epoch=2, revision=1 if i == 3 and execute else 0)
    steps.append(SessionItem(kind="stop", at_ns=now + 1))
    root.mkdir()
    raw = b"".join(json_bytes_compact(step.model_dump(mode="json")) for step in steps)
    (root / "events.jsonl").write_bytes(raw)
    metadata = dict(
        format_version="model-session-replay-v1",
        synthetic=True,
        review_reference="untrained-analytical-fixture",
        stream_id="bridge-replay",
        timebase="synthetic",
        events=len(steps),
        events_bytes=len(raw),
        events_sha256=digest(raw),
    )
    data = json_bytes(metadata)
    (root / "manifest.json").write_bytes(data)
    return digest(data)


@pytest.mark.parametrize("kind", ["mlp", "tcn"])
@pytest.mark.parametrize("execute", [False, True])
def test_archive_to_real_forward_policy_and_local_application(
    tmp_path, kind, execute, monkeypatch
):
    bundle, _, sha = fixture_bundle(tmp_path, kind)
    pin = recording(tmp_path / "input", execute=execute)
    import torch

    def forbidden(*a, **k):
        pytest.fail("no training/checkpoint loading allowed")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    result = run_session(
        tmp_path / "input",
        tmp_path / "out",
        manifest_sha256=pin,
        package=tmp_path / "bundle",
        model_sha256=bundle.manifest_sha256,
        calibration_path=tmp_path / "calibration.json",
        calibration_sha256=sha,
        allow_synthetic=True,
        execute_synthetic=execute,
    )
    assert result["complete"] and result["frames"] == 84
    assert result["counts"]["policy_candidates"] == 1
    assert result["counts"].get("rejected_controls", 0) == 0
    assert result["page_sequence"] == ([1, 2] if execute else [1])
    assert result["final"]["state"] == "DISABLED" and not result["live_available"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "out/trace.jsonl").read_text().splitlines()
    ]
    warming = [r for r in rows if r.get("model_warming")]
    assert warming and all(not r.get("event") for r in warming)
    # A typed inference result is retained in its old captured epoch at binding;
    # new generation classifications are freshly recomputed, never relabelled.
    bind = next(r for r in rows if r.get("reason") == "owner_bound_model_history_reset")
    assert bind["classification"]["context_epoch"] == "1" and bind["epoch"] == 2


def test_gap_timer_fires_at_deadline_without_new_predictions(tmp_path):
    bundle, _, sha = fixture_bundle(tmp_path, "mlp")
    pin = recording(tmp_path / "input", long_gap=True)
    report = run_session(
        tmp_path / "input",
        tmp_path / "out",
        manifest_sha256=pin,
        package=tmp_path / "bundle",
        model_sha256=bundle.manifest_sha256,
        calibration_path=tmp_path / "calibration.json",
        calibration_sha256=sha,
        allow_synthetic=True,
        execute_synthetic=True,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "out/trace.jsonl").read_text().splitlines()
    ]
    deadline = next(r for r in rows if r["kind"] == "deadline")
    assert deadline["at_ns"] == 990_000_000 + 150_000_000 + 1
    assert deadline["reason"] == "model_warmup_no_progress"
    assert report["page_sequence"] == [1]


@pytest.mark.parametrize(
    "change", [dict(track_id="operator"), dict(epoch=1), dict(revision=1)]
)
def test_frame_cannot_smuggle_control_fields(change):
    with pytest.raises(ValueError):
        SessionItem(kind="tick", at_ns=0, **change)


def test_archive_version_is_not_accepted_by_old_readonly_reader(tmp_path):
    pin = recording(tmp_path / "input")
    assert SessionArchive(tmp_path / "input", expected_sha256=pin).frames == 84
    with pytest.raises(ValueError):
        ReplayArchive(tmp_path / "input", expected_sha256=pin)


def test_real_clock_source_is_shadow_only_and_bad_bindings_fail_early(tmp_path):
    bundle, calibration, sha = fixture_bundle(tmp_path, "mlp")
    options = dict(
        bundle=bundle,
        calibration=calibration,
        calibration_sha256=sha,
        clock=ReplayClock(),
        allow_synthetic=True,
        timebase="recording_relative",
        source_synthetic=False,
    )
    with pytest.raises(ValueError, match="synthetic-only"):
        ModelPresentationSession(**options, execution="synthetic_application_local")
    session = ModelPresentationSession(**options)
    try:
        h = Harness(session, options["clock"])
        h.enable()
        h.warm_and_bind()
        h.rewarm_release()
        assert h.right()[-1].event is not None
        assert session.host.backend.page == 1
    finally:
        session.close()
    wrong = calibration.model_copy(
        update={
            "source": calibration.source.model_copy(
                update={"endpoint_sha256": "0" * 64}
            )
        }
    )
    with pytest.raises(ValueError, match="inventory"):
        ModelPresentationSession(**(options | {"calibration": wrong}))


def test_busy_overlap_revokes_inflight_result_instead_of_queueing(tmp_path):
    bundle, calibration, sha = fixture_bundle(tmp_path, "mlp")
    clock = ReplayClock()
    session = ModelPresentationSession(
        bundle=bundle,
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
        execution="synthetic_application_local",
        watchdog=False,
    )
    h = Harness(session, clock)
    entered, proceed = threading.Event(), threading.Event()
    original = bundle.model.forward
    result = []
    try:
        h.enable()
        h.warm_and_bind()
        h.rewarm_release()

        def block(value):
            entered.set()
            assert proceed.wait(2)
            return original(value)

        bundle.model.forward = block
        worker = threading.Thread(
            target=lambda: result.append(session.process(h.frame(0.56)))
        )
        worker.start()
        assert entered.wait(1)
        dropped = session.process(h.frame(0.62))
        assert dropped.reason == "busy_dropped_and_stopped"
        assert session.host.dispatcher.stopped
        proceed.set()
        worker.join(2)
        assert result[0].reason == "cancelled_old_model_work"
        assert session.host.backend.page == 1
    finally:
        proceed.set()
        bundle.model.forward = original
        session.close()


def test_warmup_source_conflict_and_clock_budget_cannot_renew_authority(tmp_path):
    bundle, calibration, sha = fixture_bundle(tmp_path, "mlp")
    clock = ReplayClock()
    session = ModelPresentationSession(
        bundle=bundle,
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
        watchdog=False,
    )
    h = Harness(session, clock)
    try:
        h.enable()
        h.warm_and_bind()
        deadline = session.host.warmup.deadline_ns
        for _ in range(10):
            h.step()
        assert session.host.warmup.deadline_ns == deadline
        # Even artificially recent health cannot extend the fixed warmup lease.
        from dataclasses import replace

        clock.now = deadline
        session.host.warmup = replace(session.host.warmup, source_ns=clock.now)
        session.host._last_heartbeat = (
            clock.now
        )  # isolate the absolute warmup budget from the independent UI lease
        session.host.tick()
        assert (
            session.host.dispatcher.stopped
            and session.host.reason == "model_warmup_deadline"
        )
    finally:
        session.close()


def test_reconnect_clears_requests_and_needs_new_explicit_enable(tmp_path):
    bundle, calibration, sha = fixture_bundle(tmp_path, "mlp")
    clock = ReplayClock()
    session = ModelPresentationSession(
        bundle=bundle,
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
        watchdog=False,
    )
    h = Harness(session, clock)
    try:
        h.enable()
        for _ in range(32):
            h.step()
        assert h.request()
        old = h.frame()
        session.reconnect("another-source")
        assert not session.snapshot()["owner_request_pending"]
        assert session.snapshot()["classification"] is None
        assert session.process(old).event is None
        assert session.host.dispatcher.wait_released()
        h.enable()
        assert h.step().notice.state == "warming_up"
    finally:
        session.close()


@pytest.mark.parametrize("problem", ["missing", "multiple", "quality", "duplicate"])
def test_invalid_model_warmup_cannot_act_or_restore_owner(tmp_path, problem):
    from shared.observations import ObservationFrame

    bundle, calibration, sha = fixture_bundle(tmp_path, "mlp")
    clock = ReplayClock()
    session = ModelPresentationSession(
        bundle=bundle,
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
        execution="synthetic_application_local",
        watchdog=False,
    )
    h = Harness(session, clock)
    try:
        h.enable()
        h.warm_and_bind()
        frame = h.frame()
        if problem == "missing":
            frame = ObservationFrame(
                frame.info,
                frame.info.status("processed_no_hand", processed_at_ns=clock.now),
            )
        elif problem == "multiple":
            frame = ObservationFrame(
                frame.info,
                frame.status,
                frame.observations
                + (frame.observations[0].model_copy(update={"track_id": "other"}),),
            )
        elif problem == "quality":
            frame = ObservationFrame(
                frame.info,
                frame.status,
                (frame.observations[0].model_copy(update={"quality": 0.1}),),
            )
        elif problem == "duplicate":
            assert session.process(frame).event is None
        result = session.process(frame)
        assert result.event is None and result.ticket is None
        assert session.host.dispatcher.stopped and session.host.warmup is None
        assert session.controller.snapshot().owner_track_id is None
        assert session.host.backend.page == 1
    finally:
        session.close()


def test_typed_result_is_current_call_only_and_raw_logits_are_not_inputs(tmp_path):
    from shared.observations import ObservationFrame

    bundle, calibration, sha = fixture_bundle(tmp_path, "mlp")
    clock = ReplayClock()
    session = ModelPresentationSession(
        bundle=bundle,
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
        watchdog=False,
    )
    h = Harness(session, clock)
    try:
        h.enable()
        for _ in range(32):
            h.step()
        prediction = session._classifier.last_prediction
        assert prediction is not None and prediction.inference_timestamp_ns == clock.now
        frame = h.frame()
        missing = ObservationFrame(
            frame.info,
            frame.info.status("processed_no_hand", processed_at_ns=clock.now),
        )
        assert session.process(missing).notice.reason == "no_hand"
        assert session._classifier.last_prediction is None
        with pytest.raises(ValueError, match="whole observation"):
            session.process(prediction)
        assert session.host.dispatcher.stopped
    finally:
        session.close()


def test_pre_enable_raw_frame_cannot_warm_a_new_generation(tmp_path):
    from backend.interaction.public_replay import acknowledge

    bundle, calibration, sha = fixture_bundle(tmp_path, "mlp")
    clock = ReplayClock()
    session = ModelPresentationSession(
        bundle=bundle,
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
        watchdog=False,
    )
    h = Harness(session, clock)
    try:
        h.enable()
        old = h.frame()
        clock.now += 10_000_000
        session.stop()
        assert session.host.dispatcher.wait_released()
        assert acknowledge(session.host)
        assert not session.enable(epoch=0)
        assert session.enable(epoch=session.controller.snapshot().context_epoch)
        result = session.process(old)
        assert result.reason == "frame_predates_context" and result.notice is None
        assert h.step().notice.window_length == 1
    finally:
        session.close()


def test_watchdog_revokes_while_the_network_itself_is_blocked(tmp_path):
    import time

    bundle, calibration, sha = fixture_bundle(tmp_path, "mlp")
    clock = ReplayClock()
    session = ModelPresentationSession(
        bundle=bundle,
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
        execution="synthetic_application_local",
        watchdog=True,
    )
    h = Harness(session, clock)
    entered, proceed = threading.Event(), threading.Event()
    original = bundle.model.forward
    results = []
    worker = None
    try:
        h.enable()
        h.warm_and_bind()
        h.rewarm_release()

        def blocked(value):
            entered.set()
            assert proceed.wait(2)
            return original(value)

        bundle.model.forward = blocked
        worker = threading.Thread(
            target=lambda: results.append(session.process(h.frame(0.56)))
        )
        worker.start()
        assert entered.wait(1)
        clock.now += session.host.engine.policy.max_age_ns + 1
        deadline = time.monotonic() + 1
        while not session.host.dispatcher.stopped and time.monotonic() < deadline:
            time.sleep(0.005)
        assert session.host.dispatcher.stopped
        assert session.host.backend.page == 1 and session.snapshot()["model_busy"]
        proceed.set()
        worker.join(2)
        assert results[0].reason == "cancelled_old_model_work"
    finally:
        proceed.set()
        if worker is not None:
            worker.join(2)
        bundle.model.forward = original
        session.close()


def test_bad_pin_no_output_and_cli_entry(tmp_path, capsys):
    bundle, _, sha = fixture_bundle(tmp_path, "mlp")
    pin = recording(tmp_path / "input", execute=False)
    args = [
        "--input",
        str(tmp_path / "input"),
        "--sha256",
        pin,
        "--package",
        str(tmp_path / "bundle"),
        "--model-sha256",
        bundle.manifest_sha256,
        "--calibration",
        str(tmp_path / "calibration.json"),
        "--calibration-sha256",
        "0" * 64,
        "--output",
        str(tmp_path / "out"),
        "--allow-synthetic",
    ]
    with pytest.raises(SystemExit):
        main(args)
    assert not (tmp_path / "out").exists()
    args[args.index("--calibration-sha256") + 1] = sha
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["execution"] == "shadow"
