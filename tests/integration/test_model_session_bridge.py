"""The whole observed-frame -> real network forward -> policy -> page chain; no training."""

import threading

import pytest

from backend.interaction.model_session import ModelPresentationSession
from backend.interaction.public_replay import ReplayClock, acknowledge, synthetic_input
from shared.contracts import PerceptionStatus
from shared.observations import ObservationFrame
from tests.integration.model_bridge_fixture import fixture_bundle


class Harness:
    def __init__(self, session, clock):
        self.session, self.clock, self.index = session, clock, 0

    def frame(self, x=0.5, *, dt=30_000_000):
        self.clock.now += dt
        self.index += 1
        # Discard the old helper's artificial prediction. Only its geometry enters the bridge.
        return synthetic_input(
            self.session.controller.snapshot(),
            timestamp_ns=self.clock.now,
            frame_id=self.index,
            model_version="unused-by-bridge",
            x=x,
        )[1]

    def step(self, x=0.5, *, ack=True, dt=30_000_000):
        result = self.session.process(self.frame(x, dt=dt))
        if result.ticket:
            assert result.ticket.future.result(timeout=2).accepted
        if ack:
            assert acknowledge(self.session.host)
        return result

    def drop(self, *, dt=15_000_000, reason="latest_frame_replaced"):
        frame = self.frame(dt=dt)
        result = self.session.process(
            ObservationFrame(
                frame.info,
                frame.info.status(PerceptionStatus.NOT_PROCESSED, reason=reason),
            )
        )
        assert result.event is None and result.ticket is None
        assert acknowledge(self.session.host)
        return result

    def enable(self):
        assert acknowledge(self.session.host) and self.session.enable(
            epoch=self.session.controller.snapshot().context_epoch
        )
        assert acknowledge(self.session.host)

    def request(self, track_id="operator", epoch=None):
        return self.session.request_owner(
            track_id,
            epoch=self.session.controller.snapshot().context_epoch
            if epoch is None
            else epoch,
        )

    def warm_and_bind(self):
        for _ in range(32):
            self.step()
        assert self.session.snapshot()["eligible_owner"] == "operator"
        assert self.request()
        old_epoch = self.session.controller.snapshot().context_epoch
        result = self.step()
        assert result.reason == "owner_bound_model_history_reset"
        assert result.notice.context_epoch == str(old_epoch)
        assert self.session.controller.snapshot().context_epoch > old_epoch
        return result

    def rewarm_release(self):
        for i in range(32):
            result = self.step()
            assert result.event is None
            if i < 31:
                assert self.session.host.warmup is not None
        assert self.session.host.warmup is None
        assert self.session.host.engine.state == "WAIT_RELEASE"
        for _ in range(15):
            self.step()
        assert self.session.host.engine.state == "ACTIVE"

    def right(self):
        return [self.step(x, dt=50_000_000) for x in (0.56, 0.62, 0.68, 0.74)]


@pytest.fixture(params=["mlp", "tcn"])
def setup(request, tmp_path):
    bundle, calibration, sha = fixture_bundle(tmp_path, request.param)
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
    yield Harness(session, clock)
    session.close()
    assert session.host.dispatcher.wait_released()


def test_capture_drops_do_not_flicker_eligibility_consume_owner_or_abort_rewarm(setup):
    h = setup
    h.enable()
    for _ in range(32):
        h.step(dt=15_000_000)
        before = h.session.snapshot()
        h.drop()
        after = h.session.snapshot()
        assert after["classification"] == before["classification"]
        assert after["bridge_reason"] == before["bridge_reason"]
    assert after["eligible_owner"] == "operator"
    assert h.request()
    h.drop()
    assert h.session.snapshot()["owner_request_pending"]
    h.step(dt=15_000_000)
    assert h.session.controller.snapshot().owner_track_id == "operator"
    for i in range(32):
        warm = h.session.host.warmup
        h.drop()
        assert h.session.host.warmup == warm
        assert h.session.controller.snapshot().enabled
        result = h.step(dt=15_000_000)
        assert result.event is None
        assert result.notice.window_length == i + 1
    assert h.session.host.warmup is None
    for _ in range(15):
        h.drop()
        h.step(dt=15_000_000)
    assert h.session.host.engine.state == "ACTIVE"
    assert h.session.host.backend.page == 1


def test_only_capture_drop_ledger_preserves_result_not_perception_failure(setup):
    h = setup
    h.enable()
    for _ in range(32):
        h.step()
    assert h.session.snapshot()["eligible_owner"] == "operator"
    h.drop(reason="model_not_loaded")
    value = h.session.snapshot()
    assert value["eligible_owner"] is None
    assert value["classification"]["reason"] == "not_processed"
    assert "model_not_loaded" in value["bridge_reason"]
    assert not h.request()


def test_capture_drops_never_refresh_prediction_age_or_warmup_progress(setup):
    h = setup
    h.enable()
    for _ in range(32):
        h.step()
    source = h.session.snapshot()["classification"]["source_timestamp_ns"]
    for _ in range(10):
        h.drop()
    assert h.session.snapshot()["classification"]["source_timestamp_ns"] == source
    h.drop()
    assert h.session.snapshot()["eligible_owner"] is None
    assert not h.request()
    h.step()
    assert h.request()
    h.step()
    warm = h.session.host.warmup
    for _ in range(10):
        h.drop()
    assert h.session.host.warmup == warm
    h.drop()
    assert h.session.host.warmup is None
    assert h.session.host.dispatcher.stopped
    assert h.session.host.backend.page == 1


def test_owner_button_reports_neutral_geometry_failure_without_relaxing_gate(setup):
    h = setup
    h.enable()
    for _ in range(32):
        h.step()
    assert h.session.snapshot()["eligible_owner"] == "operator"
    frame = h.frame()
    data = frame.observations[0].model_dump()
    for mcp in (5, 9, 13, 17):
        y = data["landmarks_image"][mcp][1]
        data["landmarks_image"][mcp + 1][1] = y - 0.05
        data["landmarks_image"][mcp + 2][1] = y - 0.09
    observation = type(frame.observations[0]).model_validate(data)
    frame = type(frame)(frame.info, frame.status, (observation,))
    result = h.session.process(frame)
    assert result.notice.state == "predicted"
    assert result.reason == "owner_requires_neutral_pose"
    assert h.session.snapshot()["eligible_owner"] is None
    assert not h.request()
    assert h.session.controller.snapshot().owner_track_id is None
    assert h.session.host.backend.page == 1


@pytest.mark.parametrize("static_id", [1, 2])
def test_model_bridge_outside_live_demo_still_requires_nog(setup, static_id):
    import torch

    h = setup
    model = h.session.bundle.model
    head = model.classifier if hasattr(model, "classifier") else model.network[-1]
    with torch.no_grad():
        head.bias.zero_()
        head.bias[static_id] = 10
    h.enable()
    for _ in range(32):
        result = h.step()
    assert result.reason == "owner_requires_nog"
    assert h.session.snapshot()["eligible_owner"] is None
    assert not h.request()
    assert h.session.controller.snapshot().owner_track_id is None
    assert h.session.host.backend.page == 1


def test_owner_epoch_cold_window_does_not_deadlock_or_bypass_release(setup):
    h = setup
    h.enable()
    h.warm_and_bind()
    h.rewarm_release()
    results = h.right()
    assert [r.notice.logits.index(max(r.notice.logits)) for r in results] == [8] * 4
    assert sum(r.event is not None for r in results) == 1
    assert h.session.host.backend.page == 2
    for x in (0.68, 0.62, 0.56, 0.5):
        assert h.step(x, dt=50_000_000).event is None
    for _ in range(20):
        h.step()
    assert sum(r.event is not None for r in h.right()) == 1
    assert h.session.host.backend.page == 3
    for x in (0.68, 0.62, 0.56, 0.5):
        h.step(x, dt=50_000_000)
    for _ in range(20):
        h.step()
    left = [h.step(x, dt=50_000_000) for x in (0.44, 0.38, 0.32, 0.26)]
    assert sum(r.event is not None for r in left) == 1
    assert h.session.host.backend.page == 2


def test_default_shadow_runs_policy_but_cannot_dispatch(tmp_path):
    bundle, calibration, sha = fixture_bundle(tmp_path, "mlp")
    clock = ReplayClock()
    session = ModelPresentationSession(
        bundle=bundle,
        calibration=calibration,
        calibration_sha256=sha,
        clock=clock,
        allow_synthetic=True,
    )
    try:
        h = Harness(session, clock)
        h.enable()
        h.warm_and_bind()
        h.rewarm_release()
        result = h.right()[-1]
        assert (
            result.event is not None
            and result.ticket is None
            and result.reason == "shadow_candidate"
        )
        assert (
            session.host.backend.page == 1
            and session.host.backend.stop_requested.is_set()
        )
        assert not session.host._authorized(session.controller.snapshot())
        from shared.contracts import Action, ActionCommand, Capability

        event = result.event
        command = ActionCommand(
            command_id="shadow-must-reject",
            source_event_id=event.event_id,
            stream_id=event.stream_id,
            context_epoch=event.context_epoch,
            timebase=event.timebase,
            action=Action.NEXT_SLIDE,
            issued_at_ns=clock.now,
            expires_at_ns=clock.now + 100_000_000,
            target_context=event.target_context,
            required_capabilities=[Capability.PRESENTATION],
        )
        assert (
            not session.host.dispatcher.submit(command)
            .future.result(timeout=2)
            .accepted
        )
    finally:
        session.close()


def test_stop_does_not_wait_for_blocked_inference_and_late_output_is_discarded(setup):
    h = setup
    h.enable()
    h.warm_and_bind()
    h.rewarm_release()
    entered, proceed = threading.Event(), threading.Event()
    original = h.session.bundle.model.forward

    def blocked(value):
        entered.set()
        assert proceed.wait(2)
        return original(value)

    h.session.bundle.model.forward = blocked
    results = []
    worker = threading.Thread(
        target=lambda: results.append(h.session.process(h.frame(0.56)))
    )
    worker.start()
    try:
        assert entered.wait(1)
        stop = threading.Thread(target=h.session.stop)
        stop.start()
        stop.join(0.5)
        assert not stop.is_alive(), "stop was queued behind neural inference"
        assert h.session.host.dispatcher.stopped
        assert h.session.host.backend.page == 1
        assert h.session.host.dispatcher.wait_released()
        assert acknowledge(h.session.host)
        assert not h.session.enable(
            epoch=h.session.controller.snapshot().context_epoch
        ), "old model work must quiesce before re-enabling"
        assert h.session.snapshot()["model_busy"], (
            "stop does not pretend native model work has exited"
        )
    finally:
        proceed.set()
        worker.join(2)
        h.session.bundle.model.forward = original
    assert results[0].reason == "cancelled_old_model_work"
    assert results[0].notice is None and results[0].event is None
    assert h.session.snapshot()["classification"] is None


def test_warmup_progress_does_not_disable_timeout_or_reuse_old_prediction(setup):
    h = setup
    h.enable()
    h.warm_and_bind()
    for _ in range(10):
        h.step()
    assert h.session.host.warmup is not None
    h.clock.now += h.session.host.engine.policy.max_age_ns + 1
    h.session.host.tick()
    assert h.session.host.dispatcher.stopped and h.session.host.warmup is None
    assert h.session.host.backend.page == 1
    assert h.session.process(h.frame()).event is None


def test_owner_request_needs_fresh_preview_and_only_consumes_next_result(setup):
    h = setup
    h.enable()
    assert not h.request()
    for _ in range(32):
        h.step()
    assert not h.request("unseen-hand")
    assert not h.request(epoch=0)  # an old UI epoch cannot authorize a new session
    assert not h.request(None)
    assert not h.request(epoch=True)
    assert h.request()
    assert not h.request()
    h.clock.now += 600_000_000
    result = h.step()
    assert h.session.controller.snapshot().owner_track_id is None
    assert not h.session.snapshot()["owner_request_pending"]
    assert result.event is None
