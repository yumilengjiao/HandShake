"""Programmatic contract simulation of the native bridge, NOT real camera accuracy.

Uses the existing untrained analytical network fixtures. The in-memory provenance
copies are never exported as real models; all production synthetic gates remain.
"""

import pytest

from backend.interaction.live_demo import LocalDemoSession
from backend.interaction.public_replay import ReplayClock
from backend.native_gesture import NativeGestureSession
from backend.perception.local_demo import DemoClassifier
from backend.perception.preview import CameraFrame
from tests.integration.test_local_gesture_demo import contract_inputs, set_fixture_neutral
from tests.integration.test_model_session_bridge import Harness
from tests.unit.test_native_gesture import ASSETS, Worker, command


@pytest.mark.parametrize("kind", ["mlp", "tcn"])
@pytest.mark.parametrize("with_capture_drops", [False, True])
@pytest.mark.parametrize("neutral_id", [0, 1, 2])
def test_native_commands_to_model_policy_and_visible_pages(
    tmp_path, kind, with_capture_drops, neutral_id,
):
    clock = ReplayClock()
    bridge = NativeGestureSession(
        assets=ASSETS,
        perception_model="not-a-camera",
        worker_factory=Worker,
        clock=clock,
    )
    bridge.accept(command(bridge, "open_camera"))
    worker = bridge.worker
    args = contract_inputs(tmp_path, kind)
    session = LocalDemoSession(
        allow_application_local=True,
        deck=bridge.deck,
        **args,
        clock=clock,
        watchdog=False,
    )
    worker.adapter = DemoClassifier(
        args["bundle"],
        args["calibration"],
        args["calibration_sha256"],
        worker.stop_requested,
        clock=clock,
    )
    worker.adapter.session = session
    set_fixture_neutral(session, neutral_id)
    h = Harness(session, clock)

    def paint():
        worker.latest = CameraFrame(
            h.index, clock.now, 640, 480, b"", (), "processed_hands", 30
        )
        value = bridge.snapshot()
        bridge.accept(command(bridge, "heartbeat", value))
        bridge.accept(command(bridge, "surface_ack", value))
        return value

    def step(x=0.5, dt=30_000_000):
        if with_capture_drops:
            before_drop = bridge.snapshot()
            h.drop()
            after_drop = bridge.snapshot()
            assert after_drop["eligible_owner"] == before_drop["eligible_owner"]
            assert after_drop["policy_reason"] == before_drop["policy_reason"]
            assert after_drop["owner_pending"] == before_drop["owner_pending"]
        value = session.process(h.frame(x, dt=dt))
        if value.ticket:
            assert value.ticket.future.result(timeout=2).accepted
        paint()
        return value

    try:
        paint()
        bridge.accept(command(bridge, "enable"))
        paint()
        for _ in range(32):
            step()
        before = bridge.snapshot()
        assert before["eligible_owner"] and before["page"] == 1
        assert before["top_class"] == args["bundle"].vocabulary.tokens[neutral_id]
        assert not before["owner_bound"] and not before["actions_enabled"]
        bridge.accept(command(bridge, "choose_owner", before))
        step()
        assert bridge.snapshot()["owner_bound"]
        for _ in range(47):
            step()
        for x in [0.56, 0.62, 0.68, 0.74]:
            step(x, 50_000_000)
        value = bridge.snapshot()
        assert value["page"] == 2 and value["revision"] == "1"
        assert (
            value["scope"] == "application-local"
            and not value["external_control_available"]
        )
        assert len(session.host.commands) == 1
        # UI stop revokes even a previously rendered/eligible request.
        bridge.accept(command(bridge, "stop", value))
        bridge.accept(command(bridge, "enable", value))
        for x in [0.56, 0.62, 0.68, 0.74]:
            step(x, 50_000_000)
        value = bridge.snapshot()
        assert (
            value["page"] == 2 and not value["actions_enabled"] and not value["enabled"]
        )
        assert len(session.host.commands) == 1
    finally:
        session.close()
        bridge.close()
        assert session.host.dispatcher.wait_released()
