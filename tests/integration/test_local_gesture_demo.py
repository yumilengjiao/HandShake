"""PROGRAMMATIC CONTRACT SIMULATIONS only; never real-camera/learned-model evidence.

The in-memory provenance copies below exercise the new real-source gate. They are
not serialized as real model packages, not reported as real training, and do not
replace the negative tests proving synthetic packages are rejected at entry.
"""

from copy import deepcopy
from dataclasses import replace
import threading

import pytest

from algorithm.bundle import digest, json_bytes
from backend.interaction.live_demo import LocalDemoSession
from backend.interaction.model_session import ModelPresentationSession
from backend.interaction.public_replay import ReplayClock, acknowledge
from tests.integration.model_bridge_fixture import fixture_bundle
from tests.integration.test_model_session_bridge import Harness


def contract_inputs(tmp_path, kind="mlp"):
    bundle, calibration, _ = fixture_bundle(tmp_path, kind)
    identity = deepcopy(bundle.training_identity)
    identity["data"]["synthetic"] = False
    bundle = replace(bundle, synthetic_training=False, training_identity=identity)
    source = calibration.source.model_copy(
        update={"data_synthetic": False, "model_synthetic": False}
    )
    calibration = type(calibration).model_validate(
        calibration.model_dump() | {"source": source, "synthetic": False}
    )
    sha = digest(json_bytes(calibration.model_dump(mode="json")))
    return dict(bundle=bundle, calibration=calibration, calibration_sha256=sha)


@pytest.fixture(params=["mlp", "tcn"])
def demo(tmp_path, request):
    clock = ReplayClock()
    session = LocalDemoSession(
        allow_application_local=True,
        **contract_inputs(tmp_path, request.param),
        clock=clock,
        watchdog=False,
    )
    try:
        yield Harness(session, clock)
    finally:
        session.close()
        assert session.host.dispatcher.wait_released()


def test_live_contract_requires_separate_optin_and_never_changes_shadow_default(
    tmp_path,
):
    args = contract_inputs(tmp_path)
    with pytest.raises(ValueError, match="opt-in"):
        LocalDemoSession(**args, clock=ReplayClock(), watchdog=False)
    with pytest.raises(ValueError, match="unsupported execution"):
        ModelPresentationSession(
            **args,
            clock=ReplayClock(),
            timebase="monotonic",
            source_synthetic=False,
            execution="application_local_demo",
            watchdog=False,
        )
    for field, value in [
        ("timebase", "synthetic"),
        ("source_synthetic", True),
        ("allow_synthetic", True),
        ("execution", "shadow"),
    ]:
        with pytest.raises(ValueError, match="overridden"):
            LocalDemoSession(
                allow_application_local=True,
                **args,
                **{field: value},
                clock=ReplayClock(),
                watchdog=False,
            )


def test_synthetic_model_cannot_enter_live_demo(tmp_path):
    bundle, calibration, sha = fixture_bundle(tmp_path, "mlp")
    with pytest.raises(ValueError, match="real model"):
        LocalDemoSession(
            allow_application_local=True,
            bundle=bundle,
            calibration=calibration,
            calibration_sha256=sha,
            clock=ReplayClock(),
            watchdog=False,
        )


def test_real_clock_contract_changes_only_local_pages_after_enable_owner_and_release(
    demo,
):
    h = demo
    assert h.session.host.backend.page == 1
    assert not h.session.snapshot()["actions_enabled"]
    for _ in range(5):
        assert h.step(0.7).event is None
    h.enable()
    h.warm_and_bind()
    assert not h.session.snapshot()["actions_enabled"]
    h.rewarm_release()
    results = h.right()
    assert sum(r.event is not None for r in results) == 1
    assert h.session.host.backend.page == 2
    event = next(r.event for r in results if r.event is not None)
    assert event.timebase == "monotonic"
    commands = h.session.host.commands
    assert len(commands) == 1 and commands[0].timebase == "monotonic"
    assert commands[0].target_context.startswith("local-presentation:")
    assert h.session.snapshot()["external_control_available"] is False
    for x in (0.68, 0.62, 0.56, 0.5):
        assert h.step(x, dt=50_000_000).event is None
    for _ in range(20):
        h.step()
    assert sum(r.event is not None for r in h.right()) == 1
    assert h.session.host.backend.page == 3


def test_ui_heartbeat_never_substitutes_for_missing_model_evidence(demo):
    h = demo
    h.enable()
    h.warm_and_bind()
    h.rewarm_release()
    h.clock.now += h.session.host.engine.policy.max_age_ns + 1
    acknowledge(h.session.host)
    assert h.session.host.dispatcher.stopped
    assert not h.session.snapshot()["actions_enabled"]
    assert h.session.host.backend.page == 1


def test_stop_is_outside_inference_gate_and_late_result_cannot_change_page(demo):
    h = demo
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
        assert not stop.is_alive()
        assert h.session.host.backend.stop_requested.is_set()
        assert not h.session.snapshot()["actions_enabled"]
    finally:
        proceed.set()
        worker.join(2)
        h.session.bundle.model.forward = original
    assert results[0].event is None and results[0].notice is None
    assert h.session.host.backend.page == 1


def test_render_receipt_expiry_stops_and_old_surface_cannot_reenable(demo):
    h = demo
    h.enable()
    snapshot = h.session.snapshot()
    h.clock.now += 2_000_000_000
    h.session.host.tick()
    assert h.session.host.dispatcher.stopped
    assert not h.session.host.acknowledge_surface(
        sequence=snapshot["sequence"],
        epoch=snapshot["epoch"],
        revision=snapshot["revision"],
    )
    assert not h.session.enable(epoch=snapshot["epoch"])
