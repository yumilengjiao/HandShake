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


def set_fixture_neutral(session, class_id=2, score=10.0, *, tie=False):
    """Change only the UNTRAINED in-memory test head, never any model package."""
    import torch

    model = session.bundle.model
    head = model.classifier if hasattr(model, "classifier") else model.network[-1]
    with torch.no_grad():
        head.bias.zero_()
        head.bias[class_id] = score
        if tie:
            head.bias[1] = head.bias[2] = score


@pytest.mark.parametrize("static_id", [1, 2])
def test_static_pointing_pose_can_be_explicitly_selected_but_never_auto_flips(demo, static_id):
    h = demo
    set_fixture_neutral(h.session, static_id)
    h.enable()
    for _ in range(40):
        result = h.step()
        assert result.event is None
    value = h.session.snapshot()
    assert value["classification"]["logits"].index(max(value["classification"]["logits"])) == static_id
    assert value["eligible_owner"] == "operator"
    assert h.session.controller.snapshot().owner_track_id is None
    assert h.session.host.backend.page == 1
    assert not value["actions_enabled"]
    assert h.request()
    h.step()
    assert h.session.controller.snapshot().owner_track_id == "operator"
    assert not h.session.snapshot()["actions_enabled"]
    h.rewarm_release()
    assert h.session.snapshot()["actions_enabled"]
    for _ in range(30):
        assert h.step().event is None
    assert h.session.host.backend.page == 1
    assert not h.session.host.commands
    assert sum(r.event is not None for r in h.right()) == 1
    assert h.session.host.backend.page == 2
    h.session.stop()
    for _ in range(5):
        assert h.step(0.7).event is None
    assert h.session.host.backend.page == 2


@pytest.mark.parametrize(
    "class_id,score,tie,reason",
    [
        (2, 0.1, False, "owner_requires_confident_neutral"),
        (1, 10, True, "owner_requires_confident_neutral"),
        *[(i, 10, False, "owner_requires_static_neutral") for i in range(3, 15)],
    ],
)
def test_local_neutral_never_accepts_low_confidence_ties_or_dynamic_classes(
    demo, class_id, score, tie, reason,
):
    h = demo
    set_fixture_neutral(h.session, class_id, score, tie=tie)
    h.enable()
    for _ in range(32):
        result = h.step()
        assert result.event is None
    assert result.notice.state == "predicted"
    assert result.reason == reason
    assert h.session.snapshot()["eligible_owner"] is None
    assert not h.request()
    assert h.session.controller.snapshot().owner_track_id is None
    assert not h.session.snapshot()["actions_enabled"]
    assert h.session.host.backend.page == 1


@pytest.mark.parametrize("failure", ["stale", "pose", "class_changed"])
def test_local_static_owner_consent_still_requires_fresh_neutral_confirmation(demo, failure):
    h = demo
    set_fixture_neutral(h.session)
    h.enable()
    for _ in range(32):
        h.step()
    assert h.session.snapshot()["eligible_owner"] == "operator"
    if failure == "stale":
        h.clock.now += h.session.host.engine.policy.max_age_ns + 1
        assert h.session.snapshot()["eligible_owner"] is None
        assert not h.request()
    else:
        assert h.request()
        if failure == "class_changed":
            set_fixture_neutral(h.session, 3)
            result = h.step()
        else:
            frame = h.frame()
            data = frame.observations[0].model_dump()
            for mcp in (5, 9, 13, 17):
                y = data["landmarks_image"][mcp][1]
                data["landmarks_image"][mcp + 1][1] = y - 0.05
                data["landmarks_image"][mcp + 2][1] = y - 0.09
            observation = type(frame.observations[0]).model_validate(data)
            result = h.session.process(type(frame)(frame.info, frame.status, (observation,)))
        assert result.reason == "owner_request_rejected_or_expired"
        assert result.event is None
        assert not h.session.snapshot()["owner_request_pending"]
        set_fixture_neutral(h.session)
        h.step()
        assert h.session.snapshot()["eligible_owner"] == "operator"
        # Recovered eligibility must not replay the failed O request.
    assert h.session.controller.snapshot().owner_track_id is None
    assert not h.session.snapshot()["actions_enabled"]
    assert h.session.host.backend.page == 1
    assert not h.session.host.commands


def test_local_neutral_policy_has_separate_digest_without_changing_public_policy(demo):
    engine = demo.session.host.engine
    assert engine.neutral_tokens == ("NOG", "P1F", "P2F")
    public_digest = digest(json_bytes(engine.policy.model_dump(mode="json")))
    assert engine.policy_sha256 != public_digest
    assert engine.policy_sha256 == digest(json_bytes({
        "policy": engine.policy.model_dump(mode="json"),
        "neutral_tokens": ["NOG", "P1F", "P2F"],
        "scope": "application-local-demo",
    }))


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
