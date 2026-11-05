"""Public taxonomy safety tests with hand-authored scores/geometry, no neural training."""

from types import SimpleNamespace

import pytest

from algorithm.bundle import digest, json_bytes
from algorithm.evaluation.calibration import load_for_bundle
from backend.controller import ApplicationController
from backend.interaction.geometry import frame_geometry
from backend.interaction.public_policy import PublicGestureEngine
from backend.interaction.public_replay import (
    ReplayClock,
    synthetic_calibration,
    synthetic_input,
)
from shared.contracts import ModelPrediction
from shared.observations import ObservationFrame
from shared.public_interaction import PublicGesturePolicy


class Rig:
    def __init__(self, calibration, sha):
        self.clock = ReplayClock()
        self.controller = ApplicationController(
            "public-test",
            timebase="synthetic",
            target_context="local-presentation:test",
        )
        self.engine = PublicGestureEngine(
            self.controller, calibration=calibration, calibration_sha256=sha
        )
        self.frame = 0
        self.controller.enable(0)

    def input(self, token="NOG", x=0.5, **kwargs):
        return synthetic_input(
            self.controller.snapshot(),
            timestamp_ns=self.clock.now,
            frame_id=self.frame,
            model_version=self.engine.model_version,
            token=token,
            x=x,
            **kwargs,
        )

    def step(self, token="NOG", x=0.5, *, dt=50_000_000, ready=True, **kwargs):
        self.clock.now += dt
        self.frame += 1
        p, frame = self.input(token, x, **kwargs)
        return self.engine.update(p, frame, ready=ready, now_ns=self.clock.now)

    def arm(self):
        p, frame = self.input()
        assert (
            self.engine.arm_owner(
                "operator", p, frame, ready=True, now_ns=self.clock.now
            )
            is not None
        )

    def release(self):
        for _ in range(12):
            assert self.step() is None
        assert self.engine.state == "ACTIVE"

    def motion(self, token="TRT", mirrored=True):
        values = [0.5, 0.56, 0.62, 0.68] if token == "TRT" else [0.5, 0.44, 0.38, 0.32]
        return [
            event
            for x in values
            if (event := self.step(token, x, mirrored=mirrored)) is not None
        ]


@pytest.fixture
def rig(tmp_path):
    calibration, sha = synthetic_calibration(tmp_path)
    return Rig(calibration, sha)


def test_geometry_caches_only_fixed_thresholds_and_preserves_neutral_rule(rig, monkeypatch):
    from backend.interaction import geometry

    original = geometry.load_config
    calls = []
    geometry._neutral_thresholds.cache_clear()

    def counted():
        calls.append(True)
        return original()

    monkeypatch.setattr(geometry, "load_config", counted)
    try:
        _, frame = rig.input()
        for _ in range(20):
            result = geometry.frame_geometry(frame, "operator")
            assert result.neutral and result.quality_valid
        assert len(calls) == 1
        assert not geometry.frame_geometry(frame, "other").neutral
    finally:
        geometry._neutral_thresholds.cache_clear()


def test_unique_hand_and_nog_never_grant_owner_automatically(rig):
    for _ in range(20):
        assert rig.step() is None
    assert (
        rig.engine.state == "WAIT_OWNER"
        and rig.controller.snapshot().owner_track_id is None
    )
    rig.clock.now += 50_000_000
    rig.frame += 1  # explicit arming consumes a NEW frame, not a restamped old score
    rig.arm()
    assert rig.engine.state == "WAIT_RELEASE"
    assert rig.controller.snapshot().owner_track_id == "operator"


@pytest.mark.parametrize("token", ["OP2", "GRB", "TLT", "TRT", "P1F", "P2F"])
def test_non_neutral_source_class_cannot_arm_or_fake_wake_exit(rig, token):
    p, frame = rig.input(token)
    assert rig.engine.arm_owner("operator", p, frame, ready=True, now_ns=0) is None
    assert rig.controller.snapshot().owner_track_id is None
    assert rig.controller.snapshot().enabled


@pytest.mark.parametrize(
    "token",
    [
        "P1F",
        "P2F",
        "C1F",
        "C2F",
        "TUP",
        "TDN",
        "OP2",
        "2C1",
        "2C2",
        "ZIN",
        "ZOT",
        "GRB",
    ],
)
def test_all_unmapped_classes_produce_no_action(rig, token):
    rig.arm()
    rig.release()
    for i in range(5):
        assert rig.step(token, 0.3 + 0.08 * i) is None
    assert rig.controller.snapshot().enabled
    assert rig.engine.state == "WAIT_RELEASE"


@pytest.mark.parametrize(
    "token,action", [("TLT", "previous_slide"), ("TRT", "next_slide")]
)
@pytest.mark.parametrize("mirrored", [True, False])
def test_model_space_direction_calibration_and_event_provenance(
    rig, token, action, mirrored
):
    rig.arm()
    rig.release()
    (event,) = rig.motion(token, mirrored)
    assert event.source_token == token and event.action == action
    assert event.context_epoch == rig.controller.snapshot().context_epoch
    assert event.track_id == "operator" and event.synthetic
    assert event.model_version == rig.engine.model_version
    assert event.calibration_sha256 == rig.engine.calibration_sha256
    assert event.policy_sha256 == rig.engine.policy_sha256
    assert event.candidate_start_ns < event.source_timestamp_ns == event.commit_ns
    assert rig.engine.state == "WAIT_RELEASE"


def test_holding_and_return_motion_cannot_retrigger_after_cooldown(rig):
    rig.arm()
    rig.release()
    assert len(rig.motion()) == 1
    for _ in range(30):
        assert rig.step("TRT", 0.68) is None
    for x in (0.62, 0.56, 0.5, 0.44, 0.38, 0.32):
        assert rig.step("TLT", x) is None
    rig.release()
    assert len(rig.motion()) == 1


@pytest.mark.parametrize("token", ["P1F", "P2F"])
def test_public_policy_keeps_nog_only_release_and_original_digest(rig, token):
    assert rig.engine.neutral_tokens == ("NOG",)
    assert rig.engine.policy_sha256 == digest(
        json_bytes(rig.engine.policy.model_dump(mode="json"))
    )
    rig.arm()
    for _ in range(20):
        assert rig.step(token) is None
    assert rig.engine.state == "WAIT_RELEASE"
    rig.release()


def test_nog_without_stationary_geometric_neutral_does_not_release(rig):
    rig.arm()
    for i in range(24):
        assert rig.step("NOG", 0.4 if i % 2 else 0.6) is None
    assert rig.engine.state == "WAIT_RELEASE"
    # NOG with a four-finger-extended hand is not the independent neutral pose.
    for _ in range(12):
        rig.clock.now += 50_000_000
        rig.frame += 1
        p, frame = rig.input()
        data = frame.observations[0].model_dump()
        for mcp in (5, 9, 13, 17):
            y = data["landmarks_image"][mcp][1]
            data["landmarks_image"][mcp + 1][1] = y - 0.05
            data["landmarks_image"][mcp + 2][1] = y - 0.09
        observation = type(frame.observations[0]).model_validate(data)
        frame = ObservationFrame(frame.info, frame.status, (observation,))
        assert not frame_geometry(frame, "operator").neutral
        assert rig.engine.update(p, frame, ready=True, now_ns=rig.clock.now) is None
    assert rig.engine.state == "WAIT_RELEASE"
    rig.release()


def test_logits_alone_no_motion_or_wrong_geometry_cannot_emit(rig):
    rig.arm()
    rig.release()
    for _ in range(20):
        assert rig.step("TRT", 0.5) is None
    assert rig.engine.state == "CANDIDATE"
    # Direction opposite to predicted token is rejected, not remapped.
    assert rig.step("TRT", 0.4) is None
    assert rig.engine.reason == "motion_geometry_rejected"
    rig.release()
    assert rig.step("TRT", 0.5) is None
    assert rig.step("TRT", 0.6, y=0.8) is None
    assert rig.engine.reason == "motion_geometry_rejected"


def test_candidate_timeout_direction_change_and_time_gap_clear_history(rig):
    rig.arm()
    rig.release()
    for _ in range(32):
        assert rig.step("TRT", 0.5) is None
    assert rig.engine.reason == "motion_candidate_timeout"
    rig.release()
    assert rig.step("TRT", 0.5) is None
    assert rig.step("TLT", 0.44) is None
    assert rig.engine.reason == "motion_direction_changed"
    rig.release()
    assert rig.step("TRT", 0.5) is None
    assert rig.step("TRT", 0.7, dt=110_000_000) is None
    assert rig.engine.state == "WAIT_RELEASE"


def test_duplicate_frames_cannot_advance_release_or_candidate(rig):
    rig.arm()
    rig.clock.now += 50_000_000
    rig.frame += 1
    p, frame = rig.input()
    assert rig.engine.update(p, frame, ready=True, now_ns=rig.clock.now) is None
    assert rig.engine.update(p, frame, ready=True, now_ns=rig.clock.now) is None
    assert rig.engine.reason == "duplicate_or_reordered_prediction"
    assert rig.engine.state == "WAIT_RELEASE"
    rig.release()
    assert rig.step("TRT", 0.5) is None
    p, frame = rig.input("TRT", 0.7)
    assert rig.engine.update(p, frame, ready=True, now_ns=rig.clock.now) is None
    assert rig.engine.reason == "duplicate_or_reordered_prediction"


@pytest.mark.parametrize(
    "kind",
    [
        "wrong_epoch",
        "wrong_model",
        "future",
        "low_confidence",
        "tie",
        "not_ready",
        "low_quality",
        "wrong_owner",
        "rotation",
        "multiple",
        "missing",
    ],
)
def test_fail_closed_inputs_never_emit(rig, kind):
    rig.arm()
    rig.release()
    assert rig.step("TRT", 0.5) is None
    rig.clock.now += 50_000_000
    rig.frame += 1
    p, frame = rig.input(
        "TRT",
        0.7,
        track="other" if kind == "wrong_owner" else "operator",
        quality=0.1 if kind == "low_quality" else 1.0,
    )
    if kind == "wrong_epoch":
        p = p.model_copy(update={"context_epoch": p.context_epoch - 1})
    elif kind == "wrong_model":
        p = p.model_copy(update={"model_version": "unverified"})
    elif kind == "future":
        p = p.model_copy(update={"inference_timestamp_ns": rig.clock.now + 1})
    elif kind == "low_confidence":
        p = p.model_copy(update={"logits": [0.0] * 15})
    elif kind == "tie":
        logits = [0.0] * 15
        logits[7] = logits[8] = 10.0
        p = p.model_copy(update={"logits": logits})
    elif kind == "rotation":
        from dataclasses import replace

        transform = frame.info.transform.model_copy(update={"rotation_degrees": 90})
        info = replace(frame.info, transform=transform)
        obs = frame.observations[0].model_copy(update={"transform_metadata": transform})
        frame = ObservationFrame(info, frame.status, (obs,))
    elif kind == "multiple":
        frame = ObservationFrame(
            frame.info,
            frame.status,
            frame.observations
            + (frame.observations[0].model_copy(update={"track_id": "second"}),),
        )
    elif kind == "missing":
        frame = ObservationFrame(
            frame.info,
            frame.info.status("processed_no_hand", processed_at_ns=rig.clock.now),
        )
    assert (
        rig.engine.update(p, frame, ready=kind != "not_ready", now_ns=rig.clock.now)
        is None
    )
    assert rig.engine.state in {"WAIT_RELEASE", "SUSPENDED"}


def test_no_prediction_watchdog_expires_owner_and_requires_explicit_rearming(rig):
    rig.arm()
    rig.release()
    rig.clock.now += rig.engine.policy.max_age_ns + 1
    rig.engine.tick(rig.clock.now)
    assert (
        rig.controller.snapshot().suspended
        and rig.controller.snapshot().owner_track_id is None
    )
    for _ in range(20):
        assert rig.step() is None
    assert rig.controller.snapshot().suspended
    rig.controller.enable(rig.clock.now)
    rig.frame += 1
    rig.arm()
    rig.release()


def test_legacy_prediction_and_mismatched_whole_frame_are_rejected(rig):
    p, frame = rig.input()
    legacy = ModelPrediction(
        stream_id=p.stream_id,
        track_id=p.track_id,
        context_epoch=p.context_epoch,
        source_frame_id=p.source_frame_id,
        source_timestamp_ns=0,
        inference_timestamp_ns=0,
        timebase="synthetic",
        model_version=p.model_version,
        logits=[0.0] * 5,
    )
    with pytest.raises(ValueError, match="source classification"):
        rig.engine.update(legacy, frame, ready=True, now_ns=0)
    with pytest.raises(ValueError, match="source mismatch"):
        rig.engine.update(
            p.model_copy(update={"source_frame_id": 100}), frame, ready=True, now_ns=0
        )


def test_policy_and_calibration_pin_validation(tmp_path):
    calibration, sha = synthetic_calibration(tmp_path)
    controller = ApplicationController("s")
    with pytest.raises(ValueError, match="SHA256"):
        PublicGestureEngine(
            controller, calibration=calibration, calibration_sha256="0" * 64
        )
    invalid = calibration.model_copy(
        update={
            "rejection_status": "infeasible",
            "threshold": None,
            "accepted_count": 0,
        }
    )
    with pytest.raises(ValueError, match="feasible"):
        PublicGestureEngine(
            controller,
            calibration=invalid,
            calibration_sha256=digest(json_bytes(invalid.model_dump(mode="json"))),
        )
    for kwargs in (
        {"neutral_hold_ns": 0},
        {"max_gap_ns": 1_000_000_000},
        {"class_on": float("nan")},
        {"min_motion_samples": True},
    ):
        with pytest.raises(ValueError):
            PublicGesturePolicy(**kwargs)


def test_bundle_bound_sidecar_retains_exact_validation_provenance(tmp_path):
    calibration, sha = synthetic_calibration(tmp_path)
    source = calibration.source
    bundle = SimpleNamespace(
        manifest_sha256=source.model_sha256,
        synthetic_training=True,
        vocabulary=SimpleNamespace(vocabulary_version=source.vocabulary_version),
        normalizer=SimpleNamespace(
            stats=SimpleNamespace(
                dataset_hash=source.dataset_sha256, split_hash=source.split_sha256
            ),
            stats_hash=source.stats_sha256,
            config=SimpleNamespace(config_hash=source.config_sha256),
        ),
        training_identity={
            "data": {
                "synthetic": True,
                "val_index_sha256": source.cache_index_sha256,
                "val": {"endpoint_sha256": source.endpoint_sha256},
            }
        },
    )
    assert (
        load_for_bundle(
            tmp_path / "calibration.json", expected_sha256=sha, bundle=bundle
        )
        == calibration
    )
    bundle.manifest_sha256 = "0" * 64
    with pytest.raises(ValueError, match="binding"):
        load_for_bundle(
            tmp_path / "calibration.json", expected_sha256=sha, bundle=bundle
        )
    bundle.manifest_sha256 = source.model_sha256
    bundle.training_identity["data"]["val"]["endpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="inventory"):
        load_for_bundle(
            tmp_path / "calibration.json", expected_sha256=sha, bundle=bundle
        )


def test_audit_buffer_is_bounded(rig):
    rig.arm()
    for _ in range(600):
        rig.step()
    assert len(rig.engine.decisions) == 512
