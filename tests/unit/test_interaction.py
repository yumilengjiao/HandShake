from dataclasses import replace

import pytest

from backend.controller import ApplicationController
from backend.interaction.engine import Evidence, InteractionEngine, InteractionPolicy, State
from shared.contracts import Gesture, ModelPrediction

STEP = 50_000_000


class Timeline:
    def __init__(self):
        self.now = 0
        self.controller = ApplicationController("s", timebase="synthetic")
        self.engine = InteractionEngine(self.controller, model_version="synthetic-rule", policy=InteractionPolicy(score_kind="rule_decision", wake_hold_ns=100_000_000, exit_hold_ns=100_000_000, neutral_hold_ns=100_000_000, cooldown_ns=200_000_000))
        self.engine.enable(0)

    def evidence(self, gesture, *, x=.5, y=.5, **changes):
        self.now += STEP
        context = self.controller.snapshot()
        scores = [float(g == gesture) for g in Gesture]
        prediction = ModelPrediction(stream_id="s", track_id="t", context_epoch=context.context_epoch,
                                     source_frame_id=self.now // STEP, source_timestamp_ns=self.now,
                                     inference_timestamp_ns=self.now, logits=scores,
                                     model_version="synthetic-rule", timebase="synthetic")
        return Evidence(prediction, ready=True, quality_valid=True, visible_tracks=("t",),
                        static_pose_valid=gesture in {Gesture.WAKE_OPEN, Gesture.NO_ACTION, Gesture.EXIT_POSE},
                        neutral_valid=gesture == Gesture.NO_ACTION, palm_xy=(x, y), **changes)

    def step(self, gesture, **kwargs):
        item = self.evidence(gesture, **kwargs)
        return self.engine.update(item, now_ns=self.now)

    def wake(self):
        events = sum((self.step(Gesture.WAKE_OPEN) for _ in range(3)), [])
        assert len(events) == 1 and self.controller.snapshot().owner_track_id == "t"
        assert self.engine.state == State.WAIT_RELEASE
        return events

    def active(self):
        self.wake()
        for _ in range(3):
            self.step(Gesture.NO_ACTION)
        assert self.engine.state == State.ACTIVE

    def swipe(self, gesture=Gesture.SWIPE_RIGHT):
        xs = [.5, .57, .64, .71] if gesture == Gesture.SWIPE_RIGHT else [.5, .43, .36, .29]
        return sum((self.step(gesture, x=x) for x in xs), [])


def test_wake_release_swipe_once_and_reverse_return_never_retriggers():
    t = Timeline()
    t.active()
    assert len(t.swipe()) == 1
    assert t.swipe(Gesture.SWIPE_LEFT) == []
    for _ in range(6):
        assert t.step(Gesture.SWIPE_LEFT, x=.29) == []
    assert t.engine.state == State.WAIT_RELEASE
    # Moving back to center cannot release; three subsequent still samples can.
    t.step(Gesture.NO_ACTION, x=.5)
    for _ in range(3):
        t.step(Gesture.NO_ACTION, x=.5)
    assert t.engine.state == State.ACTIVE
    assert len(t.swipe(Gesture.SWIPE_LEFT)) == 1


def test_classifier_can_recognize_only_at_motion_completion():
    t = Timeline()
    t.active()
    for x in (.5, .57, .64):
        t.step(Gesture.NO_ACTION, x=x)
    assert len(t.step(Gesture.SWIPE_RIGHT, x=.71)) == 1


@pytest.mark.parametrize("change,reason", [({"ready": False}, "quality"), ({"quality_valid": False}, "quality"),
                                         ({"association_ambiguous": True}, "conflict"), ({"visible_tracks": ("t", "other")}, "conflict")])
def test_current_quality_and_whole_frame_conflicts_suspend(change, reason):
    t = Timeline()
    t.active()
    evidence = replace(t.evidence(Gesture.SWIPE_RIGHT), **change)
    epoch = t.controller.snapshot().context_epoch
    assert t.engine.update(evidence, now_ns=t.now) == []
    assert t.engine.state == State.SUSPENDED
    assert t.controller.snapshot().context_epoch > epoch
    assert reason in t.engine.decisions[-1]["reason"]
    t.engine.enable(t.now)
    assert t.controller.snapshot().owner_track_id is None


def test_no_action_alone_is_never_release_evidence():
    t = Timeline()
    t.wake()
    for _ in range(10):
        e = replace(t.evidence(Gesture.NO_ACTION), neutral_valid=False)
        assert not t.engine.update(e, now_ns=t.now)
    assert t.engine.state == State.WAIT_RELEASE


def test_stale_low_quality_and_gaps_cannot_add_up_wake_holds():
    t = Timeline()
    t.step(Gesture.WAKE_OPEN)
    e = t.evidence(Gesture.WAKE_OPEN)
    stale = e.prediction.model_copy(update={"context_epoch": 0})
    t.engine.update(replace(e, prediction=stale), now_ns=t.now)
    assert not t.step(Gesture.WAKE_OPEN)
    assert not t.step(Gesture.WAKE_OPEN)
    t.now += 200_000_000
    assert not t.step(Gesture.WAKE_OPEN)
    assert t.controller.snapshot().owner_track_id is None


def test_duplicate_and_reordered_evidence_cannot_confirm_wake():
    t = Timeline()
    e = t.evidence(Gesture.WAKE_OPEN)
    t.engine.update(e, now_ns=t.now)
    t.now += 100_000_000
    assert not t.engine.update(e, now_ns=t.now)
    assert t.controller.snapshot().owner_track_id is None


def test_exit_bypasses_wait_release_and_stop_is_idempotent():
    t = Timeline()
    t.wake()
    events = sum((t.step(Gesture.EXIT_POSE) for _ in range(3)), [])
    assert len(events) == 1 and events[0].gesture_id == Gesture.EXIT_POSE
    assert t.engine.state == State.DISABLED and not t.controller.snapshot().enabled
    assert not t.engine.stop(t.now)


def test_health_tick_without_predictions_expires_owner():
    t = Timeline()
    t.active()
    t.now += 151_000_000
    t.engine.tick(t.now)
    assert t.engine.state == State.SUSPENDED
    assert t.controller.snapshot().owner_track_id is None


def test_wrong_direction_or_large_vertical_motion_never_commits():
    t = Timeline()
    t.active()
    for x, y in ((.5, .5), (.57, .6), (.64, .7), (.71, .8)):
        assert not t.step(Gesture.SWIPE_RIGHT, x=x, y=y)
    for x in (.6, .5, .4, .3):
        assert not t.step(Gesture.SWIPE_RIGHT, x=x)


def test_uncalibrated_neural_scores_do_not_pass_rule_score_gate():
    t = Timeline()
    e = t.evidence(Gesture.WAKE_OPEN)
    p = e.prediction.model_copy(update={"logits": [0., 9., 0., 0., 0.]})
    assert not t.engine.update(replace(e, prediction=p), now_ns=t.now)
    assert t.engine.decisions[-1]["reason"] == "uncalibrated_neural_scores"


def test_raw_rule_scores_require_explicit_policy_and_calibrated_head_is_supported():
    t = Timeline()
    t.engine.policy = replace(t.engine.policy, score_kind="calibrated")
    e = t.evidence(Gesture.WAKE_OPEN)
    assert not t.engine.update(e, now_ns=t.now)
    assert t.engine.decisions[-1]["reason"] == "uncalibrated_neural_scores"
    for _ in range(3):
        e = t.evidence(Gesture.WAKE_OPEN)
        prediction = e.prediction.model_copy(update={"calibrated_probabilities": [.01, .96, .01, .01, .01]})
        events = t.engine.update(replace(e, prediction=prediction), now_ns=t.now)
    assert events and t.controller.snapshot().owner_track_id == "t"


def test_whole_frame_adapter_rejects_mismatched_sources_and_background_cannot_release():
    from backend.interaction.evidence import evidence_from_frame
    from shared.observations import FrameInfo, ObservationFrame
    from shared.contracts import TransformMetadata
    t = Timeline()
    e = t.evidence(Gesture.NO_ACTION)
    info = FrameInfo("s", e.prediction.source_frame_id, t.now, 640, 480, TransformMetadata(timebase="synthetic"), "synthetic")
    frame = ObservationFrame(info, info.status("processed_no_hand", processed_at_ns=t.now))
    actual = evidence_from_frame(e.prediction, frame, ready=True)
    assert not actual.quality_valid and not actual.neutral_valid and actual.visible_tracks == ()
    with pytest.raises(ValueError, match="source mismatch"):
        evidence_from_frame(e.prediction.model_copy(update={"source_frame_id": 99}), frame, ready=True)


def test_neutral_geometry_cannot_confirm_a_misclassified_wake_or_exit():
    import numpy as np
    from algorithm.smoke import _rule_case
    from backend.interaction.evidence import evidence_from_frame
    from shared.contracts import HandObservation, TransformMetadata
    from shared.observations import FrameInfo, ObservationFrame
    t = Timeline()
    p = t.evidence(Gesture.WAKE_OPEN).prediction
    def frame(pose):
        points = _rule_case(pose, 0)[0][-1, :63].reshape(21, 3) * .06 + np.asarray([.5, .35, 0.])
        info = FrameInfo("s", p.source_frame_id, t.now, 640, 640, TransformMetadata(timebase="synthetic"), "synthetic")
        obs = HandObservation(stream_id="s", source_frame_id=p.source_frame_id, source_timestamp_ns=t.now, track_id="t",
                              landmarks_image=points.tolist(), validity_mask=[True]*21, observation_valid=True,
                              handedness="unknown", quality=1., transform_metadata=info.transform, timebase="synthetic")
        return ObservationFrame(info, info.status("processed_hands", processed_at_ns=t.now), (obs,))
    neutral = evidence_from_frame(p, frame("neutral"), ready=True)
    assert neutral.neutral_valid and not neutral.static_pose_valid
    assert evidence_from_frame(p, frame("open"), ready=True).static_pose_valid
    calibrated = p.model_copy(update={"logits": [1., 0., 0., 0., 0.], "calibrated_probabilities": [.01, .96, .01, .01, .01]})
    assert evidence_from_frame(calibrated, frame("open"), ready=True).static_pose_valid
