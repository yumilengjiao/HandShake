"""Fresh whole-frame evidence drives wake/owner/release; clocks drive health timeout.

Geometric evidence is deliberately separate from class scores. no_action alone
never proves neutral release; swipe completion requires the raw causal trajectory.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from uuid import uuid4

from shared.config import load_config
from shared.contracts import Gesture, GestureEvent, ModelPrediction


class State(str, Enum):
    DISABLED = "DISABLED"
    IDLE = "IDLE"
    ARMING = "ARMING"
    ACTIVE = "ACTIVE"
    CANDIDATE = "CANDIDATE"
    WAIT_RELEASE = "WAIT_RELEASE"
    COOLDOWN = "COOLDOWN"
    SUSPENDED = "SUSPENDED"


@dataclass(frozen=True)
class InteractionPolicy:
    wake_hold_ns: int = 600_000_000
    exit_hold_ns: int = 800_000_000
    neutral_hold_ns: int = 400_000_000
    cooldown_ns: int = 500_000_000
    max_prediction_age_ns: int = 150_000_000
    max_evidence_gap_ns: int = 100_000_000
    health_timeout_ns: int = 150_000_000
    class_on: float = 0.8
    status: str = "draft"
    score_kind: str = "calibrated"

    def __post_init__(self):
        for name in ("wake_hold_ns", "exit_hold_ns", "neutral_hold_ns", "cooldown_ns", "max_prediction_age_ns", "max_evidence_gap_ns", "health_timeout_ns"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive integer ns")
        if not math.isfinite(self.class_on) or not 0 < self.class_on <= 1 or self.status != "draft" or self.score_kind not in {"calibrated", "rule_decision"}:
            raise ValueError("threshold must be finite (0,1]; policy is draft")

    @classmethod
    def defaults(cls):
        return cls(**load_config().interaction.model_dump(exclude={"status"}))


@dataclass(frozen=True)
class Evidence:
    prediction: ModelPrediction
    ready: bool
    quality_valid: bool
    # Exact visible tracks are supplied once for the WHOLE associated frame.
    visible_tracks: tuple[str, ...]
    association_ambiguous: bool = False
    static_pose_valid: bool = False
    neutral_valid: bool = False
    palm_xy: tuple[float, float] | None = None
    # Height/width needed for speed in image-width units.
    aspect_ratio: float = 0.75

    def __post_init__(self):
        for name in ("ready", "quality_valid", "association_ambiguous", "static_pose_valid", "neutral_valid"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not all(isinstance(track, str) and track for track in self.visible_tracks):
            raise ValueError("visible_tracks must contain identifiers")
        if len(set(self.visible_tracks)) != len(self.visible_tracks):
            raise ValueError("visible_tracks must be unique")
        if not math.isfinite(self.aspect_ratio) or self.aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be positive")
        if self.palm_xy is not None and (len(self.palm_xy) != 2 or not all(math.isfinite(v) for v in self.palm_xy)):
            raise ValueError("palm_xy must be finite xy")


class InteractionEngine:
    """Single logical owner; host serializes calls and drives tick independently.

    stop_handler invokes the shared dispatcher gate, including epoch invalidation.
    An engine by itself has no device backend and cannot inject anything.
    """
    def __init__(self, controller, *, policy=None, model_version, stop_handler=None, invalidate_handler=None):
        self.controller = controller
        self.policy = policy or InteractionPolicy.defaults()
        self.model_version = model_version
        self.stop_handler = stop_handler
        self.invalidate_handler = invalidate_handler
        self.state = State.DISABLED
        self.decisions = deque(maxlen=2048)
        self._epoch = -1
        self._now = -1
        self._last_source = None
        self._last_frame = None
        self._last_good = None
        self._last_evidence_source = None
        self._candidate = None
        self._neutral_since = None
        self._exit_since = None
        self._cooldown_until = 0
        self._trajectory = []
        self._last_static = None
        self._static_since = None
        self.protocol = load_config().gestures

    def _clock(self, now):
        if type(now) is not int or not 0 <= now <= 2**63 - 1 or now < self._now:
            raise ValueError("interaction clock must be nondecreasing int64 ns")
        self._now = now

    def _clear(self):
        self._candidate = self._neutral_since = self._exit_since = None
        self._trajectory.clear()
        self._last_static = self._static_since = None
        self._last_evidence_source = None

    def _sync(self):
        context = self.controller.snapshot()
        if context.context_epoch != self._epoch:
            self._epoch = context.context_epoch
            self._clear()
            self._last_source = self._last_frame = self._last_good = None
            self.state = State.DISABLED if not context.enabled else State.SUSPENDED if context.suspended else State.IDLE if context.owner_track_id is None else State.WAIT_RELEASE
        return context

    def _record(self, reason, prediction=None, event=None):
        self.decisions.append(dict(now_ns=self._now, state=self.state.value, reason=reason,
                                   source_frame_id=None if prediction is None else prediction.source_frame_id,
                                   context_epoch=self.controller.snapshot().context_epoch,
                                   event_id=None if event is None else event.event_id))
        return [] if event is None else [event]

    def enable(self, now_ns):
        self._clock(now_ns)
        self.controller.enable(now_ns)
        self._sync()
        return self._record("explicit_enable")

    def stop(self, now_ns):
        self._clock(now_ns)
        if self.stop_handler:
            self.stop_handler(now_ns)
        else:
            self.controller.stop(now_ns)
        self._sync()
        self.state = State.DISABLED
        self._clear()
        return self._record("independent_stop")

    def _suspend(self, reason):
        if self.invalidate_handler:
            self.invalidate_handler(self._now)
        else:
            self.controller.suspend(self._now)
        self._sync()
        self.state = State.SUSPENDED
        return self._record(reason)

    def tick(self, now_ns):
        self._clock(now_ns)
        context = self._sync()
        if context.enabled and not context.suspended and context.owner_track_id and (
            self._last_good is None or now_ns - self._last_good > self.policy.health_timeout_ns
        ):
            return self._suspend("health_timeout")
        return []

    def health(self, now_ns, reason="camera_or_lifecycle_failure"):
        self._clock(now_ns)
        self._sync()
        return self._suspend(reason)

    def _event(self, prediction, gesture, since, reason):
        return GestureEvent(event_id=str(uuid4()), stream_id=prediction.stream_id, track_id=prediction.track_id,
                            context_epoch=prediction.context_epoch, source_frame_id=prediction.source_frame_id,
                            source_timestamp_ns=prediction.source_timestamp_ns, gesture_id=gesture,
                            candidate_start_timestamp_ns=since, commit_timestamp_ns=self._now,
                            mode=self.controller.snapshot().mode, decision_reason=reason, timebase=prediction.timebase)

    def update(self, evidence, *, now_ns):
        self._clock(now_ns)
        context = self._sync()
        p = evidence.prediction
        if not context.enabled or context.suspended:
            self._clear()
            return self._record("disabled_or_suspended", p)
        if (p.stream_id, p.context_epoch, p.timebase, p.model_version) != (context.stream_id, context.context_epoch, context.timebase, self.model_version):
            self._clear()
            return self._record("stale_context_or_model", p)
        if not 0 <= now_ns - p.source_timestamp_ns <= self.policy.max_prediction_age_ns or p.inference_timestamp_ns > now_ns:
            self._clear()
            return self._record("expired_or_future_prediction", p)
        if self._last_source is not None and (p.source_timestamp_ns <= self._last_source or p.source_frame_id <= self._last_frame):
            self._clear()
            return self._record("duplicate_or_reordered_prediction", p)
        self._last_source, self._last_frame = p.source_timestamp_ns, p.source_frame_id
        if evidence.association_ambiguous or len(evidence.visible_tracks) != 1 or p.track_id not in evidence.visible_tracks:
            return self._suspend("whole_frame_ownership_conflict")
        if context.owner_track_id is not None and p.track_id != context.owner_track_id:
            return self._suspend("owner_not_visible")
        if not evidence.ready or not evidence.quality_valid:
            return self._suspend("invalid_current_quality_or_not_ready") if context.owner_track_id else self._reject("invalid_current_quality_or_not_ready", p)
        t = p.source_timestamp_ns
        if self._last_evidence_source is not None and t - self._last_evidence_source > self.policy.max_evidence_gap_ns:
            self._clear()
        self._last_evidence_source = t
        self._last_good = t
        probabilities = p.calibrated_probabilities
        if probabilities is None:
            # Raw score ordering works for the geometric rule baseline. It is
            # explicitly uncalibrated, so no fabricated probability threshold.
            if self.policy.score_kind != "rule_decision" or sorted(p.logits) != [0.0, 0.0, 0.0, 0.0, 1.0]:
                return self._reject("uncalibrated_neural_scores", p)
            winner = max(range(5), key=p.logits.__getitem__)
            confident = True
        else:
            winner = max(range(5), key=probabilities.__getitem__)
            confident = probabilities[winner] >= self.policy.class_on and probabilities.count(probabilities[winner]) == 1
        gesture = list(Gesture)[winner]
        if not confident:
            return self._reject("below_draft_class_threshold", p)
        if p.intent_score is not None and p.intent_score < self.policy.class_on and gesture != Gesture.EXIT_POSE:
            return self._reject("intent_rejected", p)
        static_valid = self._static(evidence)
        if context.owner_track_id is None:
            if gesture != Gesture.WAKE_OPEN or not static_valid:
                return self._reject("waiting_unique_static_wake", p)
            if self._candidate is None or self._candidate[0] != p.track_id:
                self._candidate = (p.track_id, t)
            self.state = State.ARMING
            if t - self._candidate[1] < self.policy.wake_hold_ns:
                return self._record("wake_holding", p)
            event = self._event(p, gesture, self._candidate[1], "wake_confirmed")
            # Owner changes the epoch. The control event keeps its captured old
            # epoch and is never mapped to a device action.
            self.controller.bind_owner(p.track_id, now_ns)
            self._sync()
            self._last_good = t
            self.state = State.WAIT_RELEASE
            return self._record("wake_owner_acquired", p, event)
        # Exit remains available in WAIT_RELEASE and COOLDOWN.
        if gesture == Gesture.EXIT_POSE and static_valid:
            if self._exit_since is None:
                self._exit_since = t
            if t - self._exit_since >= self.policy.exit_hold_ns:
                event = self._event(p, gesture, self._exit_since, "exit_confirmed")
                self.stop(now_ns)
                return self._record("exit_control", p, event)
            return self._record("exit_holding", p)
        self._exit_since = None
        if self.state in {State.WAIT_RELEASE, State.COOLDOWN}:
            if gesture == Gesture.NO_ACTION and evidence.neutral_valid and static_valid:
                if self._neutral_since is None:
                    self._neutral_since = t
                if t - self._neutral_since >= self.policy.neutral_hold_ns:
                    self.state = State.COOLDOWN if now_ns < self._cooldown_until else State.ACTIVE
            else:
                self._neutral_since = None
                if self.state == State.COOLDOWN:
                    self.state = State.WAIT_RELEASE
            self._trajectory.clear()
            if self.state == State.ACTIVE:
                self._append_point(evidence)
            return self._record("released" if self.state == State.ACTIVE else "waiting_neutral_and_cooldown", p)
        if gesture in {Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT}:
            event = self._swipe(evidence, gesture)
            if event:
                self.state = State.WAIT_RELEASE
                self._cooldown_until = now_ns + self.policy.cooldown_ns
                self._neutral_since = None
                self._trajectory.clear()
                return self._record("swipe_completed_once", p, event)
            self.state = State.CANDIDATE
            return self._record("swipe_incomplete", p)
        self._append_point(evidence)
        self.state = State.ACTIVE
        return self._record("no_device_gesture", p)

    def _reject(self, reason, prediction):
        self._clear()
        if self.state in {State.ARMING, State.CANDIDATE}:
            self.state = State.IDLE if self.controller.snapshot().owner_track_id is None else State.ACTIVE
        return self._record(reason, prediction)

    def _static(self, evidence):
        t, point = evidence.prediction.source_timestamp_ns, evidence.palm_xy
        previous = self._last_static
        self._last_static = None if point is None else (t, point)
        if not evidence.static_pose_valid or point is None:
            return False
        if previous is None:
            return True  # This source starts holding, never confirms on its own.
        dt = (t - previous[0]) / 1e9
        dx, dy = point[0] - previous[1][0], (point[1] - previous[1][1]) * evidence.aspect_ratio
        return dt > 0 and math.hypot(dx, dy) / dt <= self.protocol.static_max_speed_widths_per_s

    def _append_point(self, evidence):
        p, point = evidence.prediction, evidence.palm_xy
        if point is None:
            self._trajectory.clear()
            return
        if not self._trajectory or self._trajectory[-1][0] != p.source_timestamp_ns:
            self._trajectory.append((p.source_timestamp_ns, point[0], point[1]))
        while self._trajectory and p.source_timestamp_ns - self._trajectory[0][0] > self.protocol.swipe_max_duration_ns:
            self._trajectory.pop(0)
        self._trajectory = self._trajectory[-256:]

    def _swipe(self, evidence, gesture):
        p, point = evidence.prediction, evidence.palm_xy
        self._append_point(evidence)
        if point is None:
            return None
        # The classifier may recognize a swipe only at completion. Keep raw
        # pre-classification motion; do not wait another full movement/hold.
        for start in range(len(self._trajectory) - 2, -1, -1):
            segment = self._trajectory[start:]
            since, x0, _ = segment[0]
            duration = p.source_timestamp_ns - since
            if duration < self.protocol.swipe_min_duration_ns:
                continue
            vertical = max(row[2] for row in segment) - min(row[2] for row in segment)
            displacement = point[0] - x0
            signed = displacement if gesture == Gesture.SWIPE_RIGHT else -displacement
            travel = sum(abs(a[1] - b[1]) for a, b in zip(segment, segment[1:]))
            if (signed >= self.protocol.swipe_min_displacement_widths
                    and vertical <= self.protocol.swipe_max_vertical_displacement_heights
                    and travel > 0 and abs(displacement) / travel >= 0.8):
                return self._event(p, gesture, since, "causal_geometry_complete")
        return None
