"""Draft IPN HandS policy: explicit owner, calibrated scores AND causal geometry.

Pure event generation, no backend authority. TLT/TRT are application candidates,
not legacy swipe IDs; OP2/GRB never enable or stop. Host serializes with its gate.
"""

import math
from collections import deque

import numpy as np

from algorithm.bundle import digest, json_bytes
from algorithm.evaluation.calibration import Calibration
from algorithm.evaluation.classification import log_probabilities
from backend.interaction.geometry import frame_geometry
from shared.classes import IPN_HANDS_VERSION, vocabulary_for
from shared.classification import SourceClassification
from shared.public_interaction import PublicActionEvent, PublicGesturePolicy


class PublicGestureEngine:
    # The historical shadow/replay policy accepts only NOG as neutral. A
    # separately scoped local-demo subclass may declare static neutral tokens.
    neutral_tokens = ("NOG",)

    def __init__(
        self,
        controller,
        *,
        calibration,
        calibration_sha256,
        policy=None,
        stop_handler=None,
    ):
        self.controller = controller
        self.calibration = Calibration.model_validate(calibration.model_dump())
        if (
            self.calibration.source.vocabulary_version != IPN_HANDS_VERSION
            or self.calibration.rejection_status != "selected"
        ):
            raise ValueError(
                "public policy requires IPN HandS calibration with a feasible rejection threshold"
            )
        if (
            digest(json_bytes(self.calibration.model_dump(mode="json")))
            != calibration_sha256
        ):
            raise ValueError("canonical calibration SHA256 mismatch")
        self.calibration_sha256 = calibration_sha256
        self.policy = (
            PublicGesturePolicy()
            if policy is None
            else PublicGesturePolicy.model_validate(policy.model_dump())
        )
        self.policy_sha256 = digest(json_bytes(self.policy.model_dump(mode="json")))
        self.model_version = "sha256:" + self.calibration.source.model_sha256
        self.stop_handler = stop_handler
        self.state, self.reason = "DISABLED", "initial"
        self.decisions = deque(maxlen=512)
        self._epoch = -1
        self._now = -1
        self._last_source = self._last_frame = self.last_good_ns = None
        self._not_before = 0
        self._cooldown_until = 0
        self._event_index = 0
        self._clear()

    def _clear(self):
        self._point = self._candidate = self._neutral_since = None

    def _clock(self, now):
        if type(now) is not int or not 0 <= now < 2**63 or now < self._now:
            raise ValueError("public policy clock must be nondecreasing int64 ns")
        self._now = now

    def sync(self):
        context = self.controller.snapshot()
        if context.context_epoch != self._epoch:
            self._epoch = context.context_epoch
            self._not_before = (
                self.controller.changes[-1].timestamp_ns
                if self.controller.changes
                else 0
            )
            self._last_source = self._last_frame = self.last_good_ns = None
            self._cooldown_until = 0
            self._clear()
            self.state = (
                "DISABLED"
                if not context.enabled
                else "SUSPENDED"
                if context.suspended
                else "WAIT_OWNER"
                if context.owner_track_id is None
                else "WAIT_RELEASE"
            )
        return context

    def _record(self, reason, prediction=None, event=None):
        self.reason = reason
        self.decisions.append(
            {
                "now_ns": self._now,
                "state": self.state,
                "reason": reason,
                "context_epoch": self._epoch,
                "source_frame_id": None
                if prediction is None
                else prediction.source_frame_id,
                "event_id": None if event is None else event.event_id,
            }
        )
        return event

    def _wait_release(self, reason, prediction=None):
        self._clear()
        if self.state not in {"DISABLED", "SUSPENDED", "WAIT_OWNER"}:
            self.state = "WAIT_RELEASE"
        return self._record(reason, prediction)

    def suspend(self, now_ns, reason):
        self._clock(now_ns)
        if self.stop_handler:
            self.stop_handler(now_ns, suspend=True)
        elif not self.controller.snapshot().suspended:
            self.controller.suspend(now_ns)
        self.sync()
        self._clear()
        return self._record(reason)

    def tick(self, now_ns):
        self._clock(now_ns)
        context = self.sync()
        if (
            context.enabled
            and not context.suspended
            and context.owner_track_id is not None
        ):
            if (
                self.last_good_ns is None
                or now_ns - self.last_good_ns > self.policy.max_age_ns
            ):
                self.suspend(now_ns, "prediction_health_timeout")

    def _evidence(self, prediction, frame, ready):
        if not isinstance(prediction, SourceClassification) or type(ready) is not bool:
            raise ValueError(
                "public policy requires source classification and explicit readiness"
            )
        p, info = prediction, frame.info
        if (p.stream_id, p.source_frame_id, p.source_timestamp_ns, p.timebase) != (
            info.stream_id,
            info.frame_id,
            info.timestamp_ns,
            info.timebase,
        ):
            raise ValueError("public prediction/whole-frame source mismatch")
        context = self.sync()
        if not context.enabled or context.suspended:
            self._record("disabled_or_suspended", p)
            return None
        if (
            p.stream_id,
            p.context_epoch,
            p.timebase,
            p.model_version,
            p.vocabulary_version,
        ) != (
            context.stream_id,
            context.context_epoch,
            context.timebase,
            self.model_version,
            IPN_HANDS_VERSION,
        ):
            self._wait_release("stale_context_or_model", p)
            return None
        if context.mode != "presentation":
            self.suspend(self._now, "unsupported_mode")
            return None
        if (
            p.source_timestamp_ns < self._not_before
            or not 0 <= self._now - p.source_timestamp_ns <= self.policy.max_age_ns
            or p.inference_timestamp_ns > self._now
        ):
            self._wait_release("expired_or_future_prediction", p)
            return None
        if self._last_source is not None and (
            p.source_timestamp_ns <= self._last_source
            or p.source_frame_id <= self._last_frame
        ):
            self._wait_release("duplicate_or_reordered_prediction", p)
            return None
        if (
            self._last_source is not None
            and p.source_timestamp_ns - self._last_source > self.policy.max_gap_ns
        ):
            self._wait_release("evidence_gap", p)
        self._last_source, self._last_frame = p.source_timestamp_ns, p.source_frame_id
        g = frame_geometry(frame, p.track_id)
        if (
            g.ambiguous
            or g.visible_tracks != (p.track_id,)
            or context.owner_track_id not in {None, p.track_id}
        ):
            self.suspend(self._now, "whole_frame_ownership_conflict")
            return None
        if (
            not ready
            or not g.quality_valid
            or g.palm_xy is None
            or not all(0 <= v <= 1 for v in g.palm_xy)
        ):
            self.suspend(self._now, "quality_or_readiness_failure")
            return None
        if info.transform.rotation_degrees != 0:
            self.suspend(self._now, "unreviewed_coordinate_rotation")
            return None
        # Preview mirroring never changes model-space direction.
        probabilities = np.exp(
            log_probabilities([p.logits], self.calibration.temperature)
        )[0]
        winner = int(probabilities.argmax())
        threshold = max(self.policy.class_on, self.calibration.threshold)
        confident = (
            float(probabilities[winner]) >= threshold
            and int(np.count_nonzero(probabilities == probabilities[winner])) == 1
        )
        self.last_good_ns = p.source_timestamp_ns
        return g, vocabulary_for(IPN_HANDS_VERSION).tokens[winner], bool(confident)

    def arm_owner(self, track_id, prediction, frame, *, ready, now_ns):
        """Explicit host/user request, never called automatically by update()."""
        self._clock(now_ns)
        context = self.sync()
        if self.state != "WAIT_OWNER" or context.owner_track_id is not None:
            return self._record("explicit_enable_then_owner_required")
        evidence = self._evidence(prediction, frame, ready)
        if evidence is None:
            return None
        geometry, token, confident = evidence
        if (
            track_id != prediction.track_id
            or token not in self.neutral_tokens
            or not confident
            or not geometry.neutral
        ):
            return self._record(
                "owner_requires_unique_visible_neutral_hand", prediction
            )
        self.controller.bind_owner(track_id, now_ns)
        self.sync()
        self.last_good_ns = prediction.source_timestamp_ns
        # Binding invalidates the epoch; require NEW frames, never relabel this one.
        self._last_source, self._last_frame = (
            prediction.source_timestamp_ns,
            prediction.source_frame_id,
        )
        self._record("explicit_owner_bound_wait_release", prediction)
        return self.controller.snapshot()

    def update(self, prediction, frame, *, ready, now_ns):
        self.tick(now_ns)
        evidence = self._evidence(prediction, frame, ready)
        if evidence is None:
            return None
        g, token, confident = evidence
        context = self.controller.snapshot()
        if context.owner_track_id is None:
            return self._record("waiting_explicit_owner", prediction)
        if not confident:
            return self._wait_release("below_calibrated_draft_threshold", prediction)
        t, xy = prediction.source_timestamp_ns, g.palm_xy
        previous = self._point
        self._point = (t, xy)
        stationary = previous is None or (
            math.hypot(
                xy[0] - previous[1][0], (xy[1] - previous[1][1]) * g.aspect_ratio
            )
            / ((t - previous[0]) / 1e9)
            <= self.policy.stationary_speed
        )
        if self.state == "WAIT_RELEASE":
            if token in self.neutral_tokens and g.neutral and stationary:
                if self._neutral_since is None:
                    self._neutral_since = t
                if (
                    t - self._neutral_since >= self.policy.neutral_hold_ns
                    and now_ns >= self._cooldown_until
                ):
                    self.state = "ACTIVE"
                    return self._record(
                        "neutral_released_and_cooldown_complete", prediction
                    )
            else:
                self._neutral_since = None
            return self._record("waiting_geometric_neutral_and_cooldown", prediction)
        if token in self.neutral_tokens and self.state == "ACTIVE":
            return self._record("active_no_command", prediction)
        if token not in {"TLT", "TRT"}:
            return self._wait_release("source_class_has_no_action", prediction)
        sign = -1 if token == "TLT" else 1
        if self._candidate is None:
            self._candidate = (token, t, xy, xy, 1)
            self.state = "CANDIDATE"
            return self._record("motion_candidate_started", prediction)
        old_token, started, first, last, count = self._candidate
        if token != old_token:
            return self._wait_release("motion_direction_changed", prediction)
        if t - started > self.policy.max_motion_ns:
            return self._wait_release("motion_candidate_timeout", prediction)
        if (
            sign * (xy[0] - last[0]) < -self.policy.max_reverse_step
            or abs(xy[1] - first[1]) > self.policy.max_vertical_drift
        ):
            return self._wait_release("motion_geometry_rejected", prediction)
        self._candidate = (token, started, first, xy, count + 1)
        if (
            count + 1 < self.policy.min_motion_samples
            or t - started < self.policy.min_motion_ns
            or sign * (xy[0] - first[0]) + 1e-12 < self.policy.min_displacement
        ):
            return self._record("motion_incomplete", prediction)
        self._event_index += 1
        event = PublicActionEvent(
            event_id=f"public:{context.stream_id}:{context.context_epoch}:{self._event_index}",
            stream_id=prediction.stream_id,
            source_frame_id=prediction.source_frame_id,
            source_timestamp_ns=t,
            timebase=prediction.timebase,
            track_id=prediction.track_id,
            context_epoch=prediction.context_epoch,
            source_token=token,
            action="previous_slide" if token == "TLT" else "next_slide",
            target_context=context.target_context,
            candidate_start_ns=started,
            commit_ns=now_ns,
            model_version=self.model_version,
            calibration_sha256=self.calibration_sha256,
            policy_sha256=self.policy_sha256,
            synthetic=self.calibration.synthetic or prediction.timebase == "synthetic",
        )
        self._wait_release("completed_once_wait_release", prediction)
        self._cooldown_until = now_ns + self.policy.cooldown_ns
        return self._record("public_action_candidate_committed", prediction, event)
