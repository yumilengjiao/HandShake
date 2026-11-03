"""One model→policy session. Inference never holds the dispatch/stop gate.

Default shadow mode has no execution authority. This base class executes only
synthetic input/model/calibration with explicit opt-in. The separate local-demo
subclass owns its additional live, application-local-only authorization checks.
"""

from dataclasses import dataclass
from threading import Lock

from algorithm.bundle import LoadedBundle
from algorithm.evaluation.classification import log_probabilities
from algorithm.online import OnlineClassifier
from backend.interaction.geometry import frame_geometry
from backend.interaction.model_host import ModelHost
from shared.classes import IPN_HANDS_VERSION
from shared.classification import SourceClassification
from shared.observations import ObservationFrame


@dataclass(frozen=True)
class ModelStep:
    reason: str
    notice: object = None
    event: object = None
    ticket: object = None


class ModelPresentationSession:
    def __init__(
        self,
        *,
        bundle,
        calibration,
        calibration_sha256,
        clock,
        timebase="synthetic",
        source_synthetic=True,
        execution="shadow",
        allow_synthetic=False,
        watchdog=True,
        policy=None,
        stream_id=None,
    ):
        if (
            not isinstance(bundle, LoadedBundle)
            or bundle.vocabulary.vocabulary_version != IPN_HANDS_VERSION
        ):
            raise ValueError(
                "verified IPN HandS bundle required for public model bridge"
            )
        if (
            timebase not in {"synthetic", "monotonic", "recording_relative"}
            or type(source_synthetic) is not bool
        ):
            raise ValueError("explicit source timebase/provenance required")
        if stream_id is not None and (
            not isinstance(stream_id, str) or not 1 <= len(stream_id) <= 128
        ):
            raise ValueError("invalid source stream identity")
        if timebase == "synthetic" and not source_synthetic:
            raise ValueError("synthetic timebase cannot claim real observations")
        source = calibration.source
        data = bundle.training_identity["data"]
        bindings = {
            "model_sha256": bundle.manifest_sha256,
            "dataset_sha256": bundle.normalizer.stats.dataset_hash,
            "split_sha256": bundle.normalizer.stats.split_hash,
            "stats_sha256": bundle.normalizer.stats_hash,
            "config_sha256": bundle.normalizer.config.config_hash,
            "cache_index_sha256": data["val_index_sha256"],
            "endpoint_sha256": data["val"]["endpoint_sha256"],
            "vocabulary_version": bundle.vocabulary.vocabulary_version,
            "data_synthetic": data["synthetic"],
            "model_synthetic": bundle.synthetic_training,
        }
        if any(
            getattr(source, name) != expected for name, expected in bindings.items()
        ):
            raise ValueError("calibration/bundle/validation inventory mismatch")
        warmup_ns = (
            bundle.normalizer.config.max_window_span_ns
            + bundle.normalizer.config.max_age_ns
        )
        if not 0 < warmup_ns <= 5_000_000_000:
            raise ValueError("model warmup requires a bounded reviewed window span")
        self.bundle, self.clock = bundle, clock
        self.synthetic = source_synthetic or calibration.synthetic
        self.host = self._make_host(
            execution=execution,
            timebase=timebase,
            source_synthetic=source_synthetic,
            warmup_ns=warmup_ns,
            source_stream_id=stream_id,
            calibration=calibration,
            calibration_sha256=calibration_sha256,
            clock=clock,
            allow_synthetic=allow_synthetic,
            watchdog=watchdog,
            policy=policy,
        )
        self._frame_lock = Lock()
        self._generation = 0
        self._classifier = None
        self._classifier_key = None
        self._owner_request = None
        self._last_notice = None
        self._eligible_track = None
        self.reason = "initial"

    def _make_host(self, **kwargs):
        return ModelHost(**kwargs)

    @classmethod
    def load(
        cls, package, *, model_sha256, calibration_path, calibration_sha256, **kwargs
    ):
        from algorithm.bundle import load_bundle
        from algorithm.evaluation.calibration import load_for_bundle

        bundle = load_bundle(package, expected_manifest_sha256=model_sha256)
        calibration = load_for_bundle(
            calibration_path, expected_sha256=calibration_sha256, bundle=bundle
        )
        return cls(
            bundle=bundle,
            calibration=calibration,
            calibration_sha256=calibration_sha256,
            **kwargs,
        )

    @property
    def controller(self):
        return self.host.controller

    def enable(self, *, epoch):
        with self.controller.gate:
            if (
                type(epoch) is not int
                or epoch != self.controller.snapshot().context_epoch
            ):
                self.reason = "stale_enable_context"
                return False
            if self._frame_lock.locked():
                self.reason = "model_work_not_quiescent"
                return False
            if not self.host.enable():
                return False
            self._generation += 1
            self._owner_request = self._last_notice = self._eligible_track = None
            self.reason = "model_initial_warmup"
            return True

    def stop(self, reason="explicit_stop"):
        # No acquisition of _frame_lock: a blocked native forward must not block stop.
        self.host.stop(reason)
        with self.controller.gate:
            self._generation += 1
            self._owner_request = self._last_notice = self._eligible_track = None
            self.reason = reason

    def close(self):
        self.host.close()
        with self.controller.gate:
            self._generation += 1
            self._owner_request = self._last_notice = self._eligible_track = None
            self.reason = "shutdown"

    def request_owner(self, track_id, *, epoch):
        with self.controller.gate:
            context = self.controller.snapshot()
            now = self.host._read_clock()
            notice = self._last_notice
            if (
                self.host._closed.is_set()
                or type(epoch) is not int
                or epoch != context.context_epoch
                or not isinstance(track_id, str)
                or not 1 <= len(track_id) <= 128
                or not context.enabled
                or context.suspended
                or context.owner_track_id is not None
                or self._owner_request is not None
                or notice is None
                or notice.state != "predicted"
                or self._eligible_track != track_id
                or notice.context_epoch != str(context.context_epoch)
                or not 0
                <= now - int(notice.source_timestamp_ns)
                <= self.host.engine.policy.max_age_ns
            ):
                return False
            # Consent applies to ONE subsequent fresh model result, not a queued future gesture.
            self._owner_request = (
                context.context_epoch,
                track_id,
                int(notice.source_frame_id),
                now + 500_000_000,
            )
            return True

    def _cancelled(self, context, generation):
        return (
            self.host._closed.is_set()
            or generation != self._generation
            or self.controller.snapshot() != context
        )

    def process(self, frame):
        if not isinstance(frame, ObservationFrame):
            self.stop("invalid_observation_type")
            raise ValueError(
                "model bridge accepts whole observation frames, never supplied logits"
            )
        if not self._frame_lock.acquire(blocking=False):
            self.stop("overlapping_model_work")
            return ModelStep("busy_dropped_and_stopped")
        try:
            with self.controller.gate:
                self.host.tick()
                context = self.controller.snapshot()
                generation = self._generation
                if (
                    self.host._closed.is_set()
                    or not context.enabled
                    or context.suspended
                ):
                    self._owner_request = self._last_notice = self._eligible_track = (
                        None
                    )
                    return ModelStep("disabled_or_suspended")
                if (frame.info.stream_id, frame.info.timebase) != (
                    context.stream_id,
                    context.timebase,
                ):
                    self.stop("source_switch_requires_explicit_reconnect")
                    return ModelStep(self.reason)
                if (
                    self.controller.changes
                    and frame.info.timestamp_ns
                    < self.controller.changes[-1].timestamp_ns
                ):
                    return ModelStep("frame_predates_context")
                key = (context.stream_id, context.context_epoch, generation)
                if self._classifier_key != key:
                    self._classifier = OnlineClassifier(
                        self.bundle,
                        context_epoch=context.context_epoch,
                        clock=self.clock,
                        timebase=context.timebase,
                        cancelled=lambda: self._cancelled(context, generation),
                    )
                    self._classifier_key = key
                classifier = self._classifier
            # All feature/model work is outside the controller gate; one bounded in-flight call.
            notice = classifier.update(frame)
            prediction = classifier.last_prediction
            with self.controller.gate:
                if self._cancelled(context, generation):
                    return ModelStep("cancelled_old_model_work")
                self.host.tick()
                if self._cancelled(context, generation):
                    self._last_notice = self._eligible_track = None
                    return ModelStep("model_health_or_surface_expired")
                self._last_notice = notice
                self._eligible_track = None
                if self.host.warmup is not None:
                    if not self.host.warmup_progress(notice, frame):
                        self.reason = self.host.reason
                        return ModelStep(self.reason, notice)
                request, self._owner_request = self._owner_request, None
                if notice.state != "predicted":
                    if context.owner_track_id is not None:
                        self.host.stop("model_result_unavailable:" + notice.reason)
                    self.reason = notice.reason
                    return ModelStep(self.reason, notice)
                if not isinstance(prediction, SourceClassification):
                    raise ValueError("same-call typed source prediction missing")
                geometry = frame_geometry(frame, prediction.track_id)
                lp = log_probabilities(
                    [prediction.logits], self.host.engine.calibration.temperature
                )[0]
                import numpy as np

                probabilities = np.exp(lp)
                if (
                    int(probabilities.argmax()) == 0
                    and int(np.count_nonzero(probabilities == probabilities[0])) == 1
                    and probabilities[0]
                    >= max(
                        self.host.engine.policy.class_on,
                        self.host.engine.calibration.threshold,
                    )
                    and geometry.visible_tracks == (prediction.track_id,)
                    and geometry.quality_valid
                    and not geometry.ambiguous
                    and geometry.neutral
                ):
                    self._eligible_track = prediction.track_id
                if request is not None:
                    epoch, track, minimum_frame, expires = request
                    if (
                        epoch == context.context_epoch
                        and prediction.source_frame_id > minimum_frame
                        and self.host._read_clock() < expires
                        and self.host.arm_owner(track, prediction, frame, ready=True)
                    ):
                        self._last_notice = self._eligible_track = None
                        self.reason = "owner_bound_model_history_reset"
                        return ModelStep(self.reason, notice)
                    self.reason = "owner_request_rejected_or_expired"
                    return ModelStep(self.reason, notice)
                event, ticket = self.host.update(prediction, frame, ready=True)
                if event is not None and self.synthetic and not event.synthetic:
                    event = event.model_copy(update={"synthetic": True})
                self.reason = (
                    "shadow_candidate"
                    if event is not None and self.host.execution == "shadow"
                    else self.host.reason
                )
                return ModelStep(self.reason, notice, event, ticket)
        except Exception:
            self.stop("model_bridge_failure")
            raise
        finally:
            self._frame_lock.release()

    def next_deadline_ns(self):
        from backend.interaction.public_presentation import (
            HEARTBEAT_NS,
            RENDER_TIMEOUT_NS,
        )

        with self.controller.gate:
            context = self.controller.snapshot()
            if not context.enabled or self.host._closed.is_set():
                return None
            deadlines = [self.host._last_heartbeat + HEARTBEAT_NS]
            if self.host.rendered != (
                context.context_epoch,
                self.host.backend.revision,
            ):
                deadlines.append(self.host._render_since + RENDER_TIMEOUT_NS)
            warm = self.host.warmup
            if warm is not None:
                deadlines.extend(
                    (
                        warm.deadline_ns,
                        warm.source_ns + self.host.engine.policy.max_age_ns + 1,
                    )
                )
            elif (
                context.owner_track_id is not None
                and self.host.engine.last_good_ns is not None
            ):
                deadlines.append(
                    self.host.engine.last_good_ns
                    + self.host.engine.policy.max_age_ns
                    + 1
                )
            return min(deadlines)

    def reconnect(self, stream_id):
        self.stop("source_reconnect")
        with self.controller.gate:
            self.controller.reconnect(stream_id, self.host._read_clock())
            self.host.engine.sync()

    def snapshot(self):
        with self.controller.gate:
            value = self.host.snapshot()
            context = self.controller.snapshot()
            if self._owner_request is not None and (
                not context.enabled
                or context.suspended
                or self._owner_request[0] != context.context_epoch
                or self.host._read_clock() >= self._owner_request[3]
            ):
                self._owner_request = None
            notice = self._last_notice
            if notice is not None and (
                not context.enabled
                or context.suspended
                or notice.context_epoch != str(context.context_epoch)
                or self.host._read_clock() - int(notice.source_timestamp_ns)
                > self.host.engine.policy.max_age_ns
            ):
                notice = None
            return value | {
                "format_version": "model-presentation-session-v1",
                "synthetic": self.synthetic,
                "control_origin": "model_inference",
                "execution": self.host.execution,
                "scope": "shadow_no_execution"
                if self.host.execution == "shadow"
                else "application-local",
                "model_warming": self.host.warmup is not None,
                "model_busy": self._frame_lock.locked(),
                "model_version": self.bundle.model_version,
                "classification": notice.model_dump(mode="json") if notice else None,
                "eligible_owner": self._eligible_track if notice else None,
                "owner_request_pending": self._owner_request is not None,
                "bridge_reason": self.reason,
            }
