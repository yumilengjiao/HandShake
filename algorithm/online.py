"""Per-stream online classification using the exact T11 feature/window pipeline.

Single-observed-hand preview policy: multiple hands reset warm-up rather than
invent an operator. There is deliberately no reference to the action dispatcher.
"""

from __future__ import annotations

import time

from algorithm.inference import PredictionGate, TorchPredictor
from shared.features import FeatureExtractor, FeatureWindowBuilder
from shared.features.geometry import check_ns
from shared.inference_protocol import ClassificationNotice


class OnlineClassifier:
    def __init__(
        self,
        bundle,
        *,
        context_epoch: int,
        clock=time.monotonic_ns,
        timebase="monotonic",
        cancelled=lambda: False,
    ):
        self.bundle = bundle
        self.context_epoch = check_ns(context_epoch, "context_epoch")
        self.clock, self.timebase, self.cancelled = clock, timebase, cancelled
        self.predictor = TorchPredictor(
            bundle.model,
            model_version=bundle.model_version,
            input_kind=bundle.model_type,
            vocabulary=bundle.vocabulary,
            gate=PredictionGate.from_feature_config(bundle.normalizer.config),
            clock_ns=clock,
            clock_timebase=timebase,
        )
        self.extractor = FeatureExtractor(bundle.normalizer.config)
        self.builder = FeatureWindowBuilder(bundle.normalizer.config, bundle.normalizer)
        self.identity = None
        self.last_frame = self.last_timestamp = None

    def reset(self):
        self.extractor.reset()
        self.builder.reset()
        self.identity = None
        self.last_frame = self.last_timestamp = None

    def _notice(
        self, frame, state, reason, *, track=None, inference_ns=None, logits=()
    ):
        return ClassificationNotice(
            state=state,
            reason=reason,
            model_version=self.bundle.model_version,
            model_type=self.bundle.model_type,
            vocabulary_version=self.bundle.vocabulary.vocabulary_version,
            synthetic_training=self.bundle.synthetic_training,
            stream_id=frame.info.stream_id,
            context_epoch=str(self.context_epoch),
            source_frame_id=str(frame.info.frame_id),
            source_timestamp_ns=str(frame.info.timestamp_ns),
            track_id=track,
            window_length=len(self.builder.samples),
            inference_ns=None if inference_ns is None else str(inference_ns),
            logits=tuple(logits),
        )

    def update(self, frame):
        info = frame.info
        if info.timebase != self.timebase:
            raise ValueError("online source/clock timebase mismatch")
        now = check_ns(self.clock(), "clock")
        if self.cancelled():
            self.reset()
            return self._notice(frame, "rejected", "cancelled")
        stream = (info.stream_id, self.context_epoch)
        if self.identity != stream:
            self.reset()
            self.identity = stream
        if self.last_frame is not None and (
            info.frame_id <= self.last_frame or info.timestamp_ns <= self.last_timestamp
        ):
            return self._notice(frame, "rejected", "out_of_order")
        self.last_frame, self.last_timestamp = info.frame_id, info.timestamp_ns
        if now < info.timestamp_ns:
            self.reset()
            return self._notice(frame, "rejected", "future_input")
        if now - info.timestamp_ns > self.bundle.normalizer.config.max_age_ns:
            self.reset()
            return self._notice(frame, "rejected", "stale")
        status = frame.status.perception_status.value
        if status in {"not_processed", "error"}:
            self.extractor.unprocessed(
                info.timestamp_ns,
                stream_id=info.stream_id,
                context_epoch=self.context_epoch,
            )
            self.builder.unprocessed(
                info.timestamp_ns,
                stream_id=info.stream_id,
                context_epoch=self.context_epoch,
            )
            return self._notice(
                frame,
                "rejected",
                "not_processed" if status == "not_processed" else "perception_error",
            )
        if len(frame.observations) > 1:
            self.extractor.reset()
            self.builder.reset()
            return self._notice(frame, "rejected", "multiple_hands")
        if not frame.observations:
            if self.extractor.identity is not None:
                sample = self.extractor.missing(
                    info.timestamp_ns,
                    info.frame_id,
                    info.stream_id,
                    self.extractor.identity[1],
                    context_epoch=self.context_epoch,
                )
                if sample is not None:
                    self.builder.append(sample, now_ns=now)
            return self._notice(frame, "rejected", "no_hand")
        observation = frame.observations[0]
        sample = self.extractor.extract(
            observation, [info.width, info.height], context_epoch=self.context_epoch
        )
        window = self.builder.append(sample, now_ns=now)
        if not window.ready:
            state = (
                "warming_up"
                if window.ready_reason in {"warming_up", "history_uninitialized"}
                else "rejected"
            )
            return self._notice(
                frame, state, window.ready_reason, track=observation.track_id
            )
        started = time.perf_counter_ns()
        try:
            prediction = self.predictor.predict(window)
        except ValueError as error:
            # Freshness can expire during inference. Never reuse a previous score.
            reason = (
                "stale" if str(error).startswith("stale_input") else "inference_error"
            )
            return self._notice(
                frame,
                "rejected" if reason == "stale" else "error",
                reason,
                track=observation.track_id,
            )
        if self.cancelled():
            self.reset()
            return self._notice(frame, "rejected", "cancelled")
        if (
            prediction.context_epoch != self.context_epoch
            or prediction.stream_id != info.stream_id
            or prediction.source_frame_id != info.frame_id
            or prediction.model_version != self.bundle.model_version
        ):
            self.reset()
            return self._notice(frame, "error", "inference_error")
        return self._notice(
            frame,
            "predicted",
            "ready",
            track=observation.track_id,
            inference_ns=time.perf_counter_ns() - started,
            logits=prediction.logits,
        )
