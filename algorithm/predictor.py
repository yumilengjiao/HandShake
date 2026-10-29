"""T02 predictor boundary and a deliberately scripted test double, not a model."""

from typing import Protocol

from shared.contracts import FeatureWindow, GESTURE_IDS, Gesture, ModelPrediction


class Predictor(Protocol):
    def predict(self, window: FeatureWindow) -> ModelPrediction: ...


class MockPredictor:
    model_version = "synthetic-script-v1"

    def __init__(self, gesture: Gesture = Gesture.SWIPE_RIGHT):
        self.gesture = gesture

    def predict(self, window: FeatureWindow) -> ModelPrediction:
        if not window.ready:
            raise ValueError("window_not_ready: " + window.ready_reason)
        logits = [0.0] * len(GESTURE_IDS)
        logits[GESTURE_IDS[self.gesture]] = 1.0
        return ModelPrediction(
            stream_id=window.stream_id,
            track_id=window.track_id,
            context_epoch=window.context_epoch,
            source_frame_id=window.source_frame_id,
            source_timestamp_ns=window.timestamps_ns[-1],
            inference_timestamp_ns=window.timestamps_ns[-1] + 1_000_000,
            logits=logits,
            model_version=self.model_version,
            timebase=window.timebase,
        )
