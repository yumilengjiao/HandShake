"""Shared guarded FeatureWindow adapters for B0, MLP, and causal TCN.

This module does not calibrate scores, infer intent, or emit control events.
Non-monotonic recordings require an explicit clock in the same time domain.
"""

from dataclasses import dataclass
from math import isfinite
from time import monotonic_ns
from typing import Callable, Literal

import numpy as np

from algorithm.models.rule_baseline import RuleBaseline
from shared import PROTOCOL_VERSION
from shared.classification import SourceClassification
from shared.classes import load_classes
from shared.contracts import FeatureWindow, ModelPrediction, Timebase
from shared.features.schema import FeatureConfig


@dataclass(frozen=True)
class PredictionGate:
    window_size: int = 32
    max_prediction_age_ns: int = 150_000_000
    max_window_span_ns: int = 3_200_000_000
    min_quality: float = 0.5

    @classmethod
    def from_feature_config(cls, config: FeatureConfig) -> "PredictionGate":
        """Keep inference checks aligned with the window-builder configuration."""
        return cls(window_size=config.window_size, max_prediction_age_ns=config.max_age_ns,
                   max_window_span_ns=config.max_window_span_ns, min_quality=config.min_geometry_quality)

    def __post_init__(self):
        if self.window_size != 32:
            raise ValueError("all first-version predictors require a 32-observation window")
        if any(type(value) is not int or value <= 0 for value in (self.max_prediction_age_ns, self.max_window_span_ns)):
            raise ValueError("prediction age and window span must be positive integer nanoseconds")
        if isinstance(self.min_quality, bool) or not isfinite(self.min_quality) or not 0 < self.min_quality <= 1:
            raise ValueError("min_quality must be in (0,1]")


class _GuardedPredictor:
    def __init__(self, *, model_version: str, gate: PredictionGate | None, clock_ns: Callable[[], int] | None, clock_timebase: Timebase):
        if not isinstance(model_version, str) or not model_version:
            raise ValueError("model_version must be nonempty")
        if clock_timebase not in {"monotonic", "recording_relative", "synthetic"}:
            raise ValueError("unsupported clock_timebase")
        if clock_ns is None and clock_timebase != "monotonic":
            raise ValueError("non-monotonic timebases require an explicit same-domain clock_ns")
        self.model_version = model_version
        self.gate = gate or PredictionGate()
        self.clock_ns = clock_ns or monotonic_ns
        self.clock_timebase = clock_timebase

    def _clock(self, window: FeatureWindow) -> int:
        if window.timebase != self.clock_timebase:
            raise ValueError("clock_timebase_mismatch: window and clock must use the same domain")
        now = self.clock_ns()
        if type(now) is not int or not 0 <= now <= 2**63 - 1:
            raise ValueError("clock_ns must return a nonnegative integer nanosecond timestamp")
        if now < window.timestamps_ns[-1]:
            raise ValueError("future_input: source timestamp is ahead of the inference clock")
        if now - window.timestamps_ns[-1] > self.gate.max_prediction_age_ns:
            raise ValueError("stale_input: latest observation is too old")
        return now

    def _validate(self, window: FeatureWindow) -> int:
        if not isinstance(window, FeatureWindow):
            raise TypeError("predict requires a FeatureWindow")
        if not window.ready:
            raise ValueError("window_not_ready: " + window.ready_reason)
        if len(window.features) != self.gate.window_size:
            raise ValueError("window_length: all first-version predictors require exactly 32 observations")
        if window.timestamps_ns[-1] - window.timestamps_ns[0] > self.gate.max_window_span_ns:
            raise ValueError("window_span: historical observations span too long")
        current = window.features[-1]
        if current[71] < self.gate.min_quality or current[70] != 0.0 or any(mask != 1.0 for mask in current[73:]):
            raise ValueError("current_quality_invalid: latest observation must be valid")
        return self._clock(window)

    def _prediction(self, window: FeatureWindow, scores, started_at_ns: int) -> ModelPrediction | SourceClassification:
        finished_at_ns = self._clock(window)
        if finished_at_ns < started_at_ns:
            raise ValueError("clock_regressed: inference clock moved backwards")
        vocabulary = getattr(self, "vocabulary", None)
        result_type = ModelPrediction
        extra = {}
        if vocabulary is not None and vocabulary.vocabulary_version != PROTOCOL_VERSION:
            result_type = SourceClassification
            extra["vocabulary_version"] = vocabulary.vocabulary_version
        return result_type(
            **extra,
            stream_id=window.stream_id,
            track_id=window.track_id,
            context_epoch=window.context_epoch,
            source_frame_id=window.source_frame_id,
            source_timestamp_ns=window.timestamps_ns[-1],
            inference_timestamp_ns=finished_at_ns,
            logits=[float(value) for value in scores],
            model_version=self.model_version,
            timebase=window.timebase,
        )


class RulePredictor(_GuardedPredictor):
    """Adapt B0 to the same window and freshness gate used by learned models.

    Raw feature windows need no transform. For standardized windows pass
    ``inverse_features=normalizer.inverse_geometric``; timing, quality and masks
    remain unchanged. Decision scores are deliberately left uncalibrated.
    """

    def __init__(self, model: RuleBaseline | None = None, *, model_version: str = "rule-v1-draft", inverse_features: Callable | None = None, gate: PredictionGate | None = None, clock_ns: Callable[[], int] | None = None, clock_timebase: Timebase = "monotonic"):
        super().__init__(model_version=model_version, gate=gate, clock_ns=clock_ns, clock_timebase=clock_timebase)
        self.model = model or RuleBaseline()
        if self.model.thresholds.min_quality != self.gate.min_quality:
            raise ValueError("rule and prediction gate must use the same min_quality")
        self.inverse_features = inverse_features

    def predict(self, window: FeatureWindow) -> ModelPrediction:
        started = self._validate(window)
        features = np.asarray(window.features, dtype=np.float64)
        if self.inverse_features is not None:
            features = np.asarray(self.inverse_features(features), dtype=np.float64)
            if features.shape != (32, 94) or not np.isfinite(features).all():
                raise ValueError("inverse_features must return finite [32,94] features")
            if not np.array_equal(features[:, 69:], np.asarray(window.features)[:, 69:]):
                raise ValueError("inverse_features must preserve timing, quality, motion-valid and masks")
        return self._prediction(window, self.model.logits(features, window.timestamps_ns), started)


class TorchPredictor(_GuardedPredictor):
    """Inference-only adapter: MLP sees x[:, -1, :], TCN sees the full window.

    The torch dependency is imported only when constructing this adapter.
    Features have already been standardized by the shared T11 pipeline.
    """

    def __init__(self, model, *, model_version: str, input_kind: Literal["mlp", "tcn"] | None = None, gate: PredictionGate | None = None, clock_ns: Callable[[], int] | None = None, clock_timebase: Timebase = "monotonic", vocabulary=None):
        import torch

        super().__init__(model_version=model_version, gate=gate, clock_ns=clock_ns, clock_timebase=clock_timebase)
        self.vocabulary = vocabulary or load_classes()
        self.model = model
        if input_kind is None:
            from algorithm.models.mlp import GestureMLP
            from algorithm.models.tcn import GestureTCN

            if isinstance(model, GestureMLP):
                input_kind = "mlp"
            elif isinstance(model, GestureTCN):
                input_kind = "tcn"
            else:
                raise ValueError("input_kind must be explicit for a custom torch model")
        if input_kind not in {"mlp", "tcn"}:
            raise ValueError("input_kind must be mlp or tcn")
        if tuple(getattr(model, "class_tokens", ())) != tuple(self.vocabulary.tokens):
            raise ValueError("model class_tokens must match the shared vocabulary order")
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.input_kind = input_kind
        self.model.eval()

    def predict(self, window: FeatureWindow) -> ModelPrediction | SourceClassification:
        import torch

        started = self._validate(window)
        parameter = next(self.model.parameters(), None)
        options = {"dtype": parameter.dtype, "device": parameter.device} if parameter is not None else {"dtype": torch.float32, "device": "cpu"}
        values = torch.as_tensor(window.features, **options).unsqueeze(0)
        model_input = values[:, -1, :] if self.input_kind == "mlp" else values
        self.model.eval()
        with torch.no_grad():
            scores = self.model(model_input)
        if scores.shape != (1, self.vocabulary.class_count) or not torch.isfinite(scores).all().item():
            raise ValueError("model must return finite raw logits with shape [1,C]")
        return self._prediction(window, scores.detach().cpu().tolist()[0], started)
