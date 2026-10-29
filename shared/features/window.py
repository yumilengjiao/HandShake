"""Causal 32-observation windows, health-clock freshness and readiness gates."""

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from shared.contracts import FeatureWindow
from shared.features.geometry import FeatureSample, check_ns
from shared.features.schema import FeatureConfig

if TYPE_CHECKING:
    from shared.features.normalization import FeatureNormalizer


class FeatureWindowBuilder:
    def __init__(self, config: FeatureConfig | None = None, normalizer: "FeatureNormalizer | None" = None):
        self.config = config or FeatureConfig()
        self.normalizer = normalizer
        if normalizer is not None and normalizer.config.config_hash != self.config.config_hash:
            raise ValueError("window builder and normalizer config mismatch")
        self.reset()

    def reset(self) -> None:
        self._samples: deque[FeatureSample] = deque(maxlen=self.config.window_size)
        self._features: deque[np.ndarray] = deque(maxlen=self.config.window_size)
        self.identity = None
        self._timebase = None
        self._consecutive_valid = 0
        self._warmed_up = False
        self._health_timestamp_ns = None
        self._source_health_timestamp_ns = None
        self._last_valid_timestamp_ns = None

    @property
    def samples(self) -> tuple[FeatureSample, ...]:
        """Raw observations used by this window, never normalized in place."""
        return tuple(self._samples)

    def append(self, sample: FeatureSample, *, now_ns: int | None = None) -> FeatureWindow | None:
        if not isinstance(sample, FeatureSample):
            raise TypeError("append requires FeatureSample")
        if now_ns is not None:
            check_ns(now_ns, "now_ns")
            if now_ns < sample.timestamp_ns:
                raise ValueError("health time cannot precede the source sample")
        if self.identity != sample.identity or self._timebase != sample.timebase:
            self.reset()
            self.identity, self._timebase = sample.identity, sample.timebase
        if now_ns is not None and self._health_timestamp_ns is not None and now_ns < self._health_timestamp_ns:
            raise ValueError("processing health timestamps must not decrease")
        if self._source_health_timestamp_ns is not None and sample.timestamp_ns < self._source_health_timestamp_ns:
            raise ValueError("sample source time precedes the source health clock")
        gap_ns = None
        if self._samples:
            gap_ns = sample.timestamp_ns - self._samples[-1].timestamp_ns
            if gap_ns <= 0:
                raise ValueError("processed source timestamps must strictly increase")
            if gap_ns > self.config.reset_gap_ns:
                self.reset()
                self.identity, self._timebase = sample.identity, sample.timebase
                gap_ns = None
        if self._last_valid_timestamp_ns is not None and sample.timestamp_ns - self._last_valid_timestamp_ns > self.config.reset_gap_ns:
            # Frequent processed failures are still a long loss of tracking.
            # They must not preserve a previous warm-up indefinitely.
            self.reset()
            self.identity, self._timebase = sample.identity, sample.timebase
            gap_ns = None
        if not self._warmed_up:
            if not sample.observation_valid:
                self._consecutive_valid = 0
            elif gap_ns is not None and gap_ns > self.config.max_sampling_gap_ns:
                self._consecutive_valid = 1
            else:
                self._consecutive_valid += 1
            self._warmed_up = self._consecutive_valid >= self.config.window_size
        self._samples.append(sample)
        self._features.append(self.normalizer.transform(sample) if self.normalizer is not None else sample.values.copy())
        if sample.observation_valid:
            self._last_valid_timestamp_ns = sample.timestamp_ns
        self._source_health_timestamp_ns = sample.timestamp_ns
        self._health_timestamp_ns = max(self._health_timestamp_ns or 0, sample.timestamp_ns, now_ns or 0)
        return self.current()

    def unprocessed(self, timestamp_ns: int, *, stream_id: str | None = None, track_id: str | None = None, context_epoch: int | None = None) -> None:
        check_ns(timestamp_ns)
        if self.identity is not None and ((stream_id is not None and stream_id != self.identity[0])
                or (track_id is not None and track_id != self.identity[1])
                or (context_epoch is not None and context_epoch != self.identity[2])):
            self.reset()
        if self._source_health_timestamp_ns is not None and timestamp_ns < self._source_health_timestamp_ns:
            raise ValueError("source health timestamps must not decrease")
        if self._samples and timestamp_ns - self._samples[-1].timestamp_ns > self.config.reset_gap_ns:
            self.reset()
        self._source_health_timestamp_ns = timestamp_ns
        self._health_timestamp_ns = max(self._health_timestamp_ns or 0, timestamp_ns)

    def current(self, *, now_ns: int | None = None) -> FeatureWindow | None:
        if not self._samples:
            return None
        last = self._samples[-1]
        now = self._health_timestamp_ns if now_ns is None else check_ns(now_ns, "now_ns")
        if now < last.timestamp_ns or (self._health_timestamp_ns is not None and now < self._health_timestamp_ns):
            raise ValueError("health time cannot decrease or precede source")
        self._health_timestamp_ns = now
        span = last.timestamp_ns - self._samples[0].timestamp_ns
        valid_ratio = sum(sample.observation_valid for sample in self._samples) / self.config.window_size
        if now - last.timestamp_ns > self.config.reset_gap_ns:
            self.reset()
            return None
        if not last.observation_valid:
            reason = "current_observation_invalid"
        elif not last.history_initialized:
            reason = "history_uninitialized"
        elif now - last.timestamp_ns > self.config.max_age_ns:
            reason = "stale"
        elif not self._warmed_up or len(self._samples) < self.config.window_size:
            reason = "warming_up"
        elif valid_ratio < self.config.min_valid_ratio:
            reason = "insufficient_valid_ratio"
        elif span > self.config.max_window_span_ns:
            reason = "window_span_exceeded"
        else:
            reason = "ready"
        return FeatureWindow(
            stream_id=last.stream_id, track_id=last.track_id, context_epoch=last.context_epoch,
            source_frame_id=last.source_frame_id, timebase=last.timebase,
            timestamps_ns=[sample.timestamp_ns for sample in self._samples],
            features=[row.tolist() for row in self._features], ready=reason == "ready", ready_reason=reason,
        )
