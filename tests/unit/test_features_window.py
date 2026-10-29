import numpy as np
import pytest

from shared.features import FeatureConfig, FeatureExtractor, FeatureWindowBuilder
from tests.unit.test_features_geometry import observation


def warm(builder=None, extractor=None, *, start=0, step=30_000_000):
    builder, extractor = builder or FeatureWindowBuilder(), extractor or FeatureExtractor()
    windows = [builder.append(extractor.extract(observation(start + i * step, i), [640, 480])) for i in range(32)]
    return builder, extractor, windows


def test_ready_only_on_32nd_consecutive_valid_observation_and_current_end():
    builder, _, windows = warm()
    assert all(not window.ready for window in windows[:-1])
    assert windows[-1].ready and windows[-1].ready_reason == "ready"
    assert len(windows[-1].features) == 32 and len(windows[0].features) == 1
    assert windows[-1].source_frame_id == 31
    assert builder.samples[-1].timestamp_ns == windows[-1].timestamps_ns[-1]


def test_prewarm_missing_breaks_continuity_instead_of_accumulating_short_runs():
    extractor, builder = FeatureExtractor(), FeatureWindowBuilder()
    for i in range(60):
        obs = observation(i * 30_000_000, i, valid=i != 20)
        window = builder.append(extractor.extract(obs, [100, 100]))
        assert window.ready == (i >= 52)


def test_prewarm_long_sampling_gap_restarts_count_without_resampling():
    extractor, builder = FeatureExtractor(), FeatureWindowBuilder()
    timestamps = [i * 30_000_000 for i in range(20)] + [800_000_000 + i * 30_000_000 for i in range(32)]
    for i, timestamp in enumerate(timestamps):
        window = builder.append(extractor.extract(observation(timestamp, i), [100, 100]))
        assert window.ready == (i == 51)
    assert window.timestamps_ns == timestamps[-32:]


def test_short_missing_keeps_history_but_disables_current_and_recovery_checks_ratio():
    builder, extractor, windows = warm()
    for i in range(32, 40):
        missing = extractor.missing(i * 30_000_000, i, "s", "t")
        window = builder.append(missing)
        assert not window.ready and window.ready_reason == "current_observation_invalid"
    window = builder.append(extractor.extract(observation(40 * 30_000_000, 40), [100, 100]))
    assert not window.ready and window.ready_reason == "insufficient_valid_ratio"
    for i in range(41, 67):
        window = builder.append(extractor.extract(observation(i * 30_000_000, i), [100, 100]))
    assert window.ready


def test_short_missing_can_recover_ready_but_cannot_restore_owner_or_candidates():
    builder, extractor, _ = warm()
    window = builder.append(extractor.missing(960_000_000, 32, "s", "t"))
    assert not window.ready
    window = builder.append(extractor.extract(observation(990_000_000, 33), [100, 100]))
    assert window.ready
    assert not hasattr(window, "owner_track_id")


def test_unprocessed_health_does_not_append_background_and_stale_then_reset():
    builder, extractor, _ = warm()
    before = builder.current()
    builder.unprocessed(1_100_000_000)
    assert len(builder.samples) == 32
    stale = builder.current()
    assert not stale.ready and stale.ready_reason == "stale"
    assert stale.timestamps_ns == before.timestamps_ns
    builder.unprocessed(1_430_000_001)
    assert builder.current() is None and not builder.samples
    extractor.unprocessed(1_430_000_001)
    window = builder.append(extractor.extract(observation(1_460_000_001, 32), [100, 100]))
    assert not window.ready and len(window.features) == 1


@pytest.mark.parametrize("kind", ["stream", "track", "epoch", "timebase", "long_gap"])
def test_window_never_crosses_identity_or_long_gap(kind):
    builder, extractor, _ = warm()
    kwargs = {"stream": "new"} if kind == "stream" else {"track": "new"} if kind == "track" else {"timebase": "recording_relative"} if kind == "timebase" else {}
    timestamp = 1_430_000_001 if kind == "long_gap" else 960_000_000
    sample = extractor.extract(observation(timestamp, 32, **kwargs), [100, 100], context_epoch=1 if kind == "epoch" else 0)
    window = builder.append(sample)
    assert len(window.features) == 1 and not window.ready


@pytest.mark.parametrize("step", [20_000_000, 33_000_000, 100_000_000])
def test_multiple_real_sampling_intervals_are_preserved(step):
    _, _, windows = warm(step=step)
    assert windows[-1].ready
    assert np.diff(windows[-1].timestamps_ns).tolist() == [step] * 31


def test_span_gate_is_independent_of_validity_and_current_age():
    config = FeatureConfig(max_window_span_ns=1_000_000_000)
    builder = FeatureWindowBuilder(config)
    extractor = FeatureExtractor(config)
    _, _, windows = warm(builder, extractor, step=50_000_000)
    assert not windows[-1].ready and windows[-1].ready_reason == "window_span_exceeded"


def test_no_hidden_future_context_in_windows_and_duplicate_source_rejected():
    extractor, builder = FeatureExtractor(), FeatureWindowBuilder()
    sample = extractor.extract(observation(), [100, 100])
    first = builder.append(sample)
    with pytest.raises(ValueError, match="strictly increase"):
        builder.append(sample)
    for i in range(1, 40):
        builder.append(extractor.extract(observation(i * 30_000_000, i, offset=(0.1, 0, 0)), [100, 100]))
    assert len(first.features) == 1 and first.timestamps_ns == [0]


def test_future_health_time_never_makes_old_feature_ready():
    builder, _, windows = warm()
    with pytest.raises(ValueError):
        builder.current(now_ns=windows[-1].timestamps_ns[-1] - 1)
    assert builder.current(now_ns=1_100_000_000).ready_reason == "stale"
    assert builder.current(now_ns=2_000_000_000) is None


def test_processing_latency_is_not_confused_with_source_order():
    builder, extractor = FeatureWindowBuilder(), FeatureExtractor()
    for index in range(40):
        timestamp = index * 30_000_000
        sample = extractor.extract(observation(timestamp, index), [640, 480])
        window = builder.append(sample, now_ns=timestamp + 100_000_000)
    assert window.ready and window.timestamps_ns[-1] == 39 * 30_000_000


def test_continuous_long_loss_requires_a_new_complete_warmup():
    builder, extractor, _ = warm()
    for index in range(32, 64):
        window = builder.append(extractor.missing(index * 30_000_000, index, "s", "t"))
        assert not window.ready
    for index in range(64, 96):
        window = builder.append(extractor.extract(observation(index * 30_000_000, index), [640, 480]))
        assert window.ready == (index == 95)


def test_health_queries_cannot_rewind_and_make_stale_features_fresh():
    builder, _, _ = warm()
    assert builder.current(now_ns=1_100_000_000).ready_reason == "stale"
    with pytest.raises(ValueError, match="decrease"):
        builder.current(now_ns=1_000_000_000)


def test_already_expired_delayed_sample_does_not_leave_a_warmup_history():
    extractor, builder = FeatureExtractor(), FeatureWindowBuilder()
    sample = extractor.extract(observation(0), [640, 480])
    assert builder.append(sample, now_ns=500_000_001) is None
    assert not builder.samples
