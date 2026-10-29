"""Synthetic formula checks; these fixtures do not certify camera quality."""

import numpy as np
import pytest

from shared.contracts import HandObservation, TransformMetadata
from shared.features import FeatureConfig, FeatureExtractor, FeatureSample, load_feature_schema
from shared.features.schema import CX, CY, DS_DT, DT, MASK, MISSING, MOTION_VALID, QUALITY, SCALE, VX, VY


def observation(timestamp_ns=0, frame=0, *, offset=(0.0, 0.0, 0.0), factor=1.0, stream="s", track="t", valid=True, quality=1.0, points=None, timebase="synthetic"):
    if points is None:
        points = np.array([[0.05 + i % 5 * 0.025, 0.1 + i // 5 * 0.03, i * 0.001] for i in range(21)])
        points[5] = [0.11, 0.2, 0.005]
        points[9] = [0.16, 0.21, 0.009]
        points[13] = [0.2, 0.2, 0.013]
        points[17] = [0.24, 0.19, 0.017]
    points = np.asarray(points) * factor + np.asarray(offset)
    return HandObservation(stream_id=stream, source_frame_id=frame, source_timestamp_ns=timestamp_ns, timebase=timebase,
                           track_id=track, landmarks_image=points.tolist(), validity_mask=[True] * 21,
                           observation_valid=valid, handedness="right", quality=quality,
                           quality_flags=[] if valid else ["synthetic_occlusion"], transform_metadata=TransformMetadata())


def test_schema_has_all_94_ordered_fields_and_draft_rules():
    schema = load_feature_schema()
    assert schema["status"] == "draft"
    assert len(schema["fields"]) == len(schema["field_names"]) == 94
    assert [item["index"] for item in schema["fields"]] == list(range(94))
    assert schema["field_names"][63:73] == ["cx", "cy", "vx", "vy", "s", "ds_dt", "dt_seconds", "missing_ms", "q_geom", "motion_valid"]


def test_formula_aspect_ratio_local_translation_scale_and_direction():
    obs = observation()
    sample = FeatureExtractor().extract(obs, [640, 480])
    q = np.array(obs.landmarks_image)
    q[:, 1] *= 480 / 640
    s = np.linalg.norm(q[[5, 9, 17], :2] - q[0, :2], axis=1).mean()
    np.testing.assert_allclose(sample.values[:63], ((q - q[0]) / s).reshape(-1))
    np.testing.assert_allclose(sample.values[CX:CY + 1], q[[0, 5, 9, 13, 17], :2].mean(axis=0))
    assert sample.values[SCALE] == pytest.approx(s)
    translated = FeatureExtractor().extract(observation(offset=(0.2, -0.1, 0.3)), [640, 480])
    enlarged = FeatureExtractor().extract(observation(factor=2.0), [640, 480])
    np.testing.assert_allclose(translated.values[:63], sample.values[:63], atol=1e-14)
    np.testing.assert_allclose(enlarged.values[:63], sample.values[:63])
    # No left/right or rotation normalization: a horizontal reflection changes sign.
    points = np.array(obs.landmarks_image)
    points[:, 0] = 1 - points[:, 0]
    reflected = FeatureExtractor().extract(observation(points=points), [640, 480])
    np.testing.assert_allclose(reflected.values[:63].reshape(21, 3)[:, 0], -sample.values[:63].reshape(21, 3)[:, 0])
    assert sample.history_initialized and sample.observation_valid and not sample.motion_valid
    assert sample.values.shape == (94,) and np.isfinite(sample.values).all()
    assert not sample.values.flags.writeable


def test_derivatives_use_source_time_and_skip_after_missing():
    extractor = FeatureExtractor()
    first = extractor.extract(observation(1_000_000_000), [100, 100])
    second = extractor.extract(observation(1_050_000_000, 1, offset=(0.01, -0.02, 0)), [100, 100])
    assert second.motion_valid and second.values[MOTION_VALID] == 1
    np.testing.assert_allclose(second.values[VX:VY + 1], [0.2, -0.4])
    assert second.values[DT] == pytest.approx(0.05)
    assert second.values[DS_DT] == pytest.approx(0, abs=1e-12)
    missing = extractor.missing(1_100_000_000, 2, "s", "t")
    assert not missing.observation_valid and missing.history_initialized
    assert missing.values[MISSING] == 50 and missing.values[DT] == 0.05
    assert not missing.geometric_valid_mask.any()
    third = extractor.extract(observation(1_150_000_000, 3), [100, 100])
    assert not third.motion_valid and third.values[MISSING] == 0
    assert not third.geometric_valid_mask[[VX, VY, DS_DT]].any()
    assert first.geometric_valid_mask[[CX, CY, SCALE]].all()


def test_unprocessed_is_health_only_and_no_hand_cannot_create_a_track():
    extractor = FeatureExtractor()
    assert extractor.missing(0, 0, "s", "t") is None
    extractor.unprocessed(1_000_000, stream_id="s")
    assert extractor.identity is None
    extractor.extract(observation(10_000_000), [100, 100])
    extractor.unprocessed(20_000_000)
    sample = extractor.extract(observation(60_000_000, 1, offset=(0.01, 0, 0)), [100, 100])
    assert sample.values[DT] == 0.05 and sample.motion_valid
    extractor.unprocessed(600_000_001)
    sample = extractor.extract(observation(650_000_001, 2), [100, 100])
    assert sample.values[DT] == 0 and not sample.motion_valid


@pytest.mark.parametrize("change", [{"stream": "new"}, {"track": "new"}, {"context_epoch": 1}, {"timebase": "recording_relative"}])
def test_identity_and_timebase_changes_reset_history(change):
    extractor = FeatureExtractor()
    extractor.extract(observation(10), [100, 100])
    epoch = change.pop("context_epoch", 0)
    sample = extractor.extract(observation(0, **change), [100, 100], context_epoch=epoch)
    assert not sample.motion_valid and sample.values[DT] == 0


@pytest.mark.parametrize("kind", ["invalid", "small", "low_quality", "missing_key"])
def test_quality_invalid_geometry_stays_finite_and_disables_history_on_cold_start(kind):
    obs = observation(valid=kind != "invalid", factor=0.001 if kind == "small" else 1, quality=0.1 if kind == "low_quality" else 1.0)
    if kind == "missing_key":
        data = obs.model_dump()
        data["validity_mask"][5] = False
        data["observation_valid"] = False
        data["quality_flags"] = ["missing_key"]
        obs = HandObservation(**data)
    sample = FeatureExtractor().extract(obs, [640, 480])
    assert not sample.observation_valid and not sample.history_initialized
    assert sample.values[MISSING] == 0 and sample.values[MOTION_VALID] == 0
    assert not sample.values[:69].any() and not sample.values[MASK].any()
    assert np.isfinite(sample.values).all()
    if kind in {"small", "missing_key"}:
        assert sample.values[QUALITY] == 0


def test_long_gap_resets_history_and_derivatives_without_synthetic_dt():
    extractor = FeatureExtractor()
    extractor.extract(observation(0), [100, 100])
    invalid = extractor.extract(observation(500_000_001, 1, valid=False), [100, 100])
    assert invalid.values[DT] == invalid.values[MISSING] == 0
    assert not invalid.history_initialized


@pytest.mark.parametrize("timestamp", [100, 99])
def test_duplicate_or_decreasing_source_time_is_rejected(timestamp):
    extractor = FeatureExtractor()
    extractor.extract(observation(100), [100, 100])
    with pytest.raises(ValueError, match="strictly increase"):
        extractor.extract(observation(timestamp, 1), [100, 100])


@pytest.mark.parametrize("bad_size", [[0, 20], [20, -1], [20.0, 20], [True, 20], [20]])
def test_image_sizes_are_explicit(bad_size):
    with pytest.raises(ValueError):
        FeatureExtractor().extract(observation(), bad_size)


def test_future_edits_never_modify_past_feature_samples():
    def replay(future_offset):
        extractor = FeatureExtractor()
        return [extractor.extract(observation(i * 30_000_000, i, offset=(0 if i < 5 else future_offset, 0, 0)), [640, 480]) for i in range(10)]
    a, b = replay(0.0), replay(1.0)
    for left, right in zip(a[:5], b[:5]):
        np.testing.assert_array_equal(left.values, right.values)
    assert not np.array_equal(a[-1].values, b[-1].values)


@pytest.mark.parametrize("changes", [{"window_size": 31}, {"epsilon": 0}, {"min_valid_ratio": float("nan")}, {"reset_gap_ns": 100_000_000}, {"max_age_ns": True}])
def test_draft_config_rejects_bad_thresholds(changes):
    with pytest.raises(ValueError):
        FeatureConfig(**changes)


def test_sample_rejects_invalid_dimensions_times_and_masks():
    sample = FeatureExtractor().extract(observation(), [100, 100])
    data = dict(values=sample.values, timestamp_ns=0, source_frame_id=0, stream_id="s", track_id="t", context_epoch=0,
                observation_valid=True, motion_valid=False, history_initialized=True, geometric_valid_mask=sample.geometric_valid_mask)
    for change in ({"values": [0] * 93}, {"timestamp_ns": 0.1}, {"geometric_valid_mask": [1] * 69}, {"motion_valid": 1}):
        with pytest.raises(ValueError):
            FeatureSample(**(data | change))
