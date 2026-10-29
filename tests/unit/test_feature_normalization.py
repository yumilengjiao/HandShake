import json
import warnings

import numpy as np
import pytest

from shared.features import FeatureConfig, FeatureExtractor, FeatureNormalizer, FeatureStatisticsWarning
from shared.features.schema import DT, MISSING, MOTION_VALID, VX
from tests.unit.test_features_geometry import observation


def training_samples():
    extractor = FeatureExtractor()
    samples = [extractor.extract(observation(i * 30_000_000, i, offset=(i * 0.001, 0, 0)), [640, 480]) for i in range(5)]
    samples.append(extractor.missing(150_000_000, 5, "s", "t"))
    samples.append(extractor.extract(observation(180_000_000, 6), [640, 480]))
    return samples


def fit(samples=None):
    with pytest.warns(FeatureStatisticsWarning):
        return FeatureNormalizer.fit(training_samples() if samples is None else samples,
                                     dataset_hash="d" * 64, split_hash="e" * 64)


def test_only_valid_field_samples_fit_and_times_use_fixed_scaling():
    samples, normalizer = training_samples(), fit()
    assert normalizer.stats.count[0] == 6
    assert normalizer.stats.count[VX] == 4
    assert normalizer.stats.mean[VX] == pytest.approx(0.001 / 0.03)
    missing = normalizer.transform(samples[5])
    assert not missing[:69].any()
    assert missing[DT] == pytest.approx(0.3)
    assert missing[MISSING] == pytest.approx(0.03)
    assert missing[MOTION_VALID] == 0
    restored = normalizer.inverse_geometric(normalizer.transform(samples[2]))
    np.testing.assert_allclose(restored[:69], samples[2].values[:69], atol=1e-12)


def test_provenance_save_load_refuses_overwrite_or_tampering(tmp_path):
    normalizer = fit()
    path = tmp_path / "stats.json"
    normalizer.save(path)
    loaded = FeatureNormalizer.load(path, expected_dataset_hash="d" * 64, expected_split_hash="e" * 64)
    assert loaded.stats_hash == normalizer.stats_hash
    with pytest.raises(FileExistsError):
        normalizer.save(path)
    with pytest.raises(ValueError, match="dataset"):
        FeatureNormalizer.load(path, expected_dataset_hash="other")
    with pytest.raises(ValueError, match="config"):
        FeatureNormalizer.load(path, config=FeatureConfig(min_valid_ratio=0.9))
    data = json.loads(path.read_text())
    data["stats"]["mean"][0] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        FeatureNormalizer.load(path)


@pytest.mark.parametrize("split", ["val", "test", "all"])
def test_nontraining_fit_is_forbidden(split):
    with pytest.raises(ValueError, match="train"):
        FeatureNormalizer.fit(training_samples(), dataset_hash="d", split_hash="s", split=split)


def test_tensor_transform_requires_explicit_field_validity():
    normalizer = fit()
    with pytest.raises(ValueError, match="explicit"):
        normalizer.transform(np.zeros((32, 94)))
    transformed = normalizer.transform(np.zeros((32, 94)), np.zeros((32, 69), dtype=bool))
    assert np.isfinite(transformed).all() and not transformed.any()
