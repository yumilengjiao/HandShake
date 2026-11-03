"""Train-only class balancing and optimizer policy validation; no neural training."""

import math
import subprocess
import sys

import pytest

from algorithm.training.balancing import class_weights_from_counts
from algorithm.training.config import TrainConfig


def test_sqrt_weights_have_dataset_mean_one_and_moderate_ratio():
    counts = [400, 100, 0, 25]
    weights = class_weights_from_counts(counts, "sqrt_inverse")
    assert sum(n * w for n, w in zip(counts, weights)) / sum(counts) == pytest.approx(1)
    assert weights[3] / weights[0] == pytest.approx(4)
    assert weights[2] == 0
    assert all(math.isfinite(w) for w in weights)
    assert class_weights_from_counts(counts) is None
    assert class_weights_from_counts([5, 5, 5], "sqrt_inverse") == pytest.approx(
        [1, 1, 1]
    )
    assert class_weights_from_counts([0, 17], "sqrt_inverse") == [0, 1]
    assert class_weights_from_counts(
        [n * 7 for n in counts], "sqrt_inverse"
    ) == pytest.approx(weights)


@pytest.mark.parametrize(
    "counts", [[], [0, 0], [-1, 3], [True, 2], [1.5, 2], [float("nan")], [float("inf")]]
)
def test_invalid_counts_rejected_even_without_weighting(counts):
    with pytest.raises(ValueError):
        class_weights_from_counts(counts)


def test_unknown_weighting_rejected():
    with pytest.raises(ValueError):
        class_weights_from_counts([1, 2], "balanced")


def test_dry_run_balancing_does_not_import_torch():
    script = """
import sys
from algorithm.training.balancing import class_weights_from_counts
from algorithm.training.config import TrainConfig
assert class_weights_from_counts([40, 1], 'sqrt_inverse')[1] > 1
assert TrainConfig().class_weighting == 'none'
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "changes",
    [
        {"class_weighting": "balanced"},
        {"scheduler": "cosine"},
        {"lr_factor": 0},
        {"lr_factor": 1},
        {"lr_factor": float("nan")},
        {"lr_patience": -1},
        {"lr_patience": True},
        {"min_learning_rate": 0},
        {"min_learning_rate": float("inf")},
        {"scheduler": "plateau", "min_learning_rate": 0.01},
        {"scheduler": "plateau", "lr_patience": 2, "patience": 3},
    ],
)
def test_invalid_optimizer_policies(changes):
    with pytest.raises(ValueError):
        TrainConfig(**changes)


def test_defaults_preserve_original_training_recipe():
    config = TrainConfig()
    assert config.class_weighting == "none" and config.scheduler == "constant"
    assert config.learning_rate == 0.001 and config.sampling == "uniform"
    config = TrainConfig(scheduler="plateau", lr_patience=0, patience=2)
    assert config.lr_factor == 0.5
