"""Endpoint sampling contracts; these checks never perform neural training."""

import subprocess
import sys

import numpy as np
import pytest

from algorithm.training.config import TrainConfig
from algorithm.training.sampling import training_population
from backend.workflows.workspace import TrainingSettings


def test_uniform_preserves_original_endpoint_ids_including_masked_rows():
    mask = np.array([False, True, True, False, True])
    np.testing.assert_array_equal(training_population(mask), np.arange(5))
    assert TrainConfig().sampling == TrainingSettings().sampling == "uniform"


def test_supervised_population_visits_every_label_once_without_mutating_mask():
    mask = np.array([False, True, True, False, True])
    original = mask.copy()
    indexes = training_population(mask, "supervised_uniform")
    np.testing.assert_array_equal(indexes, [1, 2, 4])
    np.testing.assert_array_equal(mask, original)
    assert indexes.dtype == np.dtype("<i8") and not indexes.flags.writeable
    assert len(np.unique(indexes)) == int(mask.sum())
    with pytest.raises(ValueError):
        indexes[0] = 99


@pytest.mark.parametrize("sampling", ["uniform", "supervised_uniform"])
def test_all_valid_population_is_identical_for_both_policies(sampling):
    np.testing.assert_array_equal(
        training_population(np.ones(7, dtype=bool), sampling), np.arange(7)
    )


@pytest.mark.parametrize(
    "mask", [[], [False, False], [0, 1], [True, 1], [[True]], [float("nan")]]
)
def test_bad_or_empty_supervision_is_rejected(mask):
    with pytest.raises(ValueError):
        training_population(mask)


def test_unknown_policy_is_rejected_by_helper_and_both_configuration_layers():
    with pytest.raises(ValueError, match="sampling"):
        training_population([True], "balanced")
    for config in (TrainConfig, TrainingSettings):
        with pytest.raises(ValueError):
            config(sampling="balanced")


def test_population_checks_do_not_import_torch():
    script = """
import sys
from algorithm.training.sampling import training_population
from algorithm.training.config import TrainConfig
assert training_population([False, True], 'supervised_uniform').tolist() == [1]
assert TrainConfig(sampling='supervised_uniform').sampling == 'supervised_uniform'
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
