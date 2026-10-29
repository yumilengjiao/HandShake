"""Synthetic geometry verifies rule semantics, not real-world recognition rates."""

from dataclasses import asdict

import numpy as np
import pytest

from algorithm.models.rule_baseline import RuleBaseline, RuleThresholds
from shared.classes import load_classes
from shared.contracts import Gesture


def raw_history(pose="open", displacement=0.0, *, interval_ns=33_333_333):
    points = np.zeros((21, 3))
    for finger, mcp in enumerate((5, 9, 13, 17)):
        x = (finger - 1.5) * 0.3
        points[mcp] = (x, 1.0, 0.0)
        points[mcp + 1] = (x, 1.5, 0.0)
        folded = pose == "fist" or (pose == "neutral" and finger == 0)
        points[mcp + 2] = (x + 0.2, 1.25, 0.0) if folded else (x, 2.0, 0.0)
        points[mcp + 3] = (x + 0.1, 1.1, 0.0) if folded else (x, 2.5, 0.0)
    values = np.zeros((32, 94))
    values[:, :63] = points.ravel()
    values[:, 63] = np.linspace(0.3, 0.3 + displacement, 32)
    values[:, 64] = 0.4
    values[:, 65] = displacement / (31 * interval_ns / 1e9)
    values[:, 67] = 0.1
    values[:, 69] = interval_ns / 1e9
    values[:, 71:] = 1.0
    timestamps = 1_000_000_000 + np.arange(32, dtype=np.int64) * interval_ns
    return values, timestamps


@pytest.mark.parametrize("pose,dx,expected", [
    ("open", 0.0, Gesture.WAKE_OPEN),
    ("fist", 0.0, Gesture.EXIT_POSE),
    ("neutral", 0.0, Gesture.NO_ACTION),
    ("open", -0.4, Gesture.SWIPE_LEFT),
    ("open", 0.4, Gesture.SWIPE_RIGHT),
])
def test_each_rule_class_uses_shared_vocabulary(pose, dx, expected):
    values, timestamps = raw_history(pose, dx)
    rule = RuleBaseline()
    assert rule.classify(values, timestamps) == expected
    scores = rule.logits(values, timestamps)
    assert scores == [float(index == load_classes().id_for(expected)) for index in range(5)]


def test_swipe_depends_on_history_even_when_final_frame_is_identical():
    moving, timestamps = raw_history(displacement=0.4)
    stationary = moving.copy()
    stationary[:, 63] = moving[-1, 63]
    # Keep the very same final row, including its velocity. A one-frame rule
    # would therefore produce the same result for these two windows.
    assert np.array_equal(moving[-1], stationary[-1])
    assert RuleBaseline().classify(moving, timestamps) == Gesture.SWIPE_RIGHT
    assert RuleBaseline().classify(stationary, timestamps) == Gesture.NO_ACTION


@pytest.mark.parametrize("case", ["small_displacement", "vertical_drift", "too_fast", "too_slow", "interrupted", "bad_derivative", "inconsistent_direction", "reversed_endpoint"])
def test_swipe_rejects_insufficient_or_discontinuous_evidence(case):
    interval = 1_000_000 if case == "too_fast" else (100_000_000 if case == "too_slow" else 33_333_333)
    values, timestamps = raw_history(displacement=0.4 if case != "small_displacement" else 0.1, interval_ns=interval)
    if case == "vertical_drift":
        values[:, 64] = np.linspace(0.2, 1.0, 32)
    elif case == "too_slow":
        values[:, 63] = np.linspace(0.3, 0.5, 32)
        values[:, 65] = 0.2 / 3.1
    elif case == "interrupted":
        values[-10, 71] = 0.0
    elif case == "bad_derivative":
        values[-10, 72] = 0.0
    elif case == "inconsistent_direction":
        values[:, 63] = np.concatenate((np.linspace(0.0, 0.3, 11), np.linspace(0.3, 0.05, 12)[1:], np.linspace(0.05, 0.2, 11)[1:]))
    elif case == "reversed_endpoint":
        values[-1, 65] *= -1
    assert RuleBaseline().classify(values, timestamps) == Gesture.NO_ACTION


@pytest.mark.parametrize("index,value", [(71, 0.2), (70, 0.1), (73, 0.0), (72, 0.0)])
def test_current_invalid_or_motion_unknown_cannot_be_static_candidate(index, value):
    values, timestamps = raw_history()
    values[-1, index] = value
    assert RuleBaseline().classify(values, timestamps) == Gesture.NO_ACTION


def test_static_angle_and_distance_constraints_and_degenerate_bones():
    values, timestamps = raw_history()
    # Straight PIP but tip moved close to wrist fails the distance condition.
    values[-1, 8 * 3:8 * 3 + 3] = 0.0
    assert RuleBaseline().classify(values, timestamps) == Gesture.NO_ACTION
    values[-1, 6 * 3:6 * 3 + 3] = values[-1, 5 * 3:5 * 3 + 3]
    assert RuleBaseline().classify(values, timestamps) == Gesture.NO_ACTION


@pytest.mark.parametrize("mutation", ["short", "nan", "duplicate_time", "fractional_time"])
def test_malformed_rule_input_is_rejected(mutation):
    values, timestamps = raw_history()
    if mutation == "short":
        values, timestamps = values[-1:], timestamps[-1:]
    elif mutation == "nan":
        values[0, 0] = np.nan
    elif mutation == "duplicate_time":
        timestamps[-1] = timestamps[-2]
    else:
        timestamps = timestamps.astype(float)
    with pytest.raises(ValueError):
        RuleBaseline().classify(values, timestamps)


@pytest.mark.parametrize("override", [{"history_size": 1}, {"min_quality": 2}, {"swipe_min_duration_ns": 1.0}, {"swipe_min_direction_consistency": float("nan")}, {"folded_pip_degrees_max": 160.0}])
def test_rule_thresholds_are_validated_and_public(override):
    assert asdict(RuleThresholds())["history_size"] == 32
    with pytest.raises(ValueError):
        RuleThresholds(**override)


def test_unsigned_time_cannot_wrap_or_exceed_the_int64_contract():
    values, timestamps = raw_history()
    unsigned = timestamps.astype(np.uint64)
    unsigned[-1] = 0
    with pytest.raises(ValueError, match="increasing"):
        RuleBaseline().classify(values, unsigned)
    unsigned[-1] = 2**63
    with pytest.raises(ValueError, match="int64"):
        RuleBaseline().classify(values, unsigned)
