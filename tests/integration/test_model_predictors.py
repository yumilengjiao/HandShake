"""Same replay window, class order, quality and clock gates for all models."""

import numpy as np
import pytest

from algorithm.inference import PredictionGate, RulePredictor, TorchPredictor
from shared.classes import load_classes
from shared.contracts import FeatureWindow
from shared.features.schema import FeatureConfig, feature_schema_hash
from shared.features.normalization import FeatureNormalizer, FeatureStats

torch = pytest.importorskip("torch")

from algorithm.models.mlp import GestureMLP
from algorithm.models.tcn import GestureTCN


def raw_window():
    points = np.zeros((21, 3))
    for finger, mcp in enumerate((5, 9, 13, 17)):
        x = (finger - 1.5) * 0.3
        for offset, height in enumerate((1.0, 1.5, 2.0, 2.5)):
            points[mcp + offset] = (x, height, 0.0)
    values = np.zeros((32, 94))
    values[:, :63] = points.ravel()
    values[:, 63:65] = 0.4
    values[:, 67] = 0.1
    values[:, 69] = 1 / 30
    values[:, 71:] = 1.0
    return FeatureWindow(
        stream_id="recording-7", track_id="hand-3", context_epoch=9,
        source_frame_id=1031, timestamps_ns=[1_000_000_000 + i * 33_333_333 for i in range(32)],
        features=values.tolist(), ready=True, ready_reason="ready", timebase="recording_relative",
    )


def changed(window, **updates):
    return FeatureWindow.model_validate(window.model_dump() | updates)


def make_predictor(kind, window, *, clock_ns=None, **kwargs):
    clock_ns = clock_ns or (lambda: window.timestamps_ns[-1] + 1_000_000)
    common = {"clock_ns": clock_ns, "clock_timebase": "recording_relative"} | kwargs
    if kind == "rule":
        return RulePredictor(**common)
    model = GestureMLP() if kind == "mlp" else GestureTCN()
    return TorchPredictor(model, model_version=f"{kind}-untrained-synthetic-test", **common)


@pytest.mark.parametrize("kind", ["rule", "mlp", "tcn"])
def test_same_window_preserves_provenance_and_only_raw_outputs(kind):
    window = raw_window()
    prediction = make_predictor(kind, window).predict(window)
    assert prediction.stream_id == window.stream_id
    assert prediction.track_id == window.track_id
    assert prediction.context_epoch == window.context_epoch
    assert prediction.source_frame_id == window.source_frame_id
    assert prediction.source_timestamp_ns == window.timestamps_ns[-1]
    assert prediction.timebase == window.timebase
    assert len(prediction.logits) == load_classes().class_count
    assert np.isfinite(prediction.logits).all()
    assert prediction.calibrated_probabilities is None
    assert prediction.intent_score is None
    assert prediction.phase_scores is None
    if kind == "rule":
        assert np.argmax(prediction.logits) == load_classes().id_for("wake_open")


@pytest.mark.parametrize("kind", ["mlp", "tcn"])
def test_adapter_selects_exact_model_input_and_preserves_raw_logits(kind):
    window = raw_window()
    predictor = make_predictor(kind, window)
    values = torch.tensor(window.features, dtype=torch.float32)[None]
    with torch.no_grad():
        expected = predictor.model(values[:, -1, :] if kind == "mlp" else values)
    actual = predictor.predict(window)
    assert actual.logits == expected[0].tolist()
    assert predictor.predict(window).logits == actual.logits


@pytest.mark.parametrize("kind", ["rule", "mlp", "tcn"])
@pytest.mark.parametrize("reason", ["warming", "short", "quality", "mask", "missing", "stale", "future", "timebase", "span"])
def test_all_models_share_fail_closed_input_gates(kind, reason):
    window = raw_window()
    predictor = make_predictor(kind, window)
    if reason == "warming":
        window = changed(window, ready=False, ready_reason="warming_up")
    elif reason == "short":
        window = changed(window, features=window.features[1:], timestamps_ns=window.timestamps_ns[1:])
    elif reason in {"quality", "mask", "missing"}:
        values = np.asarray(window.features)
        index, value = {"quality": (71, 0.2), "mask": (80, 0.0), "missing": (70, 0.01)}[reason]
        values[-1, index] = value
        window = changed(window, features=values.tolist())
    elif reason == "stale":
        predictor = make_predictor(kind, window, clock_ns=lambda: window.timestamps_ns[-1] + 150_000_001)
    elif reason == "future":
        predictor = make_predictor(kind, window, clock_ns=lambda: window.timestamps_ns[-1] - 1)
    elif reason == "timebase":
        window = changed(window, timebase="synthetic")
    else:
        window = changed(window, timestamps_ns=[i * 200_000_000 for i in range(32)])
    with pytest.raises(ValueError):
        predictor.predict(window)


@pytest.mark.parametrize("kind", ["rule", "mlp", "tcn"])
@pytest.mark.parametrize("clock_values", [(1_000_000, 150_000_001), (20_000_000, 10_000_000)])
def test_inference_cannot_return_expired_or_clock_regressed_output(kind, clock_values):
    window = raw_window()
    times = iter(window.timestamps_ns[-1] + value for value in clock_values)
    predictor = make_predictor(kind, window, clock_ns=lambda: next(times))
    with pytest.raises(ValueError):
        predictor.predict(window)


def test_recording_clock_requires_explicit_same_domain_clock():
    with pytest.raises(ValueError, match="explicit"):
        RulePredictor(clock_timebase="recording_relative")
    with pytest.raises(ValueError, match="clock_timebase_mismatch"):
        RulePredictor().predict(raw_window())
    with pytest.raises(ValueError, match="clock_ns"):
        make_predictor("rule", raw_window(), clock_ns=lambda: 1.0).predict(raw_window())


def test_gate_can_follow_shared_feature_configuration():
    feature_config = FeatureConfig(max_age_ns=80_000_000, min_geometry_quality=0.7)
    gate = PredictionGate.from_feature_config(feature_config)
    assert gate.max_prediction_age_ns == feature_config.max_age_ns
    assert gate.min_quality == feature_config.min_geometry_quality
    assert gate.max_window_span_ns == feature_config.max_window_span_ns


def test_rule_inverts_geometric_standardization_on_the_same_model_window():
    window = raw_window()
    config = FeatureConfig()
    stats = FeatureStats(
        mean=np.linspace(0.1, 0.3, 69), std=np.linspace(1.0, 2.0, 69), count=np.full(69, 64),
        dataset_hash="synthetic-train-only", split_hash="synthetic-subject-split", schema_hash=feature_schema_hash(),
        config_hash=config.config_hash,
    )
    normalizer = FeatureNormalizer(stats, config)
    standardized = normalizer.transform(np.asarray(window.features), np.ones((32, 69), dtype=bool))
    same_model_window = changed(window, features=standardized.tolist())
    raw_result = make_predictor("rule", window).predict(window)
    transformed_result = make_predictor("rule", same_model_window, inverse_features=normalizer.inverse_geometric).predict(same_model_window)
    assert transformed_result.logits == raw_result.logits
    # Learned adapters consume the identical standardized FeatureWindow.
    for kind in ("mlp", "tcn"):
        assert len(make_predictor(kind, same_model_window).predict(same_model_window).logits) == 5


def test_inverse_transform_cannot_overwrite_quality_or_provenance_fields():
    window = raw_window()

    def corrupt(values):
        values[:, 71] = 0.0
        return values

    with pytest.raises(ValueError, match="preserve"):
        make_predictor("rule", window, inverse_features=corrupt).predict(window)


def test_learned_adapter_rejects_wrong_vocabulary_and_nonfinite_outputs():
    window = raw_window()
    bad_model = GestureMLP()
    bad_model.class_tokens = tuple(reversed(bad_model.class_tokens))
    with pytest.raises(ValueError, match="class_tokens"):
        TorchPredictor(bad_model, model_version="invalid")
    good_model = GestureMLP()
    with torch.no_grad():
        next(good_model.parameters()).fill_(float("nan"))
    predictor = TorchPredictor(good_model, model_version="invalid-output", clock_ns=lambda: window.timestamps_ns[-1], clock_timebase=window.timebase)
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(window)
