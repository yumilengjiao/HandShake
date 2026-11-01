"""Hand-computable score/calibration tests; no neural training or forward passes."""

import math

import numpy as np
import pytest

from algorithm.bundle import digest, json_bytes
from algorithm.evaluation.calibration import (
    Calibration,
    fit_calibration,
    load_calibration,
    save_calibration,
)
from algorithm.evaluation.classification import (
    evaluate_classification,
    log_probabilities,
)
from algorithm.evaluation.scores import (
    ScoreProvenance,
    Scores,
    load_scores,
    score_dtype,
    write_scores,
)
from shared import PROTOCOL_VERSION


def provenance(**changes):
    return ScoreProvenance(
        **(
            {
                name: "a" * 64
                for name in (
                    "model_sha256",
                    "dataset_sha256",
                    "split_sha256",
                    "stats_sha256",
                    "config_sha256",
                    "cache_index_sha256",
                    "endpoint_sha256",
                )
            }
            | dict(
                vocabulary_version=PROTOCOL_VERSION,
                split="val",
                data_synthetic=True,
                model_synthetic=True,
            )
            | changes
        )
    )


def scores_for(logits, labels, *, available=None, ready=None, **source):
    rows = np.zeros(len(labels), dtype=score_dtype(5))
    rows["label"] = labels
    rows["class_valid"] = np.asarray(labels) >= 0
    rows["ready"] = ready if ready is not None else 1
    rows["available"] = available if available is not None else 1
    rows["logits"] = np.asarray(logits).reshape(len(labels), 5)
    return Scores(rows, provenance(**source), "b" * 64)


def rejection_scores():
    return scores_for(
        [[math.log(36), 0, 0, 0, 0], [0, math.log(4), 0, 0, 0], [0] * 5],
        [0, 0, 0],
        available=[1, 1, 0],
    )


def archive(tmp_path, scores):
    root = tmp_path / "scores"
    receipt = write_scores(root, scores.provenance, scores.batches())
    return (
        root,
        receipt,
        load_scores(root, expected_manifest_sha256=receipt["manifest_sha256"]),
    )


def test_perfect_fixed_classes_matches_training_metric_without_rejection():
    from algorithm.training.metrics import classification_metrics

    scores = scores_for(np.eye(5) * 10, list(range(5)))
    result = evaluate_classification(scores)
    baseline = classification_metrics(np.arange(5), np.arange(5))
    assert result["macro_f1"] == baseline["macro_f1"] == 1
    assert result["balanced_accuracy"] == 1
    assert result["support"] == [1] * 5
    assert result["coverage"] == 1 and result["selective_risk"] == 0
    assert (
        np.asarray(result["confusion_matrix"])[:, :5].tolist()
        == baseline["confusion_matrix"]
    )


def test_unavailable_rejected_and_not_ready_keep_truth_denominator():
    scores = rejection_scores()
    result = evaluate_classification(scores, threshold=0.6)
    assert result["confusion_matrix"][0] == [1, 0, 0, 0, 0, 2]
    assert result["normalized_confusion_matrix"][0] == pytest.approx(
        [1 / 3, 0, 0, 0, 0, 2 / 3]
    )
    assert result["precision"][0] == 1 and result["recall"][0] == pytest.approx(1 / 3)
    assert result["f1"][0] == 0.5 and result["macro_f1"] == 0.1
    assert result["balanced_accuracy"] == pytest.approx(1 / 15)
    assert result["coverage"] == pytest.approx(1 / 3) and result["selective_risk"] == 0
    assert result["unavailable_labeled"] == 1 and result["abstained_labeled"] == 2
    scores.rows["ready"] = 0
    other = evaluate_classification(scores, threshold=0.6)
    assert other["ready_available_labeled"] == 0
    assert other["confusion_matrix"] == result["confusion_matrix"]


def test_uniform_probability_diagnostics_ignore_unlabeled_but_count_inventory():
    result = evaluate_classification(
        scores_for(np.zeros((6, 5)), [0, 1, 2, 3, 4, -1]), bins=5
    )
    assert result["nll_available"] == pytest.approx(math.log(5))
    assert result["brier_available"] == pytest.approx(0.8)
    assert result["ece_available"] == pytest.approx(0)
    assert result["labeled_endpoints"] == 5 and result["total_endpoints"] == 6
    assert sum(b["count"] for b in result["reliability_bins"]) == 5
    assert result["selective_risk"] == 0.8


def test_empty_and_all_wrong_and_all_missing():
    result = evaluate_classification(scores_for([], []))
    assert result["coverage"] is None and result["nll_available"] is None
    assert result["macro_f1"] == 0 and result["absent_classes"] == list(range(5))
    wrong = evaluate_classification(
        scores_for(np.eye(5)[[1, 2, 3, 4, 0]], list(range(5)))
    )
    assert wrong["macro_f1"] == 0 and wrong["selective_risk"] == 1
    missing = evaluate_classification(
        scores_for(np.zeros((5, 5)), list(range(5)), available=[0] * 5)
    )
    assert missing["coverage"] == 0 and missing["selective_risk"] is None
    assert missing["abstained_labeled"] == 5 and missing["nll_available"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0},
        {"temperature": float("nan")},
        {"threshold": 2},
        {"threshold": True},
        {"bins": 0},
        {"bins": True},
    ],
)
def test_invalid_metric_parameters(kwargs):
    with pytest.raises(ValueError):
        evaluate_classification(scores_for([], []), **kwargs)


def test_log_softmax_is_stable_and_accepts_empty():
    values = log_probabilities([[1e6, -1e6]], 0.05)
    assert np.isfinite(values).all() and np.exp(values).sum() == 1
    assert log_probabilities(np.empty((0, 5))).shape == (0, 5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("label", 8),
        ("class_valid", 2),
        ("ready", 2),
        ("available", 2),
        ("logits", np.nan),
    ],
)
def test_score_layout_rejects_invalid_values(field, value):
    scores = rejection_scores()
    scores.rows[field][0] = value
    with pytest.raises(ValueError):
        Scores(scores.rows, scores.provenance)


def test_even_empty_score_dtype_is_checked():
    with pytest.raises(ValueError, match="layout"):
        Scores(np.empty(0), provenance())


def test_archive_is_pinned_immutable_and_tamper_evident(tmp_path):
    root, receipt, scores = archive(tmp_path, rejection_scores())
    assert scores.manifest_sha256 == receipt["manifest_sha256"]
    assert evaluate_classification(scores)["labeled_endpoints"] == 3
    with pytest.raises(FileExistsError):
        write_scores(root, scores.provenance, scores.batches())
    with pytest.raises(ValueError, match="manifest hash"):
        load_scores(root, expected_manifest_sha256="0" * 64)
    with (root / "scores.bin").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="changed"):
        evaluate_classification(scores)
    with pytest.raises(ValueError, match="shape"):
        load_scores(root, expected_manifest_sha256=receipt["manifest_sha256"])


def test_partial_archive_cannot_load_and_empty_roundtrip(tmp_path):
    def broken():
        yield rejection_scores().rows
        raise ValueError("interrupted")

    with pytest.raises(ValueError, match="interrupted"):
        write_scores(tmp_path / "partial", provenance(), broken())
    with pytest.raises(ValueError, match="inventory"):
        load_scores(tmp_path / "partial", expected_manifest_sha256="0" * 64)
    _, _, empty = archive(tmp_path, scores_for([], []))
    assert len(empty.rows) == 0


def test_symlink_and_same_size_corruption_are_rejected(tmp_path):
    root, receipt, _ = archive(tmp_path, rejection_scores())
    path = root / "scores.bin"
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match="binary hash"):
        load_scores(root, expected_manifest_sha256=receipt["manifest_sha256"])
    outside = tmp_path / "outside"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="shape"):
        load_scores(root, expected_manifest_sha256=receipt["manifest_sha256"])


def test_validation_temperature_nll_choice_is_hand_computable():
    scores = scores_for([[0, math.log(4), 0, 0, 0]], [0])
    value = fit_calibration(
        scores,
        allow_synthetic=True,
        temperatures=(1.0, 2.0),
        thresholds=(0.0,),
        max_risk=1.0,
        min_coverage=1.0,
        min_accepted=1,
    )
    assert value.temperature == 2
    assert value.nlls == pytest.approx([math.log(8), math.log(6)])
    assert value.status == "draft" and value.actions_enabled is False


def test_temperature_tie_prefers_one_regardless_of_order():
    value = fit_calibration(
        scores_for([[0] * 5], [0]),
        allow_synthetic=True,
        temperatures=(2.0, 0.5, 1.0),
        thresholds=(0.0,),
        min_accepted=1,
    )
    assert value.temperature == 1


def test_rejection_selects_maximum_feasible_coverage_not_all_reject():
    value = fit_calibration(
        rejection_scores(),
        allow_synthetic=True,
        temperatures=(1.0,),
        thresholds=(1.0, 0.8, 0.6, 0.0),
        max_risk=0.0,
        min_coverage=0.3,
        min_accepted=1,
    )
    assert value.threshold == 0.6 and value.accepted_count == 1
    assert value.fit_count == 2 and value.labeled_count == 3
    impossible = fit_calibration(
        rejection_scores(),
        allow_synthetic=True,
        temperatures=(1.0,),
        thresholds=(0.0, 0.6, 1.0),
        max_risk=0.0,
        min_coverage=0.9,
        min_accepted=1,
    )
    assert impossible.rejection_status == "infeasible" and impossible.threshold is None
    assert impossible.accepted_count == 0


@pytest.mark.parametrize(
    "change", [{"split": "test"}, {"data_synthetic": False, "model_synthetic": False}]
)
def test_test_selection_and_unverified_archives_are_rejected(change):
    scores = scores_for([[0] * 5], [0], **change)
    scores.manifest_sha256 = None
    with pytest.raises(ValueError):
        fit_calibration(scores, allow_synthetic=True, min_accepted=1)


def test_synthetic_calibration_requires_opt_in_and_empty_is_rejected():
    with pytest.raises(ValueError, match="opt-in"):
        fit_calibration(rejection_scores())
    with pytest.raises(ValueError, match="no available"):
        fit_calibration(scores_for([], []), allow_synthetic=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperatures": (2.0,)},
        {"temperatures": (1.0, 1.0)},
        {"thresholds": (float("nan"),)},
        {"min_coverage": 0},
        {"min_accepted": True},
        {"max_risk": -1},
        {"thresholds": ()},
    ],
)
def test_invalid_search_budget(kwargs):
    with pytest.raises(ValueError):
        fit_calibration(rejection_scores(), allow_synthetic=True, **kwargs)


def test_calibration_roundtrip_binding_tamper_and_selection_validation(tmp_path):
    _, _, scores = archive(tmp_path, rejection_scores())
    value = fit_calibration(
        scores,
        allow_synthetic=True,
        temperatures=(1.0, 2.0),
        max_risk=1.0,
        min_coverage=0.5,
        min_accepted=1,
    )
    path = tmp_path / "calibration.json"
    sha = save_calibration(value, path)
    assert (
        load_calibration(path, expected_sha256=sha, target=scores.provenance) == value
    )
    test_source = provenance(
        split="test", cache_index_sha256="c" * 64, endpoint_sha256="d" * 64
    )
    assert load_calibration(path, expected_sha256=sha, target=test_source) == value
    with pytest.raises(ValueError, match="binding"):
        load_calibration(
            path, expected_sha256=sha, target=provenance(model_sha256="c" * 64)
        )
    with pytest.raises(ValueError, match="inventory"):
        load_calibration(
            path, expected_sha256=sha, target=provenance(endpoint_sha256="c" * 64)
        )
    with pytest.raises(ValueError, match="hash"):
        load_calibration(path, expected_sha256="0" * 64, target=scores.provenance)
    bad = value.model_dump(mode="json")
    bad["temperature"] = 2.0 if value.temperature == 1.0 else 1.0
    with pytest.raises(ValueError, match="NLL selection"):
        Calibration.model_validate_json(json_bytes(bad))
    duplicate = path.read_bytes().replace(
        b'"status": "draft"', b'"status": "draft", "status": "draft"'
    )
    path.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        load_calibration(
            path, expected_sha256=digest(duplicate), target=scores.provenance
        )
