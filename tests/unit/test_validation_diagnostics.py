"""Hand-computable offline diagnosis; no neural training or inference."""

import json

import numpy as np
import pytest

from algorithm.evaluation.calibration import fit_calibration
from algorithm.evaluation.cli import main
from algorithm.evaluation.diagnostics import diagnose_validation
from tests.unit.test_evaluation_scores import archive, rejection_scores, scores_for


def test_primary_denominator_keeps_unavailable_and_readiness_failures():
    scores = rejection_scores()
    scores.rows["ready"] = [1, 0, 0]
    before = scores.rows.copy()
    result = diagnose_validation(scores, allow_synthetic=True)
    assert result["primary"]["labeled_endpoints"] == 3
    assert result["primary"]["accuracy_all_labeled"] == 1 / 3
    assert result["primary"]["accepted_accuracy"] == 0.5
    subsets = result["readiness_diagnostics"]
    assert subsets["ready"]["accuracy_all_labeled"] == 1
    assert subsets["not_ready"]["accuracy_all_labeled"] == 0
    assert sum(subsets[k]["labeled_endpoints"] for k in ("ready", "not_ready")) == 3
    assert result["actions_enabled"] is False
    np.testing.assert_array_equal(before, scores.rows)


def test_rejection_does_not_masquerade_as_improved_full_accuracy():
    scores = rejection_scores()
    calibration = fit_calibration(
        scores,
        allow_synthetic=True,
        temperatures=(1.0,),
        thresholds=(0.0, 0.6),
        min_accepted=1,
        min_coverage=0.3,
        max_risk=0.0,
    )
    result = diagnose_validation(scores, calibration, allow_synthetic=True)
    selected = result["calibrated_rejection"]
    assert selected["accepted_accuracy"] == 1.0
    assert selected["accuracy_all_labeled"] == 1 / 3
    assert selected["coverage"] == 1 / 3
    assert selected["per_class"][0]["recall"] == 1 / 3
    assert result["policy_confidence_only"]["threshold"] == 0.8
    assert "NOT full" in result["policy_confidence_only"]["scope"]


def test_all_zero_support_classes_still_count_in_macro_f1():
    result = diagnose_validation(
        scores_for([[10, 0, 0, 0, 0]], [0]), allow_synthetic=True
    )
    assert result["primary"]["macro_f1"] == 0.2
    assert result["readiness_diagnostics"]["not_ready"]["accuracy_all_labeled"] is None
    assert result["largest_three_classes"]["support_fraction"] == 1
    assert all(not row["top_errors"] for row in result["per_class_confusions"])


def test_counts_and_percentages_of_top_confusions_are_consistent():
    scores = scores_for([[0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [9, 0, 0, 0, 0]], [0, 0, 1])
    result = diagnose_validation(scores, allow_synthetic=True)
    errors = result["per_class_confusions"][0]["top_errors"]
    assert len(errors) == 1 and errors[0]["count"] == 2
    assert errors[0]["fraction_of_true"] == 1


def test_final_test_and_unverified_or_unlabeled_archives_rejected():
    with pytest.raises(ValueError, match="validation-only"):
        diagnose_validation(
            scores_for([[0] * 5], [0], split="test"), allow_synthetic=True
        )
    scores = rejection_scores()
    with pytest.raises(ValueError, match="synthetic"):
        diagnose_validation(scores)
    scores.manifest_sha256 = None
    with pytest.raises(ValueError, match="verified"):
        diagnose_validation(scores, allow_synthetic=True)
    with pytest.raises(ValueError, match="labeled"):
        diagnose_validation(scores_for([], []), allow_synthetic=True)


def test_calibration_must_belong_to_exact_score_inventory():
    scores = rejection_scores()
    value = fit_calibration(
        scores,
        allow_synthetic=True,
        temperatures=(1.0,),
        thresholds=(0.0,),
        min_accepted=1,
        max_risk=1.0,
        min_coverage=0.3,
    )
    scores.manifest_sha256 = "c" * 64
    with pytest.raises(ValueError, match="exact validation"):
        diagnose_validation(scores, value, allow_synthetic=True)


def test_infeasible_calibration_never_reports_selective_success():
    scores = rejection_scores()
    value = fit_calibration(
        scores,
        allow_synthetic=True,
        temperatures=(1.0,),
        thresholds=(0.0,),
        min_accepted=1,
        max_risk=0.0,
        min_coverage=1.0,
    )
    result = diagnose_validation(scores, value, allow_synthetic=True)
    assert result["calibrated_rejection"] is None
    assert result["policy_confidence_only"] is None
    assert "infeasible" in result["limitations"][-1]


def test_cli_writes_immutable_json_and_rejects_final_test(tmp_path):
    root, receipt, _ = archive(tmp_path, rejection_scores())
    output = tmp_path / "diagnosis.json"
    args = [
        "diagnose",
        "--scores",
        str(root),
        "--sha256",
        receipt["manifest_sha256"],
        "--output",
        str(output),
        "--allow-synthetic",
    ]
    assert main(args) == 0
    report = json.loads(output.read_text())
    assert report["primary"]["labeled_endpoints"] == 3
    assert len(report["implementation_sha256"]) == 64
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    other = tmp_path / "test-side"
    other.mkdir()
    root, receipt, _ = archive(other, scores_for([[0] * 5], [0], split="test"))
    with pytest.raises(SystemExit):
        main(
            [
                "diagnose",
                "--scores",
                str(root),
                "--sha256",
                receipt["manifest_sha256"],
                "--output",
                str(other / "forbidden.json"),
                "--allow-synthetic",
            ]
        )
    assert not (other / "forbidden.json").exists()
