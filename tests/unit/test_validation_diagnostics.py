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


def history_cache(monkeypatch, scores, lengths, *, changes=None, verify=lambda: None):
    """Hand-computable already-verified loader fixture, no tensor allocation."""
    from algorithm.evaluation.scores import endpoint_digest

    index = {
        "format_version": "t11-cache-v3",
        "index_sha256": scores.provenance.cache_index_sha256,
        "synthetic": scores.provenance.data_synthetic,
        "files": {
            "endpoints.bin": {"sha256": "d" * 64},
            "groups.json": {"sha256": "e" * 64},
        },
    }
    scores.provenance = scores.provenance.model_copy(
        update={"endpoint_sha256": endpoint_digest(index)}
    )
    index.update(changes or {})
    arrays = {
        "y": scores.rows["label"].copy(),
        "class_valid": scores.rows["class_valid"].copy(),
        "ready": scores.rows["ready"].copy(),
        "valid_length": np.asarray(lengths),
        "_verify": verify,
    }
    monkeypatch.setattr(
        "algorithm.data.cache.load_cache", lambda *a, **k: (arrays, index)
    )
    return arrays


def test_history_separates_padding_from_full_history_quality_failures(monkeypatch):
    scores = scores_for(
        [[9, 0, 0, 0, 0], [0, 9, 0, 0, 0], [0] * 5, [0, 9, 0, 0, 0]],
        [0, 0, 0, -1],
        ready=[0, 0, 1, 0],
        available=[1, 1, 0, 1],
    )
    history_cache(monkeypatch, scores, [1, 32, 32, 3])
    before = scores.rows.copy()
    baseline = diagnose_validation(scores, allow_synthetic=True)
    result = diagnose_validation(scores, allow_synthetic=True, cache="verified-val")
    assert result["primary"] == baseline["primary"]
    parts = result["history_diagnostics"]
    keys = ("short_history", "full_not_ready", "full_ready")
    assert sum(parts[k]["labeled_endpoints"] for k in keys) == 3
    assert [parts[k]["accuracy_all_labeled"] for k in keys] == [1, 0, 0]
    assert parts["full_ready"]["unavailable_labeled"] == 1
    assert parts["short_history"]["macro_f1"] == 0.2
    np.testing.assert_array_equal(scores.rows, before)


@pytest.mark.parametrize("field", ["index_sha256", "synthetic", "files"])
def test_history_rejects_different_cache_inventory(monkeypatch, field):
    scores = rejection_scores()
    changed = {
        "index_sha256": "f" * 64,
        "synthetic": False,
        "files": {
            "endpoints.bin": {"sha256": "f" * 64},
            "groups.json": {"sha256": "e" * 64},
        },
    }
    history_cache(monkeypatch, scores, [32] * 3, changes={field: changed[field]})
    with pytest.raises(ValueError, match="exact score inventory"):
        diagnose_validation(scores, allow_synthetic=True, cache="wrong")


@pytest.mark.parametrize("field", ["y", "class_valid", "ready"])
def test_history_rejects_row_alignment_mismatch(monkeypatch, field):
    scores = rejection_scores()
    arrays = history_cache(monkeypatch, scores, [32] * 3)
    arrays[field][0] = 1 - arrays[field][0]
    with pytest.raises(ValueError, match="alignment mismatch"):
        diagnose_validation(scores, allow_synthetic=True, cache="wrong")


def test_history_rejects_wrong_count_and_legacy_cache(monkeypatch):
    scores = rejection_scores()
    arrays = history_cache(monkeypatch, scores, [32] * 3)
    arrays["y"] = arrays["y"][:-1]
    with pytest.raises(ValueError, match="count mismatch"):
        diagnose_validation(scores, allow_synthetic=True, cache="wrong")
    history_cache(
        monkeypatch, scores, [32] * 3, changes={"format_version": "t11-cache-v2"}
    )
    with pytest.raises(ValueError, match="compact v3"):
        diagnose_validation(scores, allow_synthetic=True, cache="legacy")


def test_history_preserves_cache_guard_during_subset_passes(monkeypatch):
    scores = rejection_scores()
    calls = []

    def verify():
        calls.append(1)
        # Initial alignment succeeds; a later subset pass must still check.
        if len(calls) == 5:
            raise ValueError("immutable cache changed during use")

    history_cache(monkeypatch, scores, [32] * 3, verify=verify)
    with pytest.raises(ValueError, match="immutable cache changed"):
        diagnose_validation(scores, allow_synthetic=True, cache="changed")


def test_history_preserves_score_guard_during_subset_passes(monkeypatch):
    scores = rejection_scores()
    state = {"cache_checks": 0, "changed": False}

    def verify_cache():
        state["cache_checks"] += 1
        if state["cache_checks"] == 5:
            state["changed"] = True

    def verify_scores():
        if state["changed"]:
            raise ValueError("immutable scores changed during evaluation")

    scores.guard = verify_scores
    history_cache(monkeypatch, scores, [32] * 3, verify=verify_cache)
    with pytest.raises(ValueError, match="immutable scores changed"):
        diagnose_validation(scores, allow_synthetic=True, cache="scores-changed")
