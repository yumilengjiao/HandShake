"""Validation-only error diagnosis, not tuning, training or an improved score claim.

Primary metrics always retain the full labeled endpoint denominator. Readiness
subsets and confidence-only policy analysis are explicitly secondary diagnostics.
"""

from algorithm.evaluation.calibration import Calibration
from algorithm.evaluation.classification import evaluate_classification
from shared.classes import vocabulary_for
from shared.public_interaction import PublicGesturePolicy


class _ReadinessView:
    """Filter small batches, not a whole archive; preserve its read/tamper guard."""

    def __init__(self, scores, ready):
        self.scores, self.ready = scores, ready
        self.classes = scores.classes

    def batches(self):
        for batch in self.scores.batches():
            yield batch[batch["ready"] == self.ready]


def _summary(result, tokens):
    matrix = result["confusion_matrix"]
    total = result["labeled_endpoints"]
    correct = sum(matrix[i][i] for i in range(len(tokens)))
    accepted = result["accepted_labeled"]
    return {
        "labeled_endpoints": total,
        "accuracy_all_labeled": correct / total if total else None,
        "accepted_accuracy": correct / accepted if accepted else None,
        "macro_f1": result["macro_f1"],
        "balanced_accuracy": result["balanced_accuracy"],
        "coverage": result["coverage"],
        "accepted_errors": result["accepted_errors"],
        "abstained_labeled": result["abstained_labeled"],
        "unavailable_labeled": result["unavailable_labeled"],
        "temperature": result["temperature"],
        "threshold": result["threshold"],
        "per_class": [
            {
                "token": token,
                "support": result["support"][i],
                "precision": result["precision"][i],
                "recall": result["recall"][i],
                "f1": result["f1"][i],
                "abstained": matrix[i][-1],
            }
            for i, token in enumerate(tokens)
        ],
    }


class _HistoryView:
    """Aligned endpoint subsets, with both immutable-input guards preserved."""

    def __init__(self, scores, lengths, ready, verify, subset):
        self.scores, self.lengths, self.ready = scores, lengths, ready
        self.verify, self.subset, self.classes = verify, subset, scores.classes

    def batches(self):
        cursor = 0
        self.verify()
        for batch in self.scores.batches():
            self.verify()
            stop = cursor + len(batch)
            lengths, ready = self.lengths[cursor:stop], self.ready[cursor:stop]
            if self.subset == "short_history":
                mask = lengths < 32
            elif self.subset == "full_not_ready":
                mask = (lengths == 32) & ~ready.astype(bool)
            else:
                mask = (lengths == 32) & ready.astype(bool)
            yield batch[mask]
            cursor = stop
        self.verify()


def _diagnose_history(scores, cache):
    """Join verified compact validation metadata, never infer length from ready."""
    import numpy as np

    from algorithm.data.cache import load_cache
    from algorithm.evaluation.scores import endpoint_digest

    source = scores.provenance
    arrays, index = load_cache(
        cache,
        expected_dataset_sha256=source.dataset_sha256,
        expected_split_sha256=source.split_sha256,
        expected_stats_sha256=source.stats_sha256,
        expected_config_sha256=source.config_sha256,
        expected_split="val",
        expected_vocabulary=vocabulary_for(source.vocabulary_version),
    )
    if index["format_version"] != "t11-cache-v3":
        raise ValueError("history diagnosis requires a compact v3 validation cache")
    if (
        index["index_sha256"] != source.cache_index_sha256
        or endpoint_digest(index) != source.endpoint_sha256
        or index["synthetic"] is not source.data_synthetic
        or source.data_synthetic is not source.model_synthetic
    ):
        raise ValueError("history cache must match the exact score inventory")
    verify = arrays["_verify"]
    verify()
    if len(arrays["y"]) != len(scores.rows):
        raise ValueError("history cache/score endpoint count mismatch")
    cursor = 0
    for batch in scores.batches():
        verify()
        stop = cursor + len(batch)
        for score_field, cache_field in (
            ("label", "y"),
            ("class_valid", "class_valid"),
            ("ready", "ready"),
        ):
            if not np.array_equal(batch[score_field], arrays[cache_field][cursor:stop]):
                raise ValueError("history cache/score endpoint alignment mismatch")
        cursor = stop
    verify()
    tokens = vocabulary_for(source.vocabulary_version).tokens
    report = {
        "cache_index_sha256": index["index_sha256"],
        "endpoint_sha256": source.endpoint_sha256,
        "window_frames": 32,
        "scope": "disjoint history/readiness subsets; MUST NOT replace primary all-labeled results",
        "limitations": [
            "Full history is not the same as ready; quality failures can persist after 32 frames.",
            "Subset class compositions differ; this does not establish a causal effect of padding.",
            "These are correlated cache endpoints, not independent source action counts.",
            "No labels, windows, predictions or training parameters are changed.",
        ],
    }
    for subset in ("short_history", "full_not_ready", "full_ready"):
        view = _HistoryView(
            scores, arrays["valid_length"], arrays["ready"], verify, subset
        )
        report[subset] = _summary(evaluate_classification(view), tokens)
    return report


def diagnose_validation(scores, calibration=None, *, allow_synthetic=False, cache=None):
    if scores.provenance.split != "val":
        raise ValueError(
            "diagnosis is validation-only; do not tune from final-test errors"
        )
    if scores.manifest_sha256 is None:
        raise ValueError("diagnosis requires a verified score archive")
    if scores.provenance.synthetic and allow_synthetic is not True:
        raise ValueError("synthetic diagnosis requires explicit opt-in")
    tokens = vocabulary_for(scores.provenance.vocabulary_version).tokens
    baseline = evaluate_classification(scores)
    total = baseline["labeled_endpoints"]
    if not total:
        raise ValueError("diagnosis requires labeled validation endpoints")
    summaries = _summary(baseline, tokens)
    largest = sorted(summaries["per_class"], key=lambda r: (-r["support"], r["token"]))[
        :3
    ]
    confusion = []
    for i, token in enumerate(tokens):
        counts = baseline["confusion_matrix"][i]
        support = baseline["support"][i]
        top = sorted(
            (j for j in range(len(tokens)) if j != i and counts[j]),
            key=lambda j: (-counts[j], j),
        )[:3]
        confusion.append(
            {
                "true": token,
                "support": support,
                "top_errors": [
                    {
                        "predicted": tokens[j],
                        "count": counts[j],
                        "fraction_of_true": counts[j] / support,
                    }
                    for j in top
                ],
            }
        )
    report = {
        "format_version": "validation-diagnosis-v1",
        "synthetic": scores.provenance.synthetic,
        "source": scores.provenance.model_dump(mode="json"),
        "score_manifest_sha256": scores.manifest_sha256,
        "actions_enabled": False,
        "scope": "fixed validation cache endpoints; not action events, independent clips or test performance",
        "primary": summaries,
        "largest_three_classes": {
            "tokens": [row["token"] for row in largest],
            "support_fraction": sum(row["support"] for row in largest) / total,
        },
        "per_class_confusions": confusion,
        "readiness_diagnostics": {
            "scope": "different subsets; MUST NOT replace the primary all-labeled results",
            "not_ready": _summary(
                evaluate_classification(_ReadinessView(scores, 0)), tokens
            ),
            "ready": _summary(
                evaluate_classification(_ReadinessView(scores, 1)), tokens
            ),
        },
        "calibrated_rejection": None,
        "policy_confidence_only": None,
        "limitations": [
            "No parameters or weights changed; no training or search was run.",
            "Temperature scaling does not improve the underlying argmax classifier.",
            "Accepted accuracy excludes rejected samples; compare coverage and all-labeled recall alongside it.",
            "Readiness subsets have different class composition; this is not proof of a causal effect.",
            "Policy confidence alone omits owner, warmup, release, motion geometry, timing and cooldown.",
            "Overlapping window endpoints are not statistically independent trials.",
            "Class weights or resampling probabilities must be computed from TRAINING labels, never these validation supports.",
        ],
    }
    if calibration is not None:
        value = Calibration.model_validate(calibration.model_dump())
        if (
            value.source != scores.provenance
            or value.score_manifest_sha256 != scores.manifest_sha256
        ):
            raise ValueError(
                "diagnosis calibration must match this exact validation score archive"
            )
        if value.rejection_status == "selected":
            report["calibrated_rejection"] = _summary(
                evaluate_classification(
                    scores, temperature=value.temperature, threshold=value.threshold
                ),
                tokens,
            )
            threshold = max(value.threshold, PublicGesturePolicy().class_on)
            report["policy_confidence_only"] = _summary(
                evaluate_classification(
                    scores, temperature=value.temperature, threshold=threshold
                ),
                tokens,
            )
            report["policy_confidence_only"]["scope"] = (
                "confidence filter only, NOT full gesture-policy events"
            )
        else:
            report["limitations"].append(
                "Calibration constraints were infeasible; no rejection/policy success is claimed."
            )
    if cache is not None:
        report["history_diagnostics"] = _diagnose_history(scores, cache)
    return report
