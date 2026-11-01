"""Fixed-denominator classification, probability diagnostics and abstention metrics."""

import math

import numpy as np


def log_probabilities(logits, temperature=1.0):
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or not 0.05 <= temperature <= 100
    ):
        raise ValueError("temperature must be finite in [0.05,100]")
    values = np.asarray(logits, dtype=np.float64)
    if (
        values.ndim != 2
        or not values.shape[1]
        or not np.isfinite(values).all()
        or np.any(np.abs(values) > 1e6)
    ):
        raise ValueError("invalid logits")
    scaled = values / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    return scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))


def evaluate_classification(scores, *, temperature=1.0, threshold=0.0, bins=15):
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("confidence threshold must be in [0,1]")
    if type(bins) is not int or not 1 <= bins <= 100:
        raise ValueError("ECE bins must be in [1,100]")
    # Validate temperature even for empty input.
    log_probabilities(np.zeros((1, scores.classes)), temperature)
    count = scores.classes
    matrix = np.zeros((count, count + 1), dtype=np.int64)
    bin_counts = np.zeros(bins, dtype=np.int64)
    bin_correct, bin_confidence = np.zeros(bins), np.zeros(bins)
    total = labeled = available = ready = accepted = errors = 0
    nll = brier = 0.0
    for batch in scores.batches():
        total += len(batch)
        mask = batch["class_valid"].astype(bool)
        labels = batch["label"][mask].astype(np.int64)
        labeled += len(labels)
        present = batch["available"][mask].astype(bool)
        predicted = np.full(len(labels), count, dtype=np.int64)
        if present.any():
            lp = log_probabilities(batch["logits"][mask][present], temperature)
            probabilities = np.exp(lp)
            targets = labels[present]
            guesses = probabilities.argmax(axis=1)
            confidence = probabilities.max(axis=1)
            correct = guesses == targets
            keep = confidence >= threshold
            predicted[np.flatnonzero(present)[keep]] = guesses[keep]
            available += len(targets)
            ready += int(batch["ready"][mask][present].sum())
            accepted += int(keep.sum())
            errors += int((keep & ~correct).sum())
            nll -= float(lp[np.arange(len(targets)), targets].sum())
            brier += float(
                (
                    np.square(probabilities).sum(axis=1)
                    - 2 * probabilities[np.arange(len(targets)), targets]
                    + 1
                ).sum()
            )
            positions = np.minimum((confidence * bins).astype(int), bins - 1)
            np.add.at(bin_counts, positions, 1)
            np.add.at(bin_correct, positions, correct.astype(float))
            np.add.at(bin_confidence, positions, confidence)
        np.add.at(matrix, (labels, predicted), 1)
    support = matrix.sum(axis=1)
    predicted_count = matrix[:, :count].sum(axis=0)
    tp = matrix[:, :count].diagonal()
    precision = np.divide(
        tp, predicted_count, out=np.zeros(count), where=predicted_count > 0
    )
    recall = np.divide(tp, support, out=np.zeros(count), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(count),
        where=precision + recall > 0,
    )
    normalized = np.divide(
        matrix,
        support[:, None],
        out=np.zeros_like(matrix, dtype=float),
        where=support[:, None] > 0,
    )
    reliability = []
    ece = 0.0
    for i in range(bins):
        n = int(bin_counts[i])
        accuracy = float(bin_correct[i] / n) if n else None
        confidence = float(bin_confidence[i] / n) if n else None
        if n:
            ece += n * abs(accuracy - confidence)
        reliability.append(
            {
                "lower": i / bins,
                "upper": (i + 1) / bins,
                "count": n,
                "accuracy": accuracy,
                "mean_confidence": confidence,
            }
        )
    return {
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "abstain_column": count,
        "support": support.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "absent_classes": np.flatnonzero(support == 0).tolist(),
        "total_endpoints": total,
        "labeled_endpoints": labeled,
        "available_labeled": available,
        "ready_available_labeled": ready,
        "unavailable_labeled": labeled - available,
        "accepted_labeled": accepted,
        "abstained_labeled": labeled - accepted,
        "coverage": accepted / labeled if labeled else None,
        "selective_risk": errors / accepted if accepted else None,
        "accepted_errors": errors,
        "temperature": temperature,
        "threshold": threshold,
        "nll_available": nll / available if available else None,
        "brier_available": brier / available if available else None,
        "ece_available": ece / available if available else None,
        "reliability_bins": reliability,
        "probability_metric_scope": "available labeled endpoints only; coverage denominator is all labeled endpoints",
        "ready_policy": "reported separately, never used to remove classification truth endpoints",
    }
