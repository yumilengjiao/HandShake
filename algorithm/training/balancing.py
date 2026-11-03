"""Train-label-only weights; usable by dry-run without importing torch.

Weights have an expected value of one under the supervised training distribution.
The loss must average weighted per-example CE, NOT divide by each batch's weight
sum (which would cancel weighting entirely for a singleton supervised batch).
"""

import math


def class_weights_from_counts(counts, mode="none"):
    """Return fixed weights, or None for the original unweighted objective.

    Counts must include only class_valid training endpoints, in vocabulary order.
    Unsupported classes get zero weight; an observed target with zero weight must
    be rejected by the loss. No validation/test counts, smoothing or resampling.
    """
    if mode not in {"none", "sqrt_inverse"}:
        raise ValueError("unsupported class weighting")
    if (
        not isinstance(counts, (list, tuple))
        or not counts
        or any(type(n) is not int or n < 0 for n in counts)
        or not sum(counts)
    ):
        raise ValueError(
            "class counts must be nonnegative integers with positive support"
        )
    if mode == "none":
        return None
    raw = [1 / math.sqrt(n) if n else 0.0 for n in counts]
    expectation = sum(n * w for n, w in zip(counts, raw)) / sum(counts)
    return [w / expectation for w in raw]
