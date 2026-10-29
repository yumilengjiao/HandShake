"""T12 single-frame baseline; outputs raw logits, never probabilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from shared.classes import ClassVocabulary, load_classes


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dropout_probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("dropout must be finite and in [0, 1)")
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("dropout must be finite and in [0, 1)")
    return float(value)


def _resolve_vocabulary(
    num_classes: int | None, vocabulary: ClassVocabulary | None
) -> ClassVocabulary:
    canonical = load_classes()
    resolved = canonical if vocabulary is None else vocabulary
    if not isinstance(resolved, ClassVocabulary):
        raise ValueError("vocabulary must be a validated ClassVocabulary")
    if resolved.tokens != canonical.tokens:
        raise ValueError("vocabulary must match the current shared class IDs")
    if num_classes is not None:
        _positive_int("num_classes", num_classes)
        if num_classes != resolved.class_count:
            raise ValueError("num_classes does not match the shared vocabulary")
    return resolved


def _validate_features(x: Tensor, *, ndim: int, input_dim: int) -> None:
    if not isinstance(x, Tensor):
        raise ValueError("features must be a torch Tensor")
    if x.ndim != ndim or x.shape[-1] != input_dim:
        raise ValueError(f"features must have {ndim} dimensions and D={input_dim}")
    if any(size <= 0 for size in x.shape):
        raise ValueError("feature batch and time dimensions must be nonempty")
    if not x.is_floating_point():
        raise ValueError("features must have floating-point dtype")
    if not torch.isfinite(x).all().item():
        raise ValueError("features must contain only finite values")


class GestureMLP(nn.Module):
    """94 → 128 → 64 → C baseline using the last frame of a ready window.

    Window readiness and quality gating belong to the shared inference adapter;
    this network accepts only an already selected batch of single frames.
    """

    def __init__(
        self,
        input_dim: int = 94,
        num_classes: int | None = None,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
        *,
        vocabulary: ClassVocabulary | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim)
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two widths")
        widths = tuple(_positive_int("hidden width", width) for width in hidden_dims)
        probability = _dropout_probability(dropout)
        vocab = _resolve_vocabulary(num_classes, vocabulary)
        self.class_tokens = vocab.tokens
        self.num_classes = vocab.class_count
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, widths[0]),
            nn.ReLU(),
            nn.Dropout(probability),
            nn.Linear(widths[0], widths[1]),
            nn.ReLU(),
            nn.Dropout(probability),
            nn.Linear(widths[1], self.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        _validate_features(x, ndim=2, input_dim=self.input_dim)
        return self.network(x)
