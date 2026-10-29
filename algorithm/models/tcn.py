"""T13 single-convolution residual TCN with strictly left-sided context."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn
from torch.nn import functional as F

from shared.classes import ClassVocabulary

from .mlp import (
    _dropout_probability,
    _positive_int,
    _resolve_vocabulary,
    _validate_features,
)


class CausalConv1d(nn.Conv1d):
    """Length-preserving convolution with no access to a future time step."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        in_channels = _positive_int("in_channels", in_channels)
        out_channels = _positive_int("out_channels", out_channels)
        kernel_size = _positive_int("kernel_size", kernel_size)
        dilation = _positive_int("dilation", dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=dilation,
            padding=0,
            groups=1,
            bias=bias,
        )
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class CausalResidualBlock(nn.Module):
    """One causal conv → per-time channel LN → ReLU → dropout → residual."""

    def __init__(
        self,
        channels: int = 64,
        dilation: int = 1,
        dropout: float = 0.2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        channels = _positive_int("channels", channels)
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(_dropout_probability(dropout))

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv(x)
        # LayerNorm sees only C at a single (batch, time), never [C, T].
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.dropout(F.relu(h))
        return F.relu(x + h)


class GestureTCN(nn.Module):
    """Window-end classifier with a diagnostic causal ``forward_all`` path.

    The default 4 blocks have receptive field 31 and 56,325 parameters. Regular
    inference/training requires exactly 32 frames. Diagnostic ``forward_all``
    accepts any nonempty prefix; its short-prefix outputs include left padding
    and must not be used to silently widen the endpoint supervision set.
    """

    def __init__(
        self,
        input_dim: int = 94,
        num_classes: int | None = None,
        channels: int = 64,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.2,
        kernel_size: int = 3,
        window_size: int = 32,
        *,
        vocabulary: ClassVocabulary | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim)
        self.channels = _positive_int("channels", channels)
        self.kernel_size = _positive_int("kernel_size", kernel_size)
        self.dilations = tuple(_positive_int("dilation", d) for d in dilations)
        if not self.dilations:
            raise ValueError("dilations must not be empty")
        self.window_size = _positive_int("window_size", window_size)
        self.receptive_field = 1 + (self.kernel_size - 1) * sum(self.dilations)
        if self.window_size < self.receptive_field:
            raise ValueError("window_size must cover the full receptive field")
        probability = _dropout_probability(dropout)
        vocab = _resolve_vocabulary(num_classes, vocabulary)
        self.class_tokens = vocab.tokens
        self.num_classes = vocab.class_count
        self.input_projection = nn.Linear(self.input_dim, self.channels)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(self.channels, dilation, probability, self.kernel_size)
            for dilation in self.dilations
        )
        self.classifier = nn.Linear(self.channels, self.num_classes)

    def forward_all(self, x: Tensor) -> Tensor:
        """Return raw logits for every time step, including padded prefixes."""
        _validate_features(x, ndim=3, input_dim=self.input_dim)
        h = F.relu(self.input_projection(x)).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return self.classifier(h.transpose(1, 2))

    def forward(self, x: Tensor) -> Tensor:
        _validate_features(x, ndim=3, input_dim=self.input_dim)
        if x.shape[1] != self.window_size:
            raise ValueError(f"forward requires exactly {self.window_size} frames")
        return self.forward_all(x)[:, -1, :]
