"""Endpoint supervision that explicitly skips entirely unlabelled batches."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_cross_entropy(
    logits: Tensor, targets: Tensor, class_valid: Tensor
) -> Tensor | None:
    """Mean CE on valid labels, or ``None`` to tell a trainer to skip a step.

    ``logits`` is [B,C] (or diagnostic [B,T,C]); the other tensors match all
    leading dimensions. Unsupervised targets may contain -1 or other sentinels,
    while every supervised target must be an integer in [0,C).
    """
    if not all(isinstance(value, Tensor) for value in (logits, targets, class_valid)):
        raise ValueError("logits, targets and class_valid must be torch Tensors")
    if logits.ndim not in (2, 3) or any(size <= 0 for size in logits.shape):
        raise ValueError("logits must have nonempty shape [B,C] or [B,T,C]")
    if not logits.is_floating_point() or not torch.isfinite(logits).all().item():
        raise ValueError("logits must have a floating dtype and only finite values")
    if targets.shape != logits.shape[:-1] or class_valid.shape != targets.shape:
        raise ValueError("targets and class_valid must match the leading logits shape")
    if targets.dtype != torch.long:
        raise ValueError("targets must have torch.long dtype")
    if class_valid.dtype != torch.bool:
        raise ValueError("class_valid must have torch.bool dtype")
    if targets.device != logits.device or class_valid.device != logits.device:
        raise ValueError("logits, targets and class_valid must be on the same device")
    if not class_valid.any().item():
        return None
    selected = targets[class_valid]
    if ((selected < 0) | (selected >= logits.shape[-1])).any().item():
        raise ValueError("supervised targets must be in the class vocabulary range")
    # Half/bfloat logits are promoted for the reduction; gradients still reach
    # the original tensor and float64 callers keep their requested precision.
    supervised_logits = logits[class_valid]
    if supervised_logits.dtype in (torch.float16, torch.bfloat16):
        supervised_logits = supervised_logits.float()
    loss = F.cross_entropy(supervised_logits, selected)
    if not torch.isfinite(loss).item():
        raise ValueError("cross entropy produced a non-finite loss")
    return loss
