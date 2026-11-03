"""Train-only endpoint populations, without changing labels or window history.

The trainer permutes this population once per epoch using its checkpointed torch
Generator. No replacement, class resampling, readiness gating, or test data.
This module is also usable in read-only checks without importing torch.
"""

import numpy as np


def training_population(class_valid, sampling="uniform"):
    mask = np.asarray(class_valid)
    if mask.ndim != 1 or mask.dtype.kind != "b":
        raise ValueError("class_valid must be a one-dimensional boolean vector")
    if not mask.any():
        raise ValueError("training requires nonempty supervised endpoints")
    if sampling == "uniform":
        indexes = np.arange(len(mask), dtype="<i8")
    elif sampling == "supervised_uniform":
        indexes = np.flatnonzero(mask).astype("<i8", copy=False)
    else:
        raise ValueError("unknown training sampling policy")
    indexes.setflags(write=False)
    return indexes
