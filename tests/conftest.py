import os
import sys

import pytest

from backend.mock import run_synthetic
from shared.config import load_config


@pytest.fixture(autouse=True)
def no_neural_training_guard(monkeypatch):
    """uv run check must fail, not silently train, if a selected test regresses."""
    if os.environ.get("HANDSHAKE_NO_TRAINING") != "1":
        return
    torch = sys.modules.get("torch")
    if torch is not None:

        def forbidden(*args, **kwargs):
            pytest.fail(
                "neural training is forbidden in uv run check; use --training on an authorized machine"
            )

        monkeypatch.setattr(torch.Tensor, "backward", forbidden)
        monkeypatch.setattr(torch.autograd, "backward", forbidden)
        monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def smoke_records(config):
    return run_synthetic(config, code_version="test-only")
