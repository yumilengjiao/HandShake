"""Optional real accelerator qualification for `uv run check --training`.

Synthetic caches only. Default `check` excludes this file entirely. A missing
accelerator is explicitly skipped, never reported as a successful GPU test.
"""

import os

import pytest
import torch

from algorithm.training.config import TrainConfig
from algorithm.training.trainer import train
from tests.unit.test_training import caches  # noqa: F401 - synthetic cache fixture, no training on import


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("model", ["mlp", "tcn"])
def test_accelerator_epoch_resume_matches_uninterrupted(
    caches, tmp_path, device, model
):
    if os.environ.get("HANDSHAKE_HARDWARE_TRAINING") != "1":
        pytest.skip(
            "real accelerator training requires explicit uv run check --training"
        )
    if os.environ.get("HANDSHAKE_NO_TRAINING") == "1":
        pytest.skip("neural training is disabled in this check")
    available = (
        torch.cuda.is_available()
        if device == "cuda"
        else hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"no {device} hardware/runtime available; not certified")
    config = TrainConfig(model_type=model, device=device, max_epochs=2, batch_size=4)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(*caches, full, config=config)
    train(*caches, resumed, config=config, stop_after_epoch=1)
    result = train(*caches, resumed, config=config, resume=resumed / "last.pt")
    assert result["status"] == "completed"
    assert result["identity"]["environment"]["device"].startswith(device)
    a, b = [
        torch.load(root / "last.pt", map_location="cpu", weights_only=True)
        for root in (full, resumed)
    ]
    assert a["history"] == b["history"]
    for name in a["state_dict"]:
        assert torch.equal(a["state_dict"][name], b["state_dict"][name]), name
        assert torch.equal(a["best_state_dict"][name], b["best_state_dict"][name]), name
