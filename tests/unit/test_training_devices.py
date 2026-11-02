"""Device choice and evaluation forward passes only; no backward/optimizer calls."""

from types import SimpleNamespace

import numpy as np
import pytest

from algorithm.training.config import TrainConfig
from algorithm.training.devices import resolve_device


def torch_stub(cuda=False, mps=False, count=1):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: count),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (False, False, "cpu"),
        (True, False, "cuda:0"),
        (False, True, "mps"),
        (True, True, "cuda:0"),
    ],
)
def test_auto_device_policy(cuda, mps, expected, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert resolve_device("auto", torch_module=torch_stub(cuda, mps)) == expected


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "mps", "gpu", "cuda:-1"])
def test_requested_unavailable_or_invalid_device_never_silently_falls_back(device):
    with pytest.raises(ValueError):
        resolve_device(device, torch_module=torch_stub())


def test_accelerator_config_preserves_backward_compatible_cpu_default():
    assert TrainConfig().device == "cpu"
    assert TrainConfig(device="auto").device == "auto"
    with pytest.raises(ValueError):
        TrainConfig(device="invented")


def test_cuda_index_and_deterministic_environment(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert (
        resolve_device("cuda:1", torch_module=torch_stub(cuda=True, count=2))
        == "cuda:1"
    )
    with pytest.raises(ValueError, match="does not exist"):
        resolve_device("cuda:2", torch_module=torch_stub(cuda=True, count=2))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(ValueError, match="CUBLAS"):
        resolve_device("cuda", torch_module=torch_stub(cuda=True))


def test_training_guard_fails_before_input_or_model_allocation(monkeypatch):
    import algorithm.training.trainer as trainer

    monkeypatch.setenv("HANDSHAKE_NO_TRAINING", "1")
    monkeypatch.setattr(
        trainer,
        "load_training_inputs",
        lambda *a: pytest.fail("must not read training inputs"),
    )
    with pytest.raises(RuntimeError, match="forbids neural training"):
        trainer.train("unused-train", "unused-val", "must-not-be-created")


def test_cpu_evaluation_and_rng_roundtrip_do_not_train(monkeypatch):
    import torch
    from algorithm.training.trainer import _evaluate, _restore_rng, _rng_state
    from algorithm.model_config import build_model

    monkeypatch.setattr(
        torch.Tensor, "backward", lambda *a, **k: pytest.fail("no backward")
    )
    model = build_model("mlp").eval()
    data = {
        "x": np.zeros((4, 32, 94), np.float32),
        "y": np.zeros(4, np.int64),
        "class_valid": np.ones(4, dtype=bool),
    }
    report = _evaluate(model, data, "mlp", 2)
    assert np.isfinite(report["loss"])
    generator = torch.Generator().manual_seed(42)
    state = _rng_state(generator)
    expected = torch.randperm(10, generator=generator)
    _restore_rng(state, generator)
    assert torch.equal(torch.randperm(10, generator=generator), expected)
