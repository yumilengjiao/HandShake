"""Explicit accelerator selection; never silently downgrade a requested device."""

import os
import re


def resolve_device(requested="auto", *, torch_module=None):
    if not isinstance(requested, str) or not re.fullmatch(
        r"auto|cpu|cuda(?::[0-9]+)?|mps", requested
    ):
        raise ValueError("device must be auto, cpu, cuda, cuda:N or mps")
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    mps = getattr(torch.backends, "mps", None)
    if requested == "auto":
        requested = (
            "cuda:0"
            if torch.cuda.is_available()
            else "mps"
            if mps is not None and mps.is_available()
            else "cpu"
        )
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA unavailable: check NVIDIA driver and run 'uv run setup --gpu'; or choose device='cpu'"
            )
        index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        requested = f"cuda:{index}"
        # Required for deterministic cuBLAS; must be set before the first CUDA op.
        setting = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if setting not in {":4096:8", ":16:8"}:
            raise ValueError(
                "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic training"
            )
    if requested == "mps" and (mps is None or not mps.is_available()):
        raise ValueError(
            "MPS unavailable: requires a supported Apple device/macOS/PyTorch; choose device='cpu'"
        )
    return requested


def device_identity(device):
    import torch

    result = {"device": device}
    if device.startswith("cuda"):
        result.update(
            accelerator_name=torch.cuda.get_device_name(device),
            cuda_runtime=torch.version.cuda,
        )
    elif device == "mps":
        result.update(accelerator_name="Apple Metal/MPS")
    return result
