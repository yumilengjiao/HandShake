"""Shared models; neural exports load the optional ``training`` extra lazily.

Importing a pure rule baseline must not require PyTorch to be installed.
"""

from importlib import import_module

__all__ = [
    "CausalConv1d",
    "CausalResidualBlock",
    "GestureMLP",
    "GestureTCN",
    "masked_cross_entropy",
]


def __getattr__(name: str):
    modules = {
        "CausalConv1d": ".tcn",
        "CausalResidualBlock": ".tcn",
        "GestureMLP": ".mlp",
        "GestureTCN": ".tcn",
        "masked_cross_entropy": ".losses",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
