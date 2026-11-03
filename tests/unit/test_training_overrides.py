"""Per-run tuning controls are testable without constructing an optimizer or training."""

import pytest

from backend.workflows.cli import parser
from backend.workflows import training
from backend.workflows.workspace import Settings, Workspace


@pytest.fixture
def ws(tmp_path, monkeypatch):
    workspace = Workspace(
        tmp_path,
        Settings.model_validate(
            {
                "train": {
                    "learning_rate": 0.002,
                    "weight_decay": 0.003,
                    "patience": 12,
                    "seed": 77,
                }
            }
        ),
    )
    monkeypatch.setattr(
        training, "dataset_receipt", lambda _: {"receipt_sha256": "a" * 64}
    )
    monkeypatch.setattr(
        "algorithm.training.inputs.load_training_inputs",
        lambda *a: (
            None,
            None,
            {
                "synthetic": False,
                "train": {"supervised": 30, "class_counts": [2] * 15},
                "val": {"supervised": 15},
            },
        ),
    )
    return workspace


def config(ws, flags):
    args = parser().parse_args(["train", "--dry-run", *flags])
    return training.preflight(ws, args, emit=lambda _: None)[0]


def test_unspecified_values_keep_local_settings(ws):
    value = config(ws, [])
    assert (value.learning_rate, value.weight_decay, value.patience, value.seed) == (
        0.002,
        0.003,
        12,
        77,
    )


def test_overrides_are_run_local_and_zero_is_not_replaced_by_default(ws):
    original = ws.settings.model_dump()
    value = config(
        ws,
        [
            "--learning-rate",
            ".0003",
            "--weight-decay",
            "0",
            "--patience",
            "6",
            "--seed",
            "0",
        ],
    )
    assert (value.learning_rate, value.weight_decay, value.patience, value.seed) == (
        0.0003,
        0,
        6,
        0,
    )
    assert ws.settings.model_dump() == original
    assert not ws.runs.exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--learning-rate", "0"],
        ["--learning-rate", "-.1"],
        ["--learning-rate", "nan"],
        ["--learning-rate", "inf"],
        ["--weight-decay", "-1"],
        ["--weight-decay", "nan"],
        ["--weight-decay", "inf"],
        ["--seed", "-1"],
        ["--seed", str(2**32)],
    ],
)
def test_invalid_overrides_fail_before_training(ws, flags):
    with pytest.raises(ValueError):
        config(ws, flags)
    assert not ws.runs.exists()


def test_resume_cannot_silently_change_learning_parameters():
    with pytest.raises(SystemExit):
        parser().parse_args(["resume", "--learning-rate", ".0003"])


def test_dry_run_with_overrides_never_enters_dependency_or_training_execution(
    ws, monkeypatch
):
    monkeypatch.setattr(
        training,
        "ensure_training",
        lambda *a, **k: pytest.fail("dry-run must not enter execution"),
    )
    args = parser().parse_args(["train", "--learning-rate", ".0003", "--dry-run"])
    messages = []
    assert (
        training.run_train(ws, args, ["train", "--dry-run"], emit=messages.append) == 0
    )
    assert any("lr=0.0003" in m for m in messages)
    assert any("未导入训练器" in m for m in messages)
    assert not ws.runs.exists()
