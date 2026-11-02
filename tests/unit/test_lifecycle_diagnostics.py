"""Bounded lifecycle metadata, no paths/images or exception messages."""

import json

import pytest

from backend.perception.lifecycle import lifecycle


def test_lifecycle_sanitizes_exception_and_uses_only_stderr(capsys):
    lifecycle("model_failed", error=ValueError("/private/user/model.npz secret"))
    captured = capsys.readouterr()
    assert not captured.out and "/private" not in captured.err and "secret" not in captured.err
    value = json.loads(captured.err.removeprefix("HS_DIAG "))
    assert value["phase"] == "model_failed"
    assert value["error_type"] == "ValueError"
    assert int(value["monotonic_ns"]) > 0


def test_unregistered_stage_cannot_be_used_to_emit_arbitrary_text():
    with pytest.raises(ValueError):
        lifecycle("/private/filename")
