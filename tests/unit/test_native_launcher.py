"""Launch-mode contracts only: no Flutter process, camera or model is started."""

import pytest

from backend import native_launcher


@pytest.mark.parametrize("debug", [False, True])
def test_daily_launch_is_release_and_debug_is_explicit(tmp_path, monkeypatch, debug):
    calls = []
    monkeypatch.setattr(native_launcher.shutil, "which", lambda _: "/tools/flutter")
    monkeypatch.setattr(
        native_launcher.subprocess, "call",
        lambda command, **kwargs: calls.append((command, kwargs)) or 0,
    )
    assert native_launcher.main(["--debug"] if debug else [], project_root=tmp_path) == 0
    command, options = calls[0]
    assert command == [
        "/tools/flutter", "--no-version-check", "run", "-d", "linux",
        "--debug" if debug else "--release",
    ]
    assert options["cwd"] == tmp_path / "frontend/native"
    assert options["env"]["HANDSHAKE_CAMERA"] == "0"
    assert "HANDSHAKE_MODEL_PACKAGE" not in options["env"]


def test_built_bundle_is_not_rebuilt_and_cannot_be_called_debug(tmp_path, monkeypatch):
    executable = tmp_path / "handshake_stage"
    executable.touch()
    calls = []
    monkeypatch.setattr(
        native_launcher.subprocess, "call",
        lambda command, **kwargs: calls.append(command) or 0,
    )
    assert native_launcher.main(["--bundle", str(executable)], project_root=tmp_path) == 0
    assert calls == [[str(executable)]]
    with pytest.raises(SystemExit):
        native_launcher.main(["--bundle", str(executable), "--debug"], project_root=tmp_path)
    assert len(calls) == 1
