"""No hardware/SDK dependency: exercise subprocess failure reporting and protocol."""
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from backend.perception import probe


def run(code, **kwargs):
    return probe.run_isolated([sys.executable, "-u", "-c", code], **kwargs)


def test_success_keeps_partial_logs_and_exit_code():
    result = run('import sys; print("out"); print("err", file=sys.stderr)')
    assert result["status"] == "passed" and result["returncode"] == 0
    assert result["stdout"] == "out\n" and result["stderr"] == "err\n"
    assert result["monitor_reason"] is None and result["signal"] is None


def test_failure_is_not_successful_no_hand_detection():
    result = run('print("model_load"); raise RuntimeError("failed sdk")')
    assert result["status"] == "failed" and result["returncode"] == 1
    assert "model_load" in result["stdout"] and "failed sdk" in result["stderr"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal return codes")
def test_sigkill_is_not_claimed_as_oom_or_monitor_timeout():
    result = run('import os, signal; print("model_load"); os.kill(os.getpid(), signal.SIGKILL)')
    assert result["status"] == "failed" and result["signal"] == "SIGKILL"
    assert result["returncode"] == -signal.SIGKILL and result["monitor_reason"] is None
    assert result["stdout"] == "model_load\n"


def test_timeout_kills_and_reaps_child_retaining_stage():
    result = run('import time; print("model_load", flush=True); time.sleep(60)', timeout=.5)
    assert result["status"] == result["monitor_reason"] == "timeout"
    assert result["returncode"] is not None and result["elapsed_seconds"] < 5
    assert result["stdout"] == "model_load\n"


def test_rss_guard_is_explicit_not_external_oom(monkeypatch):
    monkeypatch.setattr(probe, "linux_memory", lambda pid: 2 * 1024 * 1024)
    result = run('import time; time.sleep(60)', rss_limit_mib=1)
    assert result["status"] == result["monitor_reason"] == "rss_limit"
    assert result["sampled_peak_rss_bytes"] == 2 * 1024 * 1024
    assert result["returncode"] is not None


def test_log_guard_and_bounded_report():
    result = run('import time; print("x" * 1100000, flush=True); time.sleep(60)')
    assert result["status"] == "log_limit" and result["stdout_truncated"]
    assert len(result["stdout"]) <= probe._LOG_TAIL


def test_launch_failure_is_reported(tmp_path):
    result = probe.run_isolated([str(tmp_path / "missing-python")])
    assert result["status"] == "failed" and result["returncode"] is None
    assert "FileNotFoundError" in result["launch_error"]


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("name", ["timeout", "rss_limit_mib"])
def test_bad_limits_do_not_launch(value, name, monkeypatch):
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="finite and positive"):
        run("pass", **{name: value})


def test_missing_asset_still_writes_failed_report(tmp_path, capsys):
    report = tmp_path / "probe.json"
    assert probe.main(["--model", str(tmp_path / "missing.task"), "--report", str(report)]) == 2
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and not result["real_hand_validated"]
    assert result["stages"][-1]["probe_stage"] == "model_load"
    assert "FileNotFoundError" in result["stderr"]
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("option", ["--timeout", "--rss-limit-mib"])
def test_bad_cli_limit_does_not_create_report(tmp_path, option, capsys):
    report = tmp_path / "absent" / "report.json"
    assert probe.main(["--model", "unused.task", option, "nan", "--report", str(report)]) == 2
    assert not report.parent.exists() and "finite and positive" in capsys.readouterr().err


def test_bad_digest_produces_failure_report_without_sdk_import(tmp_path, capsys):
    model = tmp_path / "not-a-model.task"
    model.write_bytes(b"fixture")
    report = tmp_path / "digest-report.json"
    assert probe.main(["--model", str(model), "--model-sha256", "0" * 64, "--report", str(report)]) == 2
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and "SHA256 mismatch" in result["stderr"]
    assert result["stages"][-1]["probe_stage"] == "model_load"


def test_existing_report_prevents_native_launch(tmp_path, monkeypatch, capsys):
    report = tmp_path / "probe.json"
    report.write_text("previous evidence")
    monkeypatch.setattr(probe, "probe", lambda *a, **k: pytest.fail("must not launch"))
    assert probe.main(["--model", "ignored.task", "--report", str(report)]) == 2
    assert report.read_text() == "previous evidence" and "probe error" in capsys.readouterr().err


def test_zero_exit_without_completed_protocol_fails(monkeypatch):
    captured = {}
    def isolated(command, **kwargs):
        captured.update(command=command, **kwargs)
        return {"status": "passed", "stdout": '{"probe_stage": "model_load"}\n'}
    monkeypatch.setattr(probe, "run_isolated", isolated)
    monkeypatch.setenv("PYTHONPATH", "/unrelated/site-packages")
    result = probe.probe(Path("explicit.task"), model_sha256="abc")
    assert result["status"] == "failed" and "protocol_error" in result
    assert "PYTHONPATH" not in captured["env"]
    assert captured["command"][-2:] == ["--model-sha256", "abc"]
    assert not result["real_hand_validated"]


def test_worker_infers_three_black_frames_and_closes(monkeypatch, capfd):
    import numpy as np
    from backend.perception import mediapipe_adapter
    calls = []
    class Detector:
        metadata = {"synthetic": False, "sdk_version": "fake-for-unit-test"}
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))
        def detect(self, rgb, ms):
            assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8 and not rgb.any()
            calls.append(ms)
            return []
        def close(self):
            calls.append("close")
    monkeypatch.setattr(mediapipe_adapter, "MediaPipeDetector", Detector)
    assert probe._worker(SimpleNamespace(model="fixture", model_sha256="expected")) == 0
    stages = [json.loads(line) for line in capfd.readouterr().out.splitlines()]
    assert calls == [("fixture", {"expected_sha256": "expected"}), 0, 34, 68, "close"]
    assert stages[-1]["probe_stage"] == "complete" and not stages[-1]["camera_accessed"]


def test_worker_closes_after_inference_exception_without_complete(monkeypatch, capfd):
    from backend.perception import mediapipe_adapter
    closed = []
    class Detector:
        metadata = {}
        def __init__(self, *args, **kwargs): pass
        def detect(self, *args): raise RuntimeError("inference failed")
        def close(self): closed.append(True)
    monkeypatch.setattr(mediapipe_adapter, "MediaPipeDetector", Detector)
    with pytest.raises(RuntimeError, match="inference failed"):
        probe._worker(SimpleNamespace(model="fixture", model_sha256=None))
    assert closed == [True] and '"complete"' not in capfd.readouterr().out
