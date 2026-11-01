"""Short CI coverage for the explicitly opt-in long-duration soak command."""

import json
import subprocess
import sys

import pytest

from backend.safety_soak import run_soak


def test_cycles_cover_navigation_duplicate_stall_conflict_and_release(tmp_path):
    value = run_soak(
        tmp_path / "soak", cycles=6, allow_synthetic=True, max_rss_mb=2048, pace_ms=0
    )
    assert value["complete"], value["failure"]
    assert value["counts"]["cycles"] == 6
    assert value["counts"]["application_commands"] == 18
    assert value["counts"]["duplicates_rejected"] == 18
    assert value["counts"]["source_stall_faults"] == 2
    assert value["counts"]["ownership_faults"] == 2
    assert value["stop_and_release"]["count"] == 6
    assert value["watchdog_observation"]["count"] == 2
    assert value["elapsed_seconds"] > 0 and not value["live_available"]
    with pytest.raises(FileExistsError):
        run_soak(tmp_path / "soak", cycles=1, allow_synthetic=True)


def test_partial_failure_publishes_failed_report_and_cleans_up(tmp_path, monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("injected source fault")

    monkeypatch.setattr("backend.safety_soak.synthetic_input", broken)
    result = run_soak(
        tmp_path / "failed", cycles=2, allow_synthetic=True, max_rss_mb=2048
    )
    assert result["status"] == "failed" and not result["complete"]
    assert "injected source fault" in result["failure"]
    assert (
        json.loads((tmp_path / "failed/receipt.json").read_text())["status"] == "failed"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cycles": 1},
        {"cycles": True, "allow_synthetic": True},
        {"seconds": 0, "allow_synthetic": True},
        {"seconds": 1, "cycles": 1, "allow_synthetic": True},
        {"cycles": 1, "allow_synthetic": True, "pace_ms": -1},
    ],
)
def test_explicit_consent_and_resource_limits_before_output(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_soak(tmp_path / "no-output", **kwargs)
    assert not (tmp_path / "no-output").exists()


def test_standalone_duration_is_real_and_never_imports_neural_or_camera_runtime(
    tmp_path,
):
    code = "import sys; from backend.safety_soak import main; assert not {'torch','cv2','mediapipe'}.intersection(sys.modules); raise SystemExit(main())"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--seconds",
            "1",
            "--allow-synthetic",
            "--output",
            str(tmp_path / "duration"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    value = json.loads((tmp_path / "duration/report.json").read_text())
    assert value["elapsed_seconds"] >= 1 and value["complete"]
