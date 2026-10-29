import json

from backend.perception.diagnostics import main, probe


def test_missing_asset_does_not_load_sdk_or_open_camera(tmp_path):
    report = probe(tmp_path / "missing.task")
    assert report["status"] == "missing_asset" and report["attempts"] == 0
    assert not report["camera_opened"]
    output = tmp_path / "probe.json"
    assert main(["--report", str(output)]) == 2
    assert json.loads(output.read_text())["status"] == "missing_asset"
    assert main(["--report", str(output)]) == 2


def test_wrong_hash_rejected_without_importing_sdk(tmp_path):
    asset = tmp_path / "asset.task"
    asset.write_bytes(b"invalid")
    report = probe(asset, expected_sha256="0" * 64)
    assert report["status"] == "hash_mismatch" and report["attempts"] == 0


def test_legacy_probe_delegates_to_master_bounded_worker(tmp_path, monkeypatch):
    from backend.perception import diagnostics
    asset = tmp_path / "fixture.task"
    asset.write_bytes(b"fixture-only")
    calls = []
    def canonical(model, **kwargs):
        calls.append((model, kwargs))
        return dict(status="passed", returncode=0, elapsed_seconds=.01, stdout="complete", stderr="", monitor_reason=None)
    monkeypatch.setattr(diagnostics.canonical, "probe", canonical)
    report = diagnostics.probe(asset, timeout_seconds=5)
    assert report["status"] == "passed_synthetic_image" and report["attempts"] == 1
    assert calls == [(asset.resolve(), {"model_sha256": report["asset_sha256"], "timeout": 5})]
    assert report["canonical_report"]["status"] == "passed"


def test_legacy_report_exists_prevents_native_probe(tmp_path, monkeypatch):
    from backend.perception import diagnostics
    report = tmp_path / "existing.json"
    report.write_text("keep")
    monkeypatch.setattr(diagnostics, "probe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    assert diagnostics.main(["--model", "unused", "--report", str(report)]) == 2
    assert report.read_text() == "keep"


def test_invalid_legacy_timeout_does_not_reserve_report(tmp_path):
    from backend.perception import diagnostics
    target = tmp_path / "invalid.json"
    assert diagnostics.main(["--timeout-seconds", "nan", "--report", str(target)]) == 2
    assert not target.exists()
