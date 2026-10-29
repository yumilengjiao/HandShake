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
