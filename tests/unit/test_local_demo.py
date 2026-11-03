"""Local demo entry and image budgets; no display or camera is opened by these tests."""

import json
from types import SimpleNamespace
import threading

import pytest

from backend.demo_slides import load_slide_images
from backend.gesture_demo import main, run_ui
from backend.workflows.deployment import handoff_assets
from backend.workflows.workspace import WorkflowError


def handoff(tmp_path):
    (tmp_path / "export.json").write_text(
        json.dumps(
            {
                "format_version": "model-export-receipt-v1",
                "directory": "models/model",
                "manifest_sha256": "a" * 64,
            }
        )
    )
    (tmp_path / "calibration-receipt.json").write_text(
        json.dumps(
            {
                "path": "evaluations/val/calibration.json",
                "sha256": "b" * 64,
            }
        )
    )
    return tmp_path


def test_handoff_is_independent_of_local_training_dataset(tmp_path):
    result = handoff_assets(handoff(tmp_path))
    assert result["package"] == tmp_path / "models/model"
    assert result["sha256"] == "a" * 64
    assert not (tmp_path / "latest.json").exists()


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside", "", None])
def test_handoff_refuses_escaped_or_invalid_paths(tmp_path, path):
    handoff(tmp_path)
    receipt = tmp_path / "calibration-receipt.json"
    receipt.write_text(json.dumps({"path": path, "sha256": "b" * 64}))
    with pytest.raises(WorkflowError):
        handoff_assets(tmp_path)


def test_handoff_symlink_cannot_escape_root(tmp_path):
    handoff(tmp_path)
    (tmp_path / "models").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(WorkflowError):
        handoff_assets(tmp_path)


def test_launch_requires_optin_and_verify_only_does_not_import_ui(
    tmp_path, monkeypatch
):
    with pytest.raises(SystemExit):
        main(["--run", str(tmp_path)])
    calls = []
    monkeypatch.setattr(
        "backend.workflows.deployment.verify_assets", lambda **k: calls.append(k)
    )
    monkeypatch.setattr(
        "backend.gesture_demo.run_ui", lambda **k: pytest.fail("UI must not open")
    )
    assert main(["--run", str(handoff(tmp_path)), "--verify-only"]) == 0
    assert calls[0]["shadow"] is True


def test_images_are_sorted_bounded_and_do_not_overwrite_sources(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    for name, color in [("002.png", "blue"), ("001.png", "red")]:
        Image.new("RGB", (40, 20), color).save(tmp_path / name)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    (deck, identity), images = load_slide_images(tmp_path)
    assert len(deck.slides) == len(images) == 2
    assert images[0].shape == (540, 960, 3)
    assert images[0][270, 480].tolist() == [0, 0, 255]
    assert images[1][270, 480].tolist() == [255, 0, 0]
    assert len(identity) == 64
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_slide_path_count_format_and_pixel_limits(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    with pytest.raises(ValueError, match="1..32"):
        load_slide_images(tmp_path)
    path = tmp_path / "001.png"
    path.write_bytes(b"not an image")
    with pytest.raises(OSError):
        load_slide_images(tmp_path)
    Image.new("RGB", (10, 10)).save(path)
    monkeypatch.setattr("backend.demo_slides.MAX_PIXELS", 99)
    with pytest.raises(ValueError, match="8 megapixels"):
        load_slide_images(tmp_path)
    path.unlink()
    path.symlink_to(tmp_path.parent / "outside.png")
    with pytest.raises(ValueError, match="regular files"):
        load_slide_images(tmp_path)


@pytest.mark.parametrize(
    "keys,count",
    [([ord("q")], 0), ([ord("c"), ord("q")], 1), ([ord("c"), ord("c"), ord("q")], 1)],
)
def test_ui_requires_camera_key_and_never_starts_two_workers(monkeypatch, keys, count):
    cv = pytest.importorskip("cv2")
    created = []

    class Worker:
        def __init__(self, **kwargs):
            created.append(self)
            self.kwargs = kwargs
            self.adapter = None
            self.error = None
            self.done = threading.Event()
            self.stop_requested = threading.Event()
            self.thread = SimpleNamespace(
                join=lambda timeout: None, is_alive=lambda: False
            )

        def start(self):
            self.started = True

        def take(self):
            return None

        def request_stop(self):
            self.stop_requested.set()
            self.done.set()

    monkeypatch.setattr("backend.perception.local_demo.DemoCameraWorker", Worker)
    for name in ("namedWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv, name, lambda *a, **k: None)
    monkeypatch.setattr(cv, "getWindowProperty", lambda *a: 1)
    sequence = iter(keys)
    monkeypatch.setattr(cv, "waitKey", lambda *a: next(sequence))
    assets = dict(
        package="package",
        sha256="a" * 64,
        calibration="calibration",
        calibration_sha256="b" * 64,
    )
    assert run_ui(assets, perception_model="perception") == 0
    assert len(created) == count
    for worker in created:
        assert worker.started and worker.stop_requested.is_set()
        assert worker.kwargs["allow_application_local"] is True
