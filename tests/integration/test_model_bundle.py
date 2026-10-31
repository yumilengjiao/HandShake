"""T16 tests train tiny synthetic fixtures, never download data or open devices."""

import json
import shutil
from dataclasses import replace

import numpy as np
import pytest
import torch

from algorithm.bundle import MEMBERS, digest, export_bundle, json_bytes, load_bundle
from algorithm.data.cache import write_cache
from algorithm.data.dataset import (
    ObservationRecord,
    SplitManifest,
    WindowDataset,
    records_digest,
)
from algorithm.deployment import main
from algorithm.online import OnlineClassifier
from algorithm.training.config import TrainConfig
from algorithm.training.trainer import train
from shared import PROTOCOL_VERSION
from shared.classes import IPN_HANDS_VERSION, vocabulary_for
from shared.contracts import FrameStatus
from shared.observations import FrameInfo, ObservationFrame
from tests.integration.test_feature_dataset import example_records, normalizer_for


@pytest.fixture(
    scope="module", params=[("mlp", PROTOCOL_VERSION), ("tcn", IPN_HANDS_VERSION)]
)
def trained(request, tmp_path_factory):
    kind, version = request.param
    root = tmp_path_factory.mktemp("model-bundle")
    vocabulary = vocabulary_for(version)
    records = [
        ObservationRecord.model_validate(
            row.model_dump()
            | {
                "vocabulary_version": version,
                "class_id": i % vocabulary.class_count,
            }
        )
        for i, row in enumerate(example_records(count=40))
    ]
    manifest = SplitManifest(
        dataset_version="synthetic-bundle-test",
        dataset_sha256=records_digest(records),
        synthetic=True,
        status="draft",
        preregistration_reference="synthetic-fixture",
        subject_splits={"A": "train", "B": "val", "C": "test"},
        vocabulary_version=version,
    )
    normalizer = normalizer_for(records, manifest)
    for split in ("train", "val"):
        write_cache(
            WindowDataset(records, manifest, split=split, normalizer=normalizer),
            root / split,
        )
    train(
        root / "train",
        root / "val",
        root / "run",
        config=TrainConfig(model_type=kind, max_epochs=2, batch_size=16),
    )
    receipt = export_bundle(
        root / "run", root / "train/stats.json", root / "bundle", allow_synthetic=True
    )
    return root, receipt, records, manifest


def loaded(trained):
    root, receipt, *_ = trained
    return load_bundle(
        root / "bundle", expected_manifest_sha256=receipt["manifest_sha256"]
    )


def frame_for(record, *, track=None, kind=None):
    obs = record.observation
    if track:
        obs = obs.model_copy(update={"track_id": track})
    info = FrameInfo(
        obs.stream_id,
        obs.source_frame_id,
        obs.source_timestamp_ns,
        640,
        480,
        obs.transform_metadata,
        obs.timebase,
    )
    if kind is not None:
        return ObservationFrame(
            info,
            FrameStatus(
                stream_id=info.stream_id,
                source_frame_id=info.frame_id,
                source_timestamp_ns=info.timestamp_ns,
                timebase=info.timebase,
                perception_status=kind,
                processed_at_ns=info.timestamp_ns if kind != "not_processed" else None,
            ),
        )
    return ObservationFrame(info, record.frame_status, (obs,))


def repin(root):
    receipt = json.loads((root / "checksums.json").read_text())
    receipt["files"] = {
        name: {
            "sha256": digest((root / name).read_bytes()),
            "bytes": (root / name).stat().st_size,
        }
        for name in MEMBERS
    }
    data = json_bytes(receipt)
    (root / "checksums.json").write_bytes(data)
    return digest(data)


def test_export_load_parity_and_no_pickle_at_deployment(trained, monkeypatch):
    root, receipt, *_ = trained
    original = torch.load(root / "run/best.pt", weights_only=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("deployment must never unpickle")

    monkeypatch.setattr(torch, "load", forbidden)
    bundle = loaded(trained)
    assert bundle.synthetic_training
    assert bundle.model_type == receipt["model_type"]
    for key, tensor in original["state_dict"].items():
        torch.testing.assert_close(
            bundle.model.state_dict()[key], tensor, rtol=0, atol=0
        )
    assert (root / "bundle/runtime_contract.json").is_file()
    architecture = json.loads((root / "bundle/architecture.json").read_text())
    assert architecture["heads"] == ["class"] and architecture["window_shape"] == [
        32,
        94,
    ]
    assert architecture["receptive_field"] == (1 if bundle.model_type == "mlp" else 31)
    assert not (root / "bundle/weights.pt").exists()


def test_export_requires_opt_in_and_never_overwrites(trained, tmp_path):
    root = trained[0]
    with pytest.raises(ValueError, match="synthetic"):
        export_bundle(root / "run", root / "train/stats.json", tmp_path / "new")
    assert not (tmp_path / "new").exists()
    with pytest.raises(FileExistsError):
        export_bundle(
            root / "run",
            root / "train/stats.json",
            root / "bundle",
            allow_synthetic=True,
        )


def test_explicit_external_manifest_pin_required(trained):
    root = trained[0]
    with pytest.raises(ValueError, match="external SHA256"):
        load_bundle(root / "bundle", expected_manifest_sha256="")
    with pytest.raises(ValueError, match="manifest SHA256"):
        load_bundle(root / "bundle", expected_manifest_sha256="0" * 64)


@pytest.mark.parametrize(
    "member", ["weights.npz", "classes.json", "normalization.json", "golden.npz"]
)
def test_member_tampering_fails_before_model_allocation(
    trained, tmp_path, member, monkeypatch
):
    root = tmp_path / "package"
    shutil.copytree(trained[0] / "bundle", root)
    with (root / member).open("ab") as stream:
        stream.write(b"tampered")
    monkeypatch.setattr(
        "algorithm.bundle.build_model",
        lambda *a, **k: pytest.fail("model was allocated"),
    )
    with pytest.raises(ValueError, match="hash/size"):
        load_bundle(root, expected_manifest_sha256=trained[1]["manifest_sha256"])


@pytest.mark.parametrize(
    "member",
    [
        "feature_schema.json",
        "classes.json",
        "runtime_contract.json",
        "calibration.json",
        "architecture.json",
    ],
)
def test_even_rehashed_semantic_mismatch_is_rejected(trained, tmp_path, member):
    root = tmp_path / "package"
    shutil.copytree(trained[0] / "bundle", root)
    data = json.loads((root / member).read_text())
    if member == "feature_schema.json":
        data["field_names"][0:2] = reversed(data["field_names"][0:2])
    elif member == "classes.json":
        data["classes"][0]["token"] = "unknown"
    elif member == "runtime_contract.json":
        data[next(iter(data))] = "0" * 64
    elif member == "calibration.json":
        data["status"] = "calibrated"
    else:
        data["dropout"] = 0.9
    (root / member).write_bytes(json_bytes(data))
    with pytest.raises(ValueError, match="mismatch"):
        load_bundle(root, expected_manifest_sha256=repin(root))


def test_missing_extra_and_symlink_files_fail(trained, tmp_path):
    root = tmp_path / "package"
    shutil.copytree(trained[0] / "bundle", root)
    (root / "unexpected.txt").write_text("not executable")
    with pytest.raises(ValueError, match="extra"):
        load_bundle(root, expected_manifest_sha256=trained[1]["manifest_sha256"])
    (root / "unexpected.txt").unlink()
    (root / "weights.npz").unlink()
    with pytest.raises(ValueError, match="missing"):
        load_bundle(root, expected_manifest_sha256=trained[1]["manifest_sha256"])
    (root / "weights.npz").symlink_to(trained[0] / "bundle/weights.npz")
    with pytest.raises(ValueError, match="symbolic"):
        load_bundle(root, expected_manifest_sha256=trained[1]["manifest_sha256"])


@pytest.mark.parametrize("problem", ["nan", "shape", "dtype", "object", "golden"])
def test_tensor_validation_after_hashes(trained, tmp_path, problem):
    root = tmp_path / "package"
    shutil.copytree(trained[0] / "bundle", root)
    target = root / ("golden.npz" if problem == "golden" else "weights.npz")
    with np.load(target, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    key = next(iter(arrays))
    if problem == "nan":
        arrays[key].flat[0] = np.nan
    elif problem == "shape":
        arrays[key] = np.zeros((999, 999), np.float32)
    elif problem == "dtype":
        arrays[key] = arrays[key].astype(np.float64)
    elif problem == "object":
        arrays[key] = np.array(["pickle disallowed"], dtype=object)
    else:
        arrays["logits"] += 10
    np.savez(target, **arrays)
    with pytest.raises(ValueError):
        load_bundle(root, expected_manifest_sha256=repin(root))


def test_checkpoint_and_normalizer_must_belong_to_same_training_run(trained, tmp_path):
    root = trained[0]
    run = tmp_path / "run"
    shutil.copytree(root / "run", run)
    with (run / "best.pt").open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="checkpoint hash"):
        export_bundle(
            run, root / "train/stats.json", tmp_path / "out", allow_synthetic=True
        )
    assert not (tmp_path / "out").exists()
    wrong = json.loads((root / "train/stats.json").read_text())
    wrong["stats"]["dataset_hash"] = "other-dataset"
    from shared.features.schema import canonical_hash

    wrong["content_hash"] = canonical_hash(
        {k: v for k, v in wrong.items() if k != "content_hash"}
    )
    (tmp_path / "stats.json").write_bytes(json_bytes(wrong))
    with pytest.raises(ValueError, match="dataset"):
        export_bundle(
            root / "run",
            tmp_path / "stats.json",
            tmp_path / "out",
            allow_synthetic=True,
        )


def test_online_offline_logits_and_ready_are_identical(trained):
    _root, _, records, manifest = trained
    bundle = loaded(trained)
    now = [0]
    online = OnlineClassifier(
        bundle, context_epoch=0, clock=lambda: now[0], timebase="synthetic"
    )
    dataset = WindowDataset(
        records, manifest, split="train", normalizer=bundle.normalizer
    )
    for index, row in enumerate(records[:40]):
        now[0] = row.observation.source_timestamp_ns
        status = online.update(frame_for(row))
        assert (status.state == "predicted") == dataset[index]["ready"]
        if status.state == "predicted":
            x = torch.from_numpy(dataset[index]["x"]).unsqueeze(0)
            with torch.inference_mode():
                reference = bundle.model(
                    x[:, -1] if bundle.model_type == "mlp" else x
                ).numpy()[0]
            np.testing.assert_allclose(status.logits, reference, rtol=1e-4, atol=1e-5)
            assert status.source_frame_id == str(row.frame_status.source_frame_id)
            assert not status.calibrated and status.synthetic_training
    repeated = online.update(frame_for(records[39]))
    assert repeated.reason == "out_of_order" and not repeated.logits


def test_no_hand_new_track_and_multi_hand_clear_prediction(trained):
    bundle = loaded(trained)
    records = trained[2]
    now = [0]
    online = OnlineClassifier(
        bundle, context_epoch=4, clock=lambda: now[0], timebase="synthetic"
    )
    for row in records[:33]:
        now[0] = row.observation.source_timestamp_ns
        status = online.update(frame_for(row))
    assert status.state == "predicted" and status.context_epoch == "4"
    now[0] = records[33].observation.source_timestamp_ns
    assert (
        online.update(frame_for(records[33], kind="processed_no_hand")).reason
        == "no_hand"
    )
    now[0] = records[34].observation.source_timestamp_ns
    assert online.update(frame_for(records[34], track="new")).state == "warming_up"
    now[0] = records[35].observation.source_timestamp_ns
    base = frame_for(records[35])
    multi = replace(
        base,
        observations=(
            base.observations[0],
            base.observations[0].model_copy(update={"track_id": "other"}),
        ),
    )
    status = online.update(multi)
    assert status.reason == "multiple_hands" and not status.logits
    assert status.window_length == 0


def test_stale_during_inference_and_cancelled_completion_never_publish_logits(
    trained, monkeypatch
):
    bundle, records = loaded(trained), trained[2]
    now = [0]
    cancelled = [False]
    online = OnlineClassifier(
        bundle,
        context_epoch=9,
        clock=lambda: now[0],
        timebase="synthetic",
        cancelled=lambda: cancelled[0],
    )
    for row in records[:31]:
        now[0] = row.observation.source_timestamp_ns
        online.update(frame_for(row))
    original = bundle.model.forward

    def slow(x):
        result = original(x)
        now[0] += 200_000_000
        return result

    monkeypatch.setattr(bundle.model, "forward", slow)
    now[0] = records[31].observation.source_timestamp_ns
    status = online.update(frame_for(records[31]))
    assert status.reason == "stale" and not status.logits
    cancelled[0] = True
    status = online.update(frame_for(records[32]))
    assert status.reason == "cancelled" and not status.logits
    # A stop arriving inside the model call must also invalidate its completion.
    cancelled[0] = False

    def cancel_during_forward(x):
        result = original(x)
        cancelled[0] = True
        return result

    monkeypatch.setattr(bundle.model, "forward", cancel_during_forward)
    online = OnlineClassifier(
        bundle,
        context_epoch=10,
        clock=lambda: now[0],
        timebase="synthetic",
        cancelled=lambda: cancelled[0],
    )
    for row in records[:32]:
        now[0] = row.observation.source_timestamp_ns
        status = online.update(frame_for(row))
    assert status.reason == "cancelled" and not status.logits


def test_camera_worker_uses_actual_model_at_source_cadence_without_device(trained):
    import time

    from backend.capture.camera import SyntheticSource
    from backend.perception.mediapipe_adapter import RawHand
    from backend.perception.preview import CameraPreviewWorker
    from backend.perception.synthetic import skeleton

    class ProceduralCamera(SyntheticSource):
        def read(self):
            frame = super().read()
            return replace(
                frame,
                info=replace(
                    frame.info, timestamp_ns=frame.received_at_ns, timebase="monotonic"
                ),
            )

    class Detector:
        def detect(self, rgb, timestamp_ms):
            return [RawHand(skeleton(), "right", 0.9)]

        def close(self):
            pass

    root, receipt, *_ = trained
    worker = CameraPreviewWorker(
        source_factory=lambda: ProceduralCamera(frames=600, fps=60, paced=True),
        detector_factory=Detector,
        model_package=root / "bundle",
        model_sha256=receipt["manifest_sha256"],
        context_epoch=12,
    )
    worker.start()
    packet = None
    deadline = time.monotonic() + 8
    try:
        while time.monotonic() < deadline:
            candidate = worker.take()
            if (
                candidate is not None
                and candidate.classification is not None
                and candidate.classification.state == "predicted"
            ):
                packet = candidate
                break
            time.sleep(0.01)
        assert packet is not None, worker.error
        assert packet.classification.context_epoch == "12"
        assert packet.classification.source_frame_id == str(packet.frame_id)
        assert packet.classification.source_timestamp_ns == str(packet.timestamp_ns)
        assert packet.classification.synthetic_training
        assert (
            len(packet.classification.logits)
            == vocabulary_for(receipt["vocabulary_version"]).class_count
        )
    finally:
        worker.request_stop()
        assert worker.done.wait(3)


def test_failed_bundle_never_opens_camera(tmp_path):
    from backend.perception.preview import CameraPreviewWorker

    worker = CameraPreviewWorker(
        model_package=tmp_path,
        model_sha256="0" * 64,
        detector_factory=lambda: pytest.fail(
            "perception opened before model validation"
        ),
        source_factory=lambda: pytest.fail("camera opened before model validation"),
    )
    worker.start()
    assert worker.done.wait(5)
    assert worker.model_failed and worker.error == "preview_error"


def test_inspect_and_benchmark_are_explicitly_not_accuracy(trained, capsys):
    root, receipt, *_ = trained
    common = ["--package", str(root / "bundle"), "--sha256", receipt["manifest_sha256"]]
    assert main(["inspect", *common]) == 0
    info = json.loads(capsys.readouterr().out)
    assert (
        info["synthetic_training"]
        and not info["calibrated"]
        and not info["actions_enabled"]
    )
    assert main(["benchmark", *common, "--iterations", "3"]) == 0
    benchmark = json.loads(capsys.readouterr().out)["benchmark"]
    assert benchmark["p95_ms"] >= benchmark["p50_ms"] >= 0
    assert "NOT camera-to-action" in benchmark["scope"]
