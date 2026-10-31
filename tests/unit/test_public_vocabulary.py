import json
from importlib.resources import files
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from shared.classes import IPN_HANDS_VERSION, load_classes, vocabulary_for
from shared.classification import SourceClassification
from shared.contracts import FrameStatus, ModelPrediction


def test_packaged_public_vocabulary_matches_registry_without_changing_legacy():
    public = load_classes(files("shared").joinpath("defaults/classes_ipn_hands.json"))
    assert public == vocabulary_for(IPN_HANDS_VERSION)
    assert (
        public.class_count == 15
        and public.id_for("TLT") == 7
        and public.id_for("TRT") == 8
    )
    assert load_classes().class_count == 5 and load_classes().id_for("swipe_left") == 2
    assert public.digest() != load_classes().digest()


def test_imported_processing_time_is_explicitly_unavailable_not_invented():
    values = dict(
        stream_id="s",
        source_frame_id=1,
        source_timestamp_ns=0,
        perception_status="processed_no_hand",
    )
    with pytest.raises(ValidationError, match="require processed_at_ns"):
        FrameStatus(**values)
    status = FrameStatus(**values, processing_time_origin="unavailable_external")
    assert status.processed_at_ns is None
    with pytest.raises(ValidationError, match="do not invent"):
        FrameStatus(
            **values, processing_time_origin="unavailable_external", processed_at_ns=0
        )


def source_prediction():
    return SourceClassification(
        stream_id="s",
        track_id="h",
        context_epoch=0,
        source_frame_id=32,
        source_timestamp_ns=1000,
        inference_timestamp_ns=1001,
        model_version="synthetic-public",
        vocabulary_version=IPN_HANDS_VERSION,
        logits=[0.0] * 15,
    )


def test_public_predictions_cannot_enter_legacy_actions():
    from backend.interaction.engine import Evidence
    from backend.interaction.evidence import evidence_from_frame

    prediction = source_prediction()
    assert not isinstance(prediction, ModelPrediction)
    with pytest.raises(ValueError, match="explicit application policy"):
        Evidence(prediction, True, True, ("h",))
    # Fail before reading any camera-frame data or interpreting the first five logits.
    with pytest.raises(ValueError, match="not a legacy"):
        evidence_from_frame(prediction, SimpleNamespace(), ready=True)
    payload = prediction.model_dump()
    payload["logits"] = [0.0] * 5
    with pytest.raises(ValidationError, match="exact source vocabulary"):
        SourceClassification.model_validate(payload)


@pytest.mark.parametrize("kind", ["mlp", "tcn"])
def test_public_torch_predictor_returns_raw_source_scores_not_legacy_gestures(kind):
    pytest.importorskip("torch")
    from algorithm.inference import TorchPredictor
    from algorithm.model_config import build_model
    from shared.contracts import FeatureWindow

    vocabulary = vocabulary_for(IPN_HANDS_VERSION)
    model = build_model(kind, vocabulary=vocabulary)
    row = [0.0] * 94
    row[71] = 1.0
    row[73:] = [1.0] * 21
    window = FeatureWindow(
        stream_id="s",
        track_id="h",
        context_epoch=0,
        source_frame_id=32,
        timestamps_ns=list(range(1, 33)),
        features=[row] * 32,
        ready=True,
        ready_reason="fixture",
    )
    predictor = TorchPredictor(
        model,
        vocabulary=vocabulary,
        model_version="synthetic-public",
        clock_ns=lambda: 33,
    )
    result = predictor.predict(window)
    assert isinstance(result, SourceClassification) and len(result.logits) == 15
    assert result.source_timestamp_ns == 32 and result.inference_timestamp_ns == 33
    assert not hasattr(result, "intent_score") and not hasattr(
        result, "calibrated_probabilities"
    )


def test_legacy_record_hash_does_not_change_due_to_new_default_fields():
    from algorithm.data.dataset import (
        ObservationRecord,
        canonical_digest,
        records_digest,
    )

    record = ObservationRecord(
        recording_id="r",
        subject_id="p",
        session_id="s",
        lineage_id="r",
        original_sha256="0" * 64,
        visible_subjects=["p"],
        track_id="h",
        context_epoch=0,
        image_size=[640, 480],
        frame_status=FrameStatus(
            stream_id="s",
            source_frame_id=1,
            source_timestamp_ns=0,
            perception_status="processed_no_hand",
            processed_at_ns=1,
        ),
        observation=None,
        class_id=0,
        class_valid=True,
    )
    original = record.model_dump(mode="json")
    original.pop("vocabulary_version")
    original["frame_status"].pop("processing_time_origin")
    assert records_digest([record]) == canonical_digest([original])


def test_public_flip_requires_separate_verified_semantics():
    from algorithm.data.augmentation import augment_training_records
    from algorithm.data.dataset import ObservationRecord

    record = ObservationRecord(
        recording_id="r",
        subject_id="p",
        session_id="s",
        lineage_id="r",
        original_sha256="0" * 64,
        visible_subjects=["p"],
        track_id="h",
        context_epoch=0,
        image_size=[640, 480],
        frame_status=FrameStatus(
            stream_id="s",
            source_frame_id=1,
            source_timestamp_ns=0,
            perception_status="processed_no_hand",
            processed_at_ns=1,
        ),
        observation=None,
        class_id=14,
        class_valid=True,
        vocabulary_version=IPN_HANDS_VERSION,
    )
    with pytest.raises(ValueError, match="flip semantics"):
        augment_training_records(
            [record], source_split="train", seed=1, horizontal_flip=True
        )
    assert (
        augment_training_records([record], source_split="train", seed=1)[0].class_id
        == 14
    )


def test_data_cli_hash_failure_keeps_report_and_refuses_overwrite(tmp_path, capsys):
    from backend.importing.commands import main

    catalog = {
        "schema_version": "public-dataset-sources-v1",
        "datasets": [
            {
                "id": "ipn-hands",
                "license": {"id": "CC-BY-4.0"},
                "files": [
                    {
                        "filename": "IPN_HandS_annotations.zip",
                        "expected_sha256": "0" * 64,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(catalog))
    ann = tmp_path / "bad.zip"
    ann.write_bytes(b"not a zip")
    report = tmp_path / "report.json"
    args = [
        "audit-ipn",
        "--catalog",
        str(path),
        "--annotations",
        str(ann),
        "--report",
        str(report),
    ]
    assert main(args) == 2
    result = json.loads(report.read_text())
    assert result["status"] == "blocked" and "SHA256 mismatch" in result["error"]
    assert not result["ready_for_training"]
    assert main(args) == 2 and json.loads(report.read_text()) == result
    assert "report error" in capsys.readouterr().err
