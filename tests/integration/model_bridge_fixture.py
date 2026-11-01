"""UNTRAINED analytical MLP/TCN fixtures. Assign known weights; never optimize them."""

import json

import numpy as np
import torch

from algorithm.bundle import MEMBERS, _arrays_bytes, digest, json_bytes, load_bundle
from algorithm.data.dataset import canonical_digest
from algorithm.evaluation.calibration import fit_calibration, save_calibration
from algorithm.evaluation.scores import (
    ScoreProvenance,
    load_scores,
    score_dtype,
    write_scores,
)
from algorithm.model_config import build_model
from shared.classes import IPN_HANDS_VERSION, vocabulary_for
from shared.features import FeatureNormalizer
from shared.features.normalization import FeatureStats
from shared.features.schema import FIELD_NAMES, VX, FeatureConfig, feature_schema_hash
from tests.integration.test_evaluation import zero_bundle


def fixture_bundle(root, kind):
    config = FeatureConfig()
    vocabulary = vocabulary_for(IPN_HANDS_VERSION)
    stats = FeatureStats(
        np.zeros(69),
        np.ones(69),
        np.zeros(69, dtype=np.int64),
        digest(b"untrained-bridge-dataset"),
        digest(b"untrained-bridge-split"),
        feature_schema_hash(),
        config.config_hash,
        tuple(FIELD_NAMES[:69]),
    )
    normalizer = FeatureNormalizer(stats, config, vocabulary=vocabulary)
    index = dict(
        dataset_sha256=stats.dataset_hash,
        split_sha256=stats.split_hash,
        stats_sha256=normalizer.stats_hash,
        config_sha256=config.config_hash,
        synthetic=True,
        vocabulary_version=IPN_HANDS_VERSION,
        classes_sha256=vocabulary.digest(),
        index_sha256=digest(b"bridge-fixture-index"),
    )
    package = root / "bundle"
    zero_bundle(package, kind, normalizer, index)
    with np.load(package / "weights.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    first = "network.0" if kind == "mlp" else "input_projection"
    last = "network.6" if kind == "mlp" else "classifier"
    arrays[first + ".weight"][0, VX] = 1
    arrays[first + ".weight"][1, VX] = -1
    arrays[first + ".bias"][:2] = -0.2
    if kind == "mlp":
        arrays["network.3.weight"][0, 0] = arrays["network.3.weight"][1, 1] = 1
    # Raw x-speed -> known test logits. This is NOT a learned IPN gesture model.
    arrays[last + ".bias"][0] = 10
    arrays[last + ".weight"][7, 1] = arrays[last + ".weight"][8, 0] = 30
    model = build_model(kind, vocabulary=vocabulary).eval()
    model.load_state_dict({k: torch.from_numpy(v) for k, v in arrays.items()})
    golden = np.zeros((2, 32, 94), dtype=np.float32)
    with torch.inference_mode():
        golden_logits = model(
            torch.from_numpy(golden[:, -1] if kind == "mlp" else golden)
        ).numpy()
    (package / "weights.npz").write_bytes(_arrays_bytes(arrays))
    (package / "golden.npz").write_bytes(
        _arrays_bytes({"x": golden, "logits": golden_logits})
    )
    training = json.loads((package / "training_manifest.json").read_text())
    endpoint = digest(b"bridge-fixture-endpoints")
    training["identity"]["data"]["val"] = {"endpoint_sha256": endpoint}
    training["identity_sha256"] = canonical_digest(training["identity"])
    training["fixture_warning"] = (
        "UNTRAINED hand-assigned velocity fixture. Identity normalization is synthetic; no real training occurred."
    )
    (package / "training_manifest.json").write_bytes(json_bytes(training))
    (package / "model_card.md").write_text(training["fixture_warning"] + "\n")
    checks = json.loads((package / "checksums.json").read_text())
    checks["files"] = {
        name: {
            "sha256": digest((package / name).read_bytes()),
            "bytes": (package / name).stat().st_size,
        }
        for name in MEMBERS
    }
    raw = json_bytes(checks)
    (package / "checksums.json").write_bytes(raw)
    sha = digest(raw)
    bundle = load_bundle(package, expected_manifest_sha256=sha)
    samples = np.zeros((3, 32, 94), dtype=np.float32)
    samples[:, :, VX] = np.array([0, -1.2, 1.2])[:, None]
    with torch.inference_mode():
        logits = bundle.model(
            torch.from_numpy(samples[:, -1] if kind == "mlp" else samples)
        ).numpy()
    rows = np.zeros(3, dtype=score_dtype(15))
    rows["label"] = [0, 7, 8]
    rows["class_valid"] = rows["ready"] = rows["available"] = 1
    rows["logits"] = logits
    source = ScoreProvenance(
        model_sha256=sha,
        dataset_sha256=stats.dataset_hash,
        split_sha256=stats.split_hash,
        stats_sha256=normalizer.stats_hash,
        config_sha256=config.config_hash,
        cache_index_sha256=index["index_sha256"],
        endpoint_sha256=endpoint,
        vocabulary_version=IPN_HANDS_VERSION,
        split="val",
        data_synthetic=True,
        model_synthetic=True,
    )
    receipt = write_scores(root / "scores", source, [rows])
    scores = load_scores(
        root / "scores", expected_manifest_sha256=receipt["manifest_sha256"]
    )
    calibration = fit_calibration(
        scores,
        allow_synthetic=True,
        temperatures=(1.0,),
        thresholds=(0.8,),
        max_risk=0.0,
        min_coverage=1.0,
        min_accepted=3,
    )
    calibration_sha = save_calibration(calibration, root / "calibration.json")
    return bundle, calibration, calibration_sha
