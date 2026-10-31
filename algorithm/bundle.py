"""T16 integrity-pinned, inference-only CPU bundles. Deployment never unpickles.

The SHA256 must come from a trusted export receipt, not be recomputed and trusted
from an arbitrary downloaded directory. Hashes authenticate bytes, not accuracy.
"""

from __future__ import annotations

import io
import json
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from algorithm.model_config import build_model, load_model_config
from shared.classes import vocabulary_for
from shared.features import FeatureNormalizer
from shared.features.schema import (
    canonical_hash,
    feature_schema_hash,
    load_feature_schema,
)

MAX_JSON = 1_048_576
MAX_WEIGHTS = 4_194_304
MAX_CHECKPOINT = 16_777_216
MEMBERS = frozenset(
    {
        "weights.npz",
        "architecture.json",
        "classes.json",
        "feature_schema.json",
        "normalization.json",
        "calibration.json",
        "event_thresholds.json",
        "training_manifest.json",
        "model_card.md",
        "golden.npz",
        "runtime_contract.json",
    }
)
RUNTIME_FILES = (
    "shared.features.geometry",
    "shared.features.normalization",
    "shared.features.window",
    "shared.features.schema",
    "algorithm.inference",
    "algorithm.models.mlp",
    "algorithm.models.tcn",
)


def runtime_contract():
    """Pin feature/model semantics as well as dimensions; paths never come from a bundle."""
    from importlib.resources import files

    return {
        module: digest(
            files(module.rsplit(".", 1)[0])
            .joinpath(module.rsplit(".", 1)[1] + ".py")
            .read_bytes()
        )
        for module in RUNTIME_FILES
    }


CALIBRATION = {"status": "unavailable", "temperature": None, "selection_split": None}
EVENT_POLICY = {"status": "unavailable", "actions_enabled": False}


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        + "\n"
    ).encode()


def strict_json(data: bytes):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def bad_constant(_):
        raise ValueError("nonfinite JSON number")

    return json.loads(data, object_pairs_hook=unique, parse_constant=bad_constant)


def read_bounded(path: Path, limit: int) -> bytes:
    if path.is_symlink():
        raise ValueError("symbolic links are not model artifacts")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("artifact must be a bounded regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("artifact grew beyond budget")
    return data


def _arrays_bytes(arrays):
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _arrays_load(data, shapes):
    """Validate each NPY header/shape before allocating arrays, never allow pickle."""
    if len(data) > MAX_WEIGHTS:
        raise ValueError("tensor archive exceeds budget")
    result = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        expected = {name + ".npy" for name in shapes}
        if (
            len(members) != len(expected)
            or {item.filename for item in members} != expected
        ):
            raise ValueError("tensor names/duplicates mismatch")
        if sum(item.file_size for item in members) > MAX_WEIGHTS:
            raise ValueError("expanded tensor budget exceeded")
        for item in members:
            if item.flag_bits & 1 or item.file_size > MAX_WEIGHTS:
                raise ValueError("encrypted/oversized tensor")
            raw = archive.read(item)
            stream = io.BytesIO(raw)
            if np.lib.format.read_magic(stream) != (1, 0):
                raise ValueError("unsupported NPY format")
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=4096
            )
            name = item.filename[:-4]
            if shape != shapes[name] or fortran or dtype != np.dtype("float32"):
                raise ValueError("tensor shape/dtype mismatch")
            if len(raw) - stream.tell() != int(np.prod(shape)) * 4:
                raise ValueError("tensor payload size mismatch")
            array = np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096)
            if not np.isfinite(array).all():
                raise ValueError("nonfinite tensor")
            result[name] = array
    return result


def architecture_for(kind, vocabulary):
    return load_model_config(kind).model_dump(mode="json") | {
        "vocabulary_version": vocabulary.vocabulary_version,
        "num_classes": vocabulary.class_count,
    }


def deployment_architecture(kind, vocabulary):
    return architecture_for(kind, vocabulary) | {
        "heads": ["class"],
        "dtype": "float32",
        "device": "cpu",
        "window_shape": [32, 94],
        "network_input_shape": [1, 94] if kind == "mlp" else [1, 32, 94],
        "receptive_field": 1 if kind == "mlp" else 31,
    }


def _normalizer_document(data, identity, vocabulary):
    normalizer = FeatureNormalizer.from_dict(
        data,
        expected_dataset_hash=identity["data"]["dataset_sha256"],
        expected_split_hash=identity["data"]["split_sha256"],
    )
    if (
        normalizer.vocabulary != vocabulary
        or normalizer.stats_hash != identity["data"]["stats_sha256"]
        or normalizer.config.config_hash != identity["data"]["config_sha256"]
    ):
        raise ValueError("normalization provenance mismatch")
    return normalizer


def _check_training(manifest):
    if (
        manifest.get("format_version") != "training-run-v1"
        or manifest.get("status") != "completed"
        or manifest.get("final_test_used") is not False
        or type(manifest.get("synthetic")) is not bool
    ):
        raise ValueError("requires a completed run without test-set selection")
    identity = manifest["identity"]
    if manifest["identity_sha256"] != canonical_hash(identity):
        raise ValueError("training identity hash mismatch")
    if identity["data"]["synthetic"] is not manifest["synthetic"]:
        raise ValueError("synthetic training provenance mismatch")
    if identity["config"]["shuffle_labels_diagnostic"] is not False:
        raise ValueError("label-shuffling diagnostics cannot be deployed")
    vocabulary = vocabulary_for(identity["data"]["vocabulary_version"])
    kind = identity["config"]["model_type"]
    if kind not in {"mlp", "tcn"} or identity["architecture"] != architecture_for(
        kind, vocabulary
    ):
        raise ValueError("unsupported training architecture")
    if (
        identity["schema_sha256"] != feature_schema_hash()
        or identity["classes_sha256"] != vocabulary.digest()
        or canonical_hash(identity["classes"]) != vocabulary.digest()
        or identity["data"]["classes_sha256"] != vocabulary.digest()
    ):
        raise ValueError("training schema/vocabulary mismatch")
    return identity, vocabulary, kind


@dataclass(frozen=True)
class LoadedBundle:
    model: object
    normalizer: FeatureNormalizer
    model_type: str
    model_version: str
    synthetic_training: bool
    manifest_sha256: str

    @property
    def vocabulary(self):
        return self.normalizer.vocabulary


def load_bundle(directory, *, expected_manifest_sha256: str) -> LoadedBundle:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("an external SHA256 pin is required")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("expected a regular model directory")
    receipt = read_bounded(root / "checksums.json", MAX_JSON)
    if digest(receipt) != expected_manifest_sha256:
        raise ValueError("model manifest SHA256 mismatch")
    manifest = strict_json(receipt)
    if (
        set(manifest) != {"format_version", "status", "files"}
        or manifest["format_version"] != "inference-bundle-v1"
        or manifest["status"] != "uncalibrated_preview_only"
        or set(manifest["files"]) != MEMBERS
    ):
        raise ValueError("unsupported model manifest")
    if {item.name for item in root.iterdir()} != MEMBERS | {"checksums.json"}:
        raise ValueError("missing or extra model files")
    contents = {}
    for name in sorted(MEMBERS):
        entry = manifest["files"][name]
        if (
            set(entry) != {"sha256", "bytes"}
            or type(entry["bytes"]) is not int
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            raise ValueError("invalid checksum entry")
        data = read_bounded(
            root / name, MAX_WEIGHTS if name.endswith(".npz") else MAX_JSON
        )
        if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
            raise ValueError("model member hash/size mismatch: " + name)
        contents[name] = data
    training = strict_json(contents["training_manifest.json"])
    identity, vocabulary, kind = _check_training(training)
    if (
        strict_json(contents["runtime_contract.json"]) != runtime_contract()
        or strict_json(contents["architecture.json"])
        != deployment_architecture(kind, vocabulary)
        or canonical_hash(strict_json(contents["classes.json"])) != vocabulary.digest()
        or strict_json(contents["feature_schema.json"]) != load_feature_schema()
        or strict_json(contents["calibration.json"]) != CALIBRATION
        or strict_json(contents["event_thresholds.json"]) != EVENT_POLICY
    ):
        raise ValueError("model class/schema/architecture/calibration mismatch")
    normalizer = _normalizer_document(
        strict_json(contents["normalization.json"]), identity, vocabulary
    )
    # No torch import or model allocation until the entire package is verified.
    import torch

    model = build_model(kind, vocabulary=vocabulary).cpu().eval()
    state = _arrays_load(
        contents["weights.npz"],
        {name: tuple(t.shape) for name, t in model.state_dict().items()},
    )
    model.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in state.items()},
        strict=True,
    )
    golden = _arrays_load(
        contents["golden.npz"],
        {"x": (2, 32, 94), "logits": (2, vocabulary.class_count)},
    )
    with torch.inference_mode():
        x = torch.from_numpy(golden["x"])
        actual = model(x[:, -1] if kind == "mlp" else x).numpy()
    if not np.allclose(actual, golden["logits"], atol=1e-5, rtol=1e-4):
        raise ValueError("golden logits mismatch")
    return LoadedBundle(
        model,
        normalizer,
        kind,
        "sha256:" + expected_manifest_sha256,
        training["synthetic"],
        expected_manifest_sha256,
    )


def export_bundle(run_directory, normalization, output, *, allow_synthetic=False):
    """Only export trusted local trainer output; deployment contains tensor data only."""
    root, destination = Path(run_directory), Path(output)
    if destination.exists():
        raise FileExistsError(
            "model bundles are immutable; select a new output directory"
        )
    run = strict_json(read_bounded(root / "run_manifest.json", MAX_JSON))
    identity, vocabulary, kind = _check_training(run)
    from algorithm.training.trainer import implementation_hash

    if identity["code_sha256"] != implementation_hash():
        raise ValueError("export must use the same implementation as training")
    if run["synthetic"] and not allow_synthetic:
        raise ValueError("synthetic training requires explicit allow_synthetic")
    stats_data = strict_json(read_bounded(Path(normalization), MAX_JSON))
    _normalizer_document(stats_data, identity, vocabulary)
    checkpoint_bytes = read_bounded(root / "best.pt", MAX_CHECKPOINT)
    if digest(checkpoint_bytes) != run["artifacts"]["best.pt"]:
        raise ValueError("best checkpoint hash mismatch")
    import torch

    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
    )
    if (
        checkpoint["format_version"] != "trainer-best-v1"
        or checkpoint["run_id"] != run["run_id"]
        or checkpoint["identity"] != identity
        or checkpoint["epoch"] != run["best_epoch"]
        or checkpoint["val_macro_f1"] != run["best_val_macro_f1"]
    ):
        raise ValueError("best checkpoint/training manifest mismatch")
    model = build_model(kind, vocabulary=vocabulary).cpu().eval()
    if any(
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
        or not torch.isfinite(tensor).all()
        for tensor in checkpoint["state_dict"].values()
    ):
        raise ValueError("checkpoint must contain finite float32 tensors")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
    }
    if any(
        value.dtype != np.float32 or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise ValueError("deployment parameters must be finite float32")
    # Fixed diagnostic inputs, not dataset examples or accuracy measurements.
    x = np.random.default_rng(16).normal(0, 0.1, (2, 32, 94)).astype(np.float32)
    x[:, :, 69:73] = [0.33, 0, 1, 1]
    x[:, :, 73:] = 1
    with torch.inference_mode():
        logits = model(torch.from_numpy(x[:, -1] if kind == "mlp" else x)).numpy()
    if not np.isfinite(logits).all():
        raise ValueError("export produced nonfinite golden logits")
    contents = {
        "weights.npz": _arrays_bytes(arrays),
        "runtime_contract.json": json_bytes(runtime_contract()),
        "architecture.json": json_bytes(deployment_architecture(kind, vocabulary)),
        "classes.json": json_bytes(vocabulary.to_dict()),
        "feature_schema.json": json_bytes(load_feature_schema()),
        "normalization.json": json_bytes(stats_data),
        "calibration.json": json_bytes(CALIBRATION),
        "event_thresholds.json": json_bytes(EVENT_POLICY),
        "training_manifest.json": json_bytes(run),
        "golden.npz": _arrays_bytes({"x": x, "logits": logits}),
        "model_card.md": (
            f"# {kind.upper()} / {vocabulary.vocabulary_version}\n\n"
            f"Synthetic training: {run['synthetic']}\n\n"
            "UNVALIDATED / UNCALIBRATED / PREVIEW ONLY. No action authority.\n"
            "Class head only; no intent or phase probabilities.\n"
            "Golden tensors verify numerical consistency, not recognition accuracy.\n"
            f"Training run: {run['run_id']}; selected validation epoch: {run['best_epoch']}.\n"
            "Mobile conversion, event evaluation and real-world performance not verified.\n"
        ).encode(),
    }
    receipt = json_bytes(
        {
            "format_version": "inference-bundle-v1",
            "status": "uncalibrated_preview_only",
            "files": {
                name: {"sha256": digest(data), "bytes": len(data)}
                for name, data in contents.items()
            },
        }
    )
    destination.mkdir(parents=True, exist_ok=False)
    # Receipt is written last. Any interrupted export remains unloadable.
    for name, data in contents.items():
        with (destination / name).open("xb") as stream:
            stream.write(data)
    with (destination / "checksums.json").open("xb") as stream:
        stream.write(receipt)
    return {
        "format_version": "model-export-receipt-v1",
        "manifest_sha256": digest(receipt),
        "synthetic_training": run["synthetic"],
        "model_type": kind,
        "vocabulary_version": vocabulary.vocabulary_version,
        "status": "uncalibrated_preview_only",
    }
