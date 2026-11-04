"""Real compact loader and CLI wiring on tiny synthetic data; no model forward."""

import json

import numpy as np
import pytest

from algorithm.data.cache import load_cache
from algorithm.evaluation.cli import main
from algorithm.evaluation.scores import (
    ScoreProvenance,
    endpoint_digest,
    score_dtype,
    write_scores,
)
from backend.importing.stream_features import build_stream_cache
from shared.features import FeatureStatisticsWarning
from tests.integration.test_public_data import public_sources as source_fixture
from tests.integration.test_public_data import run_import

pytestmark = pytest.mark.filterwarnings(
    f"ignore::{FeatureStatisticsWarning.__module__}.{FeatureStatisticsWarning.__name__}"
)


@pytest.fixture
def sources(tmp_path):
    return source_fixture.__wrapped__(tmp_path)


def test_history_cli_pins_real_compact_metadata_and_rejects_tampering(
    sources, tmp_path
):
    imported = tmp_path / "imported"
    run_import(sources, imported)
    options = {
        "dataset_version": "synthetic-history-v1",
        "review_reference": "synthetic-only",
    }
    build_stream_cache(imported, tmp_path / "train", split="train", **options)
    cache = tmp_path / "val"
    build_stream_cache(
        imported, cache, split="val", stats=tmp_path / "train/stats.json", **options
    )
    header = json.loads((cache / "index.json").read_text())
    arrays, index = load_cache(
        cache,
        expected_dataset_sha256=header["dataset_sha256"],
        expected_split_sha256=header["split_sha256"],
        expected_stats_sha256=header["stats_sha256"],
        expected_config_sha256=header["config_sha256"],
        expected_split="val",
    )
    source = ScoreProvenance(
        **{
            key: index[key]
            for key in (
                "dataset_sha256",
                "split_sha256",
                "stats_sha256",
                "config_sha256",
                "vocabulary_version",
                "split",
            )
        },
        model_sha256="a" * 64,
        cache_index_sha256=index["index_sha256"],
        endpoint_sha256=endpoint_digest(index),
        data_synthetic=True,
        model_synthetic=True,
    )
    rows = np.zeros(len(arrays["y"]), dtype=score_dtype(15))
    for score_field, cache_field in (
        ("label", "y"),
        ("class_valid", "class_valid"),
        ("ready", "ready"),
    ):
        rows[score_field] = arrays[cache_field]
    rows["available"] = 1
    rows["logits"][:, 0] = 9
    receipt = write_scores(tmp_path / "scores", source, [rows])
    output = tmp_path / "history.json"
    args = [
        "diagnose",
        "--scores",
        str(tmp_path / "scores"),
        "--sha256",
        receipt["manifest_sha256"],
        "--cache",
        str(cache),
        "--allow-synthetic",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    report = json.loads(output.read_text())
    parts = report["history_diagnostics"]
    assert parts["cache_index_sha256"] == index["index_sha256"]
    keys = ("short_history", "full_not_ready", "full_ready")
    assert (
        sum(parts[k]["labeled_endpoints"] for k in keys)
        == report["primary"]["labeled_endpoints"]
    )
    assert parts["short_history"]["labeled_endpoints"] > 0
    assert report["actions_enabled"] is False
    with (cache / "endpoints.bin").open("ab") as stream:
        stream.write(b"tampered")
    args[-1] = str(tmp_path / "rejected.json")
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert not (tmp_path / "rejected.json").exists()
