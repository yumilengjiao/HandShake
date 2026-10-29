from pathlib import Path

import pytest
import yaml

from algorithm.model_config import build_model, load_model_config


@pytest.mark.parametrize("kind,parameters", [("mlp", 20741), ("tcn", 56325)])
def test_packaged_model_config_and_repo_links_agree(kind, parameters):
    root = Path(__file__).resolve().parents[2]
    config = load_model_config(kind)
    assert config == load_model_config(kind, root / f"configs/training/{kind}.yaml")
    assert config.status == "draft"
    torch = pytest.importorskip("torch")
    assert sum(p.numel() for p in build_model(kind).parameters()) == parameters


@pytest.mark.parametrize("kind,changes", [
    ("mlp", {"hidden_dims": [64, 32]}), ("mlp", {"dropout": float("nan")}),
    ("tcn", {"dilations": [1, 2, 4, 4]}), ("tcn", {"input_dim": 93}),
    ("tcn", {"model_type": "mlp"}), ("mlp", {"unknown": 1}),
])
def test_incompatible_or_malformed_model_config_fails(tmp_path, kind, changes):
    data = load_model_config(kind).model_dump(mode="json") | changes
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_model_config(kind, path)
