from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError

from shared.config import ProjectConfig, load_config


def test_packaged_and_repo_defaults_match(config):
    root = Path(__file__).resolve().parents[2]
    assert load_config(root / "configs") == config
    assert config.runtime.mode == "dry_run"
    assert not config.runtime.enabled_on_start
    assert not config.platform.live_supported
    assert len(config.digest()) == 64


@pytest.mark.parametrize("section,field,bad", [
    ("runtime", "command_ttl_ns", -1), ("runtime", "enabled_on_start", True),
    ("runtime", "mode", "live"), ("runtime", "max_hands", 3),
    ("runtime", "target_context", "arbitrary-app"),
    ("interaction", "wake_hold_ns", "600ms"), ("interaction", "cooldown_ns", -1),
    ("interaction", "cooldown_ns", True), ("gestures", "class_count", 6),
    ("training", "feature_schema_version", "changed"),
    ("privacy", "consent_required", False), ("runtime", "unknown_option", 1),
])
def test_bad_configuration_is_rejected(config, section, field, bad):
    data = config.model_dump(mode="json")
    data[section][field] = bad
    with pytest.raises(ValidationError):
        ProjectConfig(**data)


@pytest.mark.parametrize("kind", ["unknown_action", "missing_mapping", "wake_action", "capability", "gesture_id", "duplicate_allowlist"])
def test_cross_file_consistency(config, kind):
    data = config.model_dump(mode="json")
    if kind == "unknown_action":
        data["actions"]["mappings"]["presentation"]["swipe_left"] = "shutdown"
    elif kind == "missing_mapping":
        del data["actions"]["mappings"]["presentation"]["swipe_left"]
    elif kind == "wake_action":
        data["actions"]["mappings"]["presentation"]["wake_open"] = "next_slide"
    elif kind == "capability":
        data["platform"]["capabilities"] = []
    elif kind == "gesture_id":
        data["gestures"]["gestures"][2]["id"] = 3
    else:
        data["actions"]["allowlist"].append("next_slide")
    with pytest.raises(ValidationError):
        ProjectConfig(**data)


def test_yaml_duplicate_keys_fail(tmp_path):
    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root / "configs", tmp_path / "config")
    path = tmp_path / "config/runtime.yaml"
    path.write_text(path.read_text() + "\nmode: live\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_config(tmp_path / "config")
