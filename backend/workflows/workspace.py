"""Workspace discovery, strict defaults and small atomic workflow receipts."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkflowError(Exception):
    def __init__(self, message, hint="", code=2):
        super().__init__(message)
        self.hint, self.code = hint, code


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RuntimeSettings(SettingsModel):
    profile: Literal["cpu", "cuda"] = "cpu"


class DataSettings(SettingsModel):
    catalog: str = "data/sources.json"
    raw: str = "data/raw/ipn-hands"
    subjects: str = "data/metadata/ipn-hands-subjects.json"
    split_mode: Literal["release", "reviewed_subjects"] = "release"
    imported: str = "data/processed/ipn-hands"
    cache: str = "data/cache/ipn-hands"
    dataset_version: str = "ipn-hands-frames-v1"
    review_reference: str = ""
    stride: int = Field(default=1, ge=1, le=1024)


class TrainingSettings(SettingsModel):
    model: Literal["mlp", "tcn"] = "tcn"
    device: str = Field(default="auto", pattern=r"^(auto|cpu|cuda(:[0-9]+)?|mps)$")
    batch_size: int = Field(default=64, ge=1, le=1024)
    epochs: int = Field(default=80, ge=1)
    patience: int = Field(default=10, ge=1)
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.0001, ge=0, allow_inf_nan=False)
    grad_clip_norm: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    min_delta: float = Field(default=0.0001, ge=0, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, lt=2**32)
    threads: int = Field(default=1, ge=1, le=64)
    runs: str = "artifacts/runs"
    auto_export: bool = True


class AppSettings(SettingsModel):
    flutter: str = "flutter"
    camera: int = Field(default=0, ge=0)
    perception_model: str = "artifacts/perception/hand_landmarker.task"


class Settings(SettingsModel):
    version: Literal[1] = 1

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("version must be an integer")
        return value

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    train: TrainingSettings = Field(default_factory=TrainingSettings)
    app: AppSettings = Field(default_factory=AppSettings)


def _toml(path):
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _merge(base, changes):
    result = dict(base)
    for key, value in changes.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


@dataclass(frozen=True)
class Workspace:
    root: Path
    settings: Settings

    def path(self, value):
        path = Path(value).expanduser()
        return (
            (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        )

    @property
    def cache(self):
        return self.path(self.settings.data.cache)

    @property
    def runs(self):
        return self.path(self.settings.train.runs)

    @property
    def latest(self):
        return self.runs / "latest.json"


def workspace(path=None):
    supplied = path or os.environ.get("HANDSHAKE_WORKSPACE")
    if supplied:
        root = Path(supplied).expanduser().resolve()
    else:
        start = Path.cwd().resolve()
        root = next(
            (p for p in (start, *start.parents) if (p / "handshake.toml").is_file()),
            start,
        )
    if not root.is_dir():
        raise WorkflowError(
            f"工作目录不存在：{root}", "进入项目目录，或设置 HANDSHAKE_WORKSPACE。"
        )
    config = {}
    for name in ("handshake.toml", "handshake.local.toml"):
        if (root / name).is_file():
            config = _merge(config, _toml(root / name))
    return Workspace(root, Settings.model_validate(config))


def read_json(path, limit=16 * 1024**2):
    from algorithm.data.source_archive import strict_json

    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise WorkflowError(f"文件超出读取预算：{path}")
    return strict_json(raw)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def operation_lock(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".{name}.lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise WorkflowError(
            f"已有 {name} 操作，或上次进程异常结束：{path}",
            "确认没有同类进程运行后，才可删除这个锁文件再重试。",
        ) from None
    try:
        with stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        path.unlink(missing_ok=True)
