"""One reviewed dataset workspace, reusing immutable import and compact cache v3."""

from __future__ import annotations

from algorithm.data.dataset import canonical_digest
from algorithm.data.ipn_hands import read_annotations
from algorithm.data.source_archive import VerifiedArchive
from backend.importing.ipn_hands import import_ipn_hands, validate_subject_plan
from backend.importing.spool import read_manifest
from backend.importing.stream_features import build_stream_cache
from backend.workflows.download import ANNOTATIONS, LANDMARKS, catalog, check_archive
from backend.workflows.workspace import (
    WorkflowError,
    operation_lock,
    read_json,
    write_json,
)


def dataset_receipt(ws, *, allow_synthetic=False, verify_splits=("train", "val")):
    from algorithm.data.compact import inspect_compact_cache

    path = ws.cache / "dataset.json"
    if not path.is_file():
        raise WorkflowError(
            f"训练数据尚未准备好：{path}",
            "先运行 uv run download，再运行 uv run prepare。",
        )
    value = read_json(path)
    expected = value.get("receipt_sha256")
    if (
        value.get("format_version") != "workflow-dataset-v1"
        or canonical_digest({k: v for k, v in value.items() if k != "receipt_sha256"})
        != expected
    ):
        raise WorkflowError(
            "数据集回执损坏或版本不兼容。",
            "恢复原始回执，或在新的缓存目录重新运行 uv run prepare。",
        )
    if value.get("synthetic") is not False and not allow_synthetic:
        raise WorkflowError(
            "默认训练入口不接受合成测试集。",
            "请准备真实、已审核的数据；不要修改 synthetic 字段冒充正式数据。",
        )
    if set(value.get("splits", {})) != {"train", "val", "test"}:
        raise WorkflowError("数据集缺少 train/val/test 的完整回执。")
    if (
        type(value.get("synthetic")) is not bool
        or value.get("preparation", {}).get("synthetic") is not value["synthetic"]
    ):
        raise WorkflowError("数据集的合成来源标记不一致。")
    for split in verify_splits:
        result = inspect_compact_cache(
            ws.cache / split, expected_index_sha256=value["splits"][split]
        )
        if result["split"] != split or result["synthetic"] is not value["synthetic"]:
            raise WorkflowError("缓存角色或来源与数据回执不一致。")
    return value


def _subject_plan(ws, annotations, expected_videos):
    with VerifiedArchive(
        annotations[0],
        expected_sha256=annotations[1],
        max_member_bytes=2 * 1024**2,
        max_total_bytes=32 * 1024**2,
    ) as archive:
        videos = read_annotations(archive, expected_videos=expected_videos)
    path = ws.path(ws.settings.data.subjects)
    if not path.exists():
        write_json(
            path,
            {
                "format_version": "subject-plan-v1",
                "status": "draft",
                "reference": "",
                "subject_splits": {},
                "videos": {
                    name: {"subject_id": "", "session_id": ""}
                    for name in sorted(videos)
                },
            },
        )
        write_json(
            path.with_name(path.stem + "-source-splits.json"),
            {
                "note": "Source video split is not evidence of participant identity. Never guess identities from filenames.",
                "videos": {
                    name: item.source_split for name, item in sorted(videos.items())
                },
            },
        )
        raise WorkflowError(
            f"已生成主体/会话分组模板：{path}",
            "根据真实来源资料填写身份和 train/val/test；保留官方 test 隔离，审核后标记 reviewed 并填写 reference。说明：docs/quickstart.md。",
        )
    try:
        plan = validate_subject_plan(read_json(path), videos)
    except ValueError as error:
        raise WorkflowError(
            f"主体划分尚未通过审核：{error}",
            f"请检查 {path}；不允许随机逐帧划分或虚构主体身份。",
        ) from error
    return path, plan


def prepare(ws, emit=print, *, expected_videos=200, synthetic=False):
    cfg = ws.settings.data
    entries = catalog(ws)
    raw = ws.path(cfg.raw)
    for name in (ANNOTATIONS, LANDMARKS):
        check_archive(raw / name, entries[name]["expected_sha256"])
    subject_path, plan = _subject_plan(
        ws,
        (raw / ANNOTATIONS, entries[ANNOTATIONS]["expected_sha256"]),
        expected_videos,
    )
    if not cfg.review_reference.strip():
        raise WorkflowError(
            "尚未填写数据质量/划分审核依据。",
            "在 handshake.local.toml 的 [data] 中填写真实 review_reference，然后再运行 uv run prepare。",
        )
    identity = {
        "annotations_sha256": entries[ANNOTATIONS]["expected_sha256"],
        "landmarks_sha256": entries[LANDMARKS]["expected_sha256"],
        "subject_plan_sha256": canonical_digest(plan),
        "dataset_version": cfg.dataset_version,
        "review_reference": cfg.review_reference,
        "stride": cfg.stride,
        "synthetic": synthetic,
    }
    with operation_lock(ws.cache, "prepare"):
        if (ws.cache / "dataset.json").exists():
            previous = dataset_receipt(
                ws, allow_synthetic=synthetic, verify_splits=("train", "val", "test")
            )
            if previous["preparation"] != identity:
                raise WorkflowError(
                    "当前输入/配置与已有缓存不一致，不会覆盖。",
                    "在配置中指定新的 imported/cache 目录，保留已有实验的数据版本。",
                )
            emit("三个缓存及数据回执校验通过，无需重新构建。下一步：uv run train")
            return previous
        imported = ws.path(cfg.imported)
        if imported.exists():
            manifest = read_manifest(imported)
            for key in (
                "annotations_sha256",
                "landmarks_sha256",
                "subject_plan_sha256",
                "synthetic",
            ):
                if manifest.get(key) != identity[key]:
                    raise WorkflowError(
                        "已有导入产物与本次来源/划分不一致。",
                        "选择新的 imported/cache 目录；不能覆盖正在使用的数据。",
                    )
            emit("复用已完成的导入；构建阶段仍会校验每个分片。")
        else:
            emit("导入关键点和连续时间标签（不打开相机）…")
            import_ipn_hands(
                raw / ANNOTATIONS,
                raw / LANDMARKS,
                subject_path,
                imported,
                annotations_sha256=identity["annotations_sha256"],
                landmarks_sha256=identity["landmarks_sha256"],
                expected_videos=expected_videos,
                synthetic=synthetic,
            )
        # A partial preparation may resume only under its original pinned inputs.
        pending = ws.cache / "preparation.json"
        if pending.exists() and read_json(pending) != identity:
            raise WorkflowError(
                "未完成缓存属于另一份输入或配置。", "使用新的 cache 目录重新准备。"
            )
        if not pending.exists():
            if any((ws.cache / split).exists() for split in ("train", "val", "test")):
                raise WorkflowError("发现无准备回执的缓存目录，不会自动信任或覆盖。")
            write_json(pending, identity)
        splits = {}
        for split in ("train", "val", "test"):
            target = ws.cache / split
            if target.exists():
                if not (target / "index.json").is_file():
                    raise WorkflowError(
                        f"发现未完成的 {split} 缓存：{target}",
                        "确认没有prepare进程后，移开这个派生缓存目录再重试；不要删除原始ZIP或审核资料。也可配置新的cache目录。",
                    )
                from algorithm.data.compact import inspect_compact_cache

                index = read_json(target / "index.json")
                inspect_compact_cache(
                    target, expected_index_sha256=index["index_sha256"]
                )
                splits[split] = index["index_sha256"]
                emit(f"复用完整 {split} 缓存。")
                continue
            emit(f"构建 {split} 紧凑缓存（统计仅由 train 拟合）…")
            result = build_stream_cache(
                imported,
                target,
                split=split,
                dataset_version=cfg.dataset_version,
                review_reference=cfg.review_reference,
                stats=None if split == "train" else ws.cache / "train" / "stats.json",
                stride=cfg.stride,
            )
            splits[split] = result["index_sha256"]
            emit(
                f"  {split}: {result['endpoints']} 个端点，ready覆盖率 {result['coverage']['ready_coverage']:.1%}"
            )
        from algorithm.data.compact import inspect_compact_cache

        import_hash = canonical_digest(read_manifest(imported))
        common = None
        for split in ("train", "val", "test"):
            target = ws.cache / split
            inspect_compact_cache(target, expected_index_sha256=splits[split])
            header = read_json(target / "index.json")
            frozen = read_json(target / "split_manifest.json")
            if (
                header["split"] != split
                or header["stride"] != cfg.stride
                or header["synthetic"] is not synthetic
                or frozen["import_manifest_sha256"] != import_hash
                or frozen["dataset_version"] != cfg.dataset_version
                or frozen["preregistration_reference"] != cfg.review_reference
            ):
                raise WorkflowError("已有缓存不属于本次准备的来源、划分或配置。")
            current = tuple(
                header[key]
                for key in (
                    "dataset_sha256",
                    "split_sha256",
                    "stats_sha256",
                    "config_sha256",
                    "classes_sha256",
                )
            )
            if common is not None and common != current:
                raise WorkflowError("train/val/test 缓存统计或来源不一致。")
            common = current
        from algorithm.training.inputs import load_training_inputs

        _, _, provenance = load_training_inputs(ws.cache / "train", ws.cache / "val")
        value = {
            "format_version": "workflow-dataset-v1",
            "synthetic": synthetic,
            "preparation": identity,
            "splits": splits,
            "vocabulary_version": provenance["vocabulary_version"],
        }
        value["receipt_sha256"] = canonical_digest(value)
        write_json(ws.cache / "dataset.json", value)
    emit("数据准备完成。预检：uv run train --dry-run；正式训练：uv run train")
    return value
