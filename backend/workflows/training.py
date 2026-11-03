"""Short training/evaluation commands with automatic immutable run locations."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from uuid import uuid4

from backend.workflows.data import dataset_receipt
from backend.workflows.runtime import ensure_training
from backend.workflows.workspace import (
    WorkflowError,
    operation_lock,
    read_json,
    write_json,
)


def run_name(model):
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{model}-{uuid4().hex[:8]}"
    )


def latest_run(ws):
    if not ws.latest.is_file():
        raise WorkflowError(
            "尚无当前工作区的训练记录。",
            "先运行 uv run train；结果自动保存在 artifacts/runs。",
        )
    pointer = read_json(ws.latest)
    if pointer.get("format_version") != "workflow-run-v1" or not isinstance(
        pointer.get("run"), str
    ):
        raise WorkflowError("最新训练指针格式不正确。")
    path = (ws.runs / pointer["run"]).resolve()
    if path == ws.runs or not path.is_relative_to(ws.runs) or not path.is_dir():
        raise WorkflowError("最新训练目录不存在或越出 runs 目录。")
    return path, pointer


def _bound_run(ws):
    data = dataset_receipt(ws)
    root, pointer = latest_run(ws)
    if pointer.get("dataset_receipt_sha256") != data["receipt_sha256"]:
        raise WorkflowError(
            "当前数据与最近训练使用的数据不一致。",
            "恢复该run对应的数据目录/配置，或新建训练；不会混用统计与模型。",
        )
    return root, data


def preflight(ws, args, emit=print):
    receipt = dataset_receipt(ws)
    from algorithm.training.config import TrainConfig
    from algorithm.training.inputs import load_training_inputs

    cfg = ws.settings.train

    def selected(name):
        value = getattr(args, name, None)
        return getattr(cfg, name) if value is None else value

    config = TrainConfig(
        model_type=args.model or cfg.model,
        device=args.device or cfg.device,
        batch_size=args.batch_size or cfg.batch_size,
        max_epochs=args.epochs or cfg.epochs,
        patience=selected("patience"),
        learning_rate=selected("learning_rate"),
        weight_decay=selected("weight_decay"),
        grad_clip_norm=cfg.grad_clip_norm,
        min_delta=cfg.min_delta,
        seed=selected("seed"),
        threads=cfg.threads,
    )
    _, _, provenance = load_training_inputs(ws.cache / "train", ws.cache / "val")
    if provenance["synthetic"]:
        raise WorkflowError("默认训练入口禁止合成数据。")
    emit(
        f"训练预检通过：{config.model_type.upper()} / device={config.device} / batch={config.batch_size} / 最多{config.max_epochs}轮"
    )
    emit(
        f"本次参数：lr={config.learning_rate:g} / weight_decay={config.weight_decay:g} / patience={config.patience} / seed={config.seed}；训练时记录到独立run，不修改默认配置。"
    )
    emit(
        f"train={provenance['train']['supervised']}、val={provenance['val']['supervised']} 个有标签端点；不会读取 test 样本。"
    )
    if provenance.get("grouping", {}).get("unit") == "source_capture_group":
        emit("评估范围：固定采集组留出；没有独立核实主体身份，不宣称跨人识别成绩。")
    absent = [
        i for i, count in enumerate(provenance["train"]["class_counts"]) if count == 0
    ]
    if absent:
        emit(f"注意：训练集类别 {absent} 没有监督样本；不应把本次结果当完整15类基准。")
    return config, receipt


def run_train(ws, args, arguments, emit=print):
    if args.command == "resume":
        root, receipt = _bound_run(ws)
        from algorithm.training.config import TrainConfig

        config = TrainConfig.model_validate(
            read_json(root / "run_manifest.json")["identity"]["config"]
        )
        if not (root / "last.pt").is_file():
            raise WorkflowError(
                "上次训练没有完整 epoch checkpoint，无法恢复。",
                "运行 uv run train 新建训练。",
            )
        resume = root / "last.pt"
    else:
        config, receipt = preflight(ws, args, emit)
        if args.dry_run:
            emit(
                "这是只读预检，未导入训练器、创建优化器或开始训练。设备可用性请运行 uv run doctor。"
            )
            return 0
        root, resume = ws.runs / run_name(config.model_type), None
    status = ensure_training(ws, arguments, emit)
    if status is not None:
        return status
    from algorithm.training.trainer import train
    from algorithm.training.devices import resolve_device

    selected = resolve_device(config.device)
    emit(f"实际设备：{selected}。结果目录：{root}")
    emit(
        "每轮自动保存 best.pt、last.pt 和指标；Ctrl+C 中断后可用 uv run resume 恢复完整轮次。"
    )
    pointer = {
        "format_version": "workflow-run-v1",
        "run": root.name,
        "dataset_receipt_sha256": receipt["receipt_sha256"],
    }

    def progress(row):
        emit(
            f"Epoch {row['epoch']}/{config.max_epochs} | loss={row['train_loss']:.5f} | val Macro-F1={row['validation']['macro_f1']:.4f}"
        )
        write_json(ws.latest, pointer)

    with operation_lock(ws.runs, "train"):
        try:
            report = train(
                ws.cache / "train",
                ws.cache / "val",
                root,
                config=config,
                resume=resume,
                progress=progress,
            )
        finally:
            if (root / "run_manifest.json").is_file():
                write_json(root / "workflow.json", pointer)
                write_json(ws.latest, pointer)
    emit(
        f"训练结束：{report['status']}；最佳 epoch={report['best_epoch']}，val Macro-F1={report['best_val_macro_f1']:.4f}"
    )
    if ws.settings.train.auto_export:
        export_run(ws, emit, bound=(root, receipt))
    emit("下一步：uv run evaluate；验证集校准：uv run calibrate")
    return 0


def model_directory(root, receipt):
    directory = receipt.get("directory", "model")
    path = (root / directory).resolve()
    if path == root or not path.is_relative_to(root):
        raise WorkflowError("模型回执中的路径越出训练目录。")
    return path


def export_run(ws, emit=print, *, bound=None):
    root, _ = bound if bound is not None else _bound_run(ws)
    from algorithm.bundle import export_bundle, load_bundle

    run = read_json(root / "run_manifest.json")
    if run.get("status") != "completed":
        raise WorkflowError(
            "最近训练尚未完成，不能导出。",
            "先运行 uv run resume 完成训练；不会把不完整checkpoint当成功模型。",
        )
    checkpoint_sha = run["artifacts"]["best.pt"]
    if not isinstance(checkpoint_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_sha
    ):
        raise WorkflowError("训练回执中的 best.pt 哈希无效。")
    receipt_path = root / "export.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("checkpoint_sha256") == checkpoint_sha:
            target = model_directory(root, receipt)
            load_bundle(target, expected_manifest_sha256=receipt["manifest_sha256"])
            emit(f"模型包已存在且通过校验：{target}")
            return receipt
    target = model_directory(
        root, {"directory": f"models/{checkpoint_sha[:16]}-{uuid4().hex[:8]}"}
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = export_bundle(root, ws.cache / "train" / "stats.json", target)
    receipt.update(
        checkpoint_sha256=checkpoint_sha, directory=target.relative_to(root).as_posix()
    )
    write_json(receipt_path, receipt)
    emit(f"推理模型已导出：{target}；仍无应用控制权限。")
    return receipt


def evaluate_run(ws, *, split="val", calibrate=False, emit=print):
    root, data = _bound_run(ws)
    package = export_run(ws, emit, bound=(root, data))
    from algorithm.evaluation.classification import evaluate_classification
    from algorithm.evaluation.collect import collect_scores
    from algorithm.evaluation.scores import load_scores

    if split == "test":
        emit(
            "注意：这是最终 test 评估，不得用结果继续选择模型/阈值。结果中保留本次访问记录。"
        )
    output = root / "evaluations" / run_name(split)
    output.mkdir(parents=True, exist_ok=False)
    scores_receipt = collect_scores(
        model_directory(root, package),
        ws.cache / split,
        output / "scores",
        model_sha256=package["manifest_sha256"],
        cache_index_sha256=data["splits"][split],
        split=split,
    )
    write_json(output / "scores.json", scores_receipt)
    scores = load_scores(
        output / "scores", expected_manifest_sha256=scores_receipt["manifest_sha256"]
    )
    result = evaluate_classification(scores)
    if data.get("grouping") is not None:
        result["grouping"] = data["grouping"]
    write_json(output / "classification.json", result)
    emit(f"{split} 分类报告：{output / 'classification.json'}")
    if calibrate:
        from algorithm.evaluation.calibration import fit_calibration, save_calibration

        value = fit_calibration(scores)
        sha = save_calibration(value, output / "calibration.json")
        write_json(
            root / "calibration-receipt.json",
            {
                "path": (output / "calibration.json").relative_to(root).as_posix(),
                "sha256": sha,
            },
        )
        emit(
            f"验证集校准：{value.rejection_status}；保存于 {output / 'calibration.json'}"
        )
        if value.rejection_status != "selected":
            raise WorkflowError(
                "默认风险/覆盖约束下无可用拒识阈值。结果已保存，但不视为可用校准。",
                "检查模型和验证集；不要用测试集调参或放宽门限冒充达标。",
            )
    return 0
