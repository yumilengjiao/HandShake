"""Inspect handed-off inference assets without a training run, cache, camera or optimizer."""

from backend.workflows.workspace import WorkflowError


def verify_assets(package, sha256, *, calibration=None, calibration_sha256=None,
                  shadow=False, allow_synthetic=False, emit=print):
    if not package or not sha256:
        raise WorkflowError("模型包和可信 SHA256 必须同时提供。",
                            "请训练者交付导出目录及 export.json 中的 manifest_sha256。")
    if bool(calibration) != bool(calibration_sha256):
        raise WorkflowError("校准文件和可信校准 SHA256 必须同时提供。")
    if shadow and not calibration:
        raise WorkflowError("影子识别需要与模型匹配的验证集校准文件。")
    import torch
    from algorithm.bundle import load_bundle

    # Small CPU models: avoid a large OpenMP pool during loading/golden checks.
    torch.set_num_threads(1)
    bundle = load_bundle(package, expected_manifest_sha256=sha256)
    if bundle.synthetic_training and not allow_synthetic:
        raise WorkflowError("这是合成诊断模型，不是正式训练成果。",
                            "只在影子诊断中显式使用 --shadow --allow-synthetic-model。")
    value = None
    if calibration:
        from algorithm.evaluation.calibration import load_for_bundle

        value = load_for_bundle(calibration, expected_sha256=calibration_sha256,
                                bundle=bundle)
    if shadow and (bundle.vocabulary.vocabulary_version != "ipn-hands-v1"
                   or value.rejection_status != "selected"):
        raise WorkflowError("影子识别需要15类模型与可用的拒识校准；不会放宽门限。")
    emit(f"模型校验通过：{bundle.model_type.upper()} / {bundle.vocabulary.class_count}类 / "
         f"synthetic={bundle.synthetic_training}")
    emit("类别：" + ", ".join(bundle.vocabulary.tokens))
    emit("校准：" + (value.rejection_status if value else "未提供（只显示原始分数）"))
    emit("已验证包哈希、运行契约、归一化及golden前向；不代表真实识别精度。动作权限始终关闭。")
    return bundle
