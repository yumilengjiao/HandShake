"""Human-readable environment/status checks and explicit, non-executing desktop launch."""

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys

from backend.workflows.workspace import WorkflowError, read_json


def doctor(ws, emit=print):
    emit(
        f"HandShake | {platform.system()} {platform.machine()} | Python {platform.python_version()}"
    )
    emit(
        f"工作区：{ws.root}\n解释器：{sys.executable}\n依赖配置：{ws.settings.runtime.profile}"
    )
    for name, module in (
        ("numpy", "numpy"),
        ("pydantic", "pydantic"),
        ("torch", "torch"),
        ("mediapipe", "mediapipe"),
        ("pytest", "pytest"),
    ):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "未安装（可选依赖按需准备）"
        emit(f"  {name}: {version}")
    if importlib.util.find_spec("torch") is not None:
        from algorithm.training.devices import resolve_device

        try:
            emit(f"  可选训练设备：{resolve_device(ws.settings.train.device)}")
        except (ValueError, RuntimeError) as error:
            emit(f"  设备未就绪：{error}")
    flutter = shutil.which(os.environ.get("FLUTTER_BIN", ws.settings.app.flutter))
    emit(f"  Flutter: {flutter or '未找到；不影响数据准备和训练'}")
    if platform.system() != "Linux":
        emit(
            "  原生界面仅有 Linux 宿主；本平台的数据/训练路径可预检，桌面app尚未交付。"
        )
    from backend.workflows.download import (
        ANNOTATIONS,
        LANDMARKS,
        catalog,
        check_archive,
    )

    ready = True
    try:
        entries = catalog(ws)
        for name in (ANNOTATIONS, LANDMARKS):
            check_archive(
                ws.path(ws.settings.data.raw) / name, entries[name]["expected_sha256"]
            )
            emit(f"  原始包通过：{name}")
    except (WorkflowError, OSError, ValueError) as error:
        ready = False
        emit(f"  数据未就绪：{error}")
    if not (ws.cache / "dataset.json").is_file():
        ready = False
        emit(
            "  尚无完整准备回执。执行顺序：uv run download → uv run prepare → uv run train"
        )
    elif ready:
        emit("  已有准备回执；深度完整性预检：uv run train --dry-run")
    emit("说明：doctor 不训练、不打开相机；依赖存在不等于硬件或正式训练已验收。")
    return 0


def launch_app(ws, *, model=False, package=None, sha256=None, calibration=None,
               calibration_sha256=None, shadow=False, allow_synthetic=False,
               bundle=None, verify_only=False, emit=print):
    if not verify_only and platform.system() != "Linux":
        raise WorkflowError(
            "此仓库目前只交付 Flutter Linux 桌面宿主。",
            "当前系统仍可准备数据和训练；Windows/macOS原生宿主及手机需另外实现/验收。",
        )
    args = []
    if not verify_only:
        native = ws.root / "frontend" / "native"
        if not (native / "pubspec.yaml").is_file():
            raise WorkflowError(
                "找不到 Flutter 源码，安装版 Python wheel 不包含原生界面。",
                "进入完整源码 checkout 运行 uv run app。",
            )
        if bundle:
            executable = ws.path(bundle)
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise WorkflowError(f"原生可执行文件不存在或不可执行：{executable}")
            args += ["--bundle", str(executable)]
        else:
            flutter = os.environ.get("FLUTTER_BIN", ws.settings.app.flutter)
            if shutil.which(flutter) is None:
                raise WorkflowError(
                    "找不到 Flutter SDK。",
                    "配置 [app] flutter 或 FLUTTER_BIN；已有发布版可用 uv run app --bundle <可执行文件>。",
                )
            args += ["--flutter", flutter]
        args += ["--camera", str(ws.settings.app.camera)]
    perception = ws.path(ws.settings.app.perception_model)
    if not verify_only and perception.is_file():
        if importlib.util.find_spec("mediapipe") is None:
            raise WorkflowError(
                "发现感知资产，但尚未安装相机依赖。",
                "先运行 uv run setup --camera；它不会打开相机。",
            )
        args += ["--perception-model", str(perception)]
    if (model or package) and not verify_only and not perception.is_file():
        raise WorkflowError(
            f"缺少感知模型：{perception}",
            "按 docs/perception_diagnostics.md 获取可信 MediaPipe 资产；不会自动下载陌生模型。",
        )
    selected_package = ws.path(package) if package else None
    selected_calibration = ws.path(calibration) if calibration else None
    if model:
        from backend.workflows.training import _bound_run, export_run, model_directory

        root, data = _bound_run(ws)
        receipt = export_run(ws, emit, bound=(root, data))
        selected_package = model_directory(root, receipt)
        sha256 = receipt["manifest_sha256"]
        calibration_receipt = root / "calibration-receipt.json"
        if calibration_receipt.exists():
            from algorithm.data.source_archive import file_sha256

            binding = read_json(calibration_receipt)
            calibration = (root / binding["path"]).resolve()
            if (
                not calibration.is_relative_to(root)
                or file_sha256(calibration) != binding["sha256"]
            ):
                raise WorkflowError(
                    "校准回执路径/哈希不一致，不启动模型预览。",
                    "恢复原始校准产物，或重新运行 uv run calibrate。",
                )
            if (
                read_json(calibration)["source"]["model_sha256"]
                != receipt["manifest_sha256"]
            ):
                raise WorkflowError(
                    "校准属于旧模型，不能混用。",
                    "运行 uv run calibrate 为当前best模型重新校准。",
                )
            selected_calibration = calibration
            calibration_sha256 = binding["sha256"]
    if package or shadow or verify_only:
        from backend.workflows.deployment import verify_assets

        verify_assets(selected_package, sha256, calibration=selected_calibration,
                      calibration_sha256=calibration_sha256, shadow=shadow,
                      allow_synthetic=allow_synthetic, emit=emit)
    if verify_only:
        emit("只读交接校验完成；未启动Flutter、相机、训练器或应用操作。")
        return 0
    if selected_package:
        args += ["--model-package", str(selected_package), "--model-sha256", sha256]
    if selected_calibration:
        args += ["--calibration", str(selected_calibration),
                 "--calibration-sha256", calibration_sha256]
    if shadow:
        args.append("--shadow-session")
    if allow_synthetic:
        args.append("--allow-synthetic-model")
    from backend.native_launcher import main

    emit("启动原生预览；保持停止。相机只能由窗口中的明确操作打开，模型不会控制应用。")
    return main(args, project_root=ws.root)


def check(ws, *, training=False, emit=print):
    if not (ws.root / "tests").is_dir():
        raise WorkflowError("check 需要源码 checkout 中的测试目录。")
    if importlib.util.find_spec("pytest") is None:
        raise WorkflowError(
            "尚未安装测试依赖。", "先运行 uv run setup，随后 uv run check。"
        )
    command = [sys.executable, "-m", "pytest"]
    if not training:
        command += [
            "--ignore=" + path
            for path in (
                "tests/unit/test_training.py",
                "tests/unit/test_model_losses.py",
                "tests/unit/test_mlp.py",
                "tests/unit/test_tcn_causality.py",
                "tests/integration/test_model_bundle.py",
                "tests/integration/test_accelerator_training.py",
            )
        ]
        command += [
            "-k",
            "not test_public_import_to_deep_training_and_resume and not test_compact_training_resume_and_model_export",
        ]
    environment = {
        k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}
    }
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["HANDSHAKE_NO_TRAINING"] = "0" if training else "1"
    environment["HANDSHAKE_HARDWARE_TRAINING"] = "1" if training else "0"
    emit(
        "运行完整训练回归（会做合成训练）"
        if training
        else "运行非训练回归；不会运行神经训练用例。"
    )
    return subprocess.call(command, cwd=ws.root, env=environment)
