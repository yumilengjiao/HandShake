"""HandShake daily commands. Defaults come from handshake.toml, not long shell arguments."""

from __future__ import annotations

import argparse
import re
import sys
import zipfile

from backend.workflows.workspace import WorkflowError, workspace


DESCRIPTIONS = {
    "doctor": "检查环境、设备与数据状态（不训练、不打开相机）",
    "status": "查看当前环境与数据状态",
    "setup": "准备项目虚拟环境；可选 --gpu / --camera",
    "download": "按来源清单下载并校验必需数据包",
    "prepare": "审核数据与分组，生成统一 train/val/test 紧凑缓存",
    "train": "使用默认 TCN 开始训练，自动保存每轮指标与最佳模型",
    "resume": "恢复最近训练的完整 epoch checkpoint",
    "evaluate": "评估最近模型；默认只用 val，不接触最终 test 评估",
    "calibrate": "为最近模型生成验证集温度/拒识校准",
    "export": "导出最近模型的可信推理包",
    "app": "启动原生只读预览（当前仅 Linux 桌面）",
    "check": "运行非训练回归；--training 显式包含训练测试",
}


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("请输入正整数")
    return number


def parser():
    value = argparse.ArgumentParser(
        prog="uv run hs",
        description="HandShake：短命令，统一配置，可追溯数据与训练结果。",
        epilog="配置：handshake.toml；本机覆盖：handshake.local.toml。所有命令可加 --workspace DIR / --debug。",
    )
    commands = value.add_subparsers(dest="command")
    for name, description in DESCRIPTIONS.items():
        child = commands.add_parser(name, help=description, description=description)
        if name == "train":
            child.add_argument(
                "--model", choices=["mlp", "tcn"], help="覆盖配置中的模型"
            )
            child.add_argument("--device", help="auto / cpu / cuda / cuda:N / mps")
            child.add_argument("--epochs", type=positive)
            child.add_argument("--batch-size", type=positive)
            child.add_argument("--learning-rate", type=float, help="覆盖本次学习率，不修改共享配置")
            child.add_argument("--weight-decay", type=float, help="覆盖本次AdamW权重衰减，可为0")
            child.add_argument("--patience", type=positive, help="本次早停耐心轮数")
            child.add_argument("--seed", type=int, help="本次预先指定的随机种子，不用于挑最好结果")
            child.add_argument(
                "--dry-run", action="store_true", help="只校验数据和参数，不开始训练"
            )
        elif name == "setup":
            child.add_argument(
                "--gpu",
                action="store_true",
                help="显式安装锁定 CUDA 环境，并记入本机配置",
            )
            child.add_argument(
                "--camera",
                action="store_true",
                help="安装相机/MediaPipe依赖，不打开相机",
            )
        elif name == "evaluate":
            child.add_argument(
                "--split",
                choices=["val", "test"],
                default="val",
                help="test 仅用于最后冻结评估",
            )
        elif name == "app":
            child.add_argument(
                "--model", action="store_true", help="显式加载本机最近模型，仍仅预览"
            )
            child.add_argument("--package", help="别人交付的推理模型目录；不要求本机训练数据")
            child.add_argument("--sha256", help="训练者提供的 checksums.json 可信SHA256")
            child.add_argument("--calibration", help="与外部模型匹配的验证集校准文件")
            child.add_argument("--calibration-sha256", help="校准文件可信SHA256")
            child.add_argument("--shadow", action="store_true", help="启用影子owner界面，只显示候选，不执行")
            child.add_argument("--allow-synthetic-model", action="store_true", help="仅供显式影子诊断")
            child.add_argument("--bundle", help="已构建的Linux可执行文件，无需Flutter SDK")
            child.add_argument("--verify-only", action="store_true", help="只校验模型，不启动界面或相机")
        elif name == "check":
            child.add_argument(
                "--training", action="store_true", help="只在允许训练的机器上启用"
            )
    return value


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace")
    common.add_argument("--debug", action="store_true")
    options, remaining = common.parse_known_args(arguments)
    cli = parser()
    args = cli.parse_args(remaining)
    if args.command is None:
        cli.print_help()
        return 0
    if args.command == "app":
        if args.model and args.package:
            cli.error("--model 与 --package 二选一，不混用本机训练和外部模型")
        if bool(args.package) != bool(args.sha256):
            cli.error("--package 与 --sha256 必须同时提供")
        if bool(args.calibration) != bool(args.calibration_sha256) or (args.calibration and not args.package):
            cli.error("外部 --calibration / --calibration-sha256 必须成对提供，并指定 --package")
        if (args.shadow or args.verify_only) and not (args.model or args.package):
            cli.error("--shadow / --verify-only 需要 --model 或 --package")
        if args.allow_synthetic_model and not args.shadow:
            cli.error("--allow-synthetic-model 仅用于 --shadow 诊断")
    execution_arguments = remaining + (["--debug"] if options.debug else [])
    emit = lambda message: print(message, flush=True)
    try:
        ws = workspace(options.workspace)
        emit(f"[{args.command}] 工作区：{ws.root}")
        if args.command in {"doctor", "status"}:
            from backend.workflows.system import doctor

            return doctor(ws, emit)
        if args.command == "setup":
            from backend.workflows.runtime import setup

            return setup(ws, gpu=args.gpu, camera=args.camera, emit=emit)
        if args.command == "download":
            from backend.workflows.download import download

            download(ws, emit)
            return 0
        if args.command == "prepare":
            from backend.workflows.data import prepare

            prepare(ws, emit)
            return 0
        if args.command in {"train", "resume"}:
            from backend.workflows.training import run_train

            return run_train(ws, args, execution_arguments, emit)
        if args.command in {"evaluate", "calibrate", "export", "check"} or (
            args.command == "app" and (args.model or args.package)
        ):
            if args.command != "check" and not (args.command == "app" and args.package):
                from backend.workflows.training import _bound_run

                _bound_run(ws)
            from backend.workflows.runtime import ensure_training

            status = ensure_training(ws, execution_arguments, emit)
            if status is not None:
                return status
        if args.command in {"evaluate", "calibrate"}:
            from backend.workflows.training import evaluate_run

            return evaluate_run(
                ws,
                split=getattr(args, "split", "val"),
                calibrate=args.command == "calibrate",
                emit=emit,
            )
        if args.command == "export":
            from backend.workflows.training import export_run

            export_run(ws, emit)
            return 0
        if args.command == "app":
            from backend.workflows.system import launch_app

            return launch_app(
                ws, model=args.model, package=args.package, sha256=args.sha256,
                calibration=args.calibration, calibration_sha256=args.calibration_sha256,
                shadow=args.shadow, allow_synthetic=args.allow_synthetic_model,
                bundle=args.bundle, verify_only=args.verify_only, emit=emit,
            )
        if args.command == "check":
            from backend.workflows.system import check

            return check(ws, training=args.training, emit=emit)
        raise WorkflowError("未知命令。", "运行 uv run hs --help 查看可用命令。")
    except KeyboardInterrupt:
        print(
            "\n操作已中断。训练若已有完整checkpoint，可用 uv run resume 恢复；未完成的数据不会被当作有效产物。",
            file=sys.stderr,
        )
        return 130
    except WorkflowError as error:
        print(f"\n未完成：{error}", file=sys.stderr)
        if error.hint:
            print(f"建议：{error.hint}", file=sys.stderr)
        if options.debug:
            raise
        return error.code
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        ImportError,
        EOFError,
        zipfile.BadZipFile,
    ) as error:
        print(
            f"\n{args.command} 未完成：{type(error).__name__}: {error}", file=sys.stderr
        )
        if isinstance(error, FileNotFoundError):
            hint = "检查配置路径；数据准备使用 uv run download / uv run prepare。"
        elif isinstance(error, ImportError):
            hint = "运行 uv run setup；相机相关依赖使用 uv run setup --camera。"
        elif "memory" in str(error).lower():
            hint = "降低 handshake.local.toml 中的 batch_size；核对可用内存/显存。不会自动更改实验参数。"
        elif "determin" in str(error).lower():
            hint = "该设备不支持当前确定性算子；改用 device='cpu' 开启新run，不静默关闭确定性保护。"
        elif re.search(r"\b(cuda|mps|cudnn|metal)\b|driver", str(error).lower()):
            hint = "检查GPU驱动和PyTorch版本；NVIDIA可运行 uv run setup --gpu，或在配置中选择 device='cpu' 新建run。"
        else:
            hint = "检查 handshake.toml 与输入完整性。使用 --debug 查看完整异常；不会自动覆盖已有产物。"
        print("建议：" + hint, file=sys.stderr)
        if options.debug:
            raise
        return 2


def _entry(command):
    return main([command, *sys.argv[1:]])


def doctor_entry():
    return _entry("doctor")


def status_entry():
    return _entry("status")


def setup_entry():
    return _entry("setup")


def download_entry():
    return _entry("download")


def prepare_entry():
    return _entry("prepare")


def train_entry():
    return _entry("train")


def resume_entry():
    return _entry("resume")


def evaluate_entry():
    return _entry("evaluate")


def calibrate_entry():
    return _entry("calibrate")


def export_entry():
    return _entry("export")


def app_entry():
    return _entry("app")


def check_entry():
    return _entry("check")
