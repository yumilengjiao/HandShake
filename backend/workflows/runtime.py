"""Optional dependency bootstrapping through uv's checked-in lock, never global pip."""

import importlib.util
import os
import platform
import shutil
import subprocess

from backend.workflows.workspace import WorkflowError, _toml


def profile_extra(ws):
    return "cuda" if ws.settings.runtime.profile == "cuda" else "training"


def _project(ws):
    path = ws.root / "pyproject.toml"
    return (
        path.is_file()
        and _toml(path).get("project", {}).get("name") == "handshake-core"
    )


def ensure_training(ws, arguments, emit=print):
    extra = profile_extra(ws)
    if os.environ.get("HANDSHAKE_WORKFLOW_EXTRA") == extra:
        if importlib.util.find_spec("torch") is None:
            raise WorkflowError(
                "训练依赖安装后仍找不到 PyTorch。", "运行 uv run setup 检查依赖环境。"
            )
        return None
    if not _project(ws):
        if importlib.util.find_spec("torch") is not None:
            return None
        raise WorkflowError(
            "安装版核心缺少训练依赖。",
            "安装 handshake-core[training]，或在源码目录使用 uv run setup。",
        )
    uv = shutil.which("uv")
    if uv is None:
        raise WorkflowError(
            "找不到 uv，无法准备锁定的训练环境。",
            "安装 uv 并加入 PATH；不会修改系统 Python。",
        )
    emit(f"检查锁定推理/训练依赖（{extra}）；首次运行需要下载，请稍候…")
    environment = {
        k: v for k, v in os.environ.items() if k not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment["HANDSHAKE_WORKFLOW_EXTRA"] = extra
    return subprocess.call(
        [
            uv,
            "run",
            "--locked",
            "--extra",
            extra,
            "python",
            "-m",
            "backend.workflows",
            "--workspace",
            str(ws.root),
            *arguments,
        ],
        cwd=ws.root,
        env=environment,
    )


def setup(ws, *, gpu=False, camera=False, emit=print):
    if (ws.runs / ".train.lock").exists():
        raise WorkflowError(
            "训练可能正在运行，暂不修改依赖环境。",
            "先结束训练，再运行setup；不要在训练期间更换PyTorch或驱动。",
        )
    if not _project(ws):
        raise WorkflowError(
            "setup 需要源码仓库中的 pyproject.toml。",
            "进入 HandShake checkout 后重试。",
        )
    if (gpu or ws.settings.runtime.profile == "cuda") and platform.system() not in {
        "Linux",
        "Windows",
    }:
        raise WorkflowError(
            "CUDA 环境仅面向受支持的 NVIDIA Linux/Windows。",
            "Apple Silicon 使用默认 setup 与 device='auto'，由 MPS 后端处理。",
        )
    uv = shutil.which("uv")
    if uv is None:
        raise WorkflowError("请先安装 uv 并加入 PATH。")
    extra = "cuda" if gpu else profile_extra(ws)
    emit(
        f"准备项目虚拟环境：{extra}"
        + (" + perception（相机依赖）" if camera else "")
        + "。不会训练或打开相机。"
    )
    command = [uv, "sync", "--locked", "--extra", extra, "--extra", "dev"]
    if camera:
        command += ["--extra", "perception"]
    environment = {
        k: v for k, v in os.environ.items() if k not in {"PYTHONHOME", "PYTHONPATH"}
    }
    result = subprocess.call(command, cwd=ws.root, env=environment)
    if result:
        raise WorkflowError(
            "uv 未能安装当前平台的依赖。",
            "检查网络、Python版本和驱动；相机依赖建议 Python 3.12。不会忽略锁文件或偷偷换版本。",
            result,
        )
    if gpu and ws.settings.runtime.profile != "cuda":
        import tomli_w

        path = ws.root / "handshake.local.toml"
        local = _toml(path) if path.exists() else {}
        local.setdefault("runtime", {})["profile"] = "cuda"
        path.write_text(tomli_w.dumps(local), encoding="utf-8")
        emit("已把本机 CUDA 选择写入 handshake.local.toml（Git 忽略）。")
    emit("环境准备完成。下一步：uv run doctor，然后 uv run download。")
    return 0
