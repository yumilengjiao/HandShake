# HandShake

基于 MediaPipe 手部关键点与 MLP / 因果 TCN 的手势识别项目，提供数据准备、模型训练、评估和 Linux 原生预览界面。

## 安装

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，获取项目源码后，在项目根目录执行：

```bash
uv run setup
uv run doctor
```

项目默认使用 Python 3.12，由 uv 管理解释器和虚拟环境，无需手动激活。

NVIDIA 训练机可选择 CUDA 环境，需安装兼容的显卡驱动：

```bash
uv run setup --gpu
```

使用相机预览时，额外安装感知依赖：

```bash
uv run setup --camera
```

原生界面还需安装 Flutter 3.47.9 和 Linux 构建依赖，见[界面安装说明](frontend/native/README.md)。训练不需要 Flutter 或摄像头。

## 使用

### 准备数据

```bash
uv run download     # 下载并校验数据
uv run prepare      # 检查分组并构建训练、验证和测试缓存
```

> 默认使用已审查、固定版本的图片帧关键点，不下载视频。分组由程序生成；标注包的旧公布SHA差异及采集组评估限制见[数据来源](data/SOURCES.md)。

### 训练与评估

```bash
uv run train --dry-run    # 训练前检查，不开始训练
uv run train              # 使用默认配置训练 TCN
uv run train --model mlp  # 训练 MLP 对照模型
uv run resume             # 恢复最近一次训练
uv run evaluate           # 评估最近模型，默认使用验证集
uv run calibrate          # 生成验证集校准
uv run export             # 导出最近模型
```

训练结果自动保存到 `artifacts/runs/` 的独立目录，包括最佳权重、最近 checkpoint 和指标，无需手动指定输出路径。

### 界面与检查

```bash
uv run app           # 启动原生预览
uv run app --model   # 加载最近模型进行只读预览
# 别人训练的模型无需本机训练数据，见 docs/deadline_desktop_handoff.md
uv run app --package /path/to/model --sha256 <可信SHA> --verify-only
uv run check         # 运行非训练回归测试
```

界面目前仅提供 Linux 宿主，启动后保持停止，不自动打开相机或执行应用操作。CPU/CUDA/MPS 的实际兼容性以目标机器验证结果为准。

允许进行合成训练测试的机器可运行 `uv run check --training`。

## 配置与帮助

默认配置位于 [`handshake.toml`](handshake.toml)，可调整模型、设备、学习率、batch 和数据目录。本机设置写入 `handshake.local.toml`，无需修改共享配置；该文件不会提交到 Git。

```bash
uv run hs            # 查看全部命令
uv run train --help  # 查看单条命令帮助
```

详细操作与训练机交接见[快速开始](docs/quickstart.md)。

## 课程报告与交付模型

[课程报告](docs/report/README.md)提供可编辑 Word、Markdown 和全黑文字 PDF，以及真实 TCN 验证曲线、混淆矩阵和逐类指标。验证准确率为 **78.85%**、Macro-F1 为 **46.52%**；这不是现场手势控制成功率。报告生成工具仅保留本机，不提交 Git。
