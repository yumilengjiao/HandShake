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
# 单次参数试验，可先加 --dry-run；不修改默认配置
uv run train --learning-rate 0.0003 --weight-decay 0.0001 --seed 42 --dry-run
uv run resume             # 恢复最近一次训练
uv run evaluate           # 评估最近模型，默认使用验证集
uv run calibrate          # 生成验证集校准
uv run export             # 导出最近模型
```

训练结果自动保存到 `artifacts/runs/` 的独立目录，包括最佳权重、最近 checkpoint 和指标，无需手动指定输出路径。

可选改进：`--class-weighting sqrt_inverse`、`--scheduler plateau`、`--sampling supervised_uniform`；默认仍保留原基线。类别不平衡及历史对照见[训练改进说明](docs/training_improvements.md)和[夜间复盘](docs/training_overnight_review.md)。

新增显式TCN变体`--model-variant motion/dropout35/motion_dropout35`，默认仍为`baseline`。2026-11-04上午训练已在12:00停止；当前验证F1最高的完整候选为因果运动TCN：**Accuracy 79.38%、Macro-F1 50.57%**。但单击两类召回下降，seed43复验只完成18/20epoch，**尚不能认定稳定提分或所有手势更好**。当前候选、旧模型保留情况及完整限制见[上午最终结果](docs/training_morning_results.md)。

### 界面与检查

```bash
uv run app           # 统一原生应用：自动读取最新已导出模型，含实时手势翻页页
uv run app --model   # 显式使用本机训练工作流选择模型
# 别人训练的模型无需本机训练数据，见 docs/deadline_desktop_handoff.md
uv run app --package /path/to/model --sha256 <可信SHA> --verify-only
uv run check         # 运行非训练回归测试
```

界面目前仅提供 Linux 宿主，启动后保持停止，不自动打开相机或执行应用操作。已有完整模型与校准时，`uv run app` 默认进入[手势翻页预览](docs/native_gesture_preview.md)：同一窗口内点击“打开相机 → 启用手势 → 选择控制手”，等待预热后，用左右抛动翻动内置示例页。空格停控制，Esc关闭相机；真实相机成功率仍待现场验收。CPU/CUDA/MPS 的实际兼容性以目标机器验证结果为准。

允许进行合成训练测试的机器可运行 `uv run check --training`。

### 独立演示兼容入口

日常使用上述统一应用即可。原[本地手势演示播放器](docs/local_gesture_demo.md)仍保留，可载入PPT导出的编号图片。它是独立的显式授权入口，**不改变影子模式、不注入系统按键**；真实相机成功率尚未验收。

```bash
uv run --extra training --extra perception --extra demo handshake-demo \
  --run artifacts/runs/20261102T174832Z-tcn-af17e513 --allow-local-demo
# 窗口内：C开相机 → E启用 → O选中立手；空格停控制，Esc关闭相机退出。
```

训练交付目录只在接收模型的本机存在；其他机器先按交接说明准备自己的模型及校准。

## 配置与帮助

默认配置位于 [`handshake.toml`](handshake.toml)，可调整模型、设备、学习率、batch 和数据目录。本机设置写入 `handshake.local.toml`，无需修改共享配置；该文件不会提交到 Git。

```bash
uv run hs            # 查看全部命令
uv run train --help  # 查看单条命令帮助
```

详细操作与训练机交接见[快速开始](docs/quickstart.md)；所有日常命令、高级工具和 `scripts/*.py` 的用途与参数见[命令参考](docs/cli_reference.md)。
