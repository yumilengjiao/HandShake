# 截止前桌面与模型交接

## 本机启动

```bash
uv run app
# 无需Flutter SDK（必须保留整个bundle目录）
uv run app --bundle frontend/native/build/linux/x64/release/bundle/handshake_stage
```

本机 `handshake.local.toml` 配置Flutter，个人路径不提交。终端 `flutter` / `dart` 可通过 `~/.local/bin` 使用。启动仍为停止，必须手动打开相机；右上角“演示”提供独立人工翻页。

## 让训练者交什么

必须使用一致的源码/运行契约，训练者在其机器完成 `uv run export`、`uv run calibrate`，交付：

1. `export.json` 中 `directory` 指向的**完整模型目录**，不能只给 `best.pt`；保留原目录内的所有文件。
2. `export.json` 的 `manifest_sha256`，通过可信聊天/交接记录另行提供。
3. `calibration-receipt.json` 指向的验证集校准文件，以及该回执中的 `sha256`。包内名为calibration.json的占位文件不是这个sidecar。
4. 用于报告的 `epochs.jsonl`、验证/最终测试 `classification.json`、配置、硬件、训练时间、随机种子和数据版本。最终测试不用于调参。

接收者不用复制训练缓存、`.venv`，也不用伪造本地 `artifacts/runs/latest.json`。

```bash
# 将路径和HASH替换为训练者交付值。此步不需要相机、Flutter或训练数据。
uv run app --package /path/to/model --sha256 MODEL_HASH \
  --calibration /path/to/validation-calibration.json \
  --calibration-sha256 CALIBRATION_HASH --verify-only

# 同样参数去掉 --verify-only：只读分类与概率界面
# 再加 --shadow：明确启用/选择owner/预热后的候选界面，仍不执行动作
# 可再加 --bundle <已构建可执行文件>：无需Flutter SDK
```

`--model` 仍表示本机最近训练，与外部 `--package` 互斥。错误哈希、错模型校准、不可行影子阈值直接拒绝。合成资产默认拒绝；程序诊断须显式 `--shadow --allow-synthetic-model`，不能作为正式模型成绩。

## 故障定位与边界

相机worker向stderr输出有限生命周期事件，前缀 `HS_DIAG`。仅含阶段、单调时钟和异常类型，不记录画面、关键点、模型路径或任意异常消息。界面断连时显示最后阶段/失败阶段和能取得的进程退出码；SDK其他stderr不会展示或持久化。既有停止、时效和预热期限未放宽。

模型加载过程中收到停止后，不再继续初始化感知器。资源释放失败仍不确认停止；断连不自动重启或授权。

本轮实现了交接和诊断能力，不声称彻底消除了所有负载下的断连。原生检查使用程序输入与手工赋权、未训练模型，不访问真实相机，也不执行外部PPT操作。
