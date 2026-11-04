# 训练机快速开始

## 默认训练数据

默认使用 **`ipn-hands-frames-v1`**：200个完整图片序列对应的MediaPipe关键点与15类时间标签，共801,476帧。**不下载视频，不需要摄像头，不执行包内抽帧脚本。** 网络训练的是图片提取后的94维关键点特征，不是直接训练RGB图像分类器。

来源和许可见 [data/SOURCES.md](../data/SOURCES.md)。标注包的旧公布SHA差异仍保留；默认采用经过实际全量核查、另行固定哈希的[项目审查快照](ipn_hands_release.md)，不是取消校验。

## 1. 准备环境

取得包含本次数据就绪改动的代码，安装uv，在项目根目录运行：

```bash
uv run setup
uv run doctor
```

仓库默认Python3.12；uv管理解释器和虚拟环境，不用手动激活，也不要复制别人的`.venv`。

NVIDIA训练机可改用以下安装命令，后续操作不变：

```bash
uv run setup --gpu
```

该命令安装锁定的CUDA12.8版PyTorch依赖，并记住本机选择，不安装系统显卡驱动。需兼容驱动，首次可能下载数GB。CPU/CUDA extras互斥，不使用 `--all-extras`。Apple Silicon使用普通setup，`device="auto"`在实际可用时选择MPS；不支持的确定性算子会明确报错，不悄悄改变实验。

远程训练机不需要Flutter、桌面或MediaPipe SDK。只有要使用相机界面时才需要 `uv run setup --camera` 及独立的Flutter/Linux系统依赖。

## 2. 下载和准备

```bash
uv run download
uv run prepare
uv run train --dry-run
```

- 下载只包含两个必需ZIP；已有正确文件自动跳过。任何未经审查的新字节、坏review或错误SHA仍拒绝。
- prepare自动生成固定采集组计划，不要求使用者填写虚假的主体表或审核编号。首次导入和构建80万帧需要时间，会显示序列/缓存进度。
- 默认116个train、32个val、52个test序列。同一来源采集组的四个序列始终在同一集合；官方test不变，val仅从官方train预先固定选择。
- **来源采集组不是独立核实的参与者身份**。缓存、模型与评估结果标记 `subject_identity_verified=false`；不要把结果称作已认证的跨人评估。
- 训练前检查不创建优化器、不开始神经训练；只验证train/val内容，不使用test样本调参。

默认目录见[数据布局](dataset_layout.md)。建议预留至少数GB数据/临时缓存空间；系统盘空间不足时在本机配置中指定数据盘。此建议不是所有硬件组合的最低资源认证。

### 需要真实主体协议时

默认不要求自录，也不猜受试者姓名。若有可靠主体/会话资料，可选择：

```toml
[data]
split_mode = "reviewed_subjects"
subjects = "data/metadata/ipn-hands-subjects.json"
review_reference = "真实完成的审核依据"
imported = "data/processed/ipn-subject-reviewed"
cache = "data/cache/ipn-subject-reviewed"
```

此高级模式继续使用严格的 `subject-plan-v1`：`status="reviewed"`，每个源序列映射真实主体/会话，三个集合非空、同主体不跨集合、官方test保持隔离。缺文件会生成draft模板并停止，不自动假造身份。使用新目录避免覆盖默认采集组实验。

## 3. 训练

在允许训练的机器上，数据预检通过后即可运行：

```bash
uv run train
```

建议另行运行 `uv run check --training` 验证训练/恢复和可用CUDA/MPS设备；无对应设备会skip，不算硬件认证，也不是正式模型成绩。完整工程检查含感知用例，缺少可选依赖时需先 `uv run setup --camera`；这些依赖和完整检查不是关键点模型训练的前置。

默认TCN、batch64、最多80轮、patience10、seed42；学习率、设备和目录在 `handshake.toml` 设置，本机覆盖写入被Git忽略的 `handshake.local.toml`。每轮显示loss和val Macro-F1，自动保存：

```text
artifacts/runs/<时间>-<模型>-<标识>/
  config.yaml / run_manifest.json / workflow.json
  epochs.jsonl / last.pt / best.pt
  export.json / models/<模型版本>/
  evaluations/<每次评估>/
```

- MLP对照：`uv run train --model mlp`，使用相同数据，新run。
- 中断恢复：`uv run resume`，只恢复完整epoch。数据/代码/配置/环境不匹配时拒绝，需开启新run。
- 单独导出：`uv run export`，新best不会被旧模型包冒充，不覆盖旧包。
- 显存不足：降低batch并开新run；不会在失败后自动改参数继续跑。
- 数据不平衡，NOG/P1F/P2F占比较高，请看每类指标、Macro-F1与覆盖率，不只看总准确率。

## 4. 评估与使用

```bash
uv run evaluate
uv run calibrate
```

默认val，自动关联模型与可信回执。校准不改变网络权重；约束不可满足时保存报告但返回失败。最终模型/参数冻结后才运行 `uv run evaluate --split test`。

Linux上 `uv run app` 会读取最新已有的完整模型/校准交接，默认进入同窗口的[实时手势翻页预览](native_gesture_preview.md)，不重新训练或导出；相机、启用和控制手选择仍需在界面明确操作。给别人部署请发送完整模型包、export回执、校准文件及代码版本；不要只发 `weights.npz` 或整个`.venv`。

## 错误与平台边界

- 下载遇到网络/配额限制：按提示重试或从官方分享页手动下载到raw目录；同一命令仍会核对SHA。没有登录cookie或认证绕过。
- 上游文件变更：保留失败信息，更新数据版本必须重新审查，不自动信任新哈希。
- 锁文件残留：确认没有同类进程后才可删除锁，不能同时覆盖正在使用的数据。
- 未完成的导入/缓存：选择新目录或移开该派生目录后重试，不删除原始ZIP。
- CPU/CUDA/MPS代码支持不等于每种硬件/驱动都已实测。Windows/macOS及GPU结果以实际CI/训练机报告为准。
- 原生界面只有Linux宿主，影子MLP历史间歇性断连仍未解决；与数据准备能否训练是两件事。

本机只做真实数据准备、预检和非训练回归，不运行神经训练、真实相机或外部应用控制。正式训练结果、最终实验与真实手势控制另行验收。
