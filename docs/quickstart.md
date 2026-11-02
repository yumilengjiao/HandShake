# 训练机快速开始

## 先看真实阻塞

**不能把这份文档理解为数据已经齐全。** 2026-11-02 本分支实际运行 `uv run download`，官方标注 ZIP 仍为82,274字节、SHA `520dfbf3…bcbaec`，不同于来源清单的 `4686a2eb…0ec71`。下载器将其隔离而非发布；主关键点包未因此继续下载。真实主体/会话分组审核也尚未提供。

训练机可以先准备环境和做软件验证，但正式训练必须等这两项解决。不能删除校验、改合成标记、猜主体身份或把测试模型冒充训练成果。

## 1. 安装与能力检查

取得同一代码提交，安装 uv。在项目根目录执行：

```bash
uv run setup
uv run doctor
```

仓库 `.python-version` 默认选择Python3.12，uv可按需准备。无需激活 `.venv`，Windows也不用改成 `.venv/bin/python`。路径含空格和非ASCII字符时请让 uv 管理解释器，不复制其他机器的虚拟环境。

NVIDIA训练机可选择：

```bash
uv run setup --gpu
```

这会通过 `uv.lock` 安装 `cuda` extra（CUDA 12.8版本的PyTorch依赖），将选择保存在被忽略的 `handshake.local.toml`；不会安装或升级系统显卡驱动。驱动须兼容该运行时，首次下载可能需要数GB。后续仍只需 `uv run train`。CPU和CUDA extras互斥，不使用 `--all-extras`。

Apple Silicon使用普通 `setup`，`device="auto"` 在PyTorch确实提供MPS时选择MPS；Intel Mac走CPU。严格确定性算子不支持时会失败并提示改用CPU开启新run，不静默切换。明确写 `device="cuda:1"` 或 `"mps"` 而设备不可用时拒绝启动。

相机依赖只在需要界面/感知测试时安装：`uv run setup --camera`。远程训练机无需摄像头、Flutter或桌面环境。

## 2. 获取数据

```bash
uv run download
```

仅获取来源清单中必需的两个文件，不下载可选GT/Holistic/原始视频。HTTPS、有限大小、30秒网络操作超时、流式SHA；相同正确文件自动跳过。

- 网络失败：提示网络/代理或官方分享页，退出非零。重试同一条命令即可。
- Google Drive公共“下载确认”页：仅处理同一文件、同一官方主机的有限表单；不运行脚本、不使用登录cookie或绕过认证。限流、登录或未知页面明确报错。
- SHA不符：保留 `.rejected-<hash>.zip` 供调查；不产生正式文件，不自动更新来源清单。
- 正在下载：`.download.lock` 阻止并发覆盖。异常断电残留锁须先确认没有进程运行，再人工删除。
- 已手动下载：把文件放进 `data/raw/ipn-hands/`，再次运行本命令会验证并复用；无需传完整路径/SHA。

解决上游版本问题必须有可追溯的发布依据；若更换数据来源，应审查并更新适配器与来源清单，不直接替换为当前下载的哈希来宣称“校验通过”。

## 3. 分组与准备

```bash
uv run prepare
```

当来源包均通过后，命令检查 `data/metadata/ipn-hands-subjects.json`。缺少该文件会生成**draft模板**及官方视频split参考文件，并停下来给出下一步提示，不伪造审核。

主体计划沿用 `subject-plan-v1`：

```json
{
  "format_version": "subject-plan-v1",
  "status": "reviewed",
  "reference": "实际完成的主体分组审核依据",
  "subject_splits": {
    "真实主体A": "train",
    "真实主体B": "val",
    "真实主体C": "test"
  },
  "videos": {
    "真实源视频ID": {"subject_id": "真实主体A", "session_id": "真实会话ID"}
  }
}
```

上面只是结构说明，不是可当作真实数据提交的计划。实际文件必须覆盖全部200个源视频、所有主体角色均被使用、三个集合非空，同一主体/来源/会话不得泄漏；官方test保持隔离。不能根据视频文件名猜主体，也不能随机逐帧切分。

真实审核完成后，在本机配置中填写：

```toml
[data]
review_reference = "实际质量与划分审核记录"
```

再运行 `uv run prepare`：导入连续标签/关键点、执行磁盘暂存审查、仅用train拟合统计、构建三个紧凑缓存，最后发布 `dataset.json`。缺失/损坏的中途产物不视为完整数据。同一输入/配置重复执行会验证并复用；切换数据版本请配置新目录，避免覆盖既有实验。

## 4. 训练与结果

```bash
uv run train --dry-run
uv run train
```

预检不创建优化器、不训练；实际设备可用性由 `doctor` 和训练开始前的设备选择确认。训练默认 TCN、batch64、最多80轮、patience10、seed42。每轮显示loss与验证集Macro-F1，自动保存最佳与最近checkpoint，不用手写输出路径。

```text
artifacts/runs/
  latest.json
  <UTC时间>-tcn-<随机标识>/
    config.yaml
    run_manifest.json
    epochs.jsonl
    last.pt
    best.pt
    workflow.json
    export.json
    calibration-receipt.json             # 指向本run内真实校准文件的路径/SHA
    models/<最佳checkpoint标识>-<随机标识>/
    evaluations/<评估标识>/
```

- 对照模型：`uv run train --model mlp`，同一数据，新run。
- 中断恢复：`uv run resume`。恢复最近run已完成epoch；不恢复半批次，不自动猜一个旧checkpoint。
- CPU/CUDA/MPS记录真实选中设备；保存相应随机状态。数据、代码、配置、环境变化会拒绝严格恢复，需开启新run。
- 显存不足：减小batch后开启新run；不在失败后自动更改实验参数继续跑。
- 默认训练结束导出推理包。单独重试：`uv run export`。模型包按best版本另存，恢复训练后的新best不会被旧包冒充。

**当前机器不执行神经训练。** 本轮训练编排通过mock测试、CPU前向/随机状态检查；完整训练、优化器恢复、CUDA/MPS真实训练应在允许训练的目标机器执行 `uv run check --training` 后再验收。该命令另包含可用CUDA/MPS设备上MLP/TCN的完整轮次恢复对照；没有设备时明确skip，不算通过硬件认证。

## 5. 评估与校准

```bash
uv run evaluate
uv run calibrate
```

默认val，自动关联最近run、可信模型导出回执、数据cache回执。每次生成独立报告目录，不覆盖历史。`uv run app --model` 会复用同一模型的有效校准回执用于只读展示；旧模型的校准会被拒绝并提示重新 `calibrate`。校准不修改网络权重；风险/覆盖约束不可满足时保留报告但返回失败，不能据此启用控制。

最终模型/参数冻结后，才使用 `uv run evaluate --split test`。test不用于选择模型或校准门限；每次访问保留报告。常规训练预检只验证train/val缓存，不读取test样本。

## 6. 发给训练者的最小清单

- 同一源码提交＋`uv.lock`＋`handshake.toml`，不要发送整个 `.venv`。
- 已审核的原始数据/主体计划，或已完成的缓存目录（train、val、test、dataset.json），通过独立渠道传递可信回执。
- 真实来源/许可/主体与质量审查记录；不要把个人身份资料推送到公开Git。
- 目标机器先 `uv run doctor`、`uv run train --dry-run`，再 `uv run check --training`。
- 根据设备一次性选择setup profile，记录最终提交、Python/PyTorch/驱动/硬件和随机种子。
- 先训练MLP对照，再训练TCN；完成val评估与校准，最后才评估test。

自包含的 `dataset.json` 提供一致性关联，不是数字签名；接收方仍需可信传输或独立核对其哈希。不要一边训练一边覆盖拷贝缓存。

## 平台矩阵：设计目标与实测分开

| 环境 | 代码/依赖路线 | 验证状态 |
| --- | --- | --- |
| Linux x86_64 CPU / Python3.12 | 默认 training extra；全部短命令 | 本机命令、数据fixture、非训练回归与CPU前向可验；本轮不运行训练 |
| Windows x86_64 CPU | 同一uv命令；POSIX清单路径、LF源码 | 新增CI目标，尚未据CI结果宣称通过 |
| macOS CPU / Apple Silicon MPS | 默认extra由PyPI匹配本平台PyTorch | 新增CI目标；真实MPS训练未实测 |
| Linux/Windows NVIDIA | 显式cuda extra、设备/随机状态/确定性配置 | 锁文件已解析；未在本机下载CUDA大包或进行GPU训练 |
| 其他架构、AMD/ROCm、移动系统 | 不在本次验证矩阵 | 不保证存在兼容依赖；CPU fallback仅在该平台实际有可用PyTorch时成立 |
| Flutter原生界面 | 只有Linux宿主与SDK路线 | 仍有影子MLP间歇性断连；不代表其他桌面或手机已交付 |

CI配置被提交只代表有测试任务，**不是CI已经运行通过**。最终兼容性应以具体OS、架构、Python、驱动组合的实际报告为准。
