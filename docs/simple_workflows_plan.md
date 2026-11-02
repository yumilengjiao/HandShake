# 简化工作流：实现与验收记录

分支：`feat/simple-workflows`，基于 `aaa71c6`。本次聚焦**日常入口、默认配置、数据/结果关联与训练机交接**，不是一次正式模型训练或全平台发布。

## 已落地

- `uv run hs/doctor/status/setup/download/prepare/train/resume/evaluate/calibrate/export/app/check` 共13个入口，全部注册在发行包中；旧 `handshake-*` 高级命令保留兼容。
- `handshake.toml` 为共享默认配置，`handshake.local.toml` 为被忽略的本机覆盖。`.python-version` 默认3.12；工作区探测、相对路径、UTF-8与POSIX清单路径统一，`.gitattributes`固定LF。
- `backend/workflows/` 只做编排，继续复用严格导入、紧凑cache v3、原训练器/评估/校准/模型包；没有复制第二套算法。
- `download` 流式HTTPS、有限大小、哈希校验、有限Google Drive公开确认、进度、隔离坏包与并发锁；不执行下载内容，不处理登录cookie或绕过认证。
- `prepare` 检查来源和真实主体计划，缺失时生成draft模板并明确停止；构建/验证三个缓存，最后发布关联回执。同一输入幂等；拒绝旧缓存混入、合成标记伪造或覆盖未完成产物。
- `train` 自动预检、按需准备锁定依赖、分配新run、显示每轮指标、保存最佳/最近权重并导出模型。`resume/evaluate/export` 自动关联最近run，不再手填多条路径与SHA。
- 导出按best checkpoint版本另存，避免恢复训练后复用旧模型包；评估每次生成新报告目录。默认val，test须显式选择，校准只能用val。`app --model`自动关联同一模型的校准回执，旧模型校准会被拒绝。
- 训练器新增显式CPU/CUDA/MPS张量放置、随机状态保存/恢复、设备身份和严格确定性配置；显式不可用设备报错。CPU/CUDA依赖extra互斥，CUDA需要明确选择。
- `check` 默认排除训练用例，并加训练入口/pytest反向传播保护；只有 `--training` 才允许完整训练回归。另备CUDA/MPS上MLP/TCN的完整轮次恢复对照测试；只在明确的 `check --training` 中启用，无设备会skip而不是冒充通过。本机未运行这些加速器训练测试。
- 原生启动实现从脚本移到 `backend/native_launcher.py`，脚本只做兼容转发；`uv run app` 仍只读、启动停止、不自动访问相机。
- 重写根README，新增[快速开始](quickstart.md)和[数据格式](dataset_layout.md)；新增Windows/macOS/Linux CPU CI配置，**尚未据其运行结果认证兼容性**。

## 实际验证（2026-11-02）

环境：Linux x86_64 / Python3.12.15 / 当前CPU PyTorch2.15.0+cpu；未下载CUDA大包，未运行神经训练、真实相机或真实应用控制。

| 检查 | 结果 |
| --- | --- |
| `uv run check` | **814 passed，4 deselected**；另6个训练/梯度文件未收集；12项合成统计回退/损坏wheel fixture警告 |
| 默认数据准备 | 合成公开格式ZIP → 严格导入 → train/val/test真实v3缓存 → 回执，幂等/篡改/缺主体计划检查通过；不代表正式数据已就位 |
| 训练入口 | 缺数据阻断、dry-run无训练、唯一run目录、失败后指针、恢复后模型包版本等mock编排检查通过 |
| 设备适配 | 选择策略/不可用设备/确定性配置mock、CPU真实前向及随机状态往返通过；没有执行CUDA/MPS训练 |
| 发行包 | 离线sdist/wheel构建通过；临时安装后在仓库外加载全部13个入口及缺数据拒绝通过。复用已有Python依赖，不是新机器完整依赖闭包认证 |
| `uv lock --check --offline` / Ruff基础规则 / diff检查 | 通过 |
| Flutter analyze / 单元widget | analyze通过，最终**95项通过** |
| 实际公开下载 | 标注ZIP仍为82,274字节，实际SHA与作者清单不符，正确隔离并返回失败；主关键点包未继续下载 |

首次Flutter复跑有一个按钮测试因固定40ms等待图像解码而失败。已将按钮/命令上下文测试与单独的真实解码/绘制门控测试分离，不再靠固定sleep猜完成；没有修改生产150ms门限或放宽原生停止期限。原生窗口的历史间歇性断连仍是独立未解事项，不能以widget通过消除。

## 仍然不能标为“全部准备好”的内容

1. **数据**：官方标注SHA不一致；主关键点包未下载；真实主体/会话分组与质量审核未提供。
2. **训练**：本机遵守训练暂停。新设备上的完整训练、优化器/断点恢复及真实MLP/TCN成果待 `uv run check --training` 和正式运行验证。
3. **平台**：Windows/macOS CPU CI未在本次会话运行，CUDA/MPS未做硬件训练。只保证明确设计和失败提示，不保证任意架构/驱动都有兼容wheel。
4. **原生产品**：只有Linux Flutter宿主；影子MLP间歇性断连未解决；Windows/macOS原生壳、移动端和独立安装器未交付。

## 给接手训练者的顺序

先按[快速开始](quickstart.md)准备环境 → 解决上游数据版本与主体审核 → `download` → `prepare` → `train --dry-run` → 在获准训练的机器执行 `check --training` → MLP/TCN正式训练 → val评估/校准 → 冻结后test评估。

不要发送开发机`.venv`，不要修改旧checkpoint身份绕过恢复检查，不要把合成模型或自动生成的draft主体模板当正式训练依据。
