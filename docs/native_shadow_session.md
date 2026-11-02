# 原生影子会话：明确选择 owner，不执行应用命令

## 范围

`--shadow-session` 将原生相机观测接到已有的 [模型会话桥](model_policy_bridge.md)，不复制 owner/释放/手势策略。启动仍为停止；必须分别打开相机、启用影子识别、选择当前合格手为 owner。绑定后切换模型 epoch、清空旧窗口，再完成 **32 个新观测的预热和中性释放**。

**这里只显示候选，不翻页、不控制外部应用。** 内置影子宿主页码固定为 1、revision 固定为 0；后端停止位保持置位，调度授权恒为否。HSP5 没有执行模式、下一页或系统输入命令。原来的人工演示页面是独立会话，不会因影子候选改变页码。

本轮没有训练神经网络、打开真实相机、保存真实图像或验证真实识别准确率。正式模型、来源方向语义审查、事件参数冻结和实机性能仍待完成。

## 启动（Linux 桌面）

在仓库根目录准备 Python 3.12 与锁定依赖。`training` extra 在这里仅提供 PyTorch **推理**依赖，不会自行开始训练：

```bash
uv sync --locked --python 3.12 --extra dev --extra perception --extra training
# 先安装 Flutter 3.47.9 及 Linux 构建依赖，并将 flutter 加入 PATH。
.venv/bin/python scripts/run_native.py --shadow-session \
  --perception-model /path/to/hand_landmarker.task \
  --model-package /path/to/verified-model \
  --model-sha256 <可信导出回执SHA256> \
  --calibration /path/to/validation-calibration.json \
  --calibration-sha256 <可信校准回执SHA256>
```

Flutter 可通过 `--flutter /path/to/flutter` 或 `FLUTTER_BIN` 指定；已有 Linux bundle 可使用 `--bundle <可执行文件>`。路径来自本次参数和当前解释器，不含开发者主机目录；启动器清除继承的旧模型/校准/影子授权环境变量。不要从任意下载目录现场计算 SHA 后冒充可信外部回执。

模型须为 `ipn-hands-v1`，校准须与模型/验证集库存绑定且阈值可用。未获明确许可的 synthetic 模型会在打开相机/检测器前失败。诊断模型必须另传 `--allow-synthetic-model`，界面持续显示合成警示；这也不会授予动作执行权限。缺少配套参数会在启动前报错，不自动下载资产、选择模型或回退到假预测。

使用顺序：

1. 打开相机预览。加载资产失败时不显示旧预测，不开放 owner 按钮。
2. 等待真实图像解码/绘制确认，再点击“启用影子识别”。
3. 单手保持中性，等待当前可信 NOG 分类和同帧图像；点击“选择当前手为影子owner”。点击只是请求，还需下一条新模型结果确认，不能排队等待未来任意手势。
4. 界面显示 owner、新模型代际、重新预热/释放状态及候选。所有候选标记“未执行”。
5. 回看禁止启用/选手/绘制确认；停止、Esc、窗口失活/隐藏、断连均撤销会话，不自动重新启用。停止确认要等待相机、感知和模型 worker 释放，不把“收到停止请求”当作“释放完成”。

## 协议与线程边界

- `shared/shadow_preview_protocol.py` / `.schema.json`：`HSP5` 二进制前缀 + 有界 JSON + JPEG；保留 v2～v4，禁止流内切换版本。计数器与纳秒仍为十进制字符串，Dart 使用 `BigInt`。
- 顶层 `epoch` 是相机生命周期；`shadow.model_epoch` 是模型/owner 代际。分类必须匹配 **模型**代际、stream、版本和图像帧号/时间，不把旧结果重贴成新代际。
- `surface_ack`、`enable_shadow`、`request_owner` 必须携带捕获时的模型 epoch/sequence/revision；owner 请求还携带可见 track。停止不要求模型令牌，因此模型卡住也可先撤销权限。
- `backend/perception/shadow.py` 复用 `ModelPresentationSession(execution="shadow")`。模型工作不持有停止闸门；只有一个处理调用、一个最新编码槽。原生子进程负责隔离不可取消的 SDK 调用，超时不伪造资源释放 ACK。
- `frontend/native/lib/stage/preview_surface.dart` 只确认实际解码并绘制的图像。绘制可落后于遥测，但只能使用原始、同代际、未过期令牌；不能改贴最新 sequence。owner 证据保留 150ms 门限，令牌保留 500ms 门限。
- JPEG 最新槽的影子编码间隔为 30ms，协议输出仍上限约 10Hz；避免两个独立 100ms 限速叠加耗尽证据寿命。处理实际频率仍由设备/模型决定，未宣称达到 30FPS。
- 冷启动连接最多等待 10s，尚无输入/授权；连接后的失联与停止确认期限仍为 2s。没有为通过测试放宽模型 100ms 间隔、150ms 新鲜度或 owner 预热总截止。

## 无相机的一键原生检查

```bash
.venv/bin/python scripts/check_native_shadow.py
# PATH 中没有 Flutter 时：
.venv/bin/python scripts/check_native_shadow.py --flutter /path/to/flutter
# 同一程序生成场景，换用未训练 TCN 的真实前向：
.venv/bin/python scripts/check_native_shadow.py --model-type tcn
```

此脚本需要源码 checkout、上述 Python extras、Flutter 和可用的 Linux 图形会话。它：

- 在系统临时目录创建**手工赋权、未训练**网络及合成校准，结束后清理；不用用户数据/权重，不调用 trainer/optimizer。
- 显式启用 `tests/fixtures/native_shadow_core.py` 程序生成输入。该入口只在测试包中，默认拒绝启动，不打包进发行 wheel；生产入口没有测试数据切换开关。
- 不继承 `HANDSHAKE_*` 资产、截图路径或 Python 注入环境；保留当前虚拟环境解释器，不解析它的符号链接改用系统 Python。
- 打开测试窗口，实际解码/绘制程序生成 JPEG、接收 Python HSP5、明确请求 owner、观察预热/候选、停止和重连。还检查生产入口缺失模型资产时的拒绝。
- 以退出码报告结果，不在失败后自动重试。`--timeout` 默认 180s，超时返回 124。

按钮点击与禁用状态由 widget 测试覆盖；原生异步测试在新鲜的**实际绘制**证据通知到达时发出一次明确 ViewModel 意图，不把跨帧 `pump` 后已过期的证据当点击成功。它不是人工可用性或真实相机测试。

## 回归与证据边界（2026-11-02）

无训练 Python 回归命令：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  --ignore=tests/unit/test_training.py \
  --ignore=tests/unit/test_model_losses.py \
  --ignore=tests/unit/test_mlp.py \
  --ignore=tests/unit/test_tcn_causality.py \
  --ignore=tests/integration/test_model_bundle.py \
  -k 'not test_public_import_to_deep_training_and_resume and not test_compact_training_resume_and_model_export'

cd frontend/native
flutter --no-version-check analyze
dart format --output=none --set-exit-if-changed lib test integration_test
flutter --no-version-check test
```

新增检查覆盖跨语言 golden/schema、伪造执行权限、不同 epoch、旧/过期绘制令牌、回看禁用、同帧概率、owner 重新预热、阻塞模型下停止/释放、缺资产与未授权合成模型前置拒绝、启动路径与临时资产清理。worker 集成使用显式虚拟源时钟避免把 CI 调度抖动误当相机性能；原生窗口检查仍使用真实单调时钟。

调试中曾出现首次 worker 预热超时、原生绘制轮询错过短暂可选手状态，以及窗口检查触发看门狗停止。已补绘制令牌回归、分离冷启动期限、缩短编码槽年龄、改用事件驱动测试意图；程序生成动作使用与会话桥一致的 1.2 图像宽度/秒诊断信号。**单次原生通过不是高负载稳定性证明**；真实摄像头帧率/调度不足时仍应拒绝或停止，不能延长阈值冒充验证通过。

实际验证环境为 Linux / Python 3.12.15 / Flutter 3.47.9。结果如下，不把失败后的历史单次成功合并成“全绿”：

| 检查 | 本轮结果 |
| --- | --- |
| 上述无训练 Python 选择集 | **748 passed，4 deselected**；另外5个训练/梯度文件未收集；8项既有合成统计/损坏wheel警告 |
| Flutter analyze、格式及单元/widget | analyze通过，**95项测试通过** |
| 原生 TCN 检查（最终诊断信号） | 两项通过：owner/预热/候选/停止/重连，以及坏资产拒绝 |
| 原生 MLP 检查 | 曾完整两项通过；**最后一轮1项失败、1项通过**，启用前核心断连导致等待超时，仍未完成稳定验收 |
| Python sdist/wheel | 离线构建通过；含HSP5 schema，不含测试源；临时安装后在仓库外启动/停止HSP5通过，未打开设备。复用了已有Python依赖，不是新机器完整依赖闭包认证 |
| 锁文件与基础静态检查 | `uv lock --check --offline`、Ruff `E9,F63,F7,F82`、`git diff --check`通过 |

**接续优先项**：定位原生 MLP 间歇性断连，给模型加载/协议心跳/进程退出补有界阶段诊断，评估模型 worker 的独立进程隔离；不能仅重试到通过或放宽安全期限。之后在干净Linux图形会话验证连续多轮与冷启动。T20/T23整体、真实手势到应用执行、Windows/macOS/手机和独立安装器仍未验收。
