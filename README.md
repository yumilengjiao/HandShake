# HandShake

基于 MediaPipe 手部关键点与自主训练时序神经网络的手势终端控制项目。

> 已实现T01/T02/T05核心与协议、T03/T04/T06相机/视频输入与授权分块采集/骨架标注，以及T11/T12/T13的94维共享特征、数据/缓存、规则/MLP/因果TCN。真实MediaPipe初始化exit137和完整实机验收仍待解决；尚无参与者训练数据、正式训练权重、产品UI或真实翻页。当前目标按native-v2采用原生多端，桌面与手机分别验收。

## 先看这些文档

- [项目总目标与技术方案](PROJECT_MASTER_PLAN.md)
- [38 个实施任务与依赖](task_plans/README.md)
- [原生多端架构与平台边界](docs/native_architecture.md)
- [任务审查记录](task_plans/REVIEW_NOTES.md)
- [文档与答辩资料](docs/README.md)
- [本轮范围与待确认项](docs/scope_freeze.md)
- [手势协议](docs/gesture_protocol.md)与[标注指南](docs/annotation_guide.md)
- [T01/T02/T05 验证记录](docs/verification_t01_t02_t05.md)
- [T11/T12/T13 特征与模型使用说明](docs/features_and_models.md)及[验证记录](docs/verification_t11_t12_t13.md)
- [采集/感知/骨架标注操作指南](docs/collection_workflow.md)
- [T03/T04/T06 验证与当前阻塞](docs/verification_t03_t04_t06.md)

课程版先完成 **手势翻页、原生桌面基础界面、自训 TCN、MLP 对照、安全控制与实验报告**。产品目标覆盖 Windows/macOS/Linux、Android/iOS/iPadOS；手机本地运行列为 T37 后续交付，其他 OS 按能力评估。可安装、本机识别、应用内控制、外部应用控制分别验收，不承诺任意系统控制任意 App。

## 根目录划分

```text
HandShake/
├── frontend/              # 原生桌面/移动界面；目前只有职责包，技术路线待 T36
├── backend/               # 运行后端：相机、感知、状态机、安全调度、终端适配
├── algorithm/             # 算法：数据加载、训练、评估、模型推理与导出
├── shared/                # 公共契约、共享特征、配置校验与数据格式
├── data/                  # 数据集：原始资料、关键点、标签、划分、特征缓存
├── artifacts/             # 产物：模型权重、训练记录、实验结果、运行日志
├── configs/               # 可版本管理的运行、训练、手势及平台配置
├── scripts/               # 薄命令入口：采集、训练、评估、回放等
├── tests/                 # 单元、集成、回放和安全测试
├── docs/                  # 设计说明、操作指南、可公开实验总结与答辩资料
├── task_plans/            # 任务拆分、依赖与实施记录
├── PROJECT_MASTER_PLAN.md
├── README.md
└── .gitignore
```

每个分区的 README 说明具体内容、负责人和交接要求。

| 目录 | 主责 | 交付重点 |
| --- | --- | --- |
| [frontend](frontend/README.md) | B | 界面和交互反馈，不直接操作设备 |
| [backend](backend/README.md) | B | 摄像头到终端控制的安全运行链路 |
| [algorithm](algorithm/README.md) | A | 自训练模型、可复现实验、统一推理接口 |
| [shared](shared/README.md) | A+B；特征 A 主责 | 两端使用同一数据契约与特征实现 |
| [data](data/README.md) | A 管版本，A+B 采集/复核 | 完整时间轴、固定划分、授权与防泄漏 |
| [artifacts](artifacts/README.md) | 按实验/运行负责人 | 权重、日志、报告及哈希关联 |
| [configs](configs/README.md) | A 训练；B 运行；公共项共同 | 显式配置、版本与安全默认值 |
| [scripts](scripts/README.md) | 按对应模块负责人 | 可复现命令，不重复业务代码 |
| [tests](tests/README.md) | A+B | 算法一致性、系统故障与安全回归 |
| [docs](docs/README.md) | A+B | 实际方案、支持范围、证据与分工 |

A/B 沿用任务规划的角色，具体由谁担任再填写；不是要求一人只做模型、另一人独自承担所有剩余工作。两人共同完成协议、采集、标注、集成和答辩，按实际投入调整分工。

## 两人如何协作

### A：算法与数据主线

主要负责 T10～T16、模型研究任务和相应测试；负责共享特征实现、数据划分、MLP/TCN、训练/评估及模型包。

向 B 交付：特征/类表版本、可信模型包与哈希、统一 Predictor 接口、校准配置、性能及限制。不能只发一个来历不明的权重文件。

### B：应用与实时系统主线

主要负责 T03/T04/T06、T17～T24 和应用扩展；负责相机、感知、界面、事件状态机、适配器及安全测试。

向 A 交付：可重载的授权采集资料、时间戳/质量/处理状态、mock 与回放入口、实机运行及失败日志。

### 共同维护的边界

1. 先约定 `shared/` 的数据对象、单位、时间基准、版本及错误类型，再并行开发。
2. 前端可以先接 mock；算法可以先用合成/pilot 数据；不相互等待全部功能完成。
3. 每次提交尽量围绕一个任务；公共接口变更同时说明影响的两端和测试。
4. 任务完成必须附实现、配置、验证和结果位置，不能只把状态改成已完成。
5. 每人都能复跑另一人的基本流程，共同完成 T23 集成和 T28 实验核查。

## 模块依赖与运行方式

```text
frontend --用户指令--> backend --调用--> algorithm 的 Predictor
frontend <--状态快照-- backend <--预测-- algorithm
                           |
                           +--> 安全事件引擎/调度器 --> 终端设备

frontend / backend / algorithm --> shared
```

- “前端/后端”是原生应用的职责分区，不要求网页或服务器；当前只有 Python 核心与 CLI。T36 冻结原生 UI/桥接路线，T20 先实现桌面，T37 实现手机本地相机、推理与应用内控制。
- 保留单仓库 Python 训练/桌面参考工程；原生移动壳及便携运行时按目标工具链构建，不要求所有端共用 Python 环境。不为目录划分强制引入微服务、数据库或 Docker；手机控制电脑是 T38 可选扩展。
- Python 发行名为 `handshake-core`，显式配置四个职责包的发现；依赖见 `pyproject.toml` 和 `uv.lock`，当前机器验证结果见 [依赖记录](docs/dependencies.md)。
- `shared/` 不反向依赖其余三个模块；`algorithm/` 不导入设备控制或界面代码；只有 `backend/` 的调度器能走真实操作后端。
- Python 训练与在线推理复用同一份共享特征。移动移植通过版本化规范和 golden fixtures 验证特征、窗口、模型与事件一致性，不各自复制修改算法；紧急停止不依赖模型返回或界面绘制。

## 与原任务路径的对应

原规划中 `src/handshake/...` 或省略前缀的路径是早期单目录示意，现在按以下分区落地，不再另外维护一套根 `src/handshake/` 副本：

| 原模块 | 当前归属 |
| --- | --- |
| `ui/` | `frontend/` |
| `app.py`、`capture/`、`perception/`、`tracking/`、`interaction/`、`actions/`、`telemetry/` | `backend/` |
| `models/`、`training/`、`inference/` | `algorithm/` |
| `contracts.py`、`features/`、公共配置/时间/记录格式 | `shared/` |
| 原源码中的 `data/` | 采集写入归 `backend/`，训练加载/增强归 `algorithm/`，共用记录读写/校验归 `shared/` |
| `personalization/` | 适配算法归 `algorithm/`，运行档案/几何控制归 `backend/`，向导归 `frontend/` |
| 根 `data/`、`artifacts/`、`configs/`、`scripts/`、`tests/`、`docs/` | 保留同名根目录职责 |

执行任务时补齐实际文件路径；这里只调整职责布局，不修改模型方案、任务编号或课程范围。

## 数据与产物管理

- `data/` 专门放数据集，`artifacts/` 专门放模型和实验产物；二者均不是源码包。
- 当前 Git 只跟踪这两个目录中的 README，其他内容默认忽略。共享数据应经授权的受控渠道传递，并核对 manifest/哈希；拉取代码不会自动获得数据或模型。
- 实名、授权原件、课程提交凭据和个人档案放本地 `private/`，已忽略，不创建公开样例。
- 可公开的脱敏说明/汇总图放 `docs/`，纯合成小测试样本放 `tests/fixtures/`；公开前仍需核查内容。
- `.gitignore` 不是访问控制或加密，不能代替授权、备份、文件权限和删除流程。

## 当前下一步

先解决真实 MediaPipe 初始化 exit137 → 补5分钟相机与人工工具验收 → 真实授权短采集/T07 双人试标。算法主线、T36 原生选型和 T20 mock 可按依赖并行；T37 手机/T38 配对后续推进。不能把合成资料当训练数据，A/B 分工与真实参与者授权仍需补齐。

## 安装与合成验证

以下基础安装原在 Linux/Python 3.10 验证；本轮 T03/T04/T06 另在 Linux/Python 3.12.14 按同一锁文件验证，见本轮记录。用 uv 创建项目 `.venv`；不以锁文件推断所有宿主兼容。

```bash
# 在 HandShake 仓库根目录运行
uv sync --locked --extra dev
.venv/bin/handshake dry_run
.venv/bin/handshake dry_run --config-dir configs --log artifacts/t02-smoke.jsonl
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
```

`dry_run` 仍只运行有限合成脚本，不访问相机、不训练、不注入事件；无参数默认 dry_run。`train/evaluate/replay/live` 仍不可用。新增显式 `preview/camera_probe/collect/inspect/recover/annotate` 工具；采集授权、格式和失败边界见操作指南，`inspect` 不等于 T23 事件回放。

当前环境的 ROS `PYTHONPATH` 和全局 pytest 插件会干扰测试，因此命令显式隔离；普通环境也可使用同一命令。基础链路不导入 MediaPipe/OpenCV/PyTorch。

可选依赖仅用于导入/CPU操作验证，安装不等于相机或模型已可用：

```bash
uv sync --locked --extra dev --extra perception --extra training
env -u PYTHONPATH .venv/bin/python scripts/check_environment.py
```

uv 配置将 Linux/Windows 的可选 torch 指向官方 CPU 索引；感知依赖只使用 `opencv-contrib-python` 一个 cv2 提供包。`shared/defaults/` 是打包默认配置，`configs/` 中的链接指向同一来源；若系统不支持符号链接，可直接使用包内默认值，或复制 YAML 为独立目录传给 `--config-dir`。

## 特征与模型模块

```bash
uv sync --locked --extra dev --extra training
env -u PYTHONPATH .venv/bin/python -m algorithm.smoke --output artifacts/model-smoke.json
```

该命令只做人造张量可过拟合/CPU前向检查和规则fixture诊断，不保存正式模型，报告文件拒绝覆盖。MLP为94→128→64→5；TCN为单卷积残差块d=1/2/4/8、56,325参数、31帧感受野。三种模型共用32帧输入就绪与新鲜度门控；MLP只取末帧但特征含速度。训练统计仅由train拟合，验证/测试构建必须复用同一统计。缓存构建命令与草案输入格式见 [模块说明](docs/features_and_models.md)，正式实验等待T04/T07/T10及T14。

## 新增工具快速开始

```bash
uv sync --locked --python 3.12 --extra dev --extra perception
# 仅短时预检，不保存摄像头画面
.venv/bin/handshake camera_probe --headless --duration 5
# 不访问相机的合成采集闭环
.venv/bin/handshake collect --source synthetic --headless --frames 90 \
  --subject synthetic-p1 --session synthetic-s1 \
  --consent-keypoints --consent-reference synthetic-only --output artifacts/collection-demo
# 验证/骨架标注刚才输出的录制目录
.venv/bin/handshake inspect --recording <录制目录>
.venv/bin/handshake annotate --recording <录制目录> --annotator A
```

真实感知先解决当前 SDK 初始化问题并提供本地 `.task`；真实采集必须另行取得授权，不能照抄合成授权引用。OpenCV工具窗口不是T20产品UI。完整命令和当前限制见 [操作指南](docs/collection_workflow.md)。

完整边界见 [架构与契约说明](docs/architecture.md)和[平台支持矩阵](docs/platform_support.md)。T01课程/设备签认、T03/T04/T06完整实机验收和T05真实双人标注仍待完成。
