# HandShake

基于 MediaPipe 手部关键点与自主训练时序神经网络的手势终端控制项目。

> 已按 T01 → T02 → T05 建立范围记录、可安装 Python 核心、配置/契约、合成 dry-run 和手势标注协议草案。尚无真实数据、训练模型、相机采集、真实翻页或 Web。最终产品采用 Web，需支持电脑和手机；本轮暂不实现 Web。

## 先看这些文档

- [项目总目标与技术方案](PROJECT_MASTER_PLAN.md)
- [35 个实施任务与依赖](task_plans/README.md)
- [任务审查记录](task_plans/REVIEW_NOTES.md)
- [文档与答辩资料](docs/README.md)
- [本轮范围与待确认项](docs/scope_freeze.md)
- [手势协议](docs/gesture_protocol.md)与[标注指南](docs/annotation_guide.md)
- [T01/T02/T05 验证记录](docs/verification_t01_t02_t05.md)

课程版先完成 **手势翻页、基础界面、自训 TCN、MLP 对照、安全控制与实验报告**；音乐、光标和研究增强按总方案优先级推进。

## 根目录划分

```text
HandShake/
├── frontend/              # 未来电脑/手机 Web 前端；本轮仅保留职责包
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

- 这里的“前端/后端”先作为职责分区；用户已确定最终为电脑/手机 Web，当前只实现可复用核心与 CLI，尚无 HTTP 服务。
- 首版采用单仓库、一个 Python 工程和共用环境；UI 与工作单元通过接口/消息协作，不为分目录强制引入微服务、数据库或 Docker。
- Python 发行名为 `handshake-core`，显式配置四个职责包的发现；依赖见 `pyproject.toml` 和 `uv.lock`，当前机器验证结果见 [依赖记录](docs/dependencies.md)。
- `shared/` 不反向依赖其余三个模块；`algorithm/` 不导入设备控制或界面代码；只有 `backend/` 的调度器能走真实操作后端。
- 共享特征只维护一份，离线训练与在线推理使用相同实现。紧急停止不依赖模型返回或界面绘制。

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

补充课程与演示设备信息 → T03/T04 相机/感知 → T06 授权采集 → T07 真实试采集和双人试标 → 后续数据、模型与安全闭环。A/B 分工由用户之后填写。Web/手机 UI 在后续 T20 实施，本轮无网页。

## 安装与合成验证

以下在 Linux x86_64 / Python 3.10.12 验证。推荐用 uv 按锁文件创建/更新本项目 `.venv`；其他 Python/宿主须重新验证，不以锁文件推断兼容性。

```bash
cd /home/soaring/programs/HandShake
uv sync --locked --extra dev
.venv/bin/handshake dry_run
.venv/bin/handshake dry_run --config-dir configs --log artifacts/t02-smoke.jsonl
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
```

`dry_run` 只运行一段有限的纯合成脚本并输出版本化 JSONL；不访问相机、不训练、不注入设备、不启动 Web。日志路径父目录须已存在，拒绝覆盖同名文件。没有参数时也默认 dry_run。`collect/train/evaluate/replay/live` 已保留 CLI 名称，当前均返回退出码2及对应未实现任务说明。

当前环境的 ROS `PYTHONPATH` 和全局 pytest 插件会干扰测试，因此命令显式隔离；普通环境也可使用同一命令。基础链路不导入 MediaPipe/OpenCV/PyTorch。

可选依赖仅用于导入/CPU操作验证，安装不等于相机或模型已可用：

```bash
uv sync --locked --extra dev --extra perception --extra training
env -u PYTHONPATH .venv/bin/python scripts/check_environment.py
```

uv 配置将 Linux/Windows 的可选 torch 指向官方 CPU 索引；感知依赖只使用 `opencv-contrib-python` 一个 cv2 提供包。`shared/defaults/` 是打包默认配置，`configs/` 中的链接指向同一来源；若系统不支持符号链接，可直接使用包内默认值，或复制 YAML 为独立目录传给 `--config-dir`。

完整边界见 [架构与契约说明](docs/architecture.md)和[平台支持矩阵](docs/platform_support.md)。T01 设备/课程确认、T02 真实相机加载和 T05 真实双人标注仍待验证，不能以合成测试替代。
