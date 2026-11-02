# tests：公共测试与回归验证

> **简化入口**：`uv run check` 默认运行非训练回归；`uv run check --training` 才包含神经训练测试，只在允许训练的机器上执行。新增工作流/下载/设备选择/数据准备回归及跨OS CPU CI任务；CI配置存在不代表已实际通过。见[新流程](../docs/quickstart.md)。

> **原生影子会话**：HSP5跨语言协议、同帧概率/独立epoch、绘制令牌、owner重新预热、停止/释放和可移植启动器回归；`scripts/check_native_shadow.py`用临时未训练模型/程序输入跑原生检查，不打开真实相机。当前[无训练回归命令与验证边界](../docs/native_shadow_session.md)。

> **公开策略阶段**：57项新增专项、607项广域非训练回归通过；验证15类独立策略、释放/回程、owner/epoch、surface/心跳、停止竞态与1→2→3→2应用内合成回放。不运行训练/梯度测试或真实相机，见[命令与边界](../docs/public_gesture_policy.md)。

> **T15评估阶段**：72项专项、550项广域非训练回归通过；使用手算分数、独立合成事件、明确未训练的零参数模型包，只做前向。广域回归另排除4项训练用例及5个训练/梯度测试文件，未运行全量pytest。命令、警告和边界见[评估记录](../docs/evaluation_and_calibration.md)。

> **本机暂停神经网络训练**：数据扩容阶段运行117项非训练回归，4项训练相关用例明确排除；缓存容量探针只做数据准备。未筛选的完整pytest包含合成训练，请不要在当前机器直接运行；本轮命令和待训练机验证项见[数据扩容记录](../docs/public_data_scale.md)。以下全量测试数量是前序阶段历史结果。

> T19内置演示阶段：171项不含训练的Python回归、56项Flutter单元/widget及1项实际原生演示集成通过。没有运行完整pytest、神经网络训练、真实相机或外部输入注入。见[演示记录](../docs/local_presentation.md)。

> T16阶段一：566项Python、51项Flutter通过；另有原生v2回归/v3坏包拒绝验证。新增模型包语义/完整性、无pickle加载、离线在线logits、取消/新鲜度/同帧绑定与原生展示测试。训练与感知注入均为合成样例，本轮不打开真实相机。见[验证说明](../docs/model_bundle_and_inference.md)。

> 相机阶段：**505项Python、38项Flutter通过**；另有常规原生集成及显式真机相机开关/重开验证。默认测试仍不访问相机；真机需`HANDSHAKE_CAMERA_TEST=1`，只记录元数据。见[运行与证据](../docs/native_camera_preview.md)。

> 2026-10-31：**484项Python回归通过，无跳过**；另有23项Flutter单元/widget与1项真实Linux原生集成通过。新增`test_desktop_bridge.py`及`frontend/native/test/`、`integration_test/`，覆盖协议/有界管道/心跳/退出、只读回看/镜像/确认与布局；不控制真实应用。见[原生验证与命令](../docs/native_stack_decision.md)。

> 最新公开数据增量：**461项通过，无失败/跳过**，含归档/时间轴/分组、15类缓存、MLP/TCN真实梯度与恢复、旧交互隔离；[JUnit](../docs/verification/pytest_public_data_pipeline.xml)。全部新增数据是合成格式样例，不是官方数据上的识别效果。

> 累计154项自动测试通过（原有131项＋23项模型探针回归），见 [修复记录](../docs/perception_diagnostics.md)。覆盖核心、采集、质量/关联、分块存储与CLI，以及探针超时/SIGKILL/日志/内存监控/资产失败报告。真实模型加载和黑图推理另行实测通过；真手效果、人工GUI和完整实机安全验收尚未通过。

## 两人分工

- A：特征、数据防泄漏、MLP/TCN、训练恢复、校准/指标和导出一致性。
- B：相机/感知状态、轨迹、界面、事件状态机、调度/后端和异常恢复。
- A+B：录制回放、端到端集成、接口兼容和最终候选回归。

## 后续布局

```text
tests/
├── README.md
├── unit/                  # 公共契约、特征、模型、指标和状态逻辑
├── integration/           # 模块拼接、配置、模型加载、UI/mock 后端
├── replay/                # 固定逻辑/调度轨迹的事件回归
├── safety/                # 停止、失跟、乱序、阻塞、副作用与释放
└── fixtures/              # 小型纯合成/经明确授权且可公开的样例
```

这些子目录按实现逐步创建；真实采集数据不因叫 fixture 就可以公开。

## 第一批重要用例

- 镜像、宽高比、尺度、时间单位、缺失掩码、归一化统计来源和 ready。
- 离线/在线同输入同特征；改变未来帧不影响因果网络过去输出。
- 窗口不跨用户/轨迹/录制，测试不进入统计拟合。
- 挥动回程不二次翻页，冷却结束但未释放不重新执行。
- 旧 stream/epoch、过期预测不执行，也不推进交互状态。
- 停止与普通提交竞态，异常但实际已按下，以及晚到副作用的释放。
- 前台是演示应用、UI 卡顿、模型/后端阻塞时的独立停止。
- 候选包冻结后重跑相应安全和回放用例。

## 原生多端测试交接

- T36 验证原生框架/平台插件和本机桥接；T20 桌面 UI 与核心独立停止、崩溃/失联处理。
- T37 用 golden fixtures 对照 Python 与移动核心的特征/窗口/ready/logits/事件，测真实机型前后摄像头、旋转/镜像、后台/锁屏/权限撤销、离线运行、热降频和电量。
- 各 OS/ABI 的构建/安装 CI 与真机相机/控制证据分列，缺机器标未验证，不默认跳过即通过。
- T38 额外测认证/配对撤销、网络延迟/乱序/重放、时钟不确定度、断连 watchdog 和重连不补动作。
- 所有平台按 T24/T29 补安全与发布回归，测试包不能自动获得 live 或公开分发授权。

## 运行边界

常规自动测试使用合成输入和 mock/dry-run，不需要真实参与者数据、相机或桌面注入权限。硬件/应用/live 测试必须显式选择，并在隔离的无敏感内容环境中授权运行。

T02/T05 当前测试运行：`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest`。覆盖契约、配置、合成标注和 mock 集成，不需要相机、模型或设备权限；具体执行结果见 [验证记录](../docs/verification_t01_t02_t05.md)。完整安全与回放尚属 [T23](../task_plans/23_replay_and_integration.md)、[T24](../task_plans/24_safety_faults_and_performance.md)。新增采集测试为 `test_capture.py`、`test_perception.py`、`test_recordings.py`、`test_collection.py` 与可选的 `test_opencv_io.py`；OpenCV测试只生成合成像素/视频，不访问真实相机。

T11/T12/T13 增加特征/统计/窗口、用户划分与缓存、规则/MLP/TCN因果/梯度及统一预测器测试。完整测试先安装`uv sync --locked --extra dev --extra training`；缺少可选torch时神经模型测试会明确skip。实际309项全通过的环境和合成边界见 [新验证记录](../docs/verification_t11_t12_t13.md)。

返回 [根目录说明](../README.md)。
