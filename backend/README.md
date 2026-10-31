# backend：实时运行与安全终端控制

> 相机预览入口`python -m backend.desktop_preview`提供v2二进制图像/骨架通路，复用既有CaptureWorker/PerceptionPipeline；需显式`start_camera`，不记录数据、不接手势分类或应用动作。见[相机阶段](../docs/native_camera_preview.md)。

> 新增`python -m backend.desktop`：Flutter合成原型的本机私有管道核心，独立状态线程、容量1输出邮箱、心跳/EOF与有界退出。它不接相机、推理、T17/T18或真实应用操作；见[原生阶段决策](../docs/native_stack_decision.md)。

> T02 核心之外，T03/T04/T06 已实现相机/视频/合成源、感知适配/质量/关联和授权采集/标注工具。T17/T18已实现旧5类交互状态机与有界安全调度；没有产品UI或真实系统操作；兼容依赖下真实模型加载/黑图推理已通过，真手/人工工具待验，见 [修复记录](../docs/perception_diagnostics.md)。

## 主责与任务

- **B 主责**；A 提供 Predictor、共享特征与模型包。
- 主要任务：T03/T04/T06、T17～T19、T21～T24、T34/T35；参与 T36 原生接口、T37 移植和 T38 可选 Agent。
- 详见 [任务导航](../task_plans/README.md)。

## 后续模块划分

| 计划模块 | 职责 |
| --- | --- |
| `capture/` | 相机/视频输入、源时间戳、最新帧槽、重连 |
| `perception/` | MediaPipe 适配、关键点质量、无手/未处理/失败区分 |
| `tracking/` | 轨迹关联、控制权与多人冲突证据 |
| `collection/` | 授权采集流程、分块写入调度与会话完整性 |
| `interaction/` | 唤醒、候选、回程/释放、冷却、连续控制 |
| `actions/` | 命令映射、唯一调度器、dry-run/mock/真实平台适配 |
| `telemetry/` | 结构化日志、健康监控、计时与事件轨迹 |
| 应用控制器 | 生命周期、运行上下文、输入/模型/模式切换、停止 |

`capture/`、`perception/`、`tracking/association.py`、`collection/` 已落地；`interaction/`、`actions/` 已实现独立模块与合成故障检查；真实集成/平台仍待T19/T23/T24。公开版UI/键盘启停与15类契约迁移未实施，见[训练与交互](../docs/training_and_interaction.md)。命令见 [采集操作指南](../docs/collection_workflow.md)。

感知诊断统一由 `backend/perception/probe.py` 执行；T14旧入口 `diagnostics.py` / `scripts/probe_perception.py` 仅兼容转发，保留原参数/摘要格式。

## 在线链路

```text
相机 → 关键点/质量/轨迹
     → shared 共享特征
     → algorithm Predictor
     → 安全状态机 → 命令调度器 → 已支持终端应用
```

前端发送用户指令，后端发布状态快照；预测不是操作授权。只有本目录中的调度器可使用真实后端，其他模块不能绕过它直接注入设备事件。

## 原生多端边界

本目录是本机运行核心，不等于 HTTP 服务器。相机、感知、权限/生命周期和系统操作通过适配器访问；平台库按需加载，未知平台仍能进入 mock/回放但不能启用未知 live 能力。T36 冻结接口，T37 复用/移植到手机本地运行时，并对照 Python 参考验证状态与停止一致性。

手机后台/锁屏/权限撤销默认停用并清旧工作，恢复仅预览；无可靠持有释放时禁用该能力。T38 才加入可选远程命令入口，目标 Agent 本地再次检查授权/能力/期限，并以独立 watchdog 处理断连，不依赖手机退出回调。架构与能力边界见 [原生多端说明](../docs/native_architecture.md)。

## 必须保留的安全约束

- 默认 dry-run，live 需人工启用、明确目标能力和控制权。
- 工作提交时捕获 stream/epoch，旧结果不得重新贴上当前代际。
- 启停、失跟、模式/模型/owner 变化清除不再有效的候选和命令。
- 紧急停止不等待推理、UI 绘制或普通事件队列；外部调用的不可撤销边界要披露。
- 对“确认按下”和“可能已按下但回执异常”都执行安全释放。
- 绝对位置可覆盖，速率单点积分，不能直接丢弃相对音量增量。
- 能力/焦点/权限不满足时拒绝执行，不自动提权或静默重试非幂等操作。

## 向 A 交接

提供带时间轴、处理状态、质量/轨迹和版本的授权观测资料；数据落 `data/`，运行日志落 `artifacts/`。共享读写格式和特征使用 `shared/`，不要各写一套。

先完成合成事件和 mock 后端，再接真实关键点/模型，最后受控 live；安全与回放测试放根 `tests/`。

返回 [根目录说明](../README.md)。
