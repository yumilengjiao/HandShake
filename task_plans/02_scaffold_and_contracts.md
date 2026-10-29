# T02 建立工程骨架、配置与公共数据契约

| 项目 | 约定 |
| --- | --- |
| 阶段 | S1 工程底座 |
| 课程优先级 | P0 |
| 建议主责 | A+B（A/B 实名以总方案为准） |
| 前置任务 | T01 |
| 总方案依据 | 第 5、16、17、19 节 |
| 状态 | 待验证；核心软件已实现并验证，真实相机加载待T03/T04 |

总约束和路径解释见 [任务导航](README.md)。下列代码、配置、命令和产物路径均为待实现约定，不代表目前已存在。新增细节属于总方案的实施起点；只有通过试采集和验证后才冻结，不得用最终测试结果反向挑选配置。

## 目标与边界

建立可安装、可验证的最小 Python 项目及跨模块契约。先做运行与 mock 的骨架，不一次创建几十个没有实现的空模块，也不把本文列出的依赖当作已兼容版本。

## 具体步骤

1. 在已建立的 `frontend/`、`backend/`、`algorithm/`、`shared/` 分区内逐步实现代码，根 `configs/`/`tests/` 按职责使用，不再另建 `src/handshake/` 副本。创建根 `pyproject.toml`，明确包名、包发现和入口，保持单工程/共用环境；不要求四套安装或 HTTP 服务。实测固定 Python、MediaPipe、OpenCV、PyTorch、NumPy、scikit-learn 和配置校验方案；UI 库由 T20 确定，PyTorch CPU 优先、ONNX 可选。
2. 增加 README 中的环境创建/安装/启动说明；在实际目标机器验证相机和感知包可加载后才锁版本。失败时记录平台与错误，不凭空填“最低兼容版本”。
3. 在 `shared/` 的契约模块定义下表字段，公共特征、时间/记录协议也由 shared 供两端复用；具体包路径随构建配置确认。内部时间统一整数 ns、单调时钟；录制相对 ns 并记录 timebase，序列化不混用秒/ms/ns。

| 对象 | 必须字段 |
| --- | --- |
| FramePacket | stream_id、frame_id、capture_timestamp_ns、image、image_size、transform_metadata |
| FrameStatus | stream_id、frame_id、source_timestamp_ns、perception_status、processed_at_ns、drop/error_reason；区分检测无手、未处理、失败 |
| HandObservation | stream_id、源帧 ID/时间、track_id、landmarks_image[21,3]、validity_mask[21]（派生有效性）、observation_valid、handedness、quality/quality_flags、变换元数据 |
| FeatureWindow | stream_id、track_id、context_epoch、timestamps_ns[T]、features[T,D]、schema_version、ready、ready_reason；首版 T=32 而非接口永久写死 |
| ModelPrediction | stream_id、track_id、context_epoch、source_frame_id、source_timestamp_ns、inference_timestamp_ns、logits、可选 calibrated_probabilities、模型版本、optional intent/phase |
| GestureEvent | event_id、stream_id、track_id、context_epoch、source_frame_id/source_timestamp_ns、gesture_id、候选起点、commit_timestamp_ns、mode、decision_reason |
| ActionCommand | command_id、source_event_id、stream_id、action、arguments、issued_at_ns、expires_at_ns、required_capabilities、context_epoch、target_context；连续控制另带 control_stream_id/update_seq |
| ActionResult | command_id、accepted、injection_status（not_attempted/confirmed/failed/uncertain）、error_code、completed_at_ns、acknowledgement_level；如保留 injected 布尔值，不得覆盖 uncertain |

运行上下文由控制器创建；工作提交时捕获 `stream_id/context_epoch` 并原样贯穿特征、预测、事件和命令，不能在旧结果返回时补写当前代际。重连/换输入创建新 stream；启停、模式/模型/owner 变更或挂起使旧 context_epoch 失效。录制回放另存上下文/健康事件，不能只存图片时间。

4. 配置分为 runtime、interaction、gestures、actions、training、privacy；加载时拒绝未知动作、缺少映射、负时长、错误 schema。配置示例中的门限均标起点，不称为最优默认值。
5. 入口预留 collect/train/evaluate/replay/dry_run/live。只有 live 可调用真实后端；其他模式后端依赖注入为 mock。模式切换必须经应用控制器，禁止 UI 直接改内部状态。
6. `.gitignore` 覆盖 `private/`、原视频、关键点、授权原件、用户档案、密钥、大权重和运行日志；授权状态与匿名清单的公开范围另行控制。代码、脱敏配置和小型合成 fixtures 可入库。日志/schema 从本任务起定义，T23 负责完整集成，不等待 T23 才具备可复现记录。

## 交付物

shared 契约与配置 schema、可加载默认配置、最小入口、依赖锁定记录、README 安装段；`tests/unit/test_contracts.py` 覆盖单位、形状、配置错误与序列化。目录职责按根 README 和总方案第16节落地。现有分区 README 与 `.gitignore` 仅是起点，不代表本任务已完成。

## 验收与交接

- [x] 无相机、无模型时入口返回可理解错误，不能顺带注入桌面事件。
- [x] 合成 FramePacket 能经过 mock predictor、engine、backend 并输出带版本的日志。
- [x] 错误维度与时间单位不会静默通过；`injected` 不被解释为应用已响应。
- [x] 接口变更更新版本及 T03/T04/T11/T16/T17/T18/T20 的调用方（现有mock调用方已同步，未实现任务交接到版本化契约）。

## 前置任务文件

[T01 冻结课程范围、平台与验收口径](01_scope_and_platform.md)

## 2026-10-29 实施记录

- 执行：Codex；成员A/B待用户填写。开始/软件验证日期2026-10-29，实机最终验收日期待填。
- 步骤1～2：`pyproject.toml`、`uv.lock`、四个职责包、`backend/cli.py` 和根 README 安装说明；基础/感知/CPU训练依赖导入实测见 [依赖记录](../docs/dependencies.md)。没有 Web依赖、网页或服务。
- 步骤3：`shared/contracts.py`、`algorithm/predictor.py`、`backend/controller.py`；ns/timebase、形状、可选头、context传播与注入状态保留。
- 步骤4：`shared/config.py`、`shared/defaults/` 和 `configs/`；拒绝未知字段/动作、重复YAML键、缺失映射、负时长与错误版本，门限标draft。
- 步骤5～6：`backend/mock.py`、`shared/records.py`、`.gitignore`；合成帧→mock预测/事件/派发→带版本JSONL可验证；无图像/身份进入日志。collect/train/evaluate/replay/live保留入口并明确不可用。
- 单元/集成测试、wheel安装冒烟、配置/锁/fixture哈希及结果见 [统一验证记录](../docs/verification_t01_t02_t05.md)。无真实数据或模型哈希，标不适用。
- 验收：无相机/模型可运行合成入口，其余入口可理解地拒绝；维度、单位标签、旧epoch、过期/重复命令与序列化通过。`mock=not_attempted` 不冒充注入/应用响应。真实相机与模型加载尚未完成，不宣称已完成T03/T04或后续安全竞态。
