# 当前工程、采集链路与原生多端边界

当前实现是单仓库Python项目，`handshake-core`为发行名，`shared/backend/algorithm/frontend`为职责包。T03/T04/T06采集/记录、T11～T13特征/模型及T14/T17/T18训练/交互/调度模块均已落盘；没有原生产品UI、移动应用、HTTP/WebSocket服务或真实系统输入实现。目标架构为[原生多端](native_architecture.md)，OpenCV工具窗口不等于T20产品UI。

T11/T12/T13使用共享几何、训练统计和窗口实现，详见[特征与模型](features_and_models.md)。下方T02链路仅指保留的零值fixture冒烟，不代表真实感知、特征或训练结果；正式端到端控制闭环仍待后续集成。

## 新增 T03/T04/T06 链路

`CameraSource → CapturedFrame（只读RGB数组）/FrameInfo → 有界最新帧与未处理账本 → VIDEO感知/质量/两手关联 → ObservationFrame → 授权后异步分块NPZ/manifest/events → 骨架标注/严格JSON导出`。

相机实时输入丢旧帧并保留状态；离线视频背压保留所有解码帧。未授权预览不保存资料，采集路径不构造动作后端。原FramePacket作为JSON边界保留，显式转换，不改旧schema。完整字段见 [感知协议](perception_contract.md) 和 [采集指南](collection_workflow.md)。

131项软件测试和短相机探针通过；历史主机MediaPipe初始化exit137；当前主机SDK初始化＋空图推理通过，人工GUI/真手/真实会话仍待验证。`inspect/recover`不等于T23完整事件回放。

## T02 既有合成链路

`FramePacket（1×1 合成 RGB） → FrameStatus（脚本） → FeatureWindow（32×94 零值 fixture） → MockPredictor → MockGestureEngine → ActionCommand → MockDispatcher → MockBackend → JSONL`

这些是保留的T02软件连接验证，processed_hands为合成状态，零值fixture未调用感知或94维特征提取。真正的特征与模型模块已由T11/T13提供；T17/T18已提供独立唤醒/释放/调度模块，T14已有训练器，详见[新增模块](training_and_interaction.md)；正式数据/模型与真实后端验收仍未完成。mock预测不提供intent、phase或校准概率。

控制器默认未启用、无 owner；合成脚本显式启用并绑定合成 track。每次启停、模式/模型/owner 变化、挂起、重连都会增加 epoch，重连还要求新 stream。提交工作前捕获上下文，后续对象保留同一 stream/epoch；预测返回时不得贴上新 epoch。mock 在发事件与派发前复检当前上下文、新鲜度/过期、目标白名单和能力，重复 command_id 被拒绝。当前检查为同步测试实现，不宣称处理了 T24 的线程竞态或迟到外部副作用。

## 契约

权威定义在 `shared/contracts.py`。所有对象带 `schema_version=1.0.0`、`time_unit=ns`、明确 timebase。T02 特征版本 `landmarks94-v1`，词表 `gesture-v1-draft`，后续修改语义必须升级版本。

| 对象 | 形状/单位/语义 |
| --- | --- |
| FramePacket | RGB8 `[H,W,3]`，image_size 为 `[width,height]`；capture_timestamp_ns 是源采集时间 |
| FrameStatus | 区分 processed_hands / processed_no_hand / not_processed / error；未处理/错误必须有原因 |
| HandObservation | 21×3有限坐标、21个 bool 派生有效性；未知 handedness_score 保留 null，不伪造关键点置信度 |
| FeatureWindow | features `[T,94]`，同长度严格递增 timestamps_ns；T 是接口变量，首版训练配置 T=32 |
| ModelPrediction | 5个 logits；可选校准概率和意图/阶段头；源帧与推理完成时间分开 |
| GestureEvent | 候选起点 ≤ 源时间 ≤ 提交时间；no_action 不能产生事件 |
| ActionCommand | 动作枚举、结构化参数、必需能力、目标、有效期；连续更新带 control_stream_id/update_seq 成对出现 |
| ActionResult | accepted 与注入状态分开；uncertain 可保留可能副作用；注入确认与应用响应确认不同 |
| ContextChange/HealthEvent | 回放所需上下文变更/健康原因；不以预测真值驱动在线状态 |

Python 内部时间是非负 int64 ns，拒绝 float、bool、字符串、负值；记录相对 recording 起点。无法仅凭整数大小识别误标单位，因此来源适配器仍须显式转换并设置 timebase；软件采集时间不是曝光真值。JSONL 支持整数无损回读；未来跨语言 IPC/网络协议须设计严格整数或十进制字符串编码，避免运行时数值精度损失。跨设备单调时钟不可直接比较，源时间与接收时间需分开，T38 另行设计新鲜度界限。

当前使用列表保证 JSON 序列化和形状校验；真实视频/大量数组性能优化在 T03/T11 适配边界实测，不将本轮小 fixture 性能当摄像头性能。模型坐标不镜像，预览镜像信息单列。缺失有效性和 no_action 类别不同。

## 配置、记录与错误

六类 YAML：runtime / interaction / gestures / actions / training / privacy，另有 platforms/mock。`shared/config.py` 拒绝未知字段/动作、映射缺失、重复 YAML 键、负时长、错误版本、平台能力和目标冲突。默认文件作为包资源提供，仓库 `configs/` 是同一来源的链接，外部可复制成独立配置目录并通过 `--config-dir` 指定。所有阈值是 draft。

`dry_run` 合成冒烟保留；新增显式 `preview/camera_probe/collect/inspect/recover/annotate` 工具。`train --train-cache ... --val-cache ... --output ...` 已接入T14；`evaluate/replay/live` 仍返回未实现/不可用，不存在隐藏live开关。

JSONL 包含运行 ID、配置 SHA256、发行版/Git 版本、dirty 标志、schema、timebase 和 synthetic 标记。文件以独占创建方式写入，拒绝覆盖既有证据；读回校验 payload 类型、schema 与 timebase。默认不写图片、身份或真值。日志读回不是 T23 的完整确定性回放。NPZ/会话读写现由T06的 `shared/recordings.py` 和 `backend/collection/` 实现；完整事件回放仍归T23。

## 后续原生桌面/手机交接

T36 冻结原生技术路线和相机/感知/推理/执行适配边界，T20 实现桌面壳；公共契约与训练代码不依赖 UI 或网络库。界面通过受控意图接口请求启停、模式与输入切换，显示核心快照，不直接执行设备操作。相机统一源时间、旋转、镜像与色彩元数据。

T37 实现 Android/iOS 本机感知、便携模型运行时、应用内控制与生命周期安全；Python 仍为训练和桌面参考，移植特征/事件需 golden fixtures 一致性验证。手机无需浏览器、电脑服务或完整训练环境即可完成本机闭环。暂停/锁屏/权限撤销默认停止，恢复不自动 live。

T38 才引入可选配对控制：明确目标、认证、序号/新鲜度、旧会话拒绝与目标端 watchdog，不能从手机发送任意系统命令。所有平台按安装、相机、本地识别、应用内/外部控制分层记录支持状态；本次没有实施任何上述新组件。
