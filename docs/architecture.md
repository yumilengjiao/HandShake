# T02 当前工程与原生多端边界

当前实现是单仓库 Python 项目，`handshake-core` 是发行名，`shared/backend/algorithm/frontend` 是职责包。安装、命令见根 README。没有原生 UI、移动应用、HTTP/WebSocket 服务或真实系统输入实现。目标架构现已改为 [原生多端](native_architecture.md)，现有源码中的旧 Web 占位说明不代表已有 Web 实现，也不覆盖最新需求；本次仅改文档。

## 当前可运行链路

`FramePacket（1×1 合成 RGB） → FrameStatus（脚本） → FeatureWindow（32×94 零值 fixture） → MockPredictor → MockGestureEngine → ActionCommand → MockDispatcher → MockBackend → JSONL`

这些都是 T02 的软件连接验证。FrameStatus 的 processed_hands 是合成状态；没有从图像检测手，没有实现 94 维特征提取、实际唤醒/释放状态机，也没有训练模型。mock 预测不提供 intent、phase 或校准概率。T11、T13、T17、T18 的生产逻辑尚待实施。

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

入口预留 collect/train/evaluate/replay/dry_run/live。只有 dry_run 合成冒烟可用；其他返回退出码2和缺失任务说明，未导入真实操作后端。没有隐藏的 live 开关。

JSONL 包含运行 ID、配置 SHA256、发行版/Git 版本、dirty 标志、schema、timebase 和 synthetic 标记。文件以独占创建方式写入，拒绝覆盖既有证据；读回校验 payload 类型、schema 与 timebase。默认不写图片、身份或真值。日志读回不是 T23 的完整确定性回放。NPZ/真实会话写入属于 T06。

## 后续原生桌面/手机交接

T36 冻结原生技术路线和相机/感知/推理/执行适配边界，T20 实现桌面壳；公共契约与训练代码不依赖 UI 或网络库。界面通过受控意图接口请求启停、模式与输入切换，显示核心快照，不直接执行设备操作。相机统一源时间、旋转、镜像与色彩元数据。

T37 实现 Android/iOS 本机感知、便携模型运行时、应用内控制与生命周期安全；Python 仍为训练和桌面参考，移植特征/事件需 golden fixtures 一致性验证。手机无需浏览器、电脑服务或完整训练环境即可完成本机闭环。暂停/锁屏/权限撤销默认停止，恢复不自动 live。

T38 才引入可选配对控制：明确目标、认证、序号/新鲜度、旧会话拒绝与目标端 watchdog，不能从手机发送任意系统命令。所有平台按安装、相机、本地识别、应用内/外部控制分层记录支持状态；本次没有实施任何上述新组件。
