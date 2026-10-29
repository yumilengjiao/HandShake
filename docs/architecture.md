# T02 当前工程与未来 Web 边界

后续进展：T11/T12/T13已新增真实公式的共享特征、数据/缓存适配、规则/MLP/TCN与受门控预测接口，详见 [特征与模型](features_and_models.md)。下方链路仍专指保留的T02合成dry-run，不将其零值fixture冒充T11的几何提取；正式相机/训练/事件闭环仍未实施。

当前实现是单仓库 Python 项目，`handshake-core` 是发行名，`shared/backend/algorithm/frontend` 是职责包。安装、命令见根 README。没有 HTTP/WebSocket 服务、网页、移动应用或真实系统输入实现。

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

Python 内部时间是非负 int64 ns，拒绝 float、bool、字符串、负值；记录相对 recording 起点。无法仅凭整数大小识别误标单位，因此来源适配器仍须显式转换并设置 timebase；软件采集时间不是曝光真值。JSONL 支持整数无损回读；未来 JS transport 要设计相对时间/十进制字符串以避免超出 Number 安全整数范围，不能直接把内部 int64 纳秒值当 JS Number。

当前使用列表保证 JSON 序列化和形状校验；真实视频/大量数组性能优化在 T03/T11 适配边界实测，不将本轮小 fixture 性能当摄像头性能。模型坐标不镜像，预览镜像信息单列。缺失有效性和 no_action 类别不同。

## 配置、记录与错误

六类 YAML：runtime / interaction / gestures / actions / training / privacy，另有 platforms/mock。`shared/config.py` 拒绝未知字段/动作、映射缺失、重复 YAML 键、负时长、错误版本、平台能力和目标冲突。默认文件作为包资源提供，仓库 `configs/` 是同一来源的链接，外部可复制成独立配置目录并通过 `--config-dir` 指定。所有阈值是 draft。

入口预留 collect/train/evaluate/replay/dry_run/live。只有 dry_run 合成冒烟可用；其他返回退出码2和缺失任务说明，未导入真实操作后端。没有隐藏的 live 开关。

JSONL 包含运行 ID、配置 SHA256、发行版/Git 版本、dirty 标志、schema、timebase 和 synthetic 标记。文件以独占创建方式写入，拒绝覆盖既有证据；读回校验 payload 类型、schema 与 timebase。默认不写图片、身份或真值。日志读回不是 T23 的完整确定性回放。NPZ/真实会话写入属于 T06。

## 后续 Web/手机交接

前端通过受控指令接口请求启停、模式与输入切换，显示后端快照；公共契约与训练代码不依赖 UI 或网络库。后续选择 transport 时要明确移动端时钟与服务端时钟的映射、配对/授权、消息排序、重连新 stream、旧 epoch 拒绝、速率限制、后台/断线挂起和独立停止；网络消息不能携带任意系统命令。相机可来自电脑或手机，输入适配器须保留源帧与变换信息。相关任务的补充要求已写入任务导航，本轮不实施这些组件。
