# T04 感知接口与质量/轨迹语义

## 实现与部署边界

实现位于 `backend/perception/`、`backend/tracking/association.py`；公共帧/观测聚合在 `shared/observations.py`。这是 MediaPipe Hand Landmarker 适配，不是现成 Gesture Recognizer；没有自训 TCN、手势分类或设备控制。真实模型加载/合成图像推理已通过兼容修复，见 [感知诊断](perception_diagnostics.md)；以下软件接口不等于已通过真实手部效果验收。

## 输入/时间/像素

- `CapturedFrame` 是内部数组快照：连续 RGB uint8 `[H,W,3]`，从源缓冲复制并标只读，避免每帧构造数十万 Python 整数。需要既有 JSON 契约时显式 `to_packet()` 转成 v1 FramePacket；并未静默改写旧 schema。
- `FrameInfo` 保留 stream、frame_id、源 ns/timebase、旋转/镜像及尺寸；`received_at_ns` 是本机单调接收时刻，和媒体时间分开。
- 相机 BGR→RGB、顺时针旋转后交给感知；模型不镜像，UI 镜像只改变显示。必须实际核对左右手标签，不把分数当身份认证。
- VIDEO API 的毫秒值从本流首帧源时间减去原点后下取整得到；原始 ns/frame_id 不变。同一毫秒或重复/倒退源时间为 `not_processed`，不伪造 +1ms。
- 新输入流必须创建新的 pipeline/MediaPipe VIDEO 实例；旧实例不接换源数据。当前同步模式没有 LIVE_STREAM 回调。
- `processed_at_ns` 在源时轴上表示源时间加本机从接收到完成的耗时；对于文件/合成这是持续时间映射，不是直接混用两个时钟，也不宣称曝光端到端延迟。

## 输出与无效性

一次返回整帧 `ObservationFrame(info, status, observations, world_landmarks)`，不在检测第一只手后提前执行。`FrameStatus` 的四个实际枚举为：

| 状态 | 含义 |
| --- | --- |
| `processed_hands` | SDK 返回手，可能包含质量不合格的观测；不等于可控制 |
| `processed_no_hand` | 成功处理且 SDK 没有输出手 |
| `not_processed` | 无模型、同毫秒、过期、最新帧替换或停止等未处理原因 |
| `error` | SDK/结构处理失败，不能改成背景或无手 |

`HandObservation` 沿用 T02 字段。坐标数量错误/NaN 的行使用有限零值占位、mask=false、明确质量原因；不抹掉失败帧或构造逐点置信度。handedness_score 仅是左右手判别分数，缺失为 None；world 是可选的21×3估计值，和图像坐标分开，不当作相机厘米距离。

## 质量与关联（draft）

`QualitySettings` 默认：掌尺度下限0.025图像宽度，边界余量0.05，速度上限4图像宽度/秒，尺度比上限2，归一化骨长变化上限1。几何使用 `(x, y*H/W, z)`；具体是试采集前起点，不是验证最优参数。当前 quality 为确定性的二值几何门控（合格1、不合格0），不是已校准概率。

质量原因覆盖点数/形状、非有限坐标、尺度过小、越界、退化骨段、异常速度/尺度/骨长、匹配歧义。归一化几何属于感知质量计算，不替代 T11 94维特征实现。

关联最多处理两只手、保留两个最近轨迹假设：掌心距离/骨架形状门控，尺度和左右手作为软代价，对整帧枚举互斥匹配。代价接近或双手中心过近时整帧拒绝关联、清轨迹假设并分配新 ID，不按数组顺序抢控制权。默认距离门限0.25、形状差1、歧义余量0.05、近距离冲突0.05，失跟500ms淘汰。ID 在流内递增不复用；图像跟踪不是人脸/用户认证，也不能保证发现漏检的第三只手。

只有质量合格的观测更新轨迹几何。T17 才拥有 owner/唤醒/暂停决策；本模块不把最近或最大的手自动授权。

## 资产与故障

`MediaPipeDetector` 显式加载本地 `.task`，计算 SHA256，可要求匹配预期值；元数据记 SDK版本、模式、最大手数、来源和许可说明。不联网自动取模型，不将加载失败回退为合成检测。Python 异常转 `error` 并结束当前采集；原生进程被杀不能靠 try/except 恢复，已开始录制的 manifest 初始为 incomplete，须人工排错/恢复检查。

核查入口：[官方 Hand Landmarker Python 指南](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)。当前已验证 fake SDK 适配、合成整帧语义及兼容依赖下的真实模型加载/黑图推理/关闭，尚未取得真手效果验收。`handshake-perception-probe` 通过独立进程留下超时/信号/日志/资源证据，不将原生故障标成无手；常规 preview/collect 仍在进程内加载模型，不能声称探针为所有原生调用提供崩溃恢复。
