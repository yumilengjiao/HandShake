# T03/T04/T06 采集、感知与标注操作指南

## 当前能力与边界

- 已实现本机相机/视频/合成输入、最新帧槽、帧状态账本、VIDEO 感知适配、质量/两手关联、授权分块落盘、完整性检查/恢复及骨架标注工具。
- 工具不加载自训手势分类器，不创建任何真实动作后端。MediaPipe 仅提取关键点；T11/T13/T17/T18/T20 的生产功能仍待实施。
- OpenCV 工具窗口只服务预检/采集/标注，不等于已完成 T20 原生产品 UI 或 T36 框架选型。
- 当前兼容组合已通过真实 Hand Landmarker 加载、合成图像推理与关闭；新增独立子进程探针保留原生失败证据。原1.2.0的SIGKILL来源尚未查明，不认定OOM；兼容修复和剩余验收见 [感知诊断](perception_diagnostics.md)。真手效果仍需人工验证。

## 环境与默认安全行为

```bash
uv sync --locked --python 3.12 --extra dev --extra perception
.venv/bin/handshake --help
.venv/bin/handshake dry_run
```

以上组合在本轮 Linux/Python 3.12 验证；Windows 使用对应 `.venv` 可执行路径，其他 OS 的相机/显示/打包仍需实测。普通无参数启动仍为原有合成 dry-run，不打开相机、不录制、不控制应用。`live/train/evaluate/replay` 仍明确不可用；`inspect` 验证记录，不冒充 T23 事件回放。

`preview/camera_probe/collect` 是显式相机工具命令，默认相机 0。所有运行都有正的有限 `--duration`（默认 30 秒）；`--headless` 不创建窗口。常规预览绝不创建数据集，`--report` 只写元数据且拒绝覆盖已有路径。

## 1. 相机预检（不保存画面）

```bash
.venv/bin/handshake camera_probe --camera 0 --headless --duration 5 \
  --report artifacts/camera-probe-new.json
# 带画面预检；需可用桌面显示，不保存关键点/原图
.venv/bin/handshake preview --camera 0 --duration 30
```

没有 `--model` 时仅预览，相应帧为 `not_processed/model_not_loaded`，不能标为无手。`--width/--height/--fps` 是请求值，报告区分 set 返回值、设备报告值、实际 RGB 尺寸和测得采集 Hz。驱动曝光时间/漏帧通常未知，不从软件“最新帧”推断曝光新鲜度。

`--rotation 0/90/180/270` 为顺时针源旋转，`--no-mirror` 关闭预览镜像。模型输入始终不镜像。相机错误停止本次运行；关闭后重新启动会生成新 stream，不自动恢复 live 或控制权。

## 2. 感知资产与不访问相机的测试

不自动联网下载模型。由用户放置可信的 Hand Landmarker `.task` 文件，建议固定 SHA256，并记录来源/许可证。本轮用于加载探针的公开资产为：

- URL：`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
- SHA256：`fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- 本地文件：`artifacts/perception/hand_landmarker.task`，已忽略，不随 Git 分发；该文件是第三方感知资产，不是自训模型。
- 再分发前核对资产许可，包许可证不自动等于模型数据许可证。

```bash
# 不依赖 MediaPipe 模型的显式合成链路
.venv/bin/handshake preview --source synthetic --frames 90 --headless
# 先检查真实模型：独立子进程，默认30秒期限，不访问相机
.venv/bin/handshake-perception-probe --model artifacts/perception/hand_landmarker.task \
  --model-sha256 fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1 \
  --report artifacts/perception-probe-new.json
# 真实 SDK + 合成图像的采集处理链路
.venv/bin/handshake preview --source synthetic --frames 10 --headless \
  --model artifacts/perception/hand_landmarker.task \
  --model-sha256 fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
# 感知预览（模型探针通过后，由操作者核对真手效果）
.venv/bin/handshake preview --camera 0 --model artifacts/perception/hand_landmarker.task
```

只有 `--source synthetic` 且没有真实模型时才选合成检测器，输出显式标 synthetic。真实相机采集没有模型会拒绝，不偷偷使用合成结果。加载进程被终止不解释为“没有手”，不将合成数据列入真实实验。

## 3. 授权采集

先取得所有参与者的真实授权。`--consent-keypoints` 是操作者对授权的显式确认，不代替授权原件；原件放受控私有目录，参数只填匿名引用。未同意或缺匿名 subject/session/consent 引用时，不打开相机、不创建录制目录。

```bash
# 可复现纯合成资料，测试工具而非训练数据
.venv/bin/handshake collect --source synthetic --frames 90 --headless \
  --subject synthetic-p1 --session synthetic-s1 \
  --consent-keypoints --consent-reference synthetic-only \
  --output artifacts/collection-demo

# 真实授权采集；先通过相机/SDK预检，再由操作者启动
.venv/bin/handshake collect --camera 0 --duration 120 \
  --model artifacts/perception/hand_landmarker.task \
  --subject P001 --session S01 --consent-reference consent-local-001 \
  --consent-keypoints --output data/recordings
```

窗口快捷键：`n` 下一个随机顺序提示，`r` 练习/重试提示，`p` 暂停/恢复提示，`q`/Esc 正常结束，`x` 取消。**暂停提示不暂停采集**，仍保存前后背景并记录 pause/resume，时间不压缩。无界面采集不自动生成精确标签；提示种子/顺序写事件流，可直接录自然活动。Ctrl-C 标取消，不将中断会话标完整。

可选 `--save-raw --consent-raw` 另行授权保存无损 RGB NPZ 块。当前不输出 MP4，也不声称完整连续原视频：只保存送达记录器的实际 RGB 帧，跳过帧仍有状态但可能无像素。每个 RGB 项通过块内帧索引对应真实时间，不重复帧伪造 FPS。未启用时连 `raw/` 目录都不创建。

## 4. 记录布局与完整性

```text
<output>/<recording UUID>/
  manifest.json              # 原子发布，初始 incomplete；唯一权威索引
  chunks/000000.npz           # 默认最多32帧/约2秒观测，绝不含原图
  events.jsonl               # start/prompt/pause/resume/retry/end/cancel/health 等
  raw/000000.npz              # 仅另行授权；frame_000000 对应本块帧表索引
  annotations/<revision>.json # 独立人工标签，每次导出新版本
  recovery-<UUID>.json        # 可选恢复检查结果，仍不标完整
```

不维护易产生并发追加冲突的全局 `recordings.jsonl`；扫描每录制 manifest 建索引。格式版本为 `recording-v1`，既有 T02 schema 保持 1.0.0。NPZ 使用 float64/int64/bool/Unicode 数组及严格 JSON 描述，`np.load(..., allow_pickle=False)`；关键点 float64 保存避免记录环节额外精度损失，训练时按特征方案转换。

帧表保留 ID、源相对时间、尺寸、变换与四类处理状态；观测表通过 frame_index 关联，保存关键点、mask、track、质量、左右手及可选 world 估计。所有可记录帧保留，检测失败/未处理不能标为无手；同流 ID/时间严格递增。manifest 保留源时基/录制起点、配置/感知哈希、匿名授权范围和设备参数；`context_available=false` 明确没有 T17 的实际交互上下文。

相机采用最新帧槽；被替换帧保留不带图片的 `not_processed` 账本，账本溢出使会话 incomplete。**本地视频文件采用有界背压而非快速解码后只保留最后一帧**：使用显式 FPS 推导或 `--video-clock decoder` 媒体时间，不自动修补倒退时间。文件处理 Hz 是吞吐量，不是实时相机性能。

写盘队列默认 128 项。满队列/磁盘错误/关停超时显式失败，不静默丢资料；每块重读验证、哈希后写入 incomplete manifest，全部写入并确认线程停止后才发布 complete。强杀/断电不保证尾块存在，重载仍检查哈希/结构，不把残片用于训练。

```bash
.venv/bin/handshake inspect --recording <output>/<UUID>
# 不完整文件默认拒绝；仅检查/人工复核时显式允许
.venv/bin/handshake inspect --recording <output>/<UUID> --allow-incomplete
.venv/bin/handshake recover --recording <output>/<UUID>
```

恢复只核验已提交块并生成新的 recovery 报告，保留 incomplete；不修改原始证据、不擅自补帧或认证为完整。采集故障退出码 2，成功工具运行 0，取消也按未完整处理。

## 5. 骨架标注与标签导出

```bash
.venv/bin/handshake annotate --recording <output>/<UUID> --annotator A
# 加载旧版本编辑；旧文件不覆盖
.venv/bin/handshake annotate --recording <output>/<UUID> --edit-labels <旧标签.json>
# 可不打开窗口，验证手工编辑的完整字段并导出新版本
.venv/bin/handshake annotate --recording <output>/<UUID> --headless --labels <标签.json>
```

滑块或 `a/d` 逐帧，空格按源时间播放，`b/e` 设置起点/终点，`g` 类别、`i` 意图、`p` 阶段，Enter 添加区间，`u` 撤回最后区间，`s` 导出新版本。标签默认不确定；右边界为所选末帧时刻+1ns，不擅自外推曝光周期。提示/取消/暂停与骨架同一时间轴显示，**不会自动产生真值**。

人工主体引用独立于预测 track。GUI 处理基础区间/不确定掩码；复杂 reference_event_time/target_command 等字段可按 [标注指南](annotation_guide.md) 编辑 JSON 后严格导入。校验 recording/subject/session、时间界限、重叠、意图与目标语义，每次保存新文件；正式双人试标与词表冻结仍归 T07。

## 待验证清单

- 5分钟headless相机稳定性与随后重开已通过；真实占用/拔插、人工窗口预览/关闭仍待验，见 [补验记录](perception_diagnostics.md)。
- 在已通过加载的兼容环境验证真手左右/镜像/遮挡/交叉与实际感知速率；黑图探针不是手部效果验收。
- 人工操作预览/标注窗口、真实授权短会话和双人标签对齐；单元测试不代替交互验收。
- 后续 T17/T18/T24 的真实控制、独立停止和竞态安全，不以此采集工具的无操作模式背书。
