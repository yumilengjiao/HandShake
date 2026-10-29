# T03/T04/T06 实施与验证记录

> **历史记录，保留初次交付结论。** 后续 `fix/capture-perception-validation` 已固定 MediaPipe 0.10.21/OpenCV 4.11 兼容组合，真实模型加载、三帧合成图像推理、关闭及154项回归通过；命令、对照证据和剩余实机验收见 [修复记录](perception_diagnostics.md)。下文的131项测试、旧依赖与exit137是修复前结果，不代表当前锁文件。

记录日期：2026-10-29。先提交原生多端文档为 `8121d0d`，随后实现本轮代码。依据 `task_plans/03_camera_and_clock.md`、`04_perception_tracking_quality.md`、`06_collection_tool.md` 及总方案课程要求；仓库内未发现独立的作业 PDF/Word 附件。没有使用 `data/` 中的第三方数据包，没有采集参与者训练数据或生成自训模型。

## 状态与实际交付

| 任务 | 已实现与验证 | 未完成的验收 |
| --- | --- | --- |
| T03 | 相机/文件/合成源、RGB数组快照、源时钟、最新帧/丢帧账本、文件背压、探针/预览；真实 V4L2 相机两次短时打开/读取/释放通过 | 约5分钟稳定性、真实占用/拔插、人工关闭窗口与完整恢复测试 |
| T04 | VIDEO API适配与资产哈希、毫秒映射、整帧质量/两手关联、未知/无手/未处理/错误分离；fake SDK、合成骨架测试通过；实际 SDK 可导入 | **真实 Hand Landmarker 初始化被终止（exit 137）**，尚无真实手部感知成功结果、精度/延迟或镜像/遮挡实机证据 |
| T06 | 显式授权、异步有界落盘、分块NPZ/hash/atomic manifest、取消/不完整恢复、随机提示、骨架标注/JSON校验与版本导出；合成闭环/磁盘故障测试通过 | 人工窗口交互、真实授权短会话、双人试标/阶段与意图核对；依赖 T04 真实感知问题解决 |

三项均标 **待验证（软件已交付）**，不是“全部已完成”。T20 产品 UI、T17/T18 的真实控制与安全竞态、自训模型不属于本次完成范围。采集工具从不创建真实动作后端。

## 自动测试

```bash
uv sync --locked --python 3.12 --extra dev --extra perception
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  --junitxml=docs/verification/pytest_t03_t04_t06.xml
uv pip check
.venv/bin/python -m compileall -q backend shared scripts
```

最终 **131 passed，0 failures/errors/skipped**：原有81项回归，加50项采集/感知/存储/CLI和真实OpenCV合成视频测试。报告见 [JUnit XML](verification/pytest_t03_t04_t06.xml)，公开前只去除 hostname，未改变测试结论。没有安装 perception extra 时两个 OpenCV 测试会明确跳过，不等同于验证通过。

覆盖：

- 时钟重复/倒退拒绝，数组副本/只读，BGR→RGB/旋转，换源新stream；相机失败与文件EOF区分。
- 慢消费者有界最新帧/账本溢出、阻塞read关停期限、文件背压全帧保留；真实OpenCV编码/解码纯合成AVI与绘制。
- 整帧手数组重排、交叉歧义、长期失跟换ID、NaN/点数/尺度/越界/骨长/速度异常；world可选值、VIDEO调用参数及SHA拒绝。
- 授权前不建目录/不开相机、原始像素另行授权、NPZ无pickle、保留无手/未处理/无效帧、相对时间对齐。
- 哈希/路径/顺序校验、取消/崩溃不纳入完整资料、恢复不擅自补帧；磁盘错误/队列满/超时不发布complete。
- 手工不确定标签、范围/重叠校验和不可覆盖的版本导出；采集提示不作为模型输入/真值。

## 本轮软件环境

Linux x86_64 / CPython **3.12.14**，项目独立 `.venv`。按已有 `uv.lock` 安装，未重新选依赖版本：

| 包 | 本轮版本 |
| --- | --- |
| NumPy | 2.5.4 |
| Pydantic / PyYAML | 2.14.1 / 6.0.3 |
| pytest | 9.2.0 |
| MediaPipe | 1.2.0 |
| opencv-contrib-python | 5.0.0.93；cv2报告5.0.0 |

`uv pip check` 核验31个包兼容。大体积感知依赖下载曾超时，最后使用同一锁文件完成（未切换版本或修改系统Python）。这与历史T02的Python3.10机器记录不同，不能合并成同一次实测。

## 真实相机：只预览，不保存画面

```bash
.venv/bin/handshake camera_probe --camera 0 --headless --duration 5 \
  --report artifacts/t03-camera-probe-final.json
```

最终探针约5.047秒，V4L2，RGB 640×480，取得46帧，实际采集 **10.015 Hz**；45帧进入消费者，1帧在停机尾部标未处理。设备接受30FPS请求且报告30FPS，**不代表实测达到30FPS**。最新帧槽峰值1，账本未溢出，正常关停确认成功。较早一次独立短探针取得51帧、约10.504Hz；两次均未写摄像头图片/视频或关键点。

最终45个本地read完成→消费者年龄样本：p50 1,256,509ns，p95 4,650,428ns，max 6,282,604ns；仅为软件缓冲延迟，不是曝光到响应延迟。没有加载模型，全部状态为 `not_processed`，不是“无手识别成功”。完整元数据报告留在本地 `artifacts/`；机器可读汇总见 [本轮环境/探针摘要](verification/t03_t04_t06_summary.json)。

## 真实 MediaPipe 阻塞（未解决）

本地下载官方 Hand Landmarker float16/v1 `.task` 资产，SHA256为 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`。路径与下载源见 [操作指南](collection_workflow.md)，文件在Git忽略的 `artifacts/perception/`，不是训练产物。

执行真实SDK＋纯合成图像预览时，在 `HandLandmarker.create_from_options` 加载阶段进程被终止、返回137，没有生成成功探针报告。设置 OMP/OPENBLAS 单线程后仍发生；另做一次1.5GiB虚拟内存限制的隔离加载探针，输出 `Created TensorFlow Lite XNNPACK delegate for CPU` 后同样137。**没有证据确定是OOM、驱动还是宿主策略，不自行归因，更不把导入成功当模型可运行。** 不再循环重试资源故障。

下一步需在可检查进程/资源限制的宿主排查原生加载，必要时另建依赖候选环境并重新验证，不覆盖现有锁或旧报告。通过后再进行真手/镜像/交叉/遮挡测试及真实授权采集。此问题不影响不访问相机的合成工具与已有核心测试。

## 打包与独立安装

`uv build --offline` 构建 sdist/wheel，通过包内容检查包含 capture/perception/tracking/collection 和 shared 新模块。在临时独立Python3.12环境从wheel安装基础依赖，从 `/tmp` 而非源码目录运行合成collect，12帧完整落盘、退出0、no-action backend、未安装OpenCV/MediaPipe也可用合成入口。

首次离线安装依赖因缺registry元数据缓存失败；改为显式 PyPI 索引后安装通过，没有把失败记录成离线安装成功。wheel是Python核心/工具包，不是跨平台原生产品安装器，也不包含真实模型/数据或 `.task` 资产。

## 数据格式/交接差异

- 内部 `CapturedFrame` 数组适配避免每帧构造巨大JSON列表，显式 `to_packet()` 仍兼容原T02契约。
- 每条录制有独立 manifest；不使用易并发冲突的全局追加索引。默认32帧/约2秒块、有界128项写队列，详见 [操作指南](collection_workflow.md)。
- 原始像素可选保存为带精确帧映射的无损NPZ；没有MP4录制，未送达记录器的像素不凭空补齐。
- 基础骨架区间编辑、撤回、加载旧版与新版本导出已实现；复杂参考事件字段经严格JSON导入编辑。人工交互测试仍待补。
- `context_available=false`，不伪造T17运行历史；恢复工具只核验已提交块并生成新报告，保留incomplete，训练读者默认拒绝。

T07 开始正式试采集前，必须先解决T04加载、完成人工工具预检和真实授权；不能直接将合成fixture作为课程训练数据。
