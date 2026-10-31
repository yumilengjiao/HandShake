# T36/T20：原生相机与同帧骨架预览

**2026-10-31 / `feat/native-camera-preview`，基于已本地提交的原型 `205a848`。**

已将相机画面与可选MediaPipe骨架接入Flutter。仍然是**预览**，没有手势分类模型、应用授权或翻页命令，不将感知SDK当作自主训练成果。原有[v1合成阶段](native_stack_decision.md)及其验证记录保留为历史。

## 启动

在仓库根目录，已安装感知依赖和Flutter的环境中：

```bash
.venv/bin/python scripts/run_native.py \
  --flutter "$HOME/.local/share/handshake-tools/flutter/bin/flutter" \
  --camera 0 \
  --perception-model artifacts/perception/hand_landmarker.task
```

也可以把Flutter加入PATH并省略`--flutter`。资产需事先按现有[感知说明](perception_diagnostics.md)准备；本入口不自动下载权重。

- **启动不打开相机**。连接后手动点击“打开相机预览”，才访问选定设备。
- 不给`--perception-model`时只显示视频，明确显示`unavailable / 未配置感知资产`。指定了错误资产/缺SDK时失败，不静默切成假骨架。
- 合成预览入口保留；切换来源先停止，等待释放后再开始。手机仍不能使用这个Python子进程方案。
- 镜像同时影响图像和骨架显示，不改变源坐标、左右词元或模型输入。
- 时间轴最多32帧，只读回看；图像与骨架仅在内存中保留。停止、换epoch、断连清空历史。
- 停止请求、正在释放、确认停止分别显示。Esc仅在本窗口有效，不是系统全局停止。

Release启动：

```bash
cd frontend/native
flutter --no-version-check build linux --release
cd ../..
.venv/bin/python scripts/run_native.py \
  --bundle frontend/native/build/linux/x64/release/bundle/handshake_stage \
  --perception-model artifacts/perception/hand_landmarker.task
```

## 实现边界

```text
OpenCVSource → CaptureWorker（最新帧槽）
   → 可选 PerceptionPipeline（既有质量/关联实现）
   → 同一 CapturedFrame 的 JPEG + ObservationFrame
   → 私有stdout二进制包 → Dart严格解码
   → 有界图像解码 → 同帧图像与骨架一起绘制
```

新增`backend/perception/preview.py`复用既有相机/感知工具，不另写关键点、质量或关联算法。输入读取、感知和JPEG编码不持有核心会话锁；协议、心跳和停止不等待推理返回。`backend/desktop_preview.py`是独立进程入口，不加载动作后端。

UI继续保持repository/ViewModel/绘制分层：`preview_packets.dart`处理二进制流；`PreviewSurface`拥有解码生命周期；`SkeletonPainter`统一图像尺寸、letterbox、镜像和有效点掩码。两只手可绘制，质量不可用用不同颜色提示，但不赋予控制权。没有使用Flutter全局ImageCache积累相机帧。

### 图像和骨架必须对应

一个包只有一个源帧ID/源时间，其JPEG和骨架来自同一帧，不用最近一组骨架叠加到任意新图像。传输后的JPEG长宽与协议一致性在分配解码像素前检查。

解码期间保留**旧图像及其自己的骨架**，不是保留旧图像再换新骨架。绘制面标注被展示的帧号。只允许同session/epoch且不落后最新来源时间750ms的成对结果；回看必须精确匹配选中帧。停止、来源代际变化时立即清除，晚到解码结果释放而不显示。至多一个解码在途和一个最新待解码目标。

## v2传输与有界性

v1仍由`python -m backend.desktop`提供，仅合成NDJSON；新UI调用`python -m backend.desktop_preview`，不会把v1改成能控制真实应用的协议。

[Python契约](../shared/preview_protocol.py) / [schema集合](../shared/preview_protocol.schema.json)：

```text
HSP2 (4 bytes)
metadata_length (uint32, big endian)
jpeg_length (uint32, big endian)
UTF-8 JSON metadata
JPEG bytes
```

- 元数据≤16,384字节、JPEG≤131,072字节；先验证前缀/长度再分配包体，支持任意分片与多个连续包，损坏/截断不尝试重新同步。
- 原始图片不放JSON/base64；时间/计数继续用规范十进制字符串，Dart用BigInt。未知版本、字段、能力以及图像/状态不匹配拒绝。
- JPEG预览最大640×480，保持原始宽高比；相机实际分辨率、驱动缓存不因请求值而视为保证。读帧时间为`software_read_complete_not_exposure`，不冒充曝光时间。
- 明确分开`perception_status`与`model_status`。前者可能是MediaPipe的有手/无手/未处理/错误；后者目前始终为手势分类模型`unavailable`。
- 命令仍是小型有界NDJSON，版本2增加`start_camera`；设备编号和资产路径只由本机启动参数配置，不允许UI消息发送脚本、任意文件请求或执行命令。
- 相机槽、编码槽、输出邮箱均有界；展示编码上限约10Hz，不等于采集或推理FPS。UI32帧历史加最新/选中引用的JPEG数据上限约4.25MiB，另有受限元数据、解码像素和相机原始缓冲。
- 部分管道写入会继续写剩余字节；输出阻塞不持有相机/会话锁。SDK stdout重定向到stderr，避免破坏二进制通道。

### 停止与失败

- 1.5秒无UI心跳请求停止；未收到第一帧的启动预算8秒，来源帧过旧750ms停止。禁止旧epoch重新开始；释放未完成时不创建第二个相机worker。
- 原生线程停止/关闭完成后才发送`stopped`确认。相机`close()`异常显式记录`release_failed`，不会因线程退出就假定释放成功。
- 释放1秒仍未确认时，独立预览核心退出，最后状态仍为`stopping`；UI显示失联/停止未确认。UI还有2秒确认期限和有界子进程终止升级。
- 预览子进程无持久化或动作后端，末尾用`os._exit`避免在SDK析构中无限阻塞。这不是通用应用执行核心的关闭方案，也不是证明真实按键已释放。
- 没有真实应用命令、录制、上传或监听端口。内存中的图像会被解码并短暂保留供绘制/回看；不宣称能控制系统交换区或操作系统自身的截屏行为。

## 验证结果

详情见[机器记录](verification/native_camera_summary.json)。

- **505项Python回归通过，无跳过**：v1兼容、新帧协议、部分写入、超长/截断包、无模型路径、RGB/缩放/骨架身份、相机启动/旧帧/心跳/释放失败与显式启用。
- **38项Flutter单元/widget通过**：跨语言合成二进制fixture、精确时间、只读JPEG/有效掩码、JPEG尺寸检查、单解码/最新替换、晚到帧释放、停止确认、360宽度/双倍字体以及非4:3投影。
- 常规Linux原生集成：合成流程在v2端点正常工作，不访问相机。
- **显式真实相机原生集成**：两轮打开→收到至少5帧→JPEG解码/绘制→停止确认→重新打开，通过。640×480；两轮短时采集均值分别约30.17Hz、29.99Hz，MediaPipe分别返回`processed_no_hand`、`processed_hands`。只记录汇总，不保存JPEG、骨架坐标或真实截图。
- 短时30Hz不能替代此前5分钟约10Hz的历史稳定性测试，也不代表UI/推理FPS或正式延迟指标。收到有手结果不等于左右手/遮挡/交叉/方向或分类准确率验收。
- Debug原生构建、Release构建、独立Python wheel与静态检查结果记录在机器记录中；不是干净系统安装、手机或其他桌面平台交付。

早期原生测试发现每次新帧到达就清空旧画面可能闪空，且低频测试pump容易始终采到解码间隙。已改为成对保留/原子替换并加对应测试；原生测试按20ms推进绘制。没有保存真实失败画面来调试。

### 复现

常规单元与UI测试不访问设备：

```bash
.venv/bin/python -m pytest
cd frontend/native
flutter --no-version-check test
flutter --no-version-check analyze
```

仅明确需要真机检查时运行以下命令（会短时打开相机两次）：

```bash
HANDSHAKE_PYTHON="$PWD/../../.venv/bin/python" \
HANDSHAKE_ROOT="$(cd ../.. && pwd)" \
HANDSHAKE_PERCEPTION_MODEL="$PWD/../../artifacts/perception/hand_landmarker.task" \
HANDSHAKE_CAMERA_TEST=1 \
flutter --no-version-check test integration_test/stage_smoke_test.dart -d linux
```

`HANDSHAKE_CAMERA_TEST`不设置时，只测合成流程。集成测试截图入口仍仅允许在合成阶段执行，真机分支不调用截图。`test/fixtures/preview_v2.bin`是程序生成的红/蓝图像和骨架，为测试相机协议分支而标注`source=camera`，不是采集数据。

## 下一步仍未完成

T16模型包/推理接入、公开类别应用策略与T19真实翻页；独立全局停止、系统休眠/权限撤销和完整T24；长时间感知稳定性、人工视觉/左右手/遮挡验收；分平台安装包与手机本地核心。T36/T20继续是进行中，不因为有视频窗口就勾选完整产品闭环。
