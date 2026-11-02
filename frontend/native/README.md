# HandShake Native Stage

Linux原生预览，Flutter 3.47.9 / Dart 3.13.7。支持合成舞台、显式相机/MediaPipe骨架、校验模型包的原始分类及可选校准概率；新增只产生候选的影子owner会话，均不控制应用。不依赖浏览器，也不把Python子进程方案当作手机运行方案。

## 启动

推荐在仓库根目录使用 `uv run app`；统一配置见 `handshake.toml`，环境检查用 `uv run doctor`。需要相机依赖时先 `uv run setup --camera`。只有Linux原生宿主已提供，训练/数据命令不依赖Flutter。下文显式Python/脚本参数保留为高级兼容入口。

安装Flutter 3.47.9，以及Linux构建所需Clang/CMake/Ninja/pkg-config/GTK3开发库；将Flutter加入PATH。先按根目录说明准备Python `.venv`，然后在项目根目录运行：

```bash
.venv/bin/python scripts/run_native.py
# Flutter不在PATH时：
.venv/bin/python scripts/run_native.py --flutter /path/to/flutter/bin/flutter
```

也可通过`FLUTTER_BIN`指定SDK可执行文件；无需放到某个开发者目录，不要将SDK提交进Git。

1. 启动只连接本机核心，保持停止；可选“开始合成预览”或“打开相机预览”，切换来源需先停止。
2. 相机骨架需启动时传`--perception-model artifacts/perception/hand_landmarker.task`；不传则明确显示仅视频，指定了错误资产则失败，不伪造骨架。
3. 点击时间轴只读回看，“返回当前”恢复展示最新帧。回看不暂停核心，也不执行操作。
4. 镜像按钮只改变显示；讲解按钮展示感知到深度学习的链路说明，未加载手势网络时不输出假预测。
5. “停止 / Esc”发送请求，等待相机/感知释放确认。Esc仅窗口内有效。关闭或失联后无自动重启。

若显示核心失联，确认启动所用的是项目`.venv/bin/python`，不要解析该符号链接后改用系统解释器；确认已安装核心依赖。截图/日志不应混入真实采集内容。

## 内置演示（application-local）

窗口右上角“演示”提供四页内置稿；先人工播放，再用按钮/左右方向键翻页。导航经唯一T18调度器，应用状态与绘制回执分开；长按不连发，Esc/隐藏/失活停用。切到演示会停止相机预览，返回观测不会自动开启相机。

这不是PowerPoint/WPS/LibreOffice控制，不接模型原始分类结果。无需相机或训练资产，当前机器不运行神经网络训练。完整边界与原生测试见[内置演示](../../docs/local_presentation.md)。

## 可选手势模型（T16阶段一）

模型包须来自可信本地训练结果并已导出，启动时同时指定`--model-package <目录>`和`--model-sha256 <可信导出回执SHA256>`，另需`--perception-model`。包在相机启动请求后校验加载；数学合成舞台不伪造模型输出。窗口显示来源类/raw logit、拒绝原因和合成训练警示，不输出未经校准的概率，也不调用动作调度器。

使用/导出/性能探针与未完成事项见[模型包与在线分类](../../docs/model_bundle_and_inference.md)。正式数据权重尚未交付；不把测试合成包当正式识别模型。

## 原生影子owner会话（不执行）

`run_native.py --shadow-session`另需模型、感知和校准资产及可信SHA；界面提供明确启用/选手、32步重新预热和候选展示。回看不能授权，停止不等待模型返回，始终不翻页。完整参数和安全边界见[HSP5指南](../../docs/native_shadow_session.md)。

无需真实相机/用户模型的一键检查，在仓库根目录运行：

```bash
.venv/bin/python scripts/check_native_shadow.py
# 同样支持 --flutter /path/to/flutter；--model-type tcn 可切换诊断网络。
```

此源码检查自动生成临时的**未训练**诊断模型和程序图像，结束清理；需要Python dev/perception/training extras、Flutter和Linux图形会话。安装training extra仅提供推理依赖，不运行训练。它不等于真实相机效果或长期稳定性验收。

## 构建与不依赖Flutter工具的启动

```bash
cd frontend/native
flutter --no-version-check pub get
flutter --no-version-check build linux --release
cd ../..
.venv/bin/python scripts/run_native.py \
  --bundle frontend/native/build/linux/x64/release/bundle/handshake_stage
```

必须保留整个`bundle/`目录，包括`lib/`与`data/`；不是只复制一个可执行文件。当前还需要本机Python环境，不是已完成安装器或手机包。

## 验证

Python验证请使用[无训练回归命令](../../docs/native_shadow_session.md#回归与证据边界2026-11-02)；未筛选的全量pytest包含训练，当前机器不要直接运行。

在本目录：

```bash
flutter --no-version-check pub get
flutter --no-version-check analyze
dart format --output=none --set-exit-if-changed lib test integration_test
flutter --no-version-check test

# 打开本应用的原生测试窗口；不访问相机或控制其他应用。
HANDSHAKE_PYTHON="$PWD/../../.venv/bin/python" \
HANDSHAKE_ROOT="$(cd ../.. && pwd)" \
flutter --no-version-check test integration_test/stage_smoke_test.dart -d linux
```

可另设`HANDSHAKE_SCREENSHOT=/absolute/path.png`导出集成测试的合成绘制面。测试逻辑画布固定1280×900，不抓取桌面或相机。`test/fixtures/snapshot.json`、`preview_v2.bin`、`model_preview_v3.bin`均为程序生成的跨语言golden，示例分数不是识别成绩。

上述测试默认不访问相机。真实相机测试需显式设置`HANDSHAKE_CAMERA_TEST=1`，感知资产另设`HANDSHAKE_PERCEPTION_MODEL`；测试会短时打开/停止/重开设备，只输出元数据、不保存真实帧或坐标。详见[相机阶段验证](../../docs/native_camera_preview.md)。

## 代码边界

- `lib/presentation/`：独立只读状态/受控人工意图、内置页码与绘制确认；不在Widget注入系统按键。
- `lib/core/`：严格只读协议、大小受限的解码、本机子进程repository。
- `lib/stage/stage_view_model.dart`：连接/确认/回看/显示状态，历史有界，不实现手势或动作状态机。
- `lib/stage/stage_screen.dart`：中央舞台、时间轴、决策轨迹、常驻停止区。
- `lib/stage/preview_surface.dart`：单解码/最新目标，成对保留和替换图像/骨架；不使用全局ImageCache积累相机帧。
- `lib/stage/skeleton_painter.dart`：统一镜像/letterbox变换、有效点和双手绘制。
- `linux/`：Flutter官方生成的Linux宿主；未生成其他平台壳来冒充支持。
- `../../backend/desktop_preview.py`：v2相机、v3分类、v4校准及v5影子通知端点，显式打开相机；v1合成NDJSON入口保留兼容。

完整范围、协议和验证见[相机阶段说明](../../docs/native_camera_preview.md)；[初次技术决策](../../docs/native_stack_decision.md)保留为历史。T36/T20整体尚未完成；下一步是正式模型/校准、应用策略、安全集成与独立打包。
