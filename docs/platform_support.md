# 平台支持与实测矩阵

## T16阶段一（2026-10-31）

本轮Linux x86_64、Python3.12.15、PyTorch2.15.0+cpu、Flutter3.47.9：566项Python/51项Flutter通过；原生v2回归和v3坏模型拒绝通过，未访问真实相机。1线程batch=1合成前向CPU探针、Release构建和wheel模型一致性见[模型包记录](model_bundle_and_inference.md)。这是未校准只读分类的软件证据，不覆盖正式识别效果、全链时延、动作权限或其他平台。

## 原生相机阶段（2026-10-31）

本机Linux/Wayland原生UI＋v2私有管道＋OpenCV/MediaPipe通过两轮短时相机打开、JPEG解码显示、停止确认及重开。均为640×480；短时采集均值约30.17/29.99Hz，SDK返回无手/有手状态。未保存真实图像或坐标。505项Python、38项Flutter通过；详细[记录与边界](native_camera_preview.md)。不将短时帧率外推为稳定性或推理FPS，也不视为全局停止/手势分类/翻页/多端验收。


## T36/T20原生合成原型（2026-10-31）

本轮机器为Linux 7.2.11-arch1-1 x86_64、Wayland、Python3.12.14、GTK3 3.24.52；Flutter3.47.9/Dart3.13.7、Clang23.1.3、CMake4.4.4。Linux Debug/Release bundle构建、真实原生窗口与Python私有管道的启停/回看/重连通过。Release显式`GDK_BACKEND=wayland`启动保持3秒、UI进程终止后子进程退出通过；有一条cursor theme加载提示，不影响本轮测试。484项Python、23项Flutter单元/widget、1项原生集成通过，见[决策与边界](native_stack_decision.md)及[机器记录](verification/native_stage_summary.json)。

这不是相机、模型、外部控制、全局停止或干净系统安装验收；X11、Windows、macOS、Android/iOS均未在本轮验证。下文历史机器信息保持原样，不代表本轮环境。

## T03/T04/T06 新增证据（2026-10-29）

当前 Linux x86_64/Python3.12.14：154项自动测试通过。兼容依赖MediaPipe0.10.21/OpenCV4.11/NumPy1.26.4下真实模型加载、黑图VIDEO推理和关闭通过。另一次300.280秒headless V4L2探针取得3003帧，640×480、实测10.0146Hz、槽峰值1，正常释放后重开通过；未保存画面/关键点。不能把设备报告的30FPS当实测速率，不能把headless采集/黑图推理当真手、人工GUI或外部应用控制验收。详细证据见 [修复与补验](perception_diagnostics.md)；131项测试、旧SDK失败和两次短探针的 [初次交付记录](verification_t03_t04_t06.md)保留为历史。

下方原有开发机器表为历史T02证据，不覆盖本轮机器或新增任务。各目标平台仍须逐项验收。


2026-10-29；本记录只覆盖当前开发机器，不代替最终演示设备清单。

## 实际开发环境

| 项目 | 观测结果 | 证据边界 |
| --- | --- | --- |
| OS/架构 | Linux 6.8.0-138-generic，x86_64 | `uname -srm` |
| 桌面 | GNOME / Wayland | `XDG_SESSION_TYPE=wayland`、`XDG_CURRENT_DESKTOP=ubuntu:GNOME` |
| CPU/内存 | Intel Core i7-13650HX；20逻辑CPU；约31GiB内存 | `lscpu`、`free -h` |
| Python | CPython 3.10.12，项目 `.venv` | 只支持已验证的安装组合，见依赖记录 |
| 摄像头 | 受限沙箱不可见；正常宿主枚举到 `/dev/video0`～`/dev/video3` | 本轮只列举节点，未打开设备；不代表四个独立物理相机，型号/采集能力未验证 |
| 显示缩放 | 待实机记录 | 不读取个人窗口内容 |
| 训练设备 | 当前以 CPU 为依赖验证目标 | 无训练实验、GPU能力未验收 |
| 演示软件/版本/焦点 | 待确认 | 未启动或操作桌面应用 |
| 手机 OS/架构/机型 | 待确认 | 历史轮次未实测手机；原生 SDK、工具链和签名条件由 T36 核查 |

## 当前支持证据

以下开发机信息为 2026-10-29 历史T02记录，不视为当前会话重新探测结果。后续原生目标未获全面支持证据；新探针结果见文首。

| 平台/应用 | 能力 | 权限/停止 | 状态 |
| --- | --- | --- | --- |
| 历史 Linux / 合成 mock target | 合成 next/previous_slide | 无相机、网络或桌面权限；命令行有限执行后退出 | 软件验证结果见 [T01/T02/T05 验证记录](verification_t01_t02_t05.md) |
| 历史 Wayland / 真实幻灯片应用 | 翻页 | 焦点查询、全局停止、按键方法均待验证 | **不支持 live** |

## 原生目标矩阵（除上述Linux合成原型外，完整能力待实现/待实测）

| 平台族 | 本机安装/相机/识别/应用内控制目标 | 外部控制与分发边界 | 责任任务 |
| --- | --- | --- | --- |
| Windows | 原生桌面应用，x64/ARM64 按实际目标分别验证 | DPI、权限等级、焦点、停止、各 ABI 安装包 | T20/T23/T24/T29；新增目标 T37 |
| macOS | 原生桌面应用，Intel/Apple Silicon 分别记录 | 相机/辅助功能权限；签名、公证、macOS 构建资源 | T36/T37 |
| Linux X11 | 发行版/架构逐项记录，相机与本机推理 | X11 后端与桌面会话权限，包依赖/显示库 | T20/T37 |
| Linux Wayland | 独立验证，不沿用 X11 注入结论 | 合法 portal/应用接口，不能默认全局键鼠；不靠 root 绕过 | T20/T37 |
| Android 手机/平板 | 原生相机、本地推理、触控启停、应用内翻页 | min/target SDK、ABI、签名包；外部接口优先，敏感能力另审 | T36/T37 |
| iOS/iPadOS | 原生相机、本地推理、触控启停、应用内翻页 | macOS/Xcode、证书/provisioning、真机；无任意跨 App 触摸注入 | T36/T37 |
| 其他 OS（如 HarmonyOS）/无界面设备 | SDK、感知、推理和构建可行性待评估 | 不假设 Android 兼容层或 Python 可用；无界面需独立控制/停止通道 | 按 T36/T37 模板扩展 |
| 已配对手机 → 电脑 | 可选远程标准命令，不是本机模式前提 | 双端授权/加密、目标端 watchdog、租约/新鲜度与撤销 | T38 |

平台名称是目标，不是已支持声明。每个实测 profile 需记录 OS 版本、ABI、机型、SDK/运行时/插件版本，并分别给出安装、相机、本地识别、应用内控制、外部控制、安全、分发的结论和证据。不能因 Android 成功就勾选 iOS，也不能把模拟器/编译通过当真机相机与模型验收。

`configs/platforms/mock.yaml` 仍只描述 mock 能力，本次不修改配置或 schema。原生能力 profile 在对应任务实施时升级契约并落盘。详细分层与权限策略见 [原生多端架构](native_architecture.md)。

真实平台验收必须补：相机参数、旋转/镜像、显示缩放/横竖屏、应用版本、实测翻页方法、目标与权限、独立停止、暂停/后台/锁屏恢复、操作白名单与模型哈希。手机另测热降频/电量与本地离线运行，跨设备另测断连释放；证据齐备前保持 dry-run。不承诺安装成功即可控制任意外部应用。
