# T02 依赖与安装验证

验证日期2026-10-29；Linux x86_64 / CPython 3.10.12 / GNOME Wayland。下表为**实际安装及导入结果**，不是仅依据在线文档填写的兼容版本。锁文件 `uv.lock` 包含精确版本与分发哈希；只对当前组合做了验证。

| 依赖 | 实测发行版本 | 实际检查 |
| --- | --- | --- |
| NumPy | 2.2.6 | 数组创建/形状 |
| Pydantic / pydantic-core | 2.14.1 / 2.50.1 | 严格字段校验、形状、序列化、JSON Schema |
| PyYAML | 6.0.3 | 安全加载、重复键拒绝及跨配置校验 |
| pytest | 9.2.0 | 81项契约/配置/标注/mock集成测试 |
| MediaPipe | 1.2.0 | 正常宿主导入与 `tasks.vision.HandLandmarker` 存在检查；未加载手部模型资产 |
| opencv-contrib-python | 5.0.0.93（cv2报告5.0.0） | 合成RGB转灰度；未打开摄像头 |
| PyTorch | 2.14.1+cpu | CPU张量与NumPy互转/求和；未训练网络 |
| scikit-learn | 1.7.2 | 导入与GroupKFold接口存在检查；未划分真实数据 |
| setuptools | 84.2.1 | editable安装、sdist/wheel构建；构建依赖也固定此版本 |

基础依赖是 NumPy/Pydantic/PyYAML。dev、perception、training 是可选组；Web/UI、ONNX、数据库及真实输入注入库均未引入。当前导入可用不等于相机、TCN或翻页已经可运行。

## 可复现命令

在仓库根目录：

```bash
uv sync --locked --extra dev
.venv/bin/handshake dry_run
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest

# 可选依赖检查，不执行采集或训练
uv sync --locked --extra dev --extra perception --extra training
env -u PYTHONPATH .venv/bin/python scripts/check_environment.py --timeout 45

# 构建（有相应缓存可加 --offline）
uv build --out-dir /tmp/handshake-t02-dist
```

正常宿主检查的机器可读证据为 [environment_t02.json](verification/environment_t02.json)。实际安装和运行命令、wheel独立环境检查见 [统一验证记录](verification_t01_t02_t05.md)。仓库默认配置使用包内资源，wheel不依赖源码当前工作目录。

## 已遇问题及处理

1. 系统预载ROS的 `PYTHONPATH` 使pytest自动加载 `launch_testing`，因缺少lark而在收集前失败。通过取消该环境变量并设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 隔离全局插件，未修改ROS或系统Python。
2. 原始感知候选同时安装 `opencv-python` 与MediaPipe所依赖的 `opencv-contrib-python`，二者共享cv2文件；移除前者后出现 `cv2.cvtColor` 缺失。已将可选依赖统一为 contrib 并重新安装该包，之后宿主合成图像操作通过。不要额外安装其他cv2提供包。
3. 受限沙箱中MediaPipe导入超时，正常宿主同一解释器的限时导入检查通过；本次真实支持证据使用宿主检查。沙箱看不到视频节点，宿主只枚举到video0～video3，未读取图像。不能把沙箱隔离现象当成设备不存在或依赖不兼容。
4. 沙箱无法解析PyPI；公开依赖下载通过允许的联网安装执行，构建随后用缓存离线完成。没有修改系统依赖，也没有启动Web或live。

## 后续锁定与来源

每次环境、词表、特征或依赖变更都重新运行核心测试；感知模型资产与实际摄像头在T03/T04记录型号、版本/哈希和失败原因。当前只有软件依赖锁，不存在模型权重锁或实验性能数字。手机浏览器兼容性在T20/T23实机验证。

实现核对依据为 [Pydantic严格校验文档](https://pydantic.dev/docs/validation/latest/concepts/strict_mode/)、[setuptools项目配置](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) 和 [MediaPipe Python设置指南](https://developers.google.com/edge/mediapipe/solutions/setup_python)。CPU来源按 [PyTorch官方安装说明](https://pytorch.org/get-started/locally/) 选择；uv对Linux/Windows的torch使用显式CPU索引。官方文档不代替上述本地实测。
