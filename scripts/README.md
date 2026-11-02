# scripts：兼容与高级诊断入口

**日常操作不再要求使用这些脚本或长参数。** 使用根目录注册的 `uv run doctor`、`uv run download`、`uv run prepare`、`uv run train`、`uv run resume`、`uv run evaluate`、`uv run calibrate`、`uv run export`、`uv run app`、`uv run check`，配置见 `handshake.toml` 与[快速开始](../docs/quickstart.md)。

日常命令实现在 `backend/workflows/`，原生启动实现在 `backend/native_launcher.py`；`run_native.py` 仅兼容转发。其余脚本保留用于明确授权的采集、离线回放、模型探针和专项验证，不将这些需要安全上下文的高级操作塞进无参数默认动作。下文为历史高级入口说明。

> 新增`check_native_shadow.py`：在临时目录生成未训练的MLP/TCN诊断资产，启动程序输入的Linux原生影子检查并清理；无相机、无训练，不继承用户资产。通过PATH、`FLUTTER_BIN`或`--flutter`定位SDK，无开发者路径依赖。生产影子入口为`run_native.py --shadow-session`，见[指南](../docs/native_shadow_session.md)。

> 新增`replay_public_policy.py` / `python -m backend.interaction.public_replay` / 安装后`handshake-public-replay`：显式`--allow-synthetic`，手写分数/几何经15类策略、调度器改变内置页码；不训练、不打开相机、不支持live。见[合成回放](../docs/public_gesture_policy.md)。

> T15入口：`scripts/evaluate.py` / `python -m algorithm.evaluation.cli` / 安装后`handshake-evaluate`，提供`score|classify|calibrate|events`。前向评分不训练，校准只用验证集，所有动作保持禁用。见[运行与验证](../docs/evaluation_and_calibration.md)。

> **本机训练暂停**：新增`probe_stream_cache.py`默认只创建合成容量样例并构建缓存；训练阶段必须显式`--with-training`，留给后续机器。本轮不运行该选项。正式缓存构建/迁移验证使用`handshake-data build-stream-cache/inspect-cache`，见[交接说明](../docs/public_data_scale.md)。

> T16模型命令：`python -m algorithm.deployment export|inspect|benchmark`（安装后`handshake-model`）；`run_native.py`可选`--model-package`与`--model-sha256`，须配合感知资产。无自动下载/训练/应用控制，见[模型包指南](../docs/model_bundle_and_inference.md)。

> 原生原型入口：`.venv/bin/python scripts/run_native.py`，可指定`--flutter`或已构建的`--bundle`；显式传入当前Python解释器和仓库目录，不使用shell、不自动打开相机；可选`--camera`/`--perception-model`，相机需UI显式启用。见[启动与验证](../frontend/native/README.md)。

> 新增已安装入口`handshake-data`（`backend.importing.commands`）：`audit-ipn`、`import-ipn`、`build-cache`共用严格来源校验与显式审查流程，不访问相机或自动联网。使用方法与内存边界见 [公开数据实现](../docs/public_data_implementation.md)。

> 现有 `check_environment.py`；T03/T06新增 `camera_probe.py`、`collect.py`、`annotate.py` 薄入口，转发已安装包的 `backend.cli`。`perception_probe.py` 转发 `backend.perception.probe`，与 `handshake-perception-probe` 共用不访问相机的真实模型诊断，见 [感知修复](../docs/perception_diagnostics.md)。T14训练器及T15离线评估已实现，正式训练与完整事件回放仍待完成。完整命令和授权要求见 [采集指南](../docs/collection_workflow.md)。

T11/T12/T13 已提供 `build_features.py`（草案观测JSON＋预分配划分→绑定版本的NPZ缓存）和 `smoke_models.py`（纯合成可过拟合/结构诊断，不保存权重）。完整用法见 [特征与模型](../docs/features_and_models.md)。`train.py`和`evaluate.py`现已提供薄入口；正式数据训练/评估与完整事件回放尚未验收。

## 原则

脚本只处理参数、加载配置和调用模块，不复制算法、状态机或平台逻辑。后续若同时提供 CLI 子命令，二者应复用同一实现，不维护两套流程。

## 计划入口与主责

| 入口 | 主责 | 调用模块 |
| --- | --- | --- |
| `collect.py`、`extract_landmarks.py` | B | backend 感知/采集与 shared 记录格式 |
| `validate_dataset.py`、`build_splits.py` | A，B 复核 | 数据校验、预登记角色与版本规则 |
| `build_features.py` | A | shared 特征和 algorithm 数据加载 |
| `train.py`、`evaluate.py` | A | algorithm 训练与指标 |
| `export_model.py` | A | algorithm 模型打包/一致性 |
| `replay.py` | A+B | backend dry-run 引擎与 algorithm Predictor |
| `benchmark.py` | B，A 复核 | 系统计时、安全与性能测试 |

主应用入口属于 backend 的应用组装层，前端只接控制器。包配置和具体命令语法在 T02/T23/T29 实现后补充。

## 共同要求

- 支持明确配置、输入、输出和版本参数，不依赖某个固定工作目录或私有绝对路径。
- 缺文件、schema 错、权限不足给出可理解错误和退出状态。
- 训练/采集/回放/evaluate 不注入真实设备操作；live 由专门受控入口启用。
- collect 的录制必须单独授权；evaluate 不默认扫描整盘数据或修改原始标签。
- 输出写入配置指定的 `data/` 或 `artifacts/`，不散落脚本目录。
- 安全回归和日常测试默认 mock，禁止测试脚本偷偷控制桌面。

返回 [根目录说明](../README.md)。
