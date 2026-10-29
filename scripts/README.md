# scripts：采集、训练与运维命令入口

> 现有 `check_environment.py`；T03/T06新增 `camera_probe.py`、`collect.py`、`annotate.py` 薄入口，均转发已安装包的 `backend.cli`。训练/评估/事件回放仍未实现。完整命令和授权要求见 [采集指南](../docs/collection_workflow.md)。

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
