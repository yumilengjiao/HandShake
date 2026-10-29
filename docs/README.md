# docs：设计说明、实验总结与答辩资料

## 当前已有内容

- [公开数据路线 public-data-v1](public_data_plan.md)、[官方下载入口](../data/SOURCES.md)、[版本/许可/校验清单](../data/sources.json)

- [范围与待确认项](scope_freeze.md)、[平台支持](platform_support.md)
- [当前架构/契约](architecture.md)、[原生多端目标与实施边界](native_architecture.md)、[依赖验证](dependencies.md)
- [手势协议](gesture_protocol.md)、[标注指南](annotation_guide.md)
- [T01/T02/T05 验证与完成边界](verification_t01_t02_t05.md)
- [T11/T12/T13 特征与模型](features_and_models.md)、[模块验证记录](verification_t11_t12_t13.md)
- [T03/T04/T06 采集操作指南](collection_workflow.md)、[感知协议](perception_contract.md)
- [T03/T04/T06 初次交付验证与旧SDK阻塞](verification_t03_t04_t06.md)
- [MediaPipe兼容修复、独立模型探针与剩余实机验收](perception_diagnostics.md)
- [通用技术架构图](assets/technical-architecture-general.png)
- [架构图生成说明](assets/technical-architecture-general.prompt.txt)

这些是方案资料，不表示图中所有模块已经实现。总目标在根 [PROJECT_MASTER_PLAN.md](../PROJECT_MASTER_PLAN.md)，具体实施顺序在 [task_plans](../task_plans/README.md)。

## 两人共同维护

- A：数据/特征/模型说明、训练复现、算法实验与模型卡。
- B：平台支持、相机/界面/控制流程、故障与系统性能说明。
- A+B：词表/标注协议、接口、方案决策、结果核查、报告/PPT 和实际分工。

## 后续建议文档

| 文档 | 内容 |
| --- | --- |
| `scope_freeze.md` | 课程范围、脱敏人员角色、节点与变更记录 |
| `native_architecture.md` | 当前原生多端目标、平台权限、便携运行时与 T36～T38 验收 |
| `architecture.md` | 当前 Python 实装、契约与后续原生交接，不混为已完成多端 |
| `annotation_guide.md`、`gesture_protocol.md` | 词元、类别/意图/阶段、确认/释放和反例 |
| `platform_support.md` | 真正实测的系统/应用/权限与限制 |
| `dataset_card.md` | 授权范围、真实规模、划分及公开程度 |
| `experiments.md` | 可复现结果、版本、统计与失败案例 |
| `demo_guide.md` | 安装、离线资产、演示脚本、停止和备份 |
| 报告/PPT | 实际完成内容、个人贡献、证据和边界 |

其中范围、架构、协议已在本轮落盘；数据卡、实验、模型、报告仍待真实工作，不生成虚构结果。

## 什么可以提交

- 可公开的设计说明、脱敏配置示例、合成示意、已获授权且不含敏感内容的实验汇总。
- 原始视频/标签/真实数据清单放 `data/`；大权重、原始运行日志和实验输出放 `artifacts/`。
- 实名报告、授权原件、姓名映射和提交凭据放本地 `private/course/` 等受控路径；公开仓库使用脱敏版本。
- 发布截图或录屏前确认人脸、旁观者、窗口标题和个人信息的处理。
- 文档区分“计划/实现/仅离线验证/实机验证”，不能把总方案复制成完成结果。

返回 [根目录说明](../README.md)。
