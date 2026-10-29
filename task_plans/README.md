# HandShake 分阶段任务规划

本目录将 [PROJECT_MASTER_PLAN.md](../PROJECT_MASTER_PLAN.md) 拆分为 **35 个独立任务文件**。现阶段只有方案与架构图；本目录不表示已采集数据、已训练模型或已实现应用。所有任务初始状态均为“待实施”。

## 如何使用

1. 先执行 T01 填写真实姓名、课程节点、设备与支持环境；角色 A/B 沿用总方案建议，实际贡献须更新。
2. 根据下面依赖完成输入，再按任务文件中的步骤实施；编号表示查阅顺序，不要求所有工作严格串行。T17 状态机和 T20 界面可先接 mock，与数据/模型开发交错推进。
3. 每个任务结束记录完成日期、实现文件、配置/数据/模型哈希、验证命令及报告、未解决问题；没有证据不标完成。规划中的产物不等于目前存在的文件。
4. 课程以 **翻页＋基础界面＋自训 TCN＋MLP 对照＋安全测试＋报告答辩** 为 P0。音乐和光标独立规划为 P1；通用适配器始终覆盖多种能力，架构不限于翻页。
5. 总方案文首课程约定优先于正文研究完整版的优先级/12周路线。P1/P2 不自动变成课程必交付，也不因课程截止而删掉数据防泄漏与停止保护。

## 阶段与退出条件

| 阶段 | 核心任务 | 通过后才推进的条件 |
| --- | --- | --- |
| S0 范围与协议 | T01、T05 | 课程差异、首平台、词表与两人分工有记录 |
| S1 工程与工具 | T02～T04、T06；T20 可先做 mock | 带时间的观测可重载、质量失效可见、采集不真实操作桌面 |
| S2 试采集与早期交互 | T07；T12 可先在 pilot 排错，T17/T18 接合成预测 | 方向/词表/标签可标，状态机能一次动作一次事件 |
| S3 主数据与主模型 | T08～T14 | 按用户固定划分，无泄漏，94维特征与32帧因果模型可训练 |
| S4 部署与主场景 | T15～T20、T23、T24 | 自训模型驱动实机翻页，界面、去重、失跟与停止有证据 |
| S5 应用/研究增强 | T21、T22、T25～T27、T31～T35 | 仅对已完成模块做对照；不阻碍主场景验收 |
| S6 实验冻结 | T28 | 验证选择与最终测试分离，结果/时长/失败案例可重算 |
| S7 交付答辩 | T29、T30 | 可复现启动、真实分工、PDF及8分钟讲述准备完整 |

阶段表示总方案的工作性质，不是固定公历周数。按实际第9周方案节点、第10周本班答辩和提交截止倒排；在 T01 确认后给任务填日期，不照搬12周研究路线。

## 课程关键路径与并行工作

数据/模型链：T01 → T02/T03/T04＋T05 → T06 → T07 → T08 → T09 → T10 → T11 → T12/T13 → T14 → T15 → T16。

应用链：T02/T04/T05 → T17 → T18 → T19；T20 可在 T02 后用 mock 开始。T16、T17～T20 汇合至 T23 → T24 → T28 → T29 → T30。

T15 先完成分类校准、指标实现与合成事件评估；T23 联调使用带 draft 标记的示例事件门限。T28 在集成后用验证连续流选最终事件门限并冻结，之后才跑最终测试。这样不会出现“必须先最终评估才能部署、必须先部署才能评估”的循环依赖。

建议 A 侧重 T10～T16 与研究模型，B 侧重 T03/T04/T06/T17～T24 和应用；两人共同设计、采集、标注复核、集成与交付。

## 文件索引与依赖

下表只列硬前置；可选增强的条件、早期 mock 实现和最终集成前置在各任务内说明。

| ID / 文件 | 任务 | 课程优先级 | 主责 | 硬前置 | 状态 |
| --- | --- | --- | --- | --- | --- |
| [T01](01_scope_and_platform.md) | 冻结课程范围、平台与验收口径 | P0 | A+B | 无 | 待实施 |
| [T02](02_scaffold_and_contracts.md) | 建立工程骨架、配置与公共数据契约 | P0 | A+B | [T01](01_scope_and_platform.md) | 待实施 |
| [T03](03_camera_and_clock.md) | 实现摄像头采集、时间戳与最新帧输入 | P0 | B | [T02](02_scaffold_and_contracts.md) | 待实施 |
| [T04](04_perception_tracking_quality.md) | 实现关键点感知、轨迹关联与质量门控 | P0 | B | [T03](03_camera_and_clock.md) | 待实施 |
| [T05](05_gesture_and_annotation_protocol.md) | 冻结手势词表、动作边界与标签规则 | P0 | A+B | [T01](01_scope_and_platform.md), [T02](02_scaffold_and_contracts.md) | 待实施 |
| [T06](06_collection_tool.md) | 实现授权采集、会话落盘与标注辅助工具 | P0 | B 主责，A 复核数据格式 | [T04](04_perception_tracking_quality.md), [T05](05_gesture_and_annotation_protocol.md) | 待实施 |
| [T07](07_pilot_collection.md) | 完成试采集并冻结数据与交互协议 | P0 | A+B | [T06](06_collection_tool.md) | 待实施 |
| [T08](08_main_dataset_collection.md) | 组织主数据采集与连续自然动作测试资料 | P0；研究规模按资源扩展 | A+B | [T07](07_pilot_collection.md) | 待实施 |
| [T09](09_annotation_and_quality_audit.md) | 标注动作时间轴并完成数据质量审核 | P0 | A+B 交叉复核 | [T05](05_gesture_and_annotation_protocol.md), [T08](08_main_dataset_collection.md) | 待实施 |
| [T10](10_splits_and_dataset_version.md) | 固定用户划分、排查泄漏并版本化数据集 | P0 | A | [T09](09_annotation_and_quality_audit.md) | 待实施 |
| [T11](11_features_and_windows.md) | 实现共享特征工程、掩码与时序数据加载 | P0 | A | [T04](04_perception_tracking_quality.md), [T05](05_gesture_and_annotation_protocol.md), [T10](10_splits_and_dataset_version.md) | 待实施 |
| [T12](12_rules_and_mlp_baselines.md) | 建立规则与单帧 MLP 对照基线 | P0：MLP；规则作为工程对照 | A | [T11](11_features_and_windows.md) | 待实施 |
| [T13](13_causal_tcn_network.md) | 搭建可验证的单分支因果 TCN 神经网络 | P0 | A | [T11](11_features_and_windows.md) | 待实施 |
| [T14](14_training_and_experiment_tracking.md) | 实现可复现训练、采样与模型选择 | P0 | A | [T12](12_rules_and_mlp_baselines.md), [T13](13_causal_tcn_network.md) | 待实施 |
| [T15](15_validation_and_calibration.md) | 建立分类与事件评估、验证集校准和门限选择 | P0 | A | [T14](14_training_and_experiment_tracking.md) | 待实施 |
| [T16](16_export_and_online_inference.md) | 打包模型并实现带新鲜度校验的在线推理 | P0：PyTorch；ONNX 选做 | A 主责，B 集成 | [T14](14_training_and_experiment_tracking.md), [T15](15_validation_and_calibration.md) | 待实施 |
| [T17](17_interaction_state_machine.md) | 实现唤醒、控制权、释放去重与独立停止状态机 | P0 | B | [T02](02_scaffold_and_contracts.md), [T04](04_perception_tracking_quality.md), [T05](05_gesture_and_annotation_protocol.md) | 待实施 |
| [T18](18_generic_action_dispatch.md) | 实现通用命令映射、调度器与 mock 适配器 | P0 | B | [T02](02_scaffold_and_contracts.md), [T05](05_gesture_and_annotation_protocol.md), [T17](17_interaction_state_machine.md) | 待实施 |
| [T19](19_presentation_application.md) | 实现演示文稿上层应用与实机翻页 | P0 | B | [T01](01_scope_and_platform.md), [T18](18_generic_action_dispatch.md) | 待实施 |
| [T20](20_desktop_ui.md) | 实现预览、模式、模型状态与启停控制界面 | P0；调试页 P1 | B | [T02](02_scaffold_and_contracts.md) | 待实施 |
| [T21](21_music_and_volume_application.md) | 实现歌曲切换、播放暂停与音量上层应用 | P1 | B 主责，A 补数据/识别 | [T18](18_generic_action_dispatch.md), [T19](19_presentation_application.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T22](22_pointer_and_continuous_control.md) | 实现光标移动、点击与可选拖拽滚动 | P1；拖拽滚动后置 | B 主责，A 补数据/识别 | [T18](18_generic_action_dispatch.md), [T19](19_presentation_application.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T23](23_replay_and_integration.md) | 实现确定性回放并打通端到端应用闭环 | P0 | A+B | [T06](06_collection_tool.md), [T16](16_export_and_online_inference.md), [T17](17_interaction_state_machine.md), [T18](18_generic_action_dispatch.md), [T19](19_presentation_application.md), [T20](20_desktop_ui.md) | 待实施 |
| [T24](24_safety_faults_and_performance.md) | 完成故障注入、停止释放与性能稳定性测试 | P0 | B 主责，A 复核 | [T23](23_replay_and_integration.md) | 待实施 |
| [T25](25_local_global_multitask.md) | 扩展局部全局双分支与意图阶段多任务网络 | P1；研究完整版主线 | A | [T09](09_annotation_and_quality_audit.md), [T13](13_causal_tcn_network.md), [T14](14_training_and_experiment_tracking.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T26](26_rejection_and_hard_negatives.md) | 研究开放集拒识并开展困难负样本迭代 | P1；系统比较为研究任务 | A 主责，B 采集/回放 | [T15](15_validation_and_calibration.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T27](27_personalization.md) | 实现几何校准、少样本适配与通用模型回退 | P1 几何；学习适配 P2（课程） | A 主责，B 向导/档案 | [T23](23_replay_and_integration.md), [T25](25_local_global_multitask.md) | 待实施 |
| [T28](28_final_experiments.md) | 冻结完整实验、消融、用户任务与失败分析 | P0 核心；P1/P2 扩展按实现范围 | A+B | [T15](15_validation_and_calibration.md), [T23](23_replay_and_integration.md), [T24](24_safety_faults_and_performance.md) | 待实施 |
| [T29](29_packaging_and_reproduction.md) | 整理可安装交付、模型卡与复现入口 | P0 | A+B | [T24](24_safety_faults_and_performance.md), [T28](28_final_experiments.md) | 待实施 |
| [T30](30_report_and_defense.md) | 完成课程 PDF 报告、PPT 与双人答辩准备 | P0 | A+B | [T28](28_final_experiments.md), [T29](29_packaging_and_reproduction.md) | 待实施 |
| [T31](31_graph_temporal_candidate.md) | 验证骨骼图卷积与因果时序候选模型 | P2 | A | [T13](13_causal_tcn_network.md), [T14](14_training_and_experiment_tracking.md), [T23](23_replay_and_integration.md), [T25](25_local_global_multitask.md) | 待实施 |
| [T32](32_causal_transformer_candidate.md) | 验证轻量因果 Transformer 候选模型 | P2 | A | [T13](13_causal_tcn_network.md), [T14](14_training_and_experiment_tracking.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T33](33_self_supervised_pretraining.md) | 探索训练侧关键点自监督预训练 | P2 | A | [T10](10_splits_and_dataset_version.md), [T11](11_features_and_windows.md), [T14](14_training_and_experiment_tracking.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T34](34_bimanual_interaction.md) | 探索双手角色绑定与协同交互 | P2 | B 主责，A 协同协议/数据 | [T04](04_perception_tracking_quality.md), [T17](17_interaction_state_machine.md), [T18](18_generic_action_dispatch.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T35](35_additional_application_backend.md) | 接入一个阅读器或其他终端应用后端 | P2 | B | [T18](18_generic_action_dispatch.md), [T19](19_presentation_application.md), [T23](23_replay_and_integration.md) | 待实施 |

## 统一技术口径

| 项目 | 首版约定/负责文件 |
| --- | --- |
| 词表 | T05：no_action/wake_open/swipe_left/swipe_right/exit_pose，C=5；扩词必须版本化 |
| 数据 | T06/T08～T10：连续观测＋无手帧表，NPZ数值数组＋JSON元数据，按用户划分后再滑窗 |
| 标签 | T05/T09：可见词元、操作意图、动作阶段和目标设备事件分开，缺标签用loss mask |
| 特征 | T11：local63＋global10＋mask21=94，实际时间间隔，无未来插值 |
| 主网络 | T13：94→64投影，4个单卷积残差块d=1/2/4/8，感受野31，32帧窗口，末帧64→C |
| 缺失/冷启动 | T11/T16：失效时不执行，长断流清历史；累计32个有效观测后允许普通事件 |
| 事件 | T17：显式启用/唤醒、owner、候选、等待释放、冷却、失跟挂起，独立停止 |
| 应用 | T18：词元→模式命令→调度→能力后端；T19翻页、T21媒体、T22光标、T35新增应用 |
| 评价 | T15/T28：分类＋事件＋有效背景时长＋真实延迟，不用帧准确率代替实际误触 |
| 配置 | 方案数值是实施起点；验证后冻结，未实现intent头不能伪造意图分数 |

94维、具体MLP层宽、局部/全局分支尺寸等是总方案的实施具体化，必须经过试采集和验证；不是已有实验结论或对第三方论文的原样复现。修改特征、词表、模型结构、标签语义时一并更新总方案、schema、任务、权重与报告。首版不同时开展所有候选研究。

课程数据起步为8人×4词元×10次×2会话≈640正例，背景约80分钟；按4/2/2分人后测试背景约20分钟，不能充分验证长期≤1次/小时。若扩到24人/8词元/3会话，需单独规划授权、词表变更、标注与划分。

## 路径、时间与发布约定

- 文中 `src/`、`configs/`、`data/`、`artifacts/`、`docs/`、`scripts/` 均相对 HandShake 仓库根目录；任务文件在本目录，来源总方案为上级 `PROJECT_MASTER_PLAN.md`。
- 内部时间统一单调ns，录制采用同一 recording 的相对ns并记录 timebase；平台API要求ms时只在适配边界转换。实际曝光时间不可得时注明软件采集近似，回放不混入墙钟时间。
- 当前仅创建规划文件，不创建虚构数据/结果、不安装依赖、不运行live、不上传报告。具体库版本与平台方法在相应实施任务实机核对后固定。
- 数据/模型/日志默认受控存储，原视频、身份映射、个人档案和大权重不进入普通Git发布；公开需符合实际授权和各自许可。
- 操作白名单与独立停止从P0起保留。停止/模式变更更新context_epoch，执行前复检；任何应用后端都不得绕开调度器。

## 每个任务的完成记录模板

```text
任务ID / 实际负责人：
开始/完成日期：
输入版本（数据/标签/split/schema/model/config/code）：
实际实现和文档路径：
验证命令及执行环境：
结果文件/日志/图表路径：
验收逐项结论（通过/未通过/不适用及原因）：
已知问题、降级与后续任务：
```

状态建议只用待实施/进行中/待验证/已完成/延期，更新索引和任务完成记录，不仅修改打勾。`已完成`需同时有实现、验证、文档与明确能力边界。

## 接口核查资料

任务依据主要为仓库总方案；补充API细节以官方资料核查，不把在线文档的当前版本号当作已锁定运行环境：

- [MediaPipe Hand Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)：调用模式、时间戳、坐标与输出字段。
- [PyTorch Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)、[LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)：卷积维度/扩张和归一化轴；因果填充由本项目实现。
- [scikit-learn 分组交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)：按参与者分组评估。
- [MPRIS](https://specifications.freedesktop.org/mpris-spec/latest/)：媒体后端实施时核对的方法与能力。
