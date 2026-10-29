# HandShake 分阶段任务规划

本目录将 [PROJECT_MASTER_PLAN.md](../PROJECT_MASTER_PLAN.md) 拆分为 **38 个独立任务文件**。T01/T02/T05 已有核心与草案；T03/T04/T06 已新增采集/感知/标注软件，154项测试通过，兼容依赖下真实模型加载/黑图推理已通过；真手/人工工具等完整实机验收未完成，见 [修复记录](../docs/perception_diagnostics.md)。没有真实参与者训练数据、自训模型、产品UI或真实应用控制。未满足验收的任务继续待验证，历史修订见 [任务审查记录](REVIEW_NOTES.md)。

当前采用 **native-v2：原生多端、桌面先交付、手机本机运行随后验收**，替代此前 Web 约定。优先覆盖 Windows/macOS/Linux、Android/iOS/iPadOS，其他系统按 SDK/运行时能力评估；不承诺所有 OS 任意跨应用控制。native-v2 文档修订时未改代码；后续 T03/T04/T06 实施记录另列，不覆盖历史验证结论。完整边界见 [原生架构](../docs/native_architecture.md) 和 [范围记录](../docs/scope_freeze.md)。新增 T36 早期选型、T37 原生手机/其他平台移植、T38 可选配对控制；T01～T35 编号保留。

## 如何使用

1. 先执行 T01 确认课程节点、设备与支持环境；实名/学号放受控课程附件，公开记录用 A/B，实际贡献须更新。
2. 根据下面依赖完成正式输入，编号不是严格串行排期。契约/合成测试、T11 特征函数、T12/T13 网络、T17/T18、T20 可并行开发；正式数据统计必须等 T10，真实模型/应用验收汇合到 T23。T29 安装冒烟和 T30 方案/报告从早期维护，不等最终实验才开始。
3. 每个任务结束记录完成日期、实现文件、配置/数据/模型哈希、验证命令及报告、未解决问题；没有证据不标完成。规划中的产物不等于目前存在的文件。
4. 课程以 **翻页＋原生桌面基础界面＋自训 TCN＋MLP 对照＋安全测试＋报告答辩** 为 P0。音乐和光标独立规划为 P1；T36 选型/接口前置到首版，T37 手机本地运行是产品 P1 交付，T38 配对控制可选 P2。课程先交付不表示删除手机目标。
5. 总方案文首课程约定优先于正文研究完整版的优先级/12周路线。P1/P2 不自动变成课程必交付，也不因课程截止而删掉数据防泄漏与停止保护。

## 阶段与退出条件

| 阶段 | 核心任务 | 通过后才推进的条件 |
| --- | --- | --- |
| S0 范围与协议 | T01、T05 | 课程差异、首平台、词表与两人分工有记录 |
| S1 工程与工具 | T02～T04、T06、T36；T20 可先做 mock | 原生路线明确，带时间的观测可重载，质量失效可见，采集不执行真实操作 |
| S2 试采集与早期交互 | T07；T12 可先在 pilot 排错，T17/T18 接合成预测 | 方向/词表/标签可标，状态机能一次动作一次事件 |
| S3 主数据与主模型 | T08～T14 | 按用户固定划分，无泄漏，94维特征与32帧因果模型可训练 |
| S4 部署与主场景 | T15～T20、T23、T24 | 自训模型驱动实机翻页，界面、去重、失跟与停止有证据 |
| S5 应用/多端/研究增强 | T21、T22、T25～T27、T31～T35、T37/T38 | 多端逐平台验收，远程可选；仅对已完成模块做对照，不阻碍首桌面交付 |
| S6 实验冻结 | T28 | 验证选择与最终测试分离，结果/时长/失败案例可重算 |
| S7 交付答辩 | T29、T30 | 可复现启动、真实分工、PDF及8分钟讲述准备完整 |

阶段表示总方案的工作性质，不是固定公历周数。按实际第9周方案节点、第10周本班答辩和提交截止倒排；在 T01 确认后给任务填日期，不照搬12周研究路线。

## 课程关键路径与并行工作

数据/模型链：T01 → T02/T03/T04＋T05 → T06 → T07 → T08 → T09 → T10 → T11 → T12/T13 → T14 → T15 → T16。

应用链：T02/T04/T05 → T17 → T18 → T19；T02 → T36 → T20，T20 草图/mock 可在 T02 后先行，正式 UI 按 T36 选定路线实施。T16、T17～T20 汇合至 T23 → T24 → T28 → T29 → T30。

多端链：T36 定接口；T03/T04/T11/T16/T17/T18/T20/T23/T36 → T37 本地原生移植；T18/T19/T24/T37 → T38 可选配对控制。T37 复用 T23/T24/T29 的检查方案，不把最终多端报告反设为桌面任务的启动前置；移动条件不足时逐平台保留未完成。

T15 完成分类校准与合成事件评估；T23 使用 draft 参数联调。T28 在验证连续流上联合选择各模型的事件参数，生成不可变 frozen 候选，调用已有 T23/T24 回归通过后再跑最终测试。回归复跑不是新增的循环任务依赖；不能用旧候选的安全报告代替新包验收。

正式数据链可以按批推进：T08 在采集前预登记用户角色，T09 边采边复核，T10 固化版本。T12 的退出交付是模型/接口和冒烟，不要求先完成依赖它的 T14 正式训练；T19/T20 的模块验收不等待 T23 的全链模型/UI 验收。T25 默认后置，但意图研究有需要时可在 T14 后并行；T27 的几何与小头适配不硬依赖多任务网络。

建议 A 侧重 T10～T16 与研究模型，B 侧重 T03/T04/T06/T17～T24 和应用；两人共同设计、采集、标注复核、集成与交付。

## 文件索引与依赖

下表列正式阶段的输入前置；合成/mock 开发可提前。对拆成“模块交付→联调验收”的任务，以模块交付解锁后继，全链证据由 T23/T24 汇总；不能把后继实验报告反设为前置任务的启动条件。条件依赖与不适用项按各任务正文记录。

| ID / 文件 | 任务 | 课程优先级 | 主责 | 硬前置 | 状态 |
| --- | --- | --- | --- | --- | --- |
| [T01](01_scope_and_platform.md) | 冻结课程范围、平台与验收口径 | P0 | A+B（用户待填） | 无 | 待验证：记录已完成，课程/设备待确认 |
| [T02](02_scaffold_and_contracts.md) | 建立工程骨架、配置与公共数据契约 | P0 | A+B（用户待填） | [T01](01_scope_and_platform.md) | 待验证：工程已验证，真实相机加载待后续 |
| [T03](03_camera_and_clock.md) | 实现摄像头采集、时间戳与最新帧输入 | P0 | B | [T02](02_scaffold_and_contracts.md) | 待验证：5分钟headless相机/重开通过，人工实机待验 |
| [T04](04_perception_tracking_quality.md) | 实现关键点感知、轨迹关联与质量门控 | P0 | B | [T03](03_camera_and_clock.md) | 待验证：真实模型加载/黑图推理通过，真手待验 |
| [T05](05_gesture_and_annotation_protocol.md) | 冻结手势词表、动作边界与标签规则 | P0 | A+B（用户待填） | [T01](01_scope_and_platform.md), [T02](02_scaffold_and_contracts.md) | 待验证：草案/合成通过，真实双人试标待T07 |
| [T06](06_collection_tool.md) | 实现授权采集、会话落盘与标注辅助工具 | P0 | B 主责，A 复核数据格式 | [T04](04_perception_tracking_quality.md), [T05](05_gesture_and_annotation_protocol.md) | 待验证：合成软件通过，GUI/真实会话待验 |
| [T07](07_pilot_collection.md) | 完成试采集并冻结数据与交互协议 | P0 | A+B | [T06](06_collection_tool.md) | 待实施 |
| [T08](08_main_dataset_collection.md) | 组织主数据采集与连续自然动作测试资料 | P0；研究规模按资源扩展 | A+B | [T07](07_pilot_collection.md) | 待实施 |
| [T09](09_annotation_and_quality_audit.md) | 标注动作时间轴并完成数据质量审核 | P0 | A+B 交叉复核 | [T05](05_gesture_and_annotation_protocol.md), [T08](08_main_dataset_collection.md) | 待实施 |
| [T10](10_splits_and_dataset_version.md) | 固定用户划分、排查泄漏并版本化数据集 | P0 | A | [T09](09_annotation_and_quality_audit.md) | 待实施 |
| [T11](11_features_and_windows.md) | 实现共享特征工程、掩码与时序数据加载 | P0 | A | [T04](04_perception_tracking_quality.md), [T05](05_gesture_and_annotation_protocol.md), [T10](10_splits_and_dataset_version.md) | 待实施 |
| [T12](12_rules_and_mlp_baselines.md) | 建立规则与单帧 MLP 对照基线 | P0：MLP；规则作为工程对照 | A | [T11](11_features_and_windows.md) | 待实施 |
| [T13](13_causal_tcn_network.md) | 搭建可验证的单分支因果 TCN 神经网络 | P0 | A | [T11](11_features_and_windows.md) | 待实施 |
| [T14](14_training_and_experiment_tracking.md) | 实现可复现训练、采样与模型选择 | P0 | A | [T12](12_rules_and_mlp_baselines.md), [T13](13_causal_tcn_network.md) | 待实施 |
| [T15](15_validation_and_calibration.md) | 建立分类与事件评估、验证集校准和门限选择 | P0 | A | [T14](14_training_and_experiment_tracking.md) | 待实施 |
| [T16](16_export_and_online_inference.md) | 打包模型并实现带新鲜度校验的在线推理 | P0 桌面参考；移动导出归 T37 | A 主责，B 集成 | [T14](14_training_and_experiment_tracking.md), [T15](15_validation_and_calibration.md) | 待实施 |
| [T17](17_interaction_state_machine.md) | 实现唤醒、控制权、释放去重与独立停止状态机 | P0 | B | [T02](02_scaffold_and_contracts.md), [T04](04_perception_tracking_quality.md), [T05](05_gesture_and_annotation_protocol.md) | 待实施 |
| [T18](18_generic_action_dispatch.md) | 实现通用命令映射、调度器与 mock 适配器 | P0 | B | [T02](02_scaffold_and_contracts.md), [T05](05_gesture_and_annotation_protocol.md), [T17](17_interaction_state_machine.md) | 待实施 |
| [T19](19_presentation_application.md) | 实现演示文稿上层应用与实机翻页 | P0 | B | [T01](01_scope_and_platform.md), [T18](18_generic_action_dispatch.md) | 待实施 |
| [T20](20_desktop_ui.md) | 实现原生桌面界面与多端交互契约 | P0 桌面；手机归 T37 | B | [T02](02_scaffold_and_contracts.md), [T36](36_native_platform_foundation.md) | 待实施 |
| [T21](21_music_and_volume_application.md) | 实现歌曲切换、播放暂停与音量上层应用 | P1 | B 主责，A 补数据/识别 | [T18](18_generic_action_dispatch.md), [T19](19_presentation_application.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T22](22_pointer_and_continuous_control.md) | 实现光标移动、点击与可选拖拽滚动 | P1；拖拽滚动后置 | B 主责，A 补数据/识别 | [T18](18_generic_action_dispatch.md), [T19](19_presentation_application.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T23](23_replay_and_integration.md) | 实现确定性回放并打通端到端应用闭环 | P0 | A+B | [T06](06_collection_tool.md), [T16](16_export_and_online_inference.md), [T17](17_interaction_state_machine.md), [T18](18_generic_action_dispatch.md), [T19](19_presentation_application.md), [T20](20_desktop_ui.md) | 待实施 |
| [T24](24_safety_faults_and_performance.md) | 完成故障注入、停止释放与性能稳定性测试 | P0 | B 主责，A 复核 | [T23](23_replay_and_integration.md) | 待实施 |
| [T25](25_local_global_multitask.md) | 扩展局部全局双分支与意图阶段多任务网络 | P1；研究完整版主线 | A | [T09](09_annotation_and_quality_audit.md), [T13](13_causal_tcn_network.md), [T14](14_training_and_experiment_tracking.md) | 待实施 |
| [T26](26_rejection_and_hard_negatives.md) | 研究开放集拒识并开展困难负样本迭代 | P1 负样本迭代；P2 系统拒识比较 | A 主责，B 采集/回放 | [T15](15_validation_and_calibration.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T27](27_personalization.md) | 实现几何校准、少样本适配与通用模型回退 | P1 几何；学习适配 P2（课程） | A 主责，B 向导/档案 | [T23](23_replay_and_integration.md) | 待实施 |
| [T28](28_final_experiments.md) | 冻结完整实验、消融、用户任务与失败分析 | P0 核心；P1/P2 扩展按实现范围 | A+B | [T15](15_validation_and_calibration.md), [T23](23_replay_and_integration.md), [T24](24_safety_faults_and_performance.md) | 待实施 |
| [T29](29_packaging_and_reproduction.md) | 整理可安装交付、模型卡与复现入口 | P0 | A+B | [T24](24_safety_faults_and_performance.md), [T28](28_final_experiments.md) | 待实施 |
| [T30](30_report_and_defense.md) | 完成课程 PDF 报告、PPT 与双人答辩准备 | P0 | A+B | [T28](28_final_experiments.md), [T29](29_packaging_and_reproduction.md) | 待实施 |
| [T31](31_graph_temporal_candidate.md) | 验证骨骼图卷积与因果时序候选模型 | P2 | A | [T13](13_causal_tcn_network.md), [T14](14_training_and_experiment_tracking.md), [T23](23_replay_and_integration.md), [T25](25_local_global_multitask.md) | 待实施 |
| [T32](32_causal_transformer_candidate.md) | 验证轻量因果 Transformer 候选模型 | P2 | A | [T13](13_causal_tcn_network.md), [T14](14_training_and_experiment_tracking.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T33](33_self_supervised_pretraining.md) | 探索训练侧关键点自监督预训练 | P2 | A | [T10](10_splits_and_dataset_version.md), [T11](11_features_and_windows.md), [T14](14_training_and_experiment_tracking.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T34](34_bimanual_interaction.md) | 探索双手角色绑定与协同交互 | P2 | B 主责，A 协同协议/数据 | [T04](04_perception_tracking_quality.md), [T17](17_interaction_state_machine.md), [T18](18_generic_action_dispatch.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T35](35_additional_application_backend.md) | 接入一个阅读器或其他终端应用后端 | P2 | B | [T18](18_generic_action_dispatch.md), [T19](19_presentation_application.md), [T23](23_replay_and_integration.md) | 待实施 |
| [T36](36_native_platform_foundation.md) | 冻结原生多端技术路线与平台接口 | P0 选型/接口 | A+B | [T02](02_scaffold_and_contracts.md) | 待实施 |
| [T37](37_native_mobile_and_portability.md) | 实现原生手机本地运行与多平台移植验收 | P1 产品交付 | A+B | [T03](03_camera_and_clock.md), [T04](04_perception_tracking_quality.md), [T11](11_features_and_windows.md), [T16](16_export_and_online_inference.md), [T17](17_interaction_state_machine.md), [T18](18_generic_action_dispatch.md), [T20](20_desktop_ui.md), [T23](23_replay_and_integration.md), [T36](36_native_platform_foundation.md) | 待实施 |
| [T38](38_paired_remote_control.md) | 实现可选的原生手机与电脑配对控制 | P2 | B 主责，A 复核 | [T18](18_generic_action_dispatch.md), [T19](19_presentation_application.md), [T24](24_safety_faults_and_performance.md), [T37](37_native_mobile_and_portability.md) | 待实施 |

## 统一技术口径

| 项目 | 首版约定/负责文件 |
| --- | --- |
| 词表 | T05：no_action/wake_open/swipe_left/swipe_right/exit_pose，C=5；扩词必须版本化 |
| 数据 | T06/T08～T10：连续观测＋所有可记录帧的处理状态，区分无手/未处理/失败；NPZ＋JSON，预登记用户划分后再滑窗 |
| 标签 | T05/T09：可见词元、意图、阶段和设备真值分开，真值不依赖检测成功；参考确认不重复叠加持有时间 |
| 特征 | T11：local63＋global10＋mask21=94；几何统计与固定时间缩放分开，缺失时长不能只在有效手帧拟合 |
| 主网络 | T13：94→64投影，4个单卷积残差块d=1/2/4/8，感受野31，32帧窗口，末帧64→C |
| 缺失/冷启动 | T11/T16：连续32个有效步完成预热，短缺失禁执行；另检查窗口有效比例/跨度，ready不等于owner恢复 |
| 上下文 | T02/T16～T18：工作提交时捕获stream/epoch并贯穿全链，旧结果不得回填新代际；整帧多手关联后才做决策 |
| 事件 | T17：显式启用/唤醒、owner、候选、回程/释放去重、冷却、失跟挂起，独立停止 |
| 连续控制/释放 | T18/T21/T22：绝对值可覆盖，速率单点积分，delta不可直接丢弃；可能已按下的异常结果也须释放 |
| 应用 | T18：词元→模式命令→调度→能力后端；T19翻页、T21媒体、T22光标、T35新增应用 |
| 评价 | T15/T28：拟提交/注入/应用响应分层；背景分母不按模型状态裁剪，记录可用时长和漏检；同事件逻辑及公平调参预算 |
| 原生多端 | T36 选型；T20 桌面壳；T37 手机本地识别/应用内控制；T38 可选手机→电脑；安装、相机、识别与外部控制分层验收 |
| 配置/发布 | 起点参数标draft；验证后生成不可变frozen候选并重跑安全回归，再最终测试；无intent头不伪造分数 |

94维、具体MLP层宽、局部/全局分支尺寸等是总方案的实施具体化，必须经过试采集和验证；不是已有实验结论或对第三方论文的原样复现。修改特征、词表、模型结构、标签语义时一并更新总方案、schema、任务、权重与报告。首版不同时开展所有候选研究。

课程数据起步为8人×4词元×10次×2会话≈640正例，背景约80分钟；按4/2/2分人后测试背景约20分钟，不能充分验证长期≤1次/小时。若扩到24人/8词元/3会话，需单独规划授权、词表变更、标注与划分。

## 路径、时间与发布约定

- 根目录职责及旧路径迁移以 [根 README](../README.md) 为准：界面归 `frontend/`，相机/感知/控制/平台归 `backend/`，训练/模型/推理归 `algorithm/`，契约/共享特征/记录格式归 `shared/`。原 `src/handshake/...` 和任务中的省略前缀路径按此映射落盘，不另建重复源码树。
- `configs/`、`data/`、`artifacts/`、`docs/`、`scripts/`、`tests/` 保留根目录职责。根 data 是数据集，源码中的数据加载/采集/读写模块按职责分别落 algorithm/backend/shared；源码 models 不存权重。
- T02核心已验证；T03/T04/T06又完成软件/合成测试和短相机探针，但真实感知模型加载失败，不能称全链实机验收。记录格式实际为recording-v1分块NPZ＋逐录制manifest，详见 [采集指南](../docs/collection_workflow.md)。
- 内部时间统一单调ns，录制采用同一 recording 的相对ns并记录 timebase；平台API要求ms时只在适配边界转换。实际曝光时间不可得时注明软件采集近似，回放不混入墙钟时间。跨语言编码须无损；跨设备时钟不可直接相减，T38 另定带不确定度的新鲜度协议。
- Python 特征保持单一参考实现；移动移植必须通过版本化规范/golden fixtures 的特征、窗口、logits、事件与停止一致性检查，不将 Python 包跨平台等同于手机可执行。
- 当前已在用户授权的 T02 范围安装并锁定依赖、执行合成测试；不运行live、不上传报告，不创建虚构真实数据/实验结果。具体环境及限制见 [依赖记录](../docs/dependencies.md)。
- 数据/模型/日志默认受控存储，原视频、身份映射、个人档案和大权重不进入普通Git发布；公开需符合实际授权和各自许可。
- 操作白名单与独立停止从P0起保留。启停/模式/模型/owner或挂起更新context_epoch，执行前串行复检；任何后端不绕调度器。已发起外部调用未必可撤销，停止接收、禁新提交和安全释放确认分别计时，不能把线程超时当系统调用已取消。
- 回放区分逻辑回放和含调度/超时轨迹的回放；预测真值不驱动运行状态，缺少历史上下文时声明人工预设压力条件。

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
