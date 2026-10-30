# T14 实现可复现训练、采样与模型选择

| 项目 | 约定 |
| --- | --- |
| 阶段 | S3/S4 训练 |
| 课程优先级 | P0 |
| 建议主责 | A（A/B 实名以总方案为准） |
| 前置任务 | T12, T13 |
| 总方案依据 | 第 14、15.9、16.5 节 |
| 状态 | 模块完成；5/15类CPU合成训练、梯度与恢复通过；正式数据训练待T07/T10 |

实施日期：2026-10-29。实际源码位于 `algorithm/training/`、`backend/interaction/`、`backend/actions/`（按本任务职责），使用与限制见 [说明](../docs/training_and_interaction.md)，证据见 [验证报告](../docs/verification_t14_t17_t18.md)。以下未验收项仍保留；规划路径以已实现职责目录为准。

**public-data-v1适配**：以[公开数据方案](../docs/public_data_plan.md)优先，不强制自采。现有模块只验证旧5类草案；15类数据须经T07来源/语义核验、统一契约迁移与T10划分后再接入。公开首版计划UI/键盘启停，不把Open twice/Grab伪映射为张掌/握拳。最新整体验证见[master适配记录](../docs/verification_master_sync.md)。

总约束和路径解释见 [任务导航](README.md)。下列代码、配置、命令和产物路径均为待实现约定，不代表目前已存在。新增细节属于总方案的实施起点；只有通过试采集和验证后才冻结，不得用最终测试结果反向挑选配置。

## 目标与输入

给 MLP 与 TCN 建立同一训练器，输入固定数据/特征版本，输出可追溯 checkpoint。验证集用于模型和门限选择，最终测试不参加 epoch、学习率或采样策略决定。供 checkpoint 选择的基础分类指标在本任务先实现并用合成样例核验，T15 扩充事件评估与校准，不要求先完成 T15 才能训练。

## 起始配置（验证前不是最优参数）

```yaml
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 64
max_epochs: 80
patience: 10
min_delta: 0.0001
grad_clip_norm: 1.0
dropout: 0.2
seed: 42
checkpoint_metric: val_macro_f1
```

先固定一种学习率策略（首版可保持常数）；显存/内存不足降低 batch 并记录。只做小范围有记录搜索，不能无穷尝试验证用户直至过拟合。研究版主模型至少 3 个种子；课程资源不足时记录实际重复数，不冒充统计稳定性。

## 实施步骤

1. 启动校验类表、schema、split 交集、数据哈希、归一化来源；固定 Python/NumPy/框架种子，记录硬件、库版本及可重现设置限制。
2. 以相同末端样本索引训练 MLP/TCN。先使用普通随机窗口采样与无权重 CE；若背景压倒少数类，再从训练集统计选择一种适度分层或类权重，不能同时强重采样＋强权重＋focal。
3. 每个 batch 使用 loss mask；记录正负分布、有效标签数、loss、梯度范数、learning rate。全批无有效监督时跳过 optimizer/scheduler 更新并计数；NaN 终止并诊断，不能静默跳过大量坏样本。
4. 每 epoch `model.eval()`、禁梯度、顺序遍历固定验证末端集合，计算包括 no_action 的每类指标与 Macro-F1。按指标选 checkpoint；同分按预定较早 epoch 规则处理，不挑最终测试最好者。
5. 早停 max_epochs/patience 生效；checkpoint 包含 state_dict、optimizer、scheduler（如有）、epoch/global_step、Python/NumPy/框架及采样器 RNG 状态、训练配置、schema、类表、split/data/代码哈希。恢复时校验版本；若不能恢复 worker/采样顺序，明确只支持 epoch 边界或统计近似复现，不声称与未中断逐步一致。禁止跨版本续训却沿用旧 run_id。
6. TensorBoard 或 JSONL 选一个主记录入口。每次试验独立 run_id，保留失败/负结果；保存至少 best 与 last，最终模型引用 best。
7. 健康检查：极小训练子集可过拟合、打乱标签诊断、意图头后续是否只识别“有手”。健康诊断使用 train/development，不读取最终测试标签做调参。

## 输出与验收

`scripts/train.py`、`training/trainer.py`、`artifacts/runs/<run_id>/config.yaml`、曲线、checkpoint 与 `run_manifest.json`。

- [x] 合成输入：同一配置可从独立进程完成训练和恢复，报告复现容差及非确定性来源。
- [x] 合成输入：最终选择过程可由验证曲线和清单解释；数据/统计/split 可追溯。
- [x] 合成输入：MLP/TCN 预算、样本与选择指标一致；所有结果区分 train/val/test。
- [ ] T15 获得冻结候选及验证流预测；不在本任务生成“最终测试最优模型”。

## 前置任务文件

公开数据增量：训练器从版本化缓存解析词表，15类计数/指标/权重与恢复身份均受校验；来源分类不能进入旧交互。见 [实现指南](../docs/public_data_implementation.md) 和 [深度学习实验计划](../docs/deep_learning_experiment_plan.md)。

[T12 建立规则与单帧 MLP 对照基线](12_rules_and_mlp_baselines.md)；[T13 搭建可验证的单分支因果 TCN 神经网络](13_causal_tcn_network.md)
