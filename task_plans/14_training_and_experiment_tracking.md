# T14 实现可复现训练、采样与模型选择

| 项目 | 约定 |
| --- | --- |
| 阶段 | S3/S4 训练 |
| 课程优先级 | P0 |
| 建议主责 | A（A/B 实名以总方案为准） |
| 前置任务 | T12, T13 |
| 总方案依据 | 第 14、15.9、16.5 节 |
| 状态 | 待实施；本文是规划，不代表已有代码、数据或结果 |

总约束和路径解释见 [任务导航](README.md)。下列代码、配置、命令和产物路径均为待实现约定，不代表目前已存在。新增细节属于总方案的实施起点；只有通过试采集和验证后才冻结，不得用最终测试结果反向挑选配置。

## 目标与输入

给 MLP 与 TCN 建立同一训练器，输入固定数据/特征版本，输出可追溯 checkpoint。验证集用于模型和门限选择，最终测试不参加 epoch、学习率或采样策略决定。

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
3. 每个 batch 使用 loss mask；记录正负分布、有效标签数、loss、梯度范数、learning rate。NaN 终止本次训练并保存诊断，不能静默跳过大量坏样本。
4. 每 epoch `model.eval()`、禁梯度、顺序遍历固定验证末端集合，计算包括 no_action 的每类指标与 Macro-F1。按指标选 checkpoint；同分按预定较早 epoch 规则处理，不挑最终测试最好者。
5. 早停 max_epochs/patience 生效；checkpoint 包含 state_dict、optimizer、epoch、seed、训练配置、schema、类表、split/data/代码哈希。断点恢复检查兼容性，禁止跨版本接着训练却沿用旧 run_id。
6. TensorBoard 或 JSONL 选一个主记录入口。每次试验独立 run_id，保留失败/负结果；保存至少 best 与 last，最终模型引用 best。
7. 健康检查：极小训练子集可过拟合、打乱标签诊断、意图头后续是否只识别“有手”。健康诊断使用 train/development，不读取最终测试标签做调参。

## 输出与验收

`scripts/train.py`、`training/trainer.py`、`artifacts/runs/<run_id>/config.yaml`、曲线、checkpoint 与 `run_manifest.json`。

- [ ] 同一配置可从独立进程完成训练和恢复，报告复现容差及非确定性来源。
- [ ] 最终选择过程可由验证曲线和清单解释；数据/统计/split 可追溯。
- [ ] MLP/TCN 预算、样本与选择指标一致；所有结果区分 train/val/test。
- [ ] T15 获得冻结候选及验证流预测；不在本任务生成“最终测试最优模型”。

## 前置任务文件

[T12 建立规则与单帧 MLP 对照基线](12_rules_and_mlp_baselines.md)；[T13 搭建可验证的单分支因果 TCN 神经网络](13_causal_tcn_network.md)
