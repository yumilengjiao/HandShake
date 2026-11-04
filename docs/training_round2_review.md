# 第二轮训练复盘与下一步优化（2026-11-03）

> 2026-11-04更新：下文提出的weight_decay=0.001与MLP对照已在夜间完成，Macro-F1分别为48.41%和38.78%，均未超过参考模型49.14%；其他夜间对照也未胜出。此文保留历史计划，最新结果与下一步方向见[夜间训练复盘](training_overnight_review.md)。

## 已完成：plateau不是这次的胜出方案

参考模型为`20261103T083328Z-tcn-113b3b10`，本轮为`20261103T123924Z-tcn-b196181f`。两轮训练manifest中的数据、架构、环境、算法代码哈希完全一致，配置仅`scheduler: constant → plateau`不同；种子都是42。因此这次比历史交付包更适合作单变量对照，但只有一个种子，不代表统计上的普遍优劣。

| 最佳checkpoint的验证指标 | 加权＋constant参考模型 | 加权＋plateau本轮 |
| --- | ---: | ---: |
| 最佳epoch | 6 | 6 |
| 完成轮数（早停） | 16 | 16 |
| 全标签Accuracy | **79.20%** | 78.82% |
| 全类Macro-F1 | **49.14%** | 48.67% |
| 平衡准确率 | **50.05%** | 49.16% |
| 未加权验证CE | **0.7853** | 0.8076 |
| C1F召回 | **30.88%** | 25.03% |
| C2F召回 | **18.49%** | 15.91% |
| 2C1召回 | 24.53% | **29.41%** |

不是所有类别都下降，也不是plateau实现没有运行。本轮实际学习率为0.001→0.0005→0.00025→0.000125→0.0000625；第一次降低发生在第5轮结束，第6轮使用0.0005。第6→16轮加权训练loss从0.4324降至0.2960，但未加权验证CE从0.8076升至0.9377，Macro-F1回落到46.77%。加权train loss与未加权val CE不能直接相减解释差距，但各自的趋势支持“后期泛化未改善”，不支持继续堆epoch。

本轮已补做全部验证端点评分和匹配的温度/拒识校准：

- 参考模型接受覆盖率69.27%、接受样本准确率92.28%。
- 本轮接受覆盖率70.78%、接受样本准确率91.17%。
- 两者都是T=1.5、阈值0.7；这是拟合所用验证集上的经验结果，不是独立测试/相机性能。接受准确率不能替代全分母Accuracy或F1。

## 当前使用哪个模型

按用户此前“哪个好就用哪个”的要求，以预设全标签验证Macro-F1为主要依据，`artifacts/runs/latest.json`已恢复指向参考模型`20261103T083328Z-tcn-113b3b10`。旧模型和本轮的best/last、日志、模型包全部保留，没有相互覆盖或混用校准。

因此短命令`evaluate/calibrate/export/app --model`现在关联参考模型，不是刚完成的plateau实验；这是有意选择，不是本轮训练丢失。

本轮评估文件：

```text
artifacts/runs/20261103T123924Z-tcn-b196181f/evaluations/20261103T140511Z-val-16427bd1/classification.json
artifacts/runs/20261103T123924Z-tcn-b196181f/evaluations/20261103T140511Z-val-16427bd1/calibration.json
```

比较和选择记录：`artifacts/diagnostics/plateau-review-20261103T140819Z.json`。模型包、归一化、校准匹配及golden前向均通过校验，不授予真实应用动作权限。没有访问test样本，也没有由本次复盘启动神经训练。

## 下一步：先检验正则化，不同时改变很多设置

若接着训练TCN，建议先从参考模型配方出发，**仅改变weight_decay：0.0001→0.001**，退回constant调度，其余设置保持不变。它检验更强AdamW权重衰减是否改善泛化；不是已经证明有效的优化，也不是下一轮必定超过49.14%。

Nushell/其他终端可直接复制单行命令：

```bash
uv run train --model tcn --class-weighting sqrt_inverse --sampling uniform --scheduler constant --learning-rate 0.001 --weight-decay 0.001 --batch-size 64 --epochs 80 --patience 10 --seed 42 --dry-run
```

确认预算后开始真实训练：

```bash
uv run train --model tcn --class-weighting sqrt_inverse --sampling uniform --scheduler constant --learning-rate 0.001 --weight-decay 0.001 --batch-size 64 --epochs 80 --patience 10 --seed 42
```

结束后先`uv run evaluate`，记录该run结果，再启动其他实验。不要用旧run的resume切换正则化。

同时仍需MLP对照来判断时序模型有没有明显收益。预算允许时单独运行：

```bash
uv run train --model mlp --class-weighting sqrt_inverse --sampling uniform --scheduler constant --learning-rate 0.001 --weight-decay 0.0001 --batch-size 64 --epochs 80 --patience 10 --seed 42
```

它与参考模型只改变模型类型。以上两项均已真实train/val只读预检通过，尚未由助手开始训练。若这些简单实验仍无改善，再根据逐类错误审查点击/双击标注与因果可辨识性，考虑关节运动特征、多尺度时间表达和语义一致的训练增强；不得删除难样本或更换验证分母抬高成绩。短名单应使用预先固定的多个种子报告波动，test继续留到模型/策略冻结后。

## 显示改进，不是算法提分

`backend/workflows/training.py`新增从完整验证混淆矩阵计算Accuracy的控制台显示；Macro-F1改为百分比显示。例如：

```text
Epoch 6/80 | loss=0.43239 | val Accuracy=78.82% | val Macro-F1=48.67% | lr=0.0005 → 0.0005
```

这个改动不修改损失、训练日志数值、模型结构或checkpoint选择。兼容无混淆矩阵的旧/模拟进度，空矩阵不除零。工作流、训练覆盖参数、采样、平衡及部署相关非训练测试123项通过。
