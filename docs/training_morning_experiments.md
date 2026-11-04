# 上午训练：因果运动分支与 dropout 对照（2026-11-04）

> 最终更新：监督器已在12:00停止。完整motion候选为79.38% Accuracy / 50.57% Macro-F1；seed43只完成18/20epoch，单击两类退步，短dropout试验未启动。以下保留实际方案和交接历史，最新选用、只读复验诊断及限制见[最终结果](training_morning_results.md)。

## 授权与当前状态

用户明确授权继续真实训练到本机 **2026-11-04 12:00（UTC+8）**。后台服务初始为 `handshake-morning-20261104.service`，性能预算交接后为 `handshake-morning-20261104-stage2.service`；08:15 开始第一轮因果运动分支训练。

- 实时状态：`artifacts/morning/2026-11-04-to-12/status.json`
- 持续更新的结果与中午总结：`artifacts/morning/2026-11-04-to-12/summary.md`
- 初始冻结计划：`artifacts/morning/2026-11-04-to-12/plan.json`
- 当前计划：`artifacts/morning/2026-11-04-to-12-stage2/plan.json`（仅性能预算修订及同一run续训）
- 任务日志：当前阶段目录下 `jobs/*/worker.log`

08:32在第2个完整epoch保存后做了一次自有服务交接：首轮每epoch约7–9分钟，原100分钟单任务上限可能提前截断20epoch。因此仅把单任务上限放宽到160分钟，从原run的完整`last.pt`精确续训；代码、数据、配置、模型身份和12:00硬截止不变。原状态/总结已备份为`status-stage1.json`/`summary-stage1.md`，原实时路径通过符号链接指向当前阶段，交接回执为`performance-handoff.json`。没有根据早期分数改配方或篡改训练清单。

本文件记录方案，不冒充已经完成的真实结果；实时成绩以状态/总结为准。目标是争取固定全类验证 Macro-F1≥52%，**不保证达到**。不读取真实 test、不打开相机、不执行应用动作、不改变标签、ready过滤或缓存端点。只替换满足预先声明条件的完整、校准匹配模型。

## 为什么换方向

[夜间复盘](training_overnight_review.md)显示：更低学习率、加强权衰减、放宽裁剪、增大batch、仅监督采样、MLP均未超过加权TCN的79.20% Accuracy / 49.14% Macro-F1。完整历史且ready的点击类别仍弱，不支持只靠删短历史或继续同类调参抬分。

因此本次针对历史运动表达与另一种正则化方式做短名单对照。原默认网络及已交付的旧模型包保持原语义。

## 明确的实验变体

入口新增 `--model-variant`，仅非默认TCN接受实验变体：

| 值 | 定义 |
| --- | --- |
| `baseline` | 原 T13：单卷积残差块、末帧类别头、dropout=0.2；默认不变 |
| `motion` | 原 TCN＋4/8/16帧的因果运动差异分支，dropout=0.2 |
| `dropout35` | 不加运动分支，只把TCN的dropout改为0.35 |
| `motion_dropout35` | 运动分支＋dropout=0.35；仅在前面没有新胜出模型且预算允许时补做 |

实现路径：`algorithm/models/motion_tcn.py`、`algorithm/model_config.py`；配置、训练身份和模型包架构都显式记录 `model_variant`。原 `algorithm/models/tcn.py` **没有修改**，原包runtime pins、参数名称/形状、golden与校准仍通过校验。新运动包额外固定运动模块的runtime摘要，不能拿默认网络误加载运动权重。

### 运动分支公式与因果边界

输入仍是 `[B,32,94]`，不重建缓存。前69维为几何/运动特征，最后21维关键点mask用于判断当前帧是否有手。对每个时刻及尺度 k∈{4,8,16}：

```text
present_t = any(keypoint_mask_t > 0)
mean_k,t = sum(present_s * geometry_s) / max(sum(present_s), 1)
           # s 从 t-k+1 到 t，只读取当前与过去
contrast_k,t = present_t * (geometry_t - mean_k,t)

branch = Linear(207,64) → per-time LayerNorm → ReLU → Dropout → Linear(64,C)
logits_t = original_TCN_logits_t + present_t * branch(concat(contrasts))
```

均值排除无手/填充帧，避免把常量手形因填充比例变化解释为运动；当前无手时残差不贡献分数。此存在门控不替代逐关键点质量审查，部分关键点缺失的影响仍须验证。最后类别层零初始化，起始eval logits与同种子的原TCN完全一致。

分支最多读16帧，原TCN感受野31帧，因此整个模型仍为31帧；不引入未来归一化或未来池化，完整历史输出与32帧重算末端保持一致。所有class_valid端点仍参与损失/验证，未ready和NOG不删除。

15类运动变体约71,390参数；原15类TCN为56,975。没有宣称更多参数必然提分。

## 实验顺序、预算与选模

预声明队列：`motion-contrasts` → `tcn-dropout35` → 条件性的 `motion-dropout35` → 新胜出配方的固定seed43复验。若第一轮已胜出且剩余时间不够同时做第二轮和复验，优先保留复验预算；不在截止前启动明显来不及完成的任务。

各候选从原参考配方出发：sqrt_inverse、uniform、constant、lr=0.001、weight_decay=0.0001、batch=64、clip=1、seed42、patience=10。**本次每轮最多20epoch**，参考原运行最多80但第16轮已早停。这是明确的中午预算变化，不能伪称与历史运行只改变一个配置字段。所有本次候选使用相同20epoch上限；确认配方只改seed。

晋升规则预先固定：

- 只允许seed42候选替换；全标签验证Macro-F1须比当前保留模型高至少 **0.1个百分点**。
- Accuracy≥**78.20%**；NOG Recall≥**76.12%**。
- C1F/C2F/2C1/2C2平均Recall≥**28.20%**，避免总体F1上升但点击能力明显塌陷。
- 有可行的匹配验证校准，接受覆盖率≥**60%**，校准搜索经验risk约束≤10%。
- 完整运行、权重/数据/架构摘要、模型包golden、归一化与校准匹配都通过。
- seed43只用于同一配方复验/报告波动，不挑最好seed晋升。两种子仍不足以认证可靠总体不确定性。

未胜出时保留 `20261103T083328Z-tcn-113b3b10`。新训练实现摘要与旧记录不同；旧参考只有明确固定的manifest/数据/包被用于比较，不篡改其清单或绕过续训检查。冻结后不修改algorithm/shared代码，改动会使队列安全停止。

## 资源、停止与验证

单进程串行CPU训练，线程1；服务CPUQuota=110%，MemoryHigh=1800M、MemoryMax=2G，另监控worker RSS、磁盘余量及单任务时间。12:00时向自己启动的进程组发SIGINT，必要时逐级终止，保留最佳checkpoint与最后完整epoch；不杀用户其他训练。服务RuntimeMax是额外停止后备。

临时阻止自动休眠仅在服务存活期间有效；电脑仍须保持开机和供电。锁、PID/start stamp、进程组与不误删外部锁的逻辑沿用已测试的夜间执行器。

已验证的软件，不是实际提分成绩：

- 变体、只读前向、CLI参数、旧包/评估兼容：58项通过。
- 因果梯度、精确resume、合成训练、5/15类导出/加载及严格架构校验：178项通过。
- 广域非训练回归：975通过、6未选中、13项既有合成/样例警告。
- 上午队列策略、截止中断和进程隔离：9项通过。
- 新运动模块及forward测试Ruff检查/格式检查通过；旧参考包与匹配校准再次验证通过。

## 已完成的首轮结果与条件性短预算补充

09:51首轮`motion`完成13epoch，最佳为第3轮：验证Accuracy **79.38%**、Macro-F1 **50.57%**，较原参考提高0.17/1.43个百分点，NOG Recall为80.92%；模型包与匹配校准通过，按原规则晋升seed42。校准接受覆盖率70.39%、接受准确率92.00%，仍不是全分母或独立测试成绩。

但逐类并非全升：C1F Recall从30.88%降到14.70%，C2F从18.49%降到11.26%；点击四类平均从30.20%降到28.98%。原平均召回底线允许这项代价，因此不能把整体F1提升描述为单击能力改善。完整逐类表在实时总结中保留。

固定seed43复验正在运行。为了保留复验时间，原计划的完整20epoch dropout对照被跳过。另在观察任何dropout候选分数之前，准备了一个条件性**6epoch真实短预算对照**：只有复验完整结束且12:00前至少31分钟仍可用时，才串行启动`dropout35`；同一代码、数据、损失与优化器配方，显式记录预算差异，不冒充同预算长期对照。

该短候选除原F1/Accuracy/NOG/校准条件外，须C1F Recall≥25.88%、C2F≥13.49%才可晋升，防止平均值掩盖单击进一步塌陷。此规则只适用于尚未观察的新短候选；不据此把已完成的运动模型改称均衡或可靠的单击模型。

准备文件/可选阶段：`artifacts/morning/2026-11-04-noon-short-trial/`；若启动，原实时状态路径继续重定向，完整复验和各阶段总结都保留。新增底线和短预算是观察首轮后对下一轮的明确修订，不是独立最终测试。

## 单行预检示例（Nushell兼容）

```bash
uv run train --model tcn --model-variant motion --class-weighting sqrt_inverse --sampling uniform --scheduler constant --learning-rate 0.001 --weight-decay 0.0001 --batch-size 64 --epochs 20 --patience 10 --seed 42 --dry-run
```

本机自动队列运行期间不要再手动启动该训练；操作锁会拒绝并发。省略变体仍为原基线，`resume`不接受临时变体切换。中午总结中的验证指标不是独立测试、真实相机性能或动作安全验收。
