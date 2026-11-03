# 训练指标诊断与可选改进（2026-11-03）

## 结论与证据

当前交付TCN的验证Accuracy为78.85%，Macro-F1为46.52%。问题不是单纯“epoch太少”：类别不平衡、后期泛化停滞和动态类别混淆都有证据。以下原模型数据来自`artifacts/runs/20261102T174832Z-tcn-af17e513/`的训练清单和每轮验证混淆矩阵，不是新训练结果。

| 现象 | 已核实数据 | 含义 |
| --- | --- | --- |
| 长尾标签 | NOG/P1F/P2F共占训练标签85.76%，最大/最小类44.80倍 | 普通CE的总贡献主要来自大类；Accuracy掩盖小类问题 |
| 点击类几乎漏识 | C1F/C2F Recall 4.37%/3.41%；分别有46.54%/53.20%被认成P1F/P2F | 静态指姿与动态点击混淆，不能只看总体准确率 |
| 其他动态类别也较弱 | TLT/TRT Recall 33.46%/46.49%；2C1/2C2 Recall 34.90%/32.58% | 需要查看各类Precision/Recall，而非只调整拒识阈值 |
| 后期过拟合迹象 | 第1→12轮train loss 0.4981→0.2290，val loss 0.7319→0.8512 | 更久训练不一定更好；调大网络暂不是首选 |
| 恒定学习率 | 12轮一直为0.001；按现有选模规则best为第2轮 | 可验证停滞后降低学习率是否有益，不能保证能解决过拟合 |
| 有效batch被稀释 | 459,704个训练端点中334,604个有监督（72.79%）；batch=64时每个小类平均常不到1条 | 没标签的行不产生CE梯度；不能把它们硬标成NOG |

第6轮Macro-F1为0.465269，比第2轮0.465223只高约0.000046，小于预设`min_delta=0.0001`，所以没有替换best。这不是保存错了checkpoint，也不足以靠缩小min_delta解决低指标。

特征包含全局中心、速度和局部手形；没有发现“只用静态手形、完全丢失位移”的实现问题。模型按因果32帧窗口训练，在30fps下最多约1.03秒历史。stride=1的相邻窗口高度重叠，几十万个端点不等于几十万个独立动作。

本机缓存的辅助检查还显示：验证中TLT、TDN的有标签端点只有约52.9%、47.1%具有完整32帧历史。但这是本机缓存统计，不能直接归因原运行的逐样本错误；也不能通过删掉短窗口/未就绪端点抬高主成绩。点击区间边界、轨迹重置和短历史的影响仍需专项实验。

## 本次实现

### 1. 可选温和的类别加权CE

入口：`--class-weighting sqrt_inverse`。默认仍为`none`，保持原基线。

对每个类别的**训练集class_valid端点数**`n_c`：

```text
raw_c = 1 / sqrt(n_c)                         # n_c=0时为0
w_c = raw_c / sum_c(n_c * raw_c / N_train)     # 训练分布下E[w]=1
loss = mean(w_target * CE(logits, target))    # 只平均有监督行
```

- 权重从训练数据生成，不使用验证/测试的类别数。
- 平方根逆频率比直接逆频率温和：本数据权重最大/最小比约6.69，而非44.80。
- 当前NOG/P1F/P2F权重约0.7758/0.5976/0.6095，C1F/C2F约3.5715/3.6611。
- 不按batch权重和重新归一化，避免仅有一个有监督样本时权重完全被约掉。
- 没有训练支持的类别权重为0；若出现这种类别的有效训练标签则报错，不能静默忽略。
- 不叠加重采样，不改变采样顺序、标签、ready过滤、特征或网络架构。
- 权重和计算来源写入`run_manifest.json → identity.objective`；每轮`train_loss_kind`区分加权/未加权。加权train loss不能直接与未加权val loss相减解释泛化差距。
- **验证CE仍未加权；Macro-F1、混淆矩阵仍统计全部有标签端点。**

这会改变优化目标，不保证Accuracy也同时提高。若提高动态类召回却造成大量NOG误判，仍不能视为部署成功；必须重新校准并检验误触。

### 2. 可选验证停滞学习率衰减

入口：`--scheduler plateau`。默认仍为`constant`。

- 使用PyTorch `ReduceLROnPlateau(mode=max)`，监控固定全类验证Macro-F1，绝对改善门槛与选模的`min_delta`一致。
- 默认`--lr-patience 2`：连续3轮没有足够改善后，学习率乘`--lr-factor 0.5`，最低`--min-learning-rate 0.000001`。
- 早停计数不因衰减重置。参数校验要求至少能在第一次降低学习率后再训练一轮；如`lr_patience=2`，早停`patience`须至少4。
- `epochs.jsonl`记录本轮实际`learning_rate`及下一轮`next_learning_rate`；batch日志不再错误地始终写初始学习率。
- scheduler状态与optimizer/RNG一起写入`last.pt`，恢复时一并加载。

### 3. 可选仅有监督端点的均匀采样

入口：`--sampling supervised_uniform`。默认仍为`uniform`，保持原基线。

- 每轮对训练集全部`class_valid=True`的端点做一次无放回随机排列，**每个有标签端点恰好访问一次**；不按类别增删样本，不改变类别比例，也不使用验证类别数。
- 只跳过无标签行作为训练窗口末端，**没有从时间轴删除历史帧**。其特征仍可出现在其他端点的32帧窗口中，时间间隔、速度和窗口填充不变。
- 有标签NOG、短历史、未ready端点全部保留；验证仍遍历同一缓存、计算全部有标签端点的未加权CE与固定全类指标。未来若加入无监督辅助损失，应重新设计采样，不能直接沿用此选项。
- 使用同一个可恢复的torch采样器Generator；采样策略、端点数量、端点ID池SHA256写入`identity.sampling`，不同采样不能混接续训。
- 每轮新增`sampling`、`sampled_endpoints`、`unsupervised_sampled`、`optimizer_steps`日志，便于比较实际优化预算。紧凑缓存仍只按批构造窗口，不物化全量窗口。

本机同一份真实训练缓存的只读预检（batch=64）：

| 项目 | uniform | supervised_uniform |
| --- | ---: | ---: |
| 每轮窗口末端数 | 459,704 | 334,604 |
| 每轮batch数（包含最后不足整批） | 7,183 | 5,229 |
| 平均每批有标签数 | 46.58 | 63.99 |
| 每轮无标签窗口末端 | 125,100 | 0 |

前向端点数减少**27.21%**，不是实测训练耗时减少27.21%，也不是已经提高分类指标。有标签的85,795个短历史端点和114,600个未ready端点均保留，二者可能重叠。

原损失已经只对有标签行取均值，所以无标签行**没有在数学上稀释CE梯度分母**；这里改善的是有效batch构成与计算预算。改采样后梯度噪声、dropout随机数消耗及每轮更新次数可能改变，不能声称与原训练逐步等价，更不能只比较train loss宣称更好。

## 建议怎么跑

先在训练机做预检。以下命令均带`--dry-run`，不会开始训练；确认设备/时间预算后去掉它才训练。其他参数从当前`handshake.toml`及本机覆盖读取，做对照时保持相同数据、batch、seed和其他配置。

```bash
# A0：在当前代码、同一份缓存上重跑原优化策略基线
uv run train --class-weighting none --scheduler constant --seed 42 --dry-run

# A1：优先只改变类别权重，观察小类Recall和Macro-F1
uv run train --class-weighting sqrt_inverse --scheduler constant --seed 42 --dry-run

# A2：与A1相比，只增加停滞学习率衰减
uv run train --class-weighting sqrt_inverse --scheduler plateau --seed 42 --dry-run

# A3：若需要验证更低初始学习率，与A2相比只改这一项
uv run train --class-weighting sqrt_inverse --scheduler plateau --learning-rate 0.0003 --seed 42 --dry-run
```

不必一开始全部跑完；先A0/A1，确认加权方向确实有益，再做后续实验。现有同数据MLP对照也仍值得补齐。最后短名单使用预先约定的种子集合（如42/43/44），报告波动，不挑最好种子。

若单独评估新采样策略，用下面一对对照；两者仅采样不同，先不叠加学习率衰减或修改batch。相同epoch代表遍历同样多的监督样本，不代表相同更新步数，需同时报告`optimizer_steps`和实际耗时。

```bash
# B0：等同A1，训练前记录同缓存、同代码、同设备的基准
uv run train --sampling uniform --class-weighting sqrt_inverse --scheduler constant --seed 42 --dry-run

# B1：只改变训练窗口末端采样；验证分母不变
uv run train --sampling supervised_uniform --class-weighting sqrt_inverse --scheduler constant --seed 42 --dry-run
```

确认B1对逐类召回、Macro-F1和NOG误报的影响后，再决定是否与plateau结合；不要同时开三个新开关后把结果归因某一项。上述仍是预检命令，去掉`--dry-run`才开始真实训练。

每次训练后记录控制台输出的独立run目录，及时执行`uv run evaluate`保存该run的验证报告，再启动下一次训练；短命令使用latest指针，不会自动替所有历史run评估。`uv run calibrate`只能为当前新模型重新校准，不复用旧模型的校准文件。模型与策略冻结前不运行`evaluate --split test`。

### 不能冒充同一次续训/同一份数据

- 改损失、采样、scheduler或代码后应新建run，不使用旧运行的`resume`。既有代码身份校验会拒绝混接，旧模型不会被覆盖。
- 本机缓存与交付训练清单的数据集、划分、统计摘要及类别数量一致，但**train/val索引哈希与端点摘要不完全一致**，本次未假定字节级等价，也未重写回执。原78.85%/46.52%可作历史参考，严格对照应在训练机同一份冻结缓存上重跑A0/A1。
- 更换设备/依赖也可能影响结果；正式实验记录代码版本和实际配置，不把测试机器的合成训练当真实指标。

## 验证状态

上一会话记录报告了174项相关测试、932项非训练回归通过。接续实现采样后，本次重新执行的结果为：

- 采样/加权损失/停滞衰减/短命令/紧凑缓存相关测试：**143通过**。其中包含MLP/TCN × 两种采样 × 两种损失/调度组合的跨进程精确续训，以及紧凑缓存续训/导出；数量与上一会话不是同一测试集合，不相加。
- 广域非训练回归：**946通过、6未选中、13项既有诊断/样例警告**；用`HANDSHAKE_NO_TRAINING=1`禁止反向传播和优化器更新。
- 真实train/val只读预检通过；独立进程确认没有导入torch或训练器、没有创建run/改变latest，旧checkpoint、epochs日志以及train/val缓存文件SHA256未变。
- 本机torch为CPU构建；未认证CUDA/MPS训练性能。本次只有小型合成训练软件测试，**尚未运行真实数据重训，没有新的真实Macro-F1/Accuracy。**

主要实现：`algorithm/training/balancing.py`、`algorithm/training/sampling.py`、`algorithm/models/losses.py`、`algorithm/training/trainer.py`；短命令及配置接入位于`backend/workflows/`。验证摘要见[training_revision_2026-11-03.json](verification/training_revision_2026-11-03.json)。
