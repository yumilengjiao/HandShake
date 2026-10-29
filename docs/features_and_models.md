# T11/T12/T13 特征、基线与因果网络

2026-10-29，`handshake-core 0.2.0`。本轮交付共享特征、训练输入适配、规则/MLP/TCN及统一预测接口；没有真实数据训练、正式模型包或Web实现。结果和限制见 [验证记录](verification_t11_t12_t13.md)。

## 特征与输入

权威实现为 `shared/features/`，完整字段、公式、单位和draft阈值在 `feature_schema.json`；版本仍为首次具体实现的 `landmarks94-v1`。类表从打包 `shared/defaults/classes.json` 加载，五类顺序与T05一致，未知/换序/增删类别拒绝。

原始图像W/H已知时，先构造 `q=(x,y·H/W,z)`；尺度为掌指根5/9/17到腕点0的平均二维距离，掌心为0/5/9/13/17的均值。local63为逐点腕相对坐标除掌尺度；global10依次为cx/cy/vx/vy/s/ds_dt/dt_seconds/missing_ms/q_geom/motion_valid；最后21维为派生mask。所有xy距离统一到图像宽度单位，不统一左右手、不旋转、不改变未镜像方向。z保留感知API的相对深度含义。

`FeatureExtractor` 每条录制/轨迹独立使用，流、轨迹、epoch或timebase变化清历史。成功感知无手或质量失败可为既有轨迹追加missing；没有真实轨迹时missing返回None，不能生成背景手。未处理帧只推进源健康时钟，不追加观测或无手预测。源帧时间严格递增；处理时钟与源顺序分别检查，允许合法的推理/处理延迟。

速度与尺度导数只在连续有效、时间间隔合理时计算。缺失后第一帧motion_valid=0。几何无效填零，missing_ms保留距最近有效观测的时长；无有效历史时另有history_initialized=false，不填无限值。

## 统计与就绪

`FeatureNormalizer.fit` 仅接受split=train。离线适配器先验证预登记用户划分、原片/recording/lineage与可见参与者无跨集合，再提取train样本。local63和cx/cy/s只从有效几何拟合，vx/vy/ds_dt还要求motion_valid；不读取类别有效性来决定几何统计。缺样本用mean0/std1，近常量用std1并告警/记fallback字段。dt和missing按固定尺度截断，q_geom/motion_valid/mask不标准化，无效几何在标准化后归零。统计保存dataset/split/schema/config哈希，加载检查版本与外部期望哈希，不覆盖已有统计文件。

`FeatureWindowBuilder` 冷启动、长断流或长失跟后需连续32个有效观测；预热期间无效或过大采样间隔重新计数。持续每帧失败超过重置时间也算长失跟，不能沿用旧预热。短缺失保留历史，但当前无效不ready；恢复检查窗口有效比例、最大跨度和当前新鲜度。ready不含owner或候选恢复授权。

健康查询与处理时钟不能倒退；`current`或带now_ns的`append`发现已超过长断流阈值时会清历史并返回None，调用方按输入不可用处理，不能当作ready窗口。

首版不伪造30Hz重采样。窗口保留源间隔；不同采样率的合成测试只验证实现，不证明真实帧率鲁棒性。

## 数据集、增强与缓存

`algorithm/data/dataset.py` 的 ObservationRecord / SplitManifest 是 **T11草案输入适配格式**，不冒充已经完成的T06/T10正式文件。记录包含匿名用户/会话/录制/track/epoch、原片与lineage引用、图像尺寸、FrameStatus、可选HandObservation、末帧类别与掩码；manifest需提供预分配subject_splits及整个解析记录集合的规范化SHA256。真实输入必须status=frozen，合成演示明确synthetic=true；本轮没有为真实参与者生成划分或统计。

`WindowDataset` 只在同录制、用户、会话、stream和track内构窗；epoch变更由共享实现重置。返回x[32,94]、y、loss_masks、source IDs/末端ns、ready/reason和valid_length；短前缀左填0，标签有效性与ready分开。固定stride按已处理末端选择，全部原始记录和末端库存另保留；无初始轨迹的no-hand时段记不可用，不创建假轨迹。coverage分母包含预选但不可表示的末端，事件真值以后不能因ready裁剪。

原始观测增强在 `algorithm/data/augmentation.py`：只接受明确train来源，支持显式的小噪声、模拟感知dropout和水平翻转。翻转x并交换左右类别，z和时间轴不变；之后重跑共享特征，速度由新轨迹计算。默认无增强；派生记录保留source_split=train、父记录集合哈希、增强配置/seed和原片/lineage，加载时与预分配用户角色复核，不能将test用户伪改成train。调用者生成新的派生数据版本/哈希再构缓存。该函数不修改原始数据，也不自动增强AnnotationTimeline或事件评估真值。时间反转、速度/时间伸缩均未启用，正式训练仍需T14验证增强不破坏类别边界。

`scripts/build_features.py` 是薄入口：

```bash
.venv/bin/python scripts/build_features.py observations.json splits.json data/features/train-v1 --split train
.venv/bin/python scripts/build_features.py observations.json splits.json data/features/val-v1 --split val --stats data/features/train-v1/stats.json
```

输入observations是上述ObservationRecord的JSON数组；splits是SplitManifest，具体字段见源码类型与 `tests/integration/test_feature_dataset.py` 的纯合成示例。未来T06格式通过适配层接入，不另写几何公式。正式原始资料留受控data目录，不提交成公开fixture。

输出features.npz、stats.json、index.json；目录须新建，已有目录拒绝覆盖。NPZ不使用pickle，缓存绑定schema/classes/config/dataset/split/stats与文件哈希；加载需提供外部期望版本，字节篡改或版本错立即拒绝。原始记录不被覆盖。

## 模型与统一预测

| 模型 | 输入/结构 | 参数/历史 | 当前边界 |
| --- | --- | --- | --- |
| B0规则 | 同一个32帧窗口；反标准化前69维后，用PIP角/归一距离、有限轨迹位移/方向/速度判类 | 0可训练参数；32帧有限历史 | 不是单帧规则；数值为draft，输出one-hot决策分数而非概率 |
| MLP | 末帧94→128→64→C；ReLU、两次Dropout(0.2) | C=5时20,741参数 | 包含T11一阶速度，所以并非完全无运动信息 |
| TCN | 每帧94→64，四个单卷积残差块d=1/2/4/8，k=3，末帧64→C | C=5时56,325参数，感受野31 | 每块只有一次卷积，32帧末端推理，不做时间全局池化 |

TCN的LayerNorm只对同一时刻通道维统计，卷积只在左侧补零。`forward_all` 供诊断并允许短前缀，正常 `forward` 严格要求完整配置窗口；训练只监督末端。`masked_cross_entropy` 仅对class_valid标签算CE，全批无监督返回None通知训练器跳过优化。没有在损失前softmax，没有预置手势分类结果。

模型配置在 `configs/training/mlp.yaml`、`tcn.yaml`（链接至打包资源），由 `algorithm/model_config.py` 校验/创建模型；C从统一类表读取。规则参数公开在RuleThresholds。B0纵向阈值单位为图像宽度，与T05以图像高度定义的标注起点参数分开；不得声称二者等同。

`algorithm/inference.py` 的RulePredictor/TorchPredictor共用窗口长度、ready、当前质量、跨度、新鲜度和同域时钟门控。MLP取最后一帧，TCN取整窗，规则对同一标准化窗反变换几何；推理前后各检查时钟，保留提交时stream/track/epoch和源帧，绝不重贴新context。录制/合成时钟必须显式注入，不能混入墙钟。输出ModelPrediction，保留intent/phase/calibrated_probabilities=None，不把类别或规则分数当执行授权。完整事件引擎待T17；各模型后续接同一引擎，分别按相同验证协议/预算选门限。

## 运行与后续

```bash
uv sync --locked --extra dev --extra training
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
env -u PYTHONPATH .venv/bin/python -m algorithm.smoke --output artifacts/model-smoke.json
```

smoke只在20个容易分离的人造张量上做100次优化，检查可过拟合、参数量和CPU前向耗时，并对5个规则几何fixture给分项诊断；不保存权重，不做正式验证集选择。输出拒绝覆盖同名报告；T14正式训练入口仍未实现。本轮不给真实F1、误触率或端到端延迟结论。

T04真实观测、T07词表/pilot与T10正式划分齐备后，T11再拟合并冻结真实统计；T14共用训练器加载这些模型，T15校准，T16完成可信模型包与在线部署。网络语义核对依据为 [PyTorch Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)、[LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)；94维公式和层宽是本项目设计，不是第三方论文的实测结论。
