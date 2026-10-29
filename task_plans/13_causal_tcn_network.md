# T13 搭建可验证的单分支因果 TCN 神经网络

| 项目 | 约定 |
| --- | --- |
| 阶段 | S3 主模型结构 |
| 课程优先级 | P0 |
| 建议主责 | A（A/B 实名以总方案为准） |
| 前置任务 | T11 |
| 总方案依据 | 第 8.2、8.4、20.1 节 |
| 状态 | 已完成（网络结构、因果性与合成CPU前后向验收）；正式训练待T14 |

总约束和路径解释见 [任务导航](README.md)。下列代码、配置、命令和产物路径均为待实现约定，不代表目前已存在。新增细节属于总方案的实施起点；只有通过试采集和验证后才冻结，不得用最终测试结果反向挑选配置。

## 目标与固定结构

实现课程主模型，不使用预置手势分类结果。首版结构与总方案一致：32 帧、4 个扩张率为 1/2/4/8 的残差块，每块仅一层核大小 3 的时序卷积；不是常见的每块两次卷积版本。

| 层 | 输入 → 输出 | 具体约定 |
| --- | --- | --- |
| 输入 | `[B,32,94]` | T11 schema；浮点特征含 mask |
| 帧级投影 | `[B,32,94]→[B,32,64]` | Linear(94,64,bias=True)、ReLU |
| 转置 | `[B,32,64]→[B,64,32]` | Conv1d 的通道在中间 |
| 残差块 ×4 | `[B,64,32]→[B,64,32]` | dilation 依次 1/2/4/8，stride=1，groups=1 |
| 末帧表示 | `[B,64,32]→[B,64]` | 只取 t=31，禁止时间全局池化 |
| 类别头 | `[B,64]→[B,C]` | Linear(64,C)，课程 C=5，原始 logits |

一个块的明确顺序为：`r=h; h=Conv1d(left_pad(h,2*d),64→64,k=3,dilation=d,padding=0); h=LayerNorm_channels(h); h=ReLU(h); h=Dropout(0.2)(h); return ReLU(r+h)`。LayerNorm 的实现为转成 `[B,T,64]` 后 `LayerNorm(64)` 再转回，只在每个时刻的通道上统计。不能直接对 `[C,T]` 归一化或用混入未来时刻的统计。

## 感受野与容量

`R=1+(3-1)×(1+2+4+8)=31`。32 帧窗口的末端有足够真实历史，冷启动由 T11/T16 门控。按以上结构且 C=5，理论参数量为：投影 6,080；4 个卷积及 LN 共 49,920；分类头 325；合计 **56,325**。参数量只适用于上述单分支结构（LayerNorm含可学习scale/bias）；T25双分支或类数变化需单独重新核算。实现时用程序重新统计，若不一致先核对 bias、归一化和层数，不直接修改报告数字。

## 实施步骤

1. 分别实现 `CausalConv1d`、`CausalResidualBlock`、`GestureTCN`，构造函数读取 D、C、channels、dilations、dropout，禁止隐式使用未来对称 padding。
2. 提供 `forward_all(x)→[B,T,C]` 供因果测试和诊断，`forward(x)` 返回末帧 logits；训练首版只使用末帧标签，防止短前缀监督和在线窗口口径不同。
3. loss 采用有效标签上的 CE；class_valid=false 不计损失。全批次都无监督时跳过优化并记数，不能用零除产生 NaN。
4. 检查 batch=1/多 batch、维度错、mask 全零、有限值、CPU 前后向，设梯度裁剪由 T14 负责。
5. `eval()` 关闭 dropout 后做因果扰动测试：固定 t 以前输入，改变未来输入，forward_all 的 ≤t 输出应在数值容差内不变。
6. 末端窗口推理作为第一版流式策略：每来新观测重算 32 帧。只有优化需要时才实现逐层状态缓存；逐层补零边界会影响短前缀，等价比较只在足够感受野后进行。

## 输出与验收

`models/tcn.py`、`configs/training/tcn.yaml`、网络结构与参数量报告、`tests/unit/test_tcn_causality.py`。

- [x] 维度、参数量、31 帧感受野与总方案对齐，代码每块确为一次卷积。
- [x] 前向/反向无 NaN；输入不足、类表错误及时报错。
- [x] 未来扰动不影响过去输出；验证窗口最后预测与逐帧窗口重算一致。
- [x] 未通过此验收不开展双分支/多任务叠加，交给 T14 训练。

实现语义核对 [PyTorch Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) 与 [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)。层数、维度和容量为本项目设计。

## 前置任务文件

[T11 实现共享特征工程、掩码与时序数据加载](11_features_and_windows.md)

## 2026-10-29 实施记录

- 执行：Codex；成员A/B待用户填写。开始/结构验收2026-10-29。
- `algorithm/models/tcn.py`实现CausalConv1d、CausalResidualBlock、GestureTCN；`configs/training/tcn.yaml`为打包draft配置，`algorithm/model_config.py`加载。每块恰一层左pad卷积，LayerNorm只在同一时刻的通道维，末帧分类无时间池化。
- 默认[32,94]→[5]，程序计数56,325参数，RF31；forward_all支持短前缀诊断，常规forward要求完整窗口。`algorithm/models/losses.py`按有效末端标签CE，全无监督返回None供T14跳步。
- `tests/unit/test_tcn_causality.py`、`test_model_losses.py`覆盖未来扰动、感受野最早影响、整流末端与32帧重算、batch独立、全零mask、有限CPU梯度、无效维度/类表/窗口；合成可过拟合也通过。
- 结构、梯度、因果性各项通过，未开展双分支/多任务；无正式checkpoint。命令/配置与实现哈希、报告见 [统一验证记录](../docs/verification_t11_t12_t13.md)。
