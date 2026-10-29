# T16 打包模型并实现带新鲜度校验的在线推理

| 项目 | 约定 |
| --- | --- |
| 阶段 | S4 部署推理 |
| 课程优先级 | P0 桌面 PyTorch 参考；移动运行时导出归 T37 |
| 建议主责 | A 主责，B 集成（A/B 实名以总方案为准） |
| 前置任务 | T14, T15 |
| 总方案依据 | 第 7.8、14.7～14.8、16.2、18 节 |
| 状态 | 待实施；本文是规划，不代表已有代码、数据或结果 |

总约束和路径解释见 [任务导航](README.md)。下列代码、配置、命令和产物路径均为待实现约定，不代表目前已存在。新增细节属于总方案的实施起点；只有通过试采集和验证后才冻结，不得用最终测试结果反向挑选配置。

## 目标与包格式

把训练结果装成可加载的版本化模型包，而不是随意复制一个权重文件。桌面首版部署 PyTorch；为 T37 提供可转换的推理图、算子/shape 说明和固定参考输入输出。移动本地运行必须验证一种便携运行时，不以 `.pt` 文件或 ONNX 导出成功冒充手机可运行；移动转换不阻塞桌面课程交付。

```text
artifacts/models/<model_version>/
  weights.pt
  architecture.json
  classes.json
  feature_schema.json
  normalization.json
  calibration.json
  event_thresholds.yaml
  training_manifest.json
  model_card.md
  checksums.json
```

`event_thresholds.yaml` 必须标记 draft/frozen 及选择来源；集成期可使用明确标注的示例起点。模型包不可原位改写：T28 验证选择后的 frozen 参数生成新的 release candidate、清单和哈希，重跑受影响安全/回放检查后才测最终测试集；测试报告作为外部关联产物，不反过来修改已测包。

若无意图/阶段头，架构配置明确 heads=[class]；预测对象对应字段为 None，不伪造恒为 1 的“意图概率”。加载自有、可信来源的权重，不以任意 pickle 为公共模型分发格式。

## 实施步骤

1. 从 best checkpoint 导出推理所需参数、结构、标签与特征协议；校验 D=94、C=类表数、T=32、感受野31和训练/配置哈希。相同维度但 schema/镜像/特征顺序不符同样拒绝。
2. Predictor 提供 prepare/reset/predict 接口；每轨迹独立 FeatureWindow，batch=1、eval、禁梯度。首版每新观测重算窗口，禁止拼接不同 track 或替换后的旧历史。
3. 冷启动或长失跟清缓存，严格复用 T11 的连续 32 个有效步预热、窗口有效比例/时间跨度及 ready_reason；不另写一套累计计数。短缺失保留掩码但当前失效帧禁执行；ready 恢复不等于控制权或候选恢复。
4. 在提交推理工作时捕获 stream_id/context_epoch/model_version，连同源 frame_id/源时间原样返回；完成时间另记。丢弃乱序、跨 stream/epoch、模型不符或超过 max_frame_age_ms 的结果，不能在旧回调完成时补上当前 epoch 使其复活。
5. 离线窗口与在线逐步重算在同一输入上比较 logits；推荐 float32 初始容差 atol=1e-5/rtol=1e-4，按实际后端记录。若改 ONNX/量化，复测分类、校准、拒识、事件和延迟，不只比较张量误差。
6. 模型热切换先暂停、清候选、释放按键、清所有窗口，校验成功后恢复预览并要求重新唤醒；失败回到已知旧包或禁用，不部分替换。
7. 缺模型、损坏包、NaN logits、维度不符提供可理解错误；不得让真实后端继续用上一条预测执行。

## 原生多端模型交接

- 提供特征/类表/校准/事件配置的共同版本与哈希，以及 golden 窗口/logits；训练框架不得渗入 UI/平台输入适配器。
- T37 根据 T36 选型验证 ONNX Runtime Mobile、LiteRT、Core ML 等一个合适候选，登记转换器、运行时、算子、CPU/加速委托、精度和每个平台资产许可。转换失败返回明确不支持，不能偷偷替换成预置手势模型。
- 推理资产须可预置离线加载；量化/委托改变结果时重新校准、评估事件/拒识与安全，再按 T28 生成新冻结包。手机不承担训练、不依赖完整 Python/PyTorch 环境。

## 交付与验收

`inference/predictor.py`、`inference/model_loader.py`、`scripts/export_model.py`、一致性报告与模型卡草案。

- [ ] 正确包可加载并预测；错类表/错 schema/损坏校验/缺文件拒绝。
- [ ] 同样历史得到相同末帧输出；乱序/过期预测无普通事件。
- [ ] 目标 CPU、batch=1 实测 p50/p95、线程数及内存；“推理 5ms”不冒充全链延迟。
- [ ] 清理和停止无需等待模型返回；交接 T17/T20/T23。

## 前置任务文件

[T14 实现可复现训练、采样与模型选择](14_training_and_experiment_tracking.md)；[T15 建立分类与事件评估、验证集校准和门限选择](15_validation_and_calibration.md)
