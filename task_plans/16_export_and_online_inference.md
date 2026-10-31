# T16 打包模型并实现带新鲜度校验的在线推理

| 项目 | 约定 |
| --- | --- |
| 阶段 | S4 部署推理 |
| 课程优先级 | P0 桌面 PyTorch 参考；移动运行时导出归 T37 |
| 建议主责 | A 主责，B 集成（A/B 实名以总方案为准） |
| 前置任务 | T14, T15 |
| 总方案依据 | 第 7.8、14.7～14.8、16.2、18 节 |
| 状态 | 阶段一完成：校验包/共享在线分类/v3只读UI已验证；正式模型、T15校准、热切换及全链仍待验 |

总约束见[任务导航](README.md)。本轮实际代码、命令、验证和信任边界见[模型包与在线分类](../docs/model_bundle_and_inference.md)。下列完整发布/动作/移动目标仍需后续验收；T15未完成时只允许明确未校准的只读预览。

## 目标与包格式

把训练结果装成可加载的版本化模型包，而不是随意复制一个权重文件。桌面首版部署 PyTorch；为 T37 提供可转换的推理图、算子/shape 说明和固定参考输入输出。移动本地运行必须验证一种便携运行时，不以 `.pt` 文件或 ONNX 导出成功冒充手机可运行；移动转换不阻塞桌面课程交付。

```text
artifacts/models/<model_version>/
  weights.npz             # 推理部署无pickle，可信本地checkpoint只在导出端读取
  architecture.json
  classes.json
  feature_schema.json
  normalization.json
  calibration.json
  event_thresholds.json   # 当前 unavailable / actions_enabled=false
  runtime_contract.json
  golden.npz
  training_manifest.json
  model_card.md
  checksums.json
```

当前`event_thresholds.json`与`calibration.json`明确为unavailable，不用示例参数冒充校准完成。模型包不可原位改写：T15/T28验证后生成新版本release candidate，再跑安全/回放与最终测试；外部报告不反过来修改已测包。加载要求来自可信导出回执的外部SHA256，不能自己重算陌生包哈希当作来源证明。

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

已交付`algorithm/bundle.py`、`algorithm/online.py`、`algorithm/deployment.py`，复用原有`algorithm/inference.py`；新增模型卡、golden、v3只读通知、Flutter展示和验证记录。没有另建重复`inference/`源码树。

- [x] 未校准预览包可加载并分类；错类表、schema/实现、哈希、文件或tensor拒绝。
- [x] 同样历史得到相同ready/末帧logits；乱序/过期/取消无可发布旧分数，动作层保持未接。
- [x] 本机CPU、batch=1/1线程合成前向p50/p95与整进程峰值RSS已测；不是全链延迟或公平效果实验。
- [ ] 清理和停止无需等待模型返回；交接 T17/T20/T23。

## 前置任务文件

[T14 实现可复现训练、采样与模型选择](14_training_and_experiment_tracking.md)；[T15 建立分类与事件评估、验证集校准和门限选择](15_validation_and_calibration.md)
