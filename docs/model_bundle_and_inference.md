# T16阶段一：可校验模型包与只读在线分类

**分支：`feat/verified-model-inference`，基于主分支 `3f4c59f`。**

本轮完成模型部署的**软件闭环**：T14可信本地训练结果 → 推理专用包 → 校验加载 → T11共享特征/窗口 → 原始logits → 原生UI只读展示。正式公开数据训练、T15校准和应用控制仍未完成；不以合成样例成绩代替识别效果。

## 交付内容

- `algorithm/bundle.py`：导出/校验CPU推理包；支持旧5类MLP和公开15类等已注册词表的MLP/TCN架构，不接收任意网络代码。
- `algorithm/online.py`：复用`FeatureExtractor`、`FeatureNormalizer`、`FeatureWindowBuilder`和`TorchPredictor`。输入不可用、乱序、失跟、过期、停止时不保留上一条logit。
- `algorithm/deployment.py` / `handshake-model`：显式export、inspect、benchmark入口；不下载数据、不开相机、不控制应用。
- `shared/inference_protocol.py`及`model_preview_protocol.py`：只读分类通知和可选HSP3协议；原有v1合成、v2相机仍保留。
- Flutter：显示来源类、**raw logit**、历史窗口长度/拒绝原因、合成训练警示。没有百分比“置信度”，也不将来源类别显示成操作授权。

## 模型包

```text
artifacts/models/<new-version>/
  weights.npz              # 只有有限float32数组；部署时绝不调用torch.load
  architecture.json
  classes.json
  feature_schema.json
  normalization.json       # 必须来自同一训练数据/划分/config/词表
  runtime_contract.json    # 共享特征、窗口和模型实现的字节哈希
  calibration.json         # 明确 unavailable，不假装温度=1已经校准
  event_thresholds.json   # 明确 unavailable / actions_enabled=false
  training_manifest.json
  model_card.md
  golden.npz              # 固定合成窗口及参考logits，不是准确率测试集
  checksums.json           # 成员大小/哈希，最后写入；它的SHA由导出回执给出
```

与初始T16规划的差异：分发权重采用有界NPZ，不采用可含pickle的`.pt`；事件策略采用严格JSON并标记不可用，暂不接受任意YAML阈值。未来有T15/T28证据后另行升级包版本，不在原包上补写“frozen”。

### 信任与完整性

1. 导出只接受本项目训练器的**可信本地**`best.pt`，核对run manifest、best SHA、结构、词表、最佳epoch和统计。要求训练已完成、未用最终test选模型、不是打乱标签诊断，且导出时实现与训练身份一致。
2. 导出端读取checkpoint使用`torch.load(weights_only=True)`；这不把任意下载pickle变成可信输入。部署端只读NPZ数组，不反序列化Python对象。
3. 加载必须提供**从可信导出回执取得的checksums.json SHA256**。不能下载一包陌生文件后自己计算SHA再声称来源可信。哈希证明字节一致，不证明数据授权、训练有效或模型准确。
4. 所有成员同一打开文件读入有界字节后再校验/解析，不校验完又重开权重。拒绝额外/缺失成员、符号链接、错误大小、重复JSON键、坏词表/字段顺序/实现哈希。
5. NPZ限4MiB，并核对展开预算、精确tensor名称、NPY头、dtype/shape、有限值；在按shape分配前检查，不允许object/pickle。元数据每文件≤1MiB。
6. 固定D=94、T=32、TCN感受野31和当前MLP/TCN配置；类数/顺序来自注册词表。加载后以`atol=1e-5, rtol=1e-4`复验golden logits。
7. 输出目录独占创建，禁止覆盖；中断写入没有有效回执就不能加载。文件系统权限仍需正常配置，不声称阻止本机恶意管理员。

`runtime_contract.json`是当前Python参考实现的严格兼容边界，源码字节变化可能需要重新导出；它不是移动端ABI。T37仍需转换器、算子及真机一致性证据。

## 使用方式

先安装当前项目及训练依赖，例如根目录的`uv sync --locked --extra dev --extra training --extra perception`。不安装训练extra时普通合成/相机入口仍可使用，模型加载则不可用。

以下路径是**已完成训练后**的使用示例，不表示仓库已有正式训练权重：

```bash
.venv/bin/python -m algorithm.deployment export \
  --run artifacts/runs/ipn-tcn-v1 \
  --normalization data/features/ipn-v1/train/stats.json \
  --output artifacts/models/ipn-tcn-preview-v1
```

合成训练必须额外显式加`--allow-synthetic`，且包/界面会持续保留此标记。导出输出包含`manifest_sha256`，请保存这份可信回执，不手改训练manifest绕过版本检查。

```bash
.venv/bin/python -m algorithm.deployment inspect \
  --package artifacts/models/ipn-tcn-preview-v1 --sha256 <导出回执中的SHA256>

.venv/bin/python -m algorithm.deployment benchmark \
  --package artifacts/models/ipn-tcn-preview-v1 --sha256 <同一SHA256> \
  --threads 1 --iterations 100

.venv/bin/python scripts/run_native.py \
  --flutter "$HOME/.local/share/handshake-tools/flutter/bin/flutter" \
  --perception-model artifacts/perception/hand_landmarker.task \
  --model-package artifacts/models/ipn-tcn-preview-v1 \
  --model-sha256 <同一SHA256>
```

安装后的`handshake-model`与`python -m algorithm.deployment`共用实现。启动仍不自动打开相机，需点击“打开相机预览”。不指定分类包时保持v2行为；指定包时选择v3。数学合成舞台不会伪装成网络推理，指定模型只用于感知输入链路。

模型校验失败时不会继续打开相机/感知SDK；已启动推理失败则明确显示不可用/拒绝或停止。切换模型目前需停止并重启核心，尚未提供UI文件选择/在线热切换。

## 在线行为与边界

- 当前策略只对**单只有效观测手**分类；两只手出现时清窗口并拒绝，不猜控制者。轨迹、stream、epoch变化重新预热，不拼接旧历史。
- 共享32步预热、缺失掩码、150ms默认新鲜度和窗口跨度等约束来自包内FeatureConfig，不另写一套“数够32就能执行”的逻辑。
- 启用模型时以感知输入节奏更新特征/网络，**只限速JPEG显示**。原来在处理循环里睡100ms会触碰100ms最大采样间隔，因此不能把UI帧率当模型采样率；设备不足时维持warming/rejected，不偷偷放宽门限。
- 推理前/后检查同一时基的新鲜度。停止在模型运行中到达时丢弃完成结果；发布时再次检查源帧年龄。图像、骨架、分类源身份不一致直接拒绝。
- context_epoch在创建worker时捕获。结果不补盖新epoch；停止/换代际清分类，UI回看只展示历史，不重发事件。
- 无意图头、阶段头、概率校准或动作映射。即使旧5类模型包含`wake_open`等词元，也只发布`ClassificationNotice`，不送入旧T17/T18。
- v3仅用于同机相机单调时钟；离线测试使用显式同域clock，不将其当成跨设备同步。T38没有因此完成。

### UI协议

v3使用HSP3前缀，继承v2的16KiB元数据、128KiB JPEG上限和私有管道。新增`classification`：绑定source frame/time、stream、epoch、model version，包含未校准原始logits、窗口长度、拒绝原因和合成训练标志。

v1/v2客户端拒绝新增版本；新客户端拒绝一条连接中途切换版本，拒绝错帧/旧代际、未知词表、非有限logits、伪校准、没有有效观测手的预测，以及非预测状态夹带旧分数。`control_available`仍只能为false。

## 验证记录

机器可读结果见[本轮摘要](verification/t16_model_inference_summary.json)。

- **566项Python测试通过，0跳过**；本轮新增61项。小型合成样例经过真实梯度更新、best选择、导出、加载与流式分类，不是仅实例化空网络。
- **51项Flutter单元/widget通过**；包括v3跨语言fixture、原始分数/合成警示显示、错epoch/帧/词表、清理停止状态等。
- 同一合成历史经离线WindowDataset和在线逐帧路径，ready及末帧logits一致；新track、多手、无手、推理中变旧/取消均无可发布旧分数。
- 受控过程相机/探测器替身验证了完整CameraPreviewWorker→真实MLP/TCN→同帧通知；**没有把这种注入称为真实摄像头识别成绩**。
- 原生Linux实跑：v2已有流程回归；v3流程和坏模型包明确拒绝。本轮这些原生测试未打开真实相机。
- Flutter Debug/Release构建、静态检查、wheel资源/模型加载验证另列摘要。源码目录外从wheel加载MLP/TCN并复验golden使用已安装CPU训练依赖，不冒充重新下载了整套依赖或干净系统部署。

### 本机CPU探针（非正式实验）

Python3.12.15、PyTorch2.15.0+cpu、x86_64/Linux、1线程、batch=1、10次预热后100次前向：

| 测试模型 | p50 | p95 | 整进程峰值RSS |
| --- | ---: | ---: | ---: |
| 5类MLP合成诊断包 | 约0.080ms | 约0.163ms | 约242MiB |
| 15类TCN合成诊断包 | 约0.888ms | 约1.346ms | 约252MiB |

输入为合成零张量，RSS包含Python/PyTorch/加载器，不是仅权重内存；两者类数不同，**不是公平效果对照，也不是相机到翻页的全链延迟**。正式比较需同一真实划分、词表、采样和设备条件。

## 尚未完成

1. 官方公开数据SHA问题、真实分组审核和全量缓存；不要求用户自录训练集。
2. 正式MLP/TCN训练、T15分类/事件评估、校准与拒识、真实手势准确率/误触率。
3. 模型UI热切换、T19应用目标与真实翻页、T23/T24全链安全/生命周期/故障实测。
4. 手机转换/本地运行、独立安装器和正式报告。

因此T16是**阶段一完成、整体待验证**，不是绕过T15宣称有可发布或可控制设备的模型。
