# 公开数据与15类深度训练管线：本轮实现

分支 `feat/public-data-training`，基于已合入T14/T17/T18的主分支继续，不重写训练器/状态机。**本轮已实现格式和软件链路，不代表官方数据已核验通过或真实识别效果已达标。**

## 实际交付

- `algorithm/data/source_archive.py`：同一个打开的文件完成SHA256核验与ZIP读取；限制成员数量/大小，拒绝路径穿越、符号链接、重复成员、重复JSON键和非有限值。不解压、不执行包内脚本、不自动下载。
- `algorithm/data/ipn_hands.py`：严格解析15类、官方视频列表、帧计数和连续区间；HandS的1-based帧/含末端区间转换为准确的`(frame_id-1)/30`相对时间，不使用原版去重帧索引。
- `backend/importing/`：核对显式审查的subject/session映射、官方test隔离，逐视频导入gzip观察分片和独立帧标签账本。失败保留incomplete，输出不覆盖。
- `shared/classes.py`及`defaults/classes_ipn_hands.json`：注册`ipn-hands-v1`，保留来源15类；旧5类ID不变，不把TLT/TRT硬塞进旧序号2/3。
- 统计、缓存、训练器、MLP/TCN跟随已验证词表；15类缓存用`t11-cache-v2`和独立`classes.json`，统计格式v2。错误/交叉版本拒绝，训练与验证必须共用同一统计及词表。
- `shared/classification.py`：公开来源模型输出`SourceClassification`，不是旧`ModelPrediction`，不冒充校准概率、意图或应用命令。旧T17明确拒绝这类结果，应用映射/按钮启停迁移仍待后续。
- `backend/perception/assembler.py`：在线与离线复用整帧质量/关联，不复制一套骨架算法。外部来源SDK完成时间通过`processing_time_origin=unavailable_external`和`processed_at_ns=null`表示，禁止伪造0ms推理延迟。

旧5类默认行为与默认记录指纹保留；新增公开数据字段/版本显式参与新数据哈希。旧checkpoint按原有实现/环境身份规则恢复，修改实现后的checkpoint不应强行跨版本恢复。

## 命令

```bash
uv sync --locked --extra dev --extra perception --extra training

# 只读核验：当前官方小包仍会因SHA不一致被拦截，这是正确失败，不是成功。
.venv/bin/handshake-data audit-ipn \
  --catalog data/sources.json \
  --annotations data/external/IPN_HandS/IPN_HandS_annotations.zip \
  --report artifacts/ipn-audit-new.json

# 仅在两个归档哈希通过且已审查主体分组后使用。
.venv/bin/handshake-data import-ipn \
  --catalog data/sources.json \
  --annotations data/external/IPN_HandS/IPN_HandS_annotations.zip \
  --landmarks data/external/IPN_HandS/IPN_HandS_landmarks_MPHands.zip \
  --subjects data/manifests/ipn-subjects-reviewed.json \
  --output data/imported/ipn-v1
```

`subject-plan-v1`结构如下（**只是字段示例，不是真实主体映射，不可直接照抄为reviewed**）：

```json
{
  "format_version": "subject-plan-v1",
  "status": "reviewed",
  "reference": "真实的分组审核记录引用",
  "subject_splits": {"P-train": "train", "P-val": "val", "P-test": "test"},
  "videos": {
    "实际来源视频ID": {"subject_id": "实际主体ID", "session_id": "实际会话ID"}
  }
}
```

必须覆盖全部来源视频；主体/会话不能从相似文件名猜测，reviewed不是自动审核或虚构授权。官方test不允许移到train/val；发现官方主体重叠时先明确重划方案/证据口径，不偷改分组通过校验。

导入结果：

```text
manifest.json                         # incomplete/imported，不自动视为可训练
recordings/<video-hash>.records.jsonl.gz  # 兼容T11观察记录，公开15类版本
recordings/<video-hash>.frames.jsonl.gz   # 每帧原始来源标签与监督覆盖，含无手/多手
```

原始档案只读。成员的原始关键点文件SHA逐视频记录，不能用整个归档同一个SHA代替所有视频的来源身份，造成虚假的跨集合重叠。

## 从审查后的小规模导入构建缓存

```bash
.venv/bin/handshake-data build-cache \
  --imported data/imported/ipn-v1 --split train --output data/features/ipn-v1/train \
  --dataset-version ipn-reviewed-v1 --review-reference <真实质量与划分审查引用>
.venv/bin/handshake-data build-cache \
  --imported data/imported/ipn-v1 --split val --output data/features/ipn-v1/val \
  --dataset-version ipn-reviewed-v1 --review-reference <同一审查引用> \
  --stats data/features/ipn-v1/train/stats.json

# 必须已有合法train/val缓存；不会读取test选模型。
.venv/bin/handshake train \
  --train-cache data/features/ipn-v1/train --val-cache data/features/ipn-v1/val \
  --training-config configs/training/trainer.yaml --output artifacts/runs/ipn-tcn-new
```

**当前缓存桥接是有界小规模实现**，默认最多10,000条观察记录；超过预算直接拒绝，不把完整80万帧语料塞进内存。全量分片/内存映射缓存与训练尚待扩展，不能通过盲目提高`--max-records`冒充已解决。导入本身按单个视频JSON和逐行输出处理；输入/输出大包仍不进入Git。

### 监督边界

- 来源标签是视频帧级别，不等于每个估计手都是执行者。首版只给“恰好一个估计手”的帧赋轨迹监督；无手/多手保留源标签账本、轨迹/缺失信息，但不猜目标手。需要单独报告覆盖率，不能称全部源帧均参与分类评价。
- 空手列表只表示外部预测未提供手，不能作为本机SDK已成功处理的实机证据；外部完成时刻未知。
- 质量差不删除原始时间和来源标签；初始无手没有真实轨迹，不伪造骨架/有效窗口。公共数据不提供本应用owner/启停/意图，不制造真值。
- 公开类别水平翻转语义尚未审核，增强入口明确禁用该选项；不凭名称猜测全部类别的镜像含义。

## 验证与尚未完成

本轮 **461项测试通过，0失败/错误/跳过**；[JUnit](verification/pytest_public_data_pipeline.xml) 与 [环境/源码哈希/检查摘要](verification/public_data_pipeline_summary.json) 已保存。修改的Python代码通过Ruff E9/F检查、编译与sdist/wheel构建；独立基础依赖环境从源码目录外导入公开词表与CLI通过。共享观察组装重构后，真实MediaPipe＋3帧合成图像预览/关闭也通过，不涉及真实摄像头或参与者资料。

`tests/integration/test_public_data.py`使用明确标记synthetic的3段、135帧、15类格式样例，检查归档、时间/标签、分组、失败保留、缓存、MLP/TCN真实梯度更新与完整epoch恢复。`tests/unit/test_public_vocabulary.py`覆盖词表、外部未知时间、来源分类与旧动作层隔离、旧记录指纹及失败报告。**这是软件诊断，不是公开数据上的准确率。**

本轮已装好锁定CPU训练依赖；正式数据仍受官方标注SHA不一致阻塞。本轮再次从官方地址得到相同82,274字节包，实际SHA仍为`520dfbf3d52ecab6535d8841bba55e94d6b512596ed1cff4505d2c1345bcbaec`，作者清单仍为`4686a2ebf70a6036166c261bc6992d050cfba08c3b18edcdd460b3fa9f30ec71`。不更改期望值假装通过。

仍需：来源版本确认、真实主体分组、全量流式缓存、公开类别应用策略、T15校准/事件评估、T16可信模型包与新模型实时闭环。**不要求用户自录训练集。**

深度学习实验与答辩线索见 [实验设计](deep_learning_experiment_plan.md)；原生界面候选与非卡片交互稿见 [界面设计](interaction_stage_design.md)。
