# 公开数据扩容：紧凑缓存与训练机交接

> 后续已完成[固定图片帧关键点版本](ipn_hands_release.md)的真实全量缓存准备与预检；旧公布SHA差异保留为审查记录，默认使用预先审核的项目快照及固定采集组。下文“仍阻塞/大包未下载”属于原扩容阶段历史状态。

> **后续工作流增量**：日常命令已统一为 `uv run download/prepare/train/resume/evaluate/export`，见[快速开始](quickstart.md)。训练代码新增CPU/CUDA/MPS显式设备选择；CUDA/MPS真实训练尚待目标机器验证，下文“仅CPU”描述保留为原扩容阶段历史范围。当前本机仍不训练，上游标注SHA问题仍未解决。

**分支 `feat/public-data-scale`；基于已拉取的master `13839c9`。**

本轮优先解决数据通路的内存问题，不继续扩展界面。按用户要求，**当前机器暂停神经网络训练**；后续只执行数据校验、缓存构建和不含训练的测试。正式数据仍受作者标注包SHA不一致阻塞，没有因此伪造训练或准确率。

## 1. 来源复核结果：仍然阻塞

2026-10-31重新读取作者README并查询Git HEAD，仍为`093729579be671df0cc9dd8cbc58b9a342663800`。从Google Drive公开direct端点得到82,274字节ZIP，206个成员CRC通过，但实际SHA仍是：

```text
520dfbf3d52ecab6535d8841bba55e94d6b512596ed1cff4505d2c1345bcbaec
```

作者公布的预期值仍是`4686a2eb...f30ec71`。`audit-ipn`仍正确返回blocked。没有修改预期SHA、绕过校验或把CRC成功当来源确认；本轮也没有下载375MB关键点大包。GitHub提交API返回403、旧uc端点一次TLS失败，不能把这些失败当作上游确认；成功的公开direct下载结果已记录在[data/sources.json](../data/sources.json)。

后续需取得上游说明/匹配的文件版本，或明确切换经过审核的数据来源；不要求用户因此自己录训练集。

## 2. 已实现的新数据路径

```text
既有严格导入器生成的完整import目录
  → 逐行读取gzip，同时校验实际消费的压缩字节SHA
  → SQLite磁盘暂存、分组/来源/时间/标签账本审查
  → 仅训练侧样本计算统计
  → 每个处理步存一份94维float32特征
  → 只在读取batch时恢复32步窗口
```

新增代码：

- `backend/importing/verified_lines.py`：有界JSONL解码、CRC/计数与所读字节SHA校验。
- `backend/importing/spool.py`：SQLite临时审查/排序；缓存页预算8MiB，不收集全量Python记录列表或逐帧ID集合。
- `backend/importing/stream_features.py`：共享特征/统计/窗口逻辑生成v3缓存。
- `algorithm/data/compact.py`：只读内存映射、严格范围/身份检查、按批窗口构造及转移预检。
- T14读取接口已适配v3，v1/v2小缓存继续兼容；不复制第二套训练器。

### 不能丢掉的审查

- 所有来源视频、subject/session分组、source test隔离沿用已审查的导入产物；`--review-reference`是实际审核引用，不是自动审核或“魔法同意字符串”。
- 检查原始成员/lineage/recording跨集合泄漏、重复帧、固定640×480和HandS的1-based/30fps时间轴。
- 每帧标签账本必须连续完整；记录数、0～2手、监督掩码和标签与账本相符。无手/多手不猜执行者，不把来源类硬映射成旧5类。
- 外部SDK完成时间保持未知，不允许将其改成虚构的本机0ms推理。
- 源ZIP、import分片只读。审查失败不发布缓存，临时SQLite清理；构建中途失败保留不可加载的未完成输出，不覆盖原缓存。

## 3. cache v3布局与兼容性

```text
index.json                # complete与所有文件大小/SHA，最后发布
features.f32              # [Nsteps,94]，little-endian float32
steps.bin                 # 每步frame_id/time/epoch，24字节
endpoints.bin             # 固定48字节的行指针、长度、标签、掩码、ready及来源身份
groups.json               # 有界轨迹分组元数据与连续行范围
inventory.jsonl.gz        # 流式端点清单，保留未表示/未选中的条目
stats.json
classes.json
split_manifest.json
```

- 新格式为`t11-cache-v3` / `t11-stream-split-v1`，dataset hash采用显式`stream-records-v1`域，绑定输入manifest和实际读取记录流。**不冒充旧版排序大列表的records_digest**；不能混用两版统计或train/val缓存。
- 逻辑窗口仍是32×94，左侧不足步数补零；从当前行向过去取值，不跨group/epoch、不读取未来。stride不丢历史特征，只控制端点选择。
- 加载先核对全部文件哈希、字段、大小、范围、数值、标签、时序和覆盖率；坏指针、缺文件、NaN、epoch串接拒绝。
- 单次最多构造1024个窗口；拒绝`np.asarray(all_windows)`这类隐式全量展开。
- 缓存必须保持不可变。批访问检查size/inode/mtime，普通中途修改会失败；这不是抵御本机恶意并发改写/截断mmap文件的沙箱。**不要向正在使用的缓存目录覆盖复制文件**。
- mmap不等于零内存：OS驻留页面、模型/梯度和batch仍占内存。训练随机排列保留O(N) int64索引，标签打乱诊断还需O(N)标签副本；本轮没有声称整个训练过程是严格常量RSS。

T14适配另将验证集指标改为按batch累加混淆矩阵，不保存全量预测列表；v3每epoch仅保留前256条batch明细，完整总批数/损失/梯度/验证指标仍记录，checkpoint不积累无界逐batch日志。CPU参考训练器仍是当前实现，**GPU训练尚未适配**。

## 4. 当前机器可执行的命令（不训练）

以下正式数据路径只有在上游校验、导入和分组审查完成后才能使用：

```bash
# 原始输入仍经过同一个严格校验入口；当前官方小包会blocked。
.venv/bin/python -m backend.importing.commands audit-ipn \
  --catalog data/sources.json --annotations data/external/IPN_HandS/IPN_HandS_annotations.zip

# 独立v3入口，不是把旧--max-records盲目调大。
.venv/bin/python -m backend.importing.commands build-stream-cache \
  --imported data/imported/ipn-v1 --output data/features/ipn-v3/train \
  --split train --dataset-version ipn-reviewed-v1 --review-reference <真实审核引用> \
  --report artifacts/ipn-v3-train-cache.json

.venv/bin/python -m backend.importing.commands build-stream-cache \
  --imported data/imported/ipn-v1 --output data/features/ipn-v3/val \
  --split val --dataset-version ipn-reviewed-v1 --review-reference <同一审核引用> \
  --stats data/features/ipn-v3/train/stats.json \
  --report artifacts/ipn-v3-val-cache.json

# 从构建回执取index_sha256；不要重算陌生缓存的哈希就当作可信来源。
.venv/bin/python -m backend.importing.commands inspect-cache \
  --cache data/features/ipn-v3/train --index-sha256 <可信构建回执中的index_sha256>
```

`inspect-cache`只验证缓存完整性，不重新证明原始数据授权或审核正确性，也不启动训练。实际正式test数据必须单独保留，不拿它选择模型/门限。

默认预算为最多2,000,000条导入记录和50,000个轨迹组，硬上限分别10,000,000和100,000。临时SQLite放在输出目录的父目录；请使用有足够空间的磁盘，不要让全量暂存落到容量很小的`/tmp` tmpfs。

## 5. 本轮验证与训练暂停

可复现的显式容量探针：

```bash
# 默认只生成合成数据并构建缓存，不调用神经网络训练。
.venv/bin/python scripts/probe_stream_cache.py \
  --output artifacts/verification/new-cache-only-probe --frames-per-video 10001
```

实际审查了30,003条**程序生成**的观察记录，构建train/val各10,001步缓存；缓存子进程用时约27秒，峰值RSS约121MiB。每侧特征区3,760,376字节（约3.59MiB）；若全部展开为float32窗口，单侧逻辑张量为120,332,032字节（约115MiB）。这是逻辑特征体积对比，**不是与压缩NPZ磁盘文件的32倍压缩率比较**；还需算上步/端点/清单文件。

这证明超过原1万条门槛的合成容量路径可运行，**不代表官方约80万帧已经全量验证，也不是识别准确率**。静态合成手型导致部分训练侧统计使用明确的std=1回退，相关字段仍保留，不掩盖为真实多样性数据。

- 用户暂停训练前，初版小样例的MLP/TCN训练、跨进程恢复和T16导出测试通过。
- 用户要求暂停后，不再执行训练测试或训练压测；最新验证只覆盖不含训练的筛选测试、缓存探针、预检与静态检查。
- 容量探针训练阶段现在需显式`--with-training`，默认和本轮运行均未开启；最新完整训练回归留给目标训练机。
- 精确测试数量、资源证据和已验/未验范围见[机器记录](verification/public_data_scale_summary.json)。

本机回归选择：

```bash
.venv/bin/python -m pytest tests/integration/test_stream_cache.py \
  tests/integration/test_feature_dataset.py tests/integration/test_public_data.py \
  tests/unit/test_features_geometry.py tests/unit/test_features_window.py \
  tests/unit/test_feature_normalization.py \
  tests/unit/test_training.py::test_metrics_fixed_classes_and_hand_computed_counts \
  -k 'not compact_training and not deep_training'
```

**不要直接在当前机器运行未筛选的完整pytest**，现有测试中包含合成神经网络训练用例。

## 6. 换到更好的机器后

1. 获取同一代码提交（合并后从master获取），安装锁定依赖；确认磁盘/RAM、Python/PyTorch版本。当前`TrainConfig.device`仅支持CPU，有GPU不表示已经会使用GPU。
2. 单独传输经过验证的正式import/缓存及可信构建回执；Git只放代码和来源清单，不传原始大包或权重。本轮合成探针目录不能当正式训练集。
3. 复制到新目录，使用`inspect-cache`核对train/val；目录不再被写入后才运行训练。
4. 在训练机执行完整pytest及v3恢复/导出检查，再使用现有`handshake train --train-cache ... --val-cache ... --output <新run目录>`。batch_size不超过1024，建议先用默认64。
5. 不把本机临时/合成checkpoint冒充正式模型；跨软硬件的旧checkpoint通常会被T14严格恢复身份检查拒绝，应新建run，不手改身份哈希绕过。
6. GPU适配、正式数据训练、T15校准/实验和实际翻页另立任务；当前不因此启用动作控制。
