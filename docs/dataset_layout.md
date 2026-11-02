# 统一数据格式、目录与迁移

本次整理**不重写已有算法格式**。日常命令统一工作区布局和回执关联；底层继续使用已测试的公开数据适配器与紧凑缓存，避免训练端/运行端维护两套特征。

## 分层

| 层 | 默认目录 | 约束 |
| --- | --- | --- |
| 原始来源 | `data/raw/ipn-hands/` | 官方ZIP只读；来源清单、许可和SHA绑定；不原地修补标签 |
| 分组计划 | `data/metadata/` | 默认 `capture-group-plan-v1` 固定采集组；有真实主体资料时可选 `subject-plan-v1`；本地文件不入Git |
| 标准化导入 | `data/processed/ipn-hands/` | `ipn-hands-import-v1`；逐视频gzip分片、源帧标签账本、全量清单 |
| 训练缓存 | `data/cache/ipn-hands/{train,val,test}/` | `t11-cache-v3`；每步特征只存一次，按batch恢复窗口 |
| 工作流回执 | `data/cache/ipn-hands/dataset.json` | 绑定来源、主体计划、审核依据、stride及三个缓存的可信构建SHA |
| 实验结果 | `artifacts/runs/<run>/` | 配置、来源、环境、权重、指标、推理包与评估分开保存；不覆盖旧run |

所有持久化清单使用POSIX相对路径与UTF-8，不记录当前开发者绝对主目录。`.gitattributes`固定源码LF，以免不同Git换行配置改变字节级模型契约。用户可在配置中显式指定外部数据盘，但这样的本机路径应放在被忽略的 `handshake.local.toml`。

## 原始标签和关键点语义

- IPN HandS **15类来源词表**（14手势＋Non-gesture），版本 `ipn-hands-v1`；不重映射成历史5类协议。
- HandS原视频640×480，`fps=30`重抽帧，源帧从1开始，标签区间含末帧；不与原IPN Hand的去重/缩放时间轴混用。
- 每帧0～2只手，每手MediaPipe顺序21个xyz点；无手保留为空，多手不能自动选操作人。
- 外部SDK处理完成时间和实际曝光时间未知，不能填成0ms或推断成实时性能。
- 默认先按完整来源采集组划分再生成窗口，组内四个序列不跨集合；有可靠身份映射时可选主体/会话模式。train拟合统计，val/test只复用。最终test不用于调参。
- 默认发布版使用 `capture-group:` 命名空间，旧记录中的 `subject_id` 只作为分组键；`grouping` 明确记录未独立核实人物身份，不能据此声称跨人评估。

## 导入层

`manifest.json`最后标记完成；中途失败留明确不完整状态。每个视频包括：

- `*.frames.jsonl.gz`：完整源帧时间轴、来源类别和手数，保留无监督/多手情况。
- `*.records.jsonl.gz`：规范化 `ObservationRecord`，含来源身份、轨迹、有效性和监督掩码。
- 清单保存压缩字节SHA、计数、主体计划SHA和原始ZIP SHA；缓存构建重新验证实际读取内容。

JSONL不是给用户逐帧手工编写的训练格式。`uv run prepare`调用同一导入器自动生成，不要求用Excel转换或预先展开巨大数组。

## 紧凑缓存层

```text
index.json              # 最后发布，包含所有文件大小/SHA和覆盖率
features.f32            # N×94，little-endian float32，每处理步只存一次
steps.bin               # 源帧号/时间/epoch
endpoints.bin           # 窗口末端、长度、监督与就绪掩码
inventory.jsonl.gz      # 包含未表示/未选中的端点清单
groups.json             # 轨迹组与行范围
stats.json              # 只在训练集合拟合的归一化统计
classes.json            # 注册的15类词表
split_manifest.json     # 固定分组/来源/审核身份
```

窗口为32×94，因果读取，不跨轨迹/会话/epoch，不读取未来。缺失历史与质量问题通过掩码和ready状态表示，不重复帧假造完整动态序列。mmap按batch构造窗口，不把所有窗口一次展开。

默认单次最多处理2,000,000条记录、50,000个轨迹组；这些是软件预算而非正式全量容量验收。更大数据可使用高级入口调整，但不能用调大上限替代资源验证。

## dataset.json 是什么，不是什么

它是日常命令的薄关联回执，**不是新的关键点/模型格式**：

- `format_version = workflow-dataset-v1`
- `preparation`：原始ZIP SHA、分组计划SHA、项目审查引用（或自定义真实审核依据）、版本、stride及synthetic声明。
- 默认发布版另带 `source_review` 和 `grouping`，保留项目审查与采集组评估范围；不能删掉分组范围后仍冒充相同数据回执。
- `splits`：train/val/test的原始 `index_sha256`。
- `receipt_sha256`：除自身外的规范化JSON摘要。

默认train只检查train/val内容；prepare验完整三侧；evaluate显式选择test时才读取test样本进行评分。回执缺失、被修改或与缓存角色/synthetic不一致会失败。回执不能证明授权、不能替代真实主体审查，也不是抵抗恶意重签内容的数字签名。

## 已有项目如何迁移

不需要删除旧数据、训练结果或高级配置：

1. 原来 `data/external/IPN_HandS/` 中的ZIP可以保持原位，在 `[data] raw` 指向该目录；或复制到默认raw目录后由download复验。
2. 旧导入目录可由 `[data] imported` 指向。清单完整且来源/主体SHA一致才能复用。
3. 日常缓存默认使用新的 `data/cache/ipn-hands/`。旧v1/v2缓存仍可经原高级训练命令使用，不自动冒充v3。
4. 新工作流不盲目认领没有prepare回执的已有缓存。默认[固定发布版](ipn_hands_release.md)已包含审查和采集组规则，可直接prepare；自定义来源仍需审查。既有capture-group导入只允许v3缓存，旧小缓存入口会拒绝，以免丢失评估范围。
5. 旧高级命令 `handshake-*` 与 `scripts/`兼容入口保留。常规操作用短命令；回放、采集授权、真实SDK探针等高级用途仍需显式参数。
6. 旧checkpoint不能通过修改identity字段强行恢复。本次训练实现有变更，旧run请保留原代码环境；新机器/新代码开启新run，交付部署使用导出的推理包。

## 禁止混淆

- 固定SHA通过 ≠ 数据来源审查完成。
- 软件合成测试通过 ≠ 正式训练已完成。
- 统计构建中的合成std回退警告 ≠ 真数据统计质量正常。
- 模型分类/校准通过 ≠ 授予应用控制权限。
- 代码支持某设备分支 ≠ 该硬件/OS组合已实测认证。
