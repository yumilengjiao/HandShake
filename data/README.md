# data：数据来源、审核和本地缓存

常规操作：

```bash
uv run download
uv run prepare
uv run train --dry-run
```

来源、许可和可信SHA见 [SOURCES.md](SOURCES.md) 与 [sources.json](sources.json)；完整流程见[快速开始](../docs/quickstart.md)，格式见[数据约定](../docs/dataset_layout.md)。

## 默认目录

```text
data/
  README.md / SOURCES.md / sources.json    # Git跟踪
  raw/ipn-hands/                           # 官方原始ZIP，只读
  metadata/                               # 真实主体/会话计划和审核资料
  processed/ipn-hands/                     # 连续标签/关键点导入分片
  cache/ipn-hands/
    train/ val/ test/                     # t11-cache-v3
    dataset.json                          # 三侧缓存与准备依据的关联回执
```

除三个说明/来源文件外，以上真实数据与派生产物均不入Git。路径可在 `handshake.local.toml` 中配置；已有旧目录不会被自动搬移或删除。

## 当前状态

**正式数据还没准备好。** 2026-11-02下载器再次发现官方标注SHA不符并隔离；主关键点包未继续下载，真实主体/会话划分尚未提供。`prepare` 的测试使用程序生成样例，不是正式数据。

下载清单不是训练集；自动生成的draft主体模板不是审核结果；不能通过修改期望SHA、猜参与者或逐帧随机划分来绕过阻塞。旧主机历史记录中出现的数据文件也不代表当前checkout已经拥有这些数据。

## 管理约束

- 公开数据优先，不要求自行录制作为训练前置；使用时保留作者归属和相应许可。
- 保留来源15类、完整时间轴、无手/多手与监督掩码，不伪造操作意图或SDK完成时间。
- 原始数据不可变；先分主体/会话，再切窗口；统计只由train拟合。
- 不把test拿来选模型或校准阈值，不边训练边覆盖缓存。
- 模型、实验指标和运行日志属于 `artifacts/`，不混入数据源码。
- `.gitignore`不等于访问控制或隐私审查；自定义目录与对外共享仍须自行核对内容。
