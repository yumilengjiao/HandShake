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

默认来源已冻结为 `ipn-hands-frames-v1`：801,476帧图片关键点、200个完整序列，未下载视频。标注包与作者旧SHA的差异通过**显式项目审查快照**处理，未删除旧公布值或开放跳过校验。详情见[版本审查](../docs/ipn_hands_release.md)。

`prepare` 默认生成固定采集组计划，无需手写主体信息；116/32/52个train/val/test序列，同组不拆分。采集键不是已核实人物身份，结果不应冒充跨人评估。需要自定义真实主体计划时选择 `split_mode="reviewed_subjects"`。

数据和缓存均不会随Git下载；拉取后依次运行上述命令。合成测试样例不能当作正式数据，训练和评估结果另行验收。

## 管理约束

- 公开数据优先，不要求自行录制作为训练前置；使用时保留作者归属和相应许可。
- 保留来源15类、完整时间轴、无手/多手与监督掩码，不伪造操作意图或SDK完成时间。
- 原始数据不可变；先划分完整来源组（有可靠主体资料时按主体/会话），再切窗口；统计只由train拟合。
- 不把test拿来选模型或校准阈值，不边训练边覆盖缓存。
- 模型、实验指标和运行日志属于 `artifacts/`，不混入数据源码。
- `.gitignore`不等于访问控制或隐私审查；自定义目录与对外共享仍须自行核对内容。
