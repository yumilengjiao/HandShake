# T15 阶段一：离线评估、验证集校准与分层事件指标

## 已实现与未完成

本轮在 `feat/evaluation-calibration` 完成离线工具和合成验收：

- `algorithm/evaluation/`：固定分母分类、分数归档、有限温度/拒识搜索、事件匹配与背景时长统计。
- `scripts/evaluate.py`、`python -m algorithm.evaluation.cli`，重新安装包后也可用 `handshake-evaluate`。
- 模型包与缓存共同校验后按批只读前向；支持旧5类及来源15类、v1/v2密集缓存及v3紧凑缓存。
- 校准作为**独立、SHA绑定的 draft sidecar**，不会修改原模型包、权重、数据或标签。

**本机没有运行神经网络训练。** 手算 logits、独立合成事件和未训练零参数模型包仅验证软件。没有正式识别成绩、真实误触/延迟成绩或最终发布校准。官方标注SHA问题仍未解除。

**当前原生预览仍显示未校准。** 本轮sidecar只用于离线报告；没有改变T16包格式/在线控制权限，没有启用手势翻页。事件引擎参数联合搜索、线上概率展示/策略接入和最终冻结分别留待T23/T28/T16后续。

## 命令与完整性约定

运行帮助（不导入torch，不打开相机）：

```bash
.venv/bin/python scripts/evaluate.py --help
.venv/bin/python -m algorithm.evaluation.cli --help
```

`score` 需要安装项目的 `training` 可选依赖提供torch，但只做 `eval()` / `inference_mode()`；不读取训练checkpoint、不构造优化器、不反向传播。其余命令只需要核心依赖。

以下变量必须取自可信构建/导出回执，**不能把任意下载目录自算的哈希当作可信来源证明**：

- `MODEL_SHA`：模型导出回执的 `manifest_sha256`（`checksums.json` 文件SHA）。
- `VAL_INDEX_SHA` / `TEST_INDEX_SHA`：缓存构建回执的 `index_sha256`（去除自身字段后的规范JSON摘要，**不是 `sha256sum index.json`**）。
- `VAL_SCORES_SHA` / `TEST_SCORES_SHA`：`score` 返回的分数清单SHA。
- `CALIBRATION_SHA`：`calibrate` 返回的校准文件SHA。

输出不可覆盖。目录先写二进制，最后发布清单；中途失败的分数目录不能加载。已有输出应保留并另选新路径，不自动清理失败产物。

### 1. 已有模型上生成验证集分数（不训练）

```bash
.venv/bin/python scripts/evaluate.py score \
  --package "$MODEL_PACKAGE" --model-sha256 "$MODEL_SHA" \
  --cache "$VAL_CACHE" --cache-index-sha256 "$VAL_INDEX_SHA" --split val \
  --batch-size 256 --threads 1 --output artifacts/reports/validation/scores-v1
```

合成数据或合成模型必须额外显式 `--allow-synthetic`，标记沿全链保留。验证缓存必须与模型训练清单记录的验证缓存完全相同；数据版本、split、归一化、特征配置及词表均绑定。测试缓存可使用同一冻结划分的test部分，但不能换数据版本或归一化。

归档为 `manifest.json` + `scores.bin`，行序与固定缓存末端一致。每行含label、class_valid、ready、available、float64 logits；包含全部缓存末端，不只保留ready/高置信/正确样本。每批最大1024个窗口，分数归档最多2,000,000行，不全量展开重叠窗口。模型输出不合法时整次失败，不静默丢样本。

**范围限制**：缓存未表示的整段无手/技术丢帧时间不能凭空恢复。分类分母是“固定缓存表示的class_valid末端”，不是完整来源时间轴或在线可用率；两者不得混称。ready作为统计项，不作为真值过滤器。全程不可修改mmap缓存/分数文件。

### 2. 验证集有限校准与拒识选择

```bash
.venv/bin/python scripts/evaluate.py calibrate \
  --scores artifacts/reports/validation/scores-v1 --sha256 "$VAL_SCORES_SHA" \
  --max-risk 0.1 --min-coverage 0.5 --min-accepted 20 \
  --output artifacts/reports/validation/calibration-v1
```

可预登记 `--temperatures 0.5 1 2 4` 和 `--thresholds 0 0.5 0.8 0.95 1`，每组最多64项；温度必须包含1，范围[0.05,100]。默认有限搜索不是连续最优化。

- 先在全部**有分数且有标签**的验证末端上最小化NLL；数值同分优先接近温度1，再按温度数值排序。
- 再在所选温度下，满足最小接受量、最小覆盖率、最大经验错误率后，选择最大覆盖率；同分优先更少错误、更低阈值。
- 覆盖率分母始终是全部有标签末端，包括无输出/拒绝样本；无可行解报告 `infeasible` / `threshold: null`，不能以全部拒绝冒充成功。
- checkpoint与校准目前复用同一验证缓存；报告明确标出这种不确定性。经验risk并非总体风险保证，概率校准也不是OOD检测。
- `test`输入在创建输出前拒绝；不能用测试输入选择温度/门限。

产物：`calibration.json`、`report.json`、`receipt.json`、`event_thresholds.yaml`。后者只有 `draft` / `actions_enabled: false` / `parameters: null` / `not_searched`，**没有虚构已选出的on/off/保持/冷却/释放门限**。报告记录温度、阈值搜索次数；事件搜索次数为0。

### 3. 冻结配置后报告分类效果

```bash
.venv/bin/python scripts/evaluate.py classify \
  --scores artifacts/reports/validation/scores-v1 --sha256 "$VAL_SCORES_SHA" \
  --calibration artifacts/reports/validation/calibration-v1/calibration.json \
  --calibration-sha256 "$CALIBRATION_SHA" \
  --output artifacts/reports/validation/classification-v1.json
```

不传校准时仅输出原始softmax基线。传校准必须同时给SHA；报告包含基线、温度缩放、可行时的拒识结果。`classify`不提供临时调温/调门限参数。

之后在训练机器完成模型与事件策略冻结，再以 `score --split test` 产生测试归档，使用相同校准文件报告；不能根据结果回头挑选配置。本工具的draft报告**不等于最终发布/完整事件候选已冻结**。

分类指标：固定类序的 C×(C+1) 计数/真实类归一化混淆矩阵（最后列为拒绝/无输出）、每类P/R/F1/support、固定全类Macro-F1/Balanced Accuracy、缺席类、覆盖率、selective risk、NLL/Brier/ECE及可靠性分箱。无样本时概率/风险指标为null；零support类保留且F1/recall为0。NLL/Brier/ECE只在有分数标签样本上计算，报告明确作用域。

## 事件输入与评估

输入契约见 `algorithm/evaluation/event_contracts.py`，小型完整示例见 [`events.synthetic.json`](../tests/fixtures/evaluation/events.synthetic.json)。**这是人为设定的20分钟轨迹，不是20分钟实机录制证据。**

```bash
# 仅对仓库内已审查的合成示例自算输入SHA；正式日志使用独立可信回执。
EVENT_SHA=$(sha256sum tests/fixtures/evaluation/events.synthetic.json | cut -d ' ' -f 1)
.venv/bin/python scripts/evaluate.py events \
  --input tests/fixtures/evaluation/events.synthetic.json --sha256 "$EVENT_SHA" \
  --output /tmp/handshake-synthetic-event-report.json
```

输入包括独立人工/任务协议目标事件、观测事件、录制时钟域，以及有效/非操作/协议启用/ACTIVE/不可用区间。所有时间为整数纳秒；区间为半开 `[start,end)`，事件容忍边界为闭区间。当前没有自动把调度日志转成目标真值的适配器，更不能从模型预测生成真值。

报告按command/control分开，再列：

| 层 | 时间与证据 |
| --- | --- |
| `submitted` | 拟提交时间；dry-run仅进入这一层 |
| `accepted` | 真实调度接受时间；不等于已经执行 |
| `external_injected` | 仅external且确认注入；failed/uncertain不计成功 |
| `application_local` | 本应用内响应确认；不是外部注入 |
| `external_application` | 有外部确认注入证据的应用响应 |

同recording/timebase/action/mode/target_context/kind内一对一匹配，**先最大化匹配数量，再最小化总绝对时间差**。稳定ID排序决定同分结果，输入顺序不影响配对。每组最多1000个真值/预测、25000条候选边，超预算拒绝而非近似匹配。匹配TP、未匹配FP/FN、重复触发、分层未观测/技术无效期事件均保留。未匹配ID带recording/timebase，不跨时钟合并。

非操作分母先取协议区间与技术有效时间交集，再排除该kind目标动作开始及容忍区间的并集。ACTIVE/拒识/不可用不会裁掉这个分母。报告另列协议启用、ACTIVE、不可用时长和命令召回；每层按该层时间统计错误事件，注入层因此单列实际错误注入。

零误触必须附有效时长，另输出平稳独立Poisson假设下的单侧95%上界 `-ln(0.05)/hours`。20分钟零次仅约9次/小时上界，不能证明≤1次/小时；短录制或非平稳过程还须更谨慎。

延迟只对该层成功匹配且具有相应证据的事件计算：source→预测、source→注入、参考确认→注入、动作开始→注入、注入→外部应用响应，以及source→应用内/外部响应。无注入的dry-run没有注入延迟；未完成样本通过FN/未观测数体现，不能只引用成功样本的延迟分位数。

## 本轮验证与复跑

专项测试：**72项通过**。手算概率/混淆矩阵/拒识，空集合、边界/乱序/重复/全错/全漏，70组随机小图与穷举最优配对对照，SHA/符号链接/文件变更/重复JSON键拒绝，完整CLI链路，以及真实MLP/TCN加载器+零参数纯前向的5/15类、旧缓存/compact缓存验证。

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_evaluation_scores.py tests/unit/test_evaluation_events.py \
  tests/integration/test_evaluation.py
```

广域非训练回归（**550项通过，4项明确排除**；另整文件排除训练及梯度测试，不宣称全量pytest通过）：

```bash
env -u PYTHONPATH -u HANDSHAKE_CAMERA_TEST PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest tests \
  --ignore=tests/unit/test_training.py \
  --ignore=tests/unit/test_mlp.py \
  --ignore=tests/unit/test_tcn_causality.py \
  --ignore=tests/unit/test_model_losses.py \
  --ignore=tests/integration/test_model_bundle.py \
  -k 'not compact_training_resume_and_model_export and not public_import_to_deep_training_and_resume'
```

紧凑合成样例的近常量统计回退警告符合预期，不是实测模型成绩。`uv build --offline`的sdist/wheel构建通过；wheel安装到临时目录后，在仓库外通过`handshake-evaluate`入口生成合成事件报告，且未导入torch。`uv lock --check --offline`及新增代码的Ruff基础/导入规则检查通过。

没有运行未筛选的pytest、神经训练、相机、Flutter构建或外部控制。

后续仍须：正式来源审核与训练、独立验证连续流及人工目标日志、T23联调、T28完整事件参数搜索/冻结和实机安全回归、正式测试/用户级不确定性、线上校准/策略接入。不能把本轮工具交付计作这些验收已完成。
