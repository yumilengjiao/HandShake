# T14/T17/T18 模块验证与 T04 当前主机诊断

实施日期：2026-10-29（Asia/Shanghai）。在合并后的现有核心上实现，未创建界面或启用真实系统操作。使用说明见 [训练与交互](training_and_interaction.md)。本轮为模块/合成验收，不能替代真实数据、正式模型、相机→应用闭环及平台安全验收。

## 完成范围

| 任务 | 已交付 | 保留边界 |
| --- | --- | --- |
| T14 | CPU统一MLP/TCN训练器、固定端点指标、mask批次、梯度校验、早停、best/last、JSONL、完整epoch恢复、配置与实现/数据/统计哈希 | 正式数据等待T07～T10；多种子正式训练/校准/导出未完成 |
| T17 | 唤醒/owner、明确中性释放、运动完成、回程去重、握拳退出、失跟挂起、独立时钟watchdog；同源整帧几何适配 | 阈值draft；保守单可见手；真手与平台生命周期/全局停止待验证 |
| T18 | 白名单模式映射、有界FIFO、最终提交复检、双重离散去重、连续序号、独立停止/释放、迟到清理、dry-run与故障mock | 无真实应用后端；平台探针需非阻塞；实际注入/外部调用安全待T19/T23/T24 |
| T04诊断 | 官方模型SHA一致，隔离初始化＋合成黑图推理退出0 | 新主机结果不解释旧主机137；未打开相机、未验真手/GUI |

实际路径：`algorithm/training/`、`backend/interaction/`、`backend/actions/`、`backend/perception/diagnostics.py`；入口 `handshake train` / `python -m algorithm.training.cli`、`scripts/train.py`、`scripts/smoke_interaction.py`、`scripts/probe_perception.py`。训练默认配置 `configs/training/trainer.yaml` 指向唯一包资源。

## 自动回归

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  --junitxml=docs/verification/t14_t17_t18/pytest.xml
```

**404项通过，无跳过，14.33秒**。机器可读结果见 [JUnit](verification/t14_t17_t18/pytest.xml)。覆盖原358项合并基线及新模块回归；原“train未实现”断言已由实际训练CLI测试替代，evaluate/replay/live仍明确不可用。

新增验证包括：

- MLP和TCN各以独立进程进行完整3 epoch训练，以及1 epoch暂停→新进程恢复；逐张量权重、best权重、训练历史与采样器状态完全一致。适用范围仅本次同环境CPU与完整epoch边界。
- 缓存内容/版本错误、test冒充val、恢复配置/权重哈希错误拒绝；全掩码批次不更新，非有限梯度注入保留失败清单；早停/同分保留早期候选。
- 唤醒后先释放，一次挥动只发一次；反向回程/长期保持/单独no_action不能重新触发；分类仅在运动完成帧出现时无需再重复等待动作。
- 陈旧/乱序/重复/低质量、错误模型与整帧多手冲突不推进控制；时钟无预测仍挂起，独立后台watchdog无需模型/UI回调。
- 未完成校准的神经原始分数不冒充概率；规则一热分数需显式策略，已提供校准字段时才使用概率门限；无手不能成为中性证据。
- 外部日志sink抛错或阻塞不妨碍停止/释放；停止与提交探针同轮发生时停止胜出；队列满/重复事件/过期命令/旧连续序号拒绝；排队后再次核验过期。相对delta逐条FIFO保留，没有最新覆盖丢量。
- 后端调用前/后异常、调用阻塞/超时、释放失败保持禁用；在途调用超时不冒充取消，晚到down与关闭后的迟到副作用再次释放。stop接收、禁新提交和安全静止分层记录。

## 真实训练器健康检查（全部合成）

```bash
.venv/bin/python -m algorithm.training.health artifacts/t14-training-health-final
```

40个train及40个development端点、5类人工编码几何，单线程CPU，AdamW lr=0.01、batch=40、60 epochs、seed42；MLP/TCN均走正式实现的缓存校验、优化/选择/保存流程。另以同一配置打乱train标签，development保持固定。没有读取最终测试资料。汇总见 [health](verification/t14_t17_t18/training_health.json)，四次run清单见同目录 `*_run.json`；完整缓存、曲线和checkpoint保留本地忽略目录。

| 模型 | 人工类编码 development Macro-F1 | 打乱train标签后 development Macro-F1 |
| --- | --- | --- |
| MLP | 1.0000 | 0.1867 |
| TCN | 1.0000 | 0.2542 |

这说明极小可区分样例能拟合、标签扰动会破坏对应关系；合成ID不代表真实参与者，这些值不是手势识别、跨用户泛化或正式测试成绩。打乱标签诊断仍按验证选择最好epoch，不将它解释为严格的随机标签统计检验。输入无32帧预热完成的ready端点，仍使用固定class_valid端点检查训练流程，并公开覆盖率。

## 新状态机→调度演示

```bash
.venv/bin/python -m backend.interaction.smoke \
  --output docs/verification/t14_t17_t18/interaction_smoke.json
```

[完整时间轴](verification/t14_t17_t18/interaction_smoke.json)包含：wake_open → swipe_right → 阻止反向回程 → 中性释放 → swipe_left → exit_pose，共4个手势事件；只生成2个拟翻页结果，均 `not_attempted`。最后未启用、无owner且已释放。采用注入合成时钟和显式规则分数，不调用摄像头/神经模型/系统注入，也不是T23真实回放验收。

## T04：当前主机SDK通过，历史原因未归因

历史T03/T04/T06环境为CPython3.12.14/NumPy2.5.4，记录中SDK初始化137；当前为Linux x86_64/CPython3.10.12、NumPy2.2.6、MediaPipe1.2.0、OpenCV5.0.0.93。本轮开始时模型资产缺失，从已有采集指南指定的官方URL下载，仅留在忽略的 `artifacts/perception/hand_landmarker.task`，不纳入Git。

资产SHA256仍为 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`，7,819,105字节。一次隔离子进程初始化并处理128×128空图成功，返回0、检测手数0，耗时约1.78秒，Linux子进程最大RSS约178MiB；这不是实际帧率或全链延迟。可读cgroup本轮无OOM计数；Matplotlib缓存路径及SDK的warning原样留在报告，不当作失败或静默删去。

[脱敏探针报告](verification/t14_t17_t18/perception_probe.json)保留前后资源限制/计数、软件版本和stdout/stderr。未修改依赖锁、未循环重试SDK、未打开相机。当前成功无法证明历史主机故障已修复或确由OOM造成；应在原主机运行同一诊断比较。T07正式试采集还须T03/T04/T06人工预检、真手/镜像/遮挡与真实授权。

## 打包与依赖

离线隔离构建因临时缓存缺setuptools而失败；改为使用已安装且版本吻合的setuptools84.2.1、`--offline --no-build-isolation --python .venv/bin/python`，构建sdist/wheel成功，没有重新安装/改变依赖。wheel安装到临时独立target，不替换项目editable；从 `/tmp` 使用 `python -I` 导入新安装路径，阻止torch导入时交互演示仍成功；移除阻止后，安装版实际训练器完成2 epoch合成训练。包内包含新模块及trainer默认YAML。`uv pip check` 检查47包兼容。

[构建/独立安装摘要与包哈希](verification/t14_t17_t18/packaging.json)。wheel仍是Python核心，不是桌面/手机原生安装包。新增代码与证据关联见 [实现清单](verification/t14_t17_t18/implementation_manifest.json)。

## 下一步

T07前完成真手和采集工具实机预检，推进T08～T10正式数据；有数据后执行T14正式实验。T15/T16可接训练器与可信候选继续开发；T19可接T18做演示目标。产品界面先完成T36选型，再T20桌面与T37手机；正式闭环/安全验收归T23/T24。本轮不把这些后继项标为完成，A/B实名分工仍由用户填写。
