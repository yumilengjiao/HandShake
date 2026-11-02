# 公开15类安全策略与内置演示合成回放

## 本轮完成了什么

后续增量：[模型会话桥](model_policy_bridge.md)已将真正的MLP/TCN前向结果接入此策略，补充owner绑定后的新epoch预热和独立停止；默认影子，仍无真实输入执行权限。下列记录保留本阶段手写分数回放的历史范围。

从已合入T15的 `master`（`bed1dbe`）建立 `feat/gesture-presentation-policy`：

- 新增 `PublicGestureEngine`，直接使用 `SourceClassification` 的15类词表；不伪装成旧5类 `ModelPrediction`。
- 有版本/SHA的draft策略与独立 `PublicActionEvent`，将经过校准、所有权、新鲜度、几何完成及释放检查的TLT/TRT变成候选命令。
- 合成专用宿主复用T18调度器和T19真实内置页码状态，验证 **1→2→3→2**。不是“只打印翻页日志”。
- 显式启用/绑定唯一轨迹，停止、超时、旧代际、界面确认、重复请求与迟到执行保护。
- 复用T15事件指标，输出源帧决策、公共类别事件、通用命令、应用结果、独立目标时间轴、分层报告和校验回执。
- `load_for_bundle()` 可将T15 sidecar绑定到已校验模型包的精确验证库存；没有修改现有模型包格式。

**仍然没有运行神经网络训练、真实相机、外部按键注入或真实手势翻页。** 当前原生窗口仍是人工翻页和只读模型预览，未增加“开启手势控制”按钮。本轮回放使用手写logits和程序生成的手部几何，不是正式模型成绩。

## 三层边界

| 层 | 能力与限制 |
| --- | --- |
| `PublicGestureEngine` | 纯策略：输出事件，无后端/相机/UI权限；校准不是用户意图 |
| `SyntheticPublicPresentation` | 必须显式 `allow_synthetic=True`、数据/模型校准均标合成；固定synthetic时钟域、固定内置稿和本应用后端，不接任意后端 |
| 现有原生桌面入口 | 不调用新合成宿主；原人工流程、v1演示IPC、v3只读分类协议保持不变 |

`data/sources.json`里TLT/TRT方向与动作语义的正式审查仍待完成。当前候选约定为**未镜像模型坐标**x减少的TLT→上一页、x增加的TRT→下一页；只有合成几何通过，不能据此宣布真实Throw left/right等价于旧swipe。预览镜像不改变命令方向；非零旋转尚未审查，策略拒绝。

OP2（Open twice）不是静态张掌唤醒，GRB（Grab）不是握拳退出。二者和其他非TLT/TRT类别都不能启停或翻页。无需增加假意图头；视觉相同的自然动作在显式激活后仍可能误触，必须用正式连续流测量。

## 运行合成演示

```bash
.venv/bin/python scripts/replay_public_policy.py \
  --allow-synthetic --output /tmp/handshake-public-replay-v1
```

同一实现也提供：

```bash
.venv/bin/python -m backend.interaction.public_replay --help
# 重新安装包后：
handshake-public-replay --allow-synthetic --output /tmp/handshake-public-replay-v2
```

不提供live开关，不读取摄像头/训练器；缺少显式合成选项立即拒绝。仅需核心依赖，不导入torch、OpenCV或MediaPipe。目录不可覆盖，末尾才写 `receipt.json`；中断产物不能当成完整验收结果。

产物包括：

- `scores/manifest.json`、`scores/scores.bin`：15条手写正确分类分数，非训练输出。
- `calibration.json`：在上述合成验证分数上有限选择，保留synthetic/draft，模型SHA标识的是合成生成器，不冒充真实权重包。
- `policy.json`：完整draft阈值、几何和时序配置。
- `event_input.json`：预先编写的目标与实际应用事件，兼容 `scripts/evaluate.py events`。
- `report.json`：页码轨迹、源帧原因、策略事件、通用命令、应用结果和T15分层统计。
- `receipt.json`：完整标记、文件SHA/大小；报告另记录策略/校准/相关实现代码的摘要。

预期 `page_sequence` 为 `[1,2,3,2]`；`submitted` 和 `application_local` 各3个TP，`external_injected` 为0。目标在运行前根据脚本编写，即使引擎漏检也保留目标；不根据检测输出生成“全对”的真值。

宿主确认是**合成headless surface ack，不是Flutter真实绘制**。复用的 `ActionResult.injection_status=confirmed` 表示内置调用确认，结合 `scope=application-local` 才能解释；T15导出中外部注入始终 `not_attempted`，不能把这个字段单独当成OS注入成功。

## 策略与安全规则

1. 初始禁用。新鲜页面确认后由宿主显式启用，再以一个**未消费的新帧**显式请求绑定operator；只看到唯一手或NOG不会自动授权。绑定要求同帧、有效质量、可靠NOG与独立几何中性姿态，随后增加epoch。
2. 绑定前的分数不能重贴新epoch继续使用。绑定后还须新鲜帧连续满足中性/低速保持；不是“点启用即执行当前手势”。
3. 每次检查stream、epoch、model SHA、词表、timebase、源帧与整帧对齐、推理完成时间、顺序、ready、质量和整帧所有权。多手/关联歧义/换轨/低质量/断流会停止或挂起并清owner；不能自动转交给另一只手。
4. 概率使用T15温度，阈值为 `max(校准拒识阈值, draft class_on)`；要求唯一胜出类。无可行拒识配置、错校准SHA或非15类校准拒绝初始化。概率下降不能作为释放证据。
5. TLT/TRT除类别外还要求同方向连续源帧、最短时长、最小净位移、样本数、纵向漂移与反向步长检查。类别变向、几何失败或候选超时取消候选；不能每帧翻一次。
6. 事件后进入WAIT_RELEASE。NOG还须满足几何中性（中区内一至三根伸指，其余明确折叠）和低速连续停留，再满足冷却后才能重新操作。保持类别、动作回程、一次NOG、低置信、缺失或乱序都不能重新武装。
7. 页面版本、context、心跳、owner和源新鲜度在入队/提交时再次检查。最多一项待处理；无surface确认时产生的事件**直接消耗并丢弃**，不能等界面恢复后补发。命令TTL不得超过原始源帧有效期。
8. 停止先设置撤销位和后端停止位，再进入共享提交闸门清队列/epoch；即使本应用后端的调用被故意阻塞，晚到恢复仍检查停止。已经实际完成的页码变化不能撤销，这与“停止之后不再提交”是不同保证。
9. watchdog独立于update运行，失去预测、页面确认或心跳会停用。注入时钟须独立推进；逻辑回放暂停时虚拟时间也暂停，不能据此声称验证了真实推理阻塞的墙钟延迟。时钟回退/异常立即撤销，拒绝新的执行；watchdog异常退出会永久关闭该宿主，不能在没有watchdog的情况下重新授权。

默认draft起点：class_on=0.8、150ms源时效、100ms最大证据间隔、150ms～1.5s动作时间、至少3样本、0.18画面宽度净位移、400ms中性保持、500ms冷却。**不是验证集选出的最佳事件参数**；配置SHA绑定报告，正式搜索与冻结留给T28。

旧5类和新15类共享不依赖类别名的整帧几何计算，旧引擎仍拒绝 `SourceClassification`；新引擎仍拒绝 `ModelPrediction`。T19后端的默认manual arm仍要求无owner；只有显式传入、与当前context完全一致的owner才能供新合成宿主使用。

## 验证

新增 **57项专项测试**，覆盖所有非映射类别、左右/镜像、保持/回程、几何释放、间断/重复/陈旧输入、错模型/代际、低置信/质量、多手/缺失/旋转、显式owner、无预测watchdog、surface与心跳租约、停止/迟到执行竞态、源TTL、校准绑定、完整页码与报告、移除运行ID后的回放一致性。

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_public_policy.py tests/integration/test_public_presentation.py
```

广域非训练回归：**607项通过，4项训练用例排除**；另外整文件排除训练/梯度用例，不宣称全量pytest通过。

```bash
env -u PYTHONPATH -u HANDSHAKE_CAMERA_TEST PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest tests \
  --ignore=tests/unit/test_training.py --ignore=tests/unit/test_mlp.py \
  --ignore=tests/unit/test_tcn_causality.py --ignore=tests/unit/test_model_losses.py \
  --ignore=tests/integration/test_model_bundle.py \
  -k 'not compact_training_resume_and_model_export and not public_import_to_deep_training_and_resume'
```

T15合成紧凑缓存的3条近常量统计回退warning符合预期。`uv build --offline`的sdist/wheel通过；wheel安装到临时目录后，在仓库外经安装入口复跑1→2→3→2并校验全部回执文件SHA，确认未导入torch/OpenCV/MediaPipe。Python `-O`回放、Ruff基础/导入规则、锁文件一致性检查通过。

此次未运行神经训练、真实相机、Flutter构建或外部操作。

## 后续仍待完成

- 正式来源/方向/语义审核、正式模型训练与验证集校准；官方标注SHA阻塞未改变。
- 真实关键点＋校验模型的逻辑/调度回放、完整输入切换、负样本与用户级实验。
- 原生UI的明确owner选择、校准状态及停止确认协议；当前合成宿主不能直接当成live入口。
- T23/T28联合事件参数搜索与冻结，之后才能授权真实相机驱动内置演示。
- 外部演示软件适配、全链实机延迟/误触、平台权限及长期安全验证。

返回 [任务T17](../task_plans/17_interaction_state_machine.md)、[任务T23](../task_plans/23_replay_and_integration.md)、[T15评估工具](evaluation_and_calibration.md)、[已有人工原生演示](local_presentation.md)。
