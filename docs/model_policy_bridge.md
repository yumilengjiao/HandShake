# T16/T17/T23：模型—策略—演示会话桥

> 后续已接入[原生影子owner会话](native_shadow_session.md)，复用本桥的启用/选手/重新预热接口，仍无真实动作权限。下文“未接原生”的描述与测试数量保留为会话桥交付时的历史范围。

## 本轮补的不是另一套手写分数回放

新增 `ModelPresentationSession`：输入只能是完整 `ObservationFrame`，经已有共享特征、32步窗口、校验包中的MLP/TCN真实前向，保留同次推理产生的 `SourceClassification`，再进入校准、owner、几何完成/释放策略与T18调度器。

默认 **shadow**：输出策略候选，不接受任何应用命令，内置页码不改变。唯一可执行模式是显式 `synthetic_application_local`，必须同时具备synthetic输入时钟/输入声明、synthetic模型/校准和 `allow_synthetic=True`。没有真实输入的执行开关，没有接入原生“启用手势”按钮，也没有外部应用控制。

软件验收用**人工设定权重**的MLP和TCN：输入关键点计算x速度，网络产生NOG/TLT/TRT诊断分数，分别经过完整会话完成 **1→2→3→2**。没有优化器、反向传播或神经网络训练，不是IPN HandS正式识别成绩。测试中的旧分数生成器仅用于生成几何，返回的手写预测被丢弃；桥不会接收外部logits。

## 修复了两个真正的集成阻塞

### 1. owner绑定后的重新预热

旧分数组件连续提供ready分数时看不出来：绑定owner增加epoch后，真正模型必须清空旧窗口并重新积累32步；旧策略却会在150ms内把这种预热当作预测失联。

新流程：

```text
页面确认 → 显式enable → 初次32步模型预热
→ 当前模型NOG＋独立几何中性/唯一手满足条件
→ 显式request_owner（500ms内仅消费下一条新结果）
→ 下一结果核验并绑定owner，epoch递增
→ 新epoch重新积累32步（无动作权限）
→ 第一条新预测后，重新计算中性保持/释放
→ ACTIVE → 模型＋几何完成 → 候选/调度 → WAIT_RELEASE
```

不修改旧样本的epoch，不重贴旧预测，也不把预热时长算进中性释放计时。绑定那条结果仍保留旧的捕获epoch，只用来说明授权请求的来源；它不会成为新代际的分类快照。

预热交接有固定截止时间：模型配置的 `max_window_span_ns + max_age_ns`，且最多5秒。只接受同owner、同stream/epoch/timebase、严格更新的帧、连续增长的窗口长度和有效整帧几何。完成的特征/窗口操作可以证明进度，但不能产生动作；相机心跳或UI心跳不能替代这种进度。缺手、多手、低质量、重复帧、断流、过期结果、非零旋转或超出总期限会停止，不能无限续租。

### 2. 推理与停止不共用阻塞锁

模型前向在控制器/调度闸门**之外**进行，每会话最多一个进行中的模型调用，不排无界队列。捕获上下文与取消代号后工作，返回时再次核对；停止、owner/源代际变化后丢弃旧结果，不把它重新包装成新代际预测。

- stop/close不等待模型调用，仍立即撤销调度权限并释放本应用后端。
- `model_busy`明确表示旧模型调用是否还在运行；不把停止应用命令冒充底层推理线程已退出。
- 旧模型调用未结束时，拒绝重新enable。重叠输入导致停止而非重排/并发调用模型。
- 新代际开始前采集的原始帧也不能进入新窗口；源切换必须显式reconnect，再重新确认/启用/选择owner。reconnect只切同一时钟域中的stream；新模型、新timebase或录制时间原点重置需要新会话，不能静默重标时间。
- 底层模型/相机资源仍由宿主进程管理，本桥不声称能取消任意卡死的本地SDK；后续原生进程集成仍需隔离与退出确认。

`OnlineClassifier.last_prediction`只保留同次成功update的原始类型/时间戳；每次update或reset先清除。现有只读v3/v4通知结构没有改变。

## API与文件回放入口

Python宿主使用 `ModelPresentationSession.load(...)` 校验模型与T15校准后创建会话。校准必须绑定模型SHA、数据/划分、归一化/配置、词表及模型选择时的验证库存。直接构造也复核这些绑定；不接受没有可行拒识阈值的公共策略。

主要接口：`enable(epoch=捕获的epoch)`、`request_owner(track_id, epoch=捕获的epoch)`、`process(frame)`、`stop()`、`reconnect(stream_id)`、`snapshot()`。owner请求的epoch在同一控制闸门内复检，不能把旧UI请求默认为当前授权。页面确认/心跳使用宿主当前快照序号、epoch和revision，继承T19的时效检查。snapshot另给出model_busy/model_warming、当前分类、eligible_owner、请求状态；不会自动选owner。

```bash
# 默认影子模式：真实或合成输入都不执行应用命令。
.venv/bin/python scripts/replay_model_session.py \
  --input "$SESSION_ARCHIVE" --sha256 "$TRUSTED_INPUT_SHA" \
  --package "$MODEL_PACKAGE" --model-sha256 "$TRUSTED_MODEL_SHA" \
  --calibration "$CALIBRATION_JSON" --calibration-sha256 "$TRUSTED_CALIBRATION_SHA" \
  --output /tmp/model-session-shadow-v1
```

输入/模型为合成时加 `--allow-synthetic`；仅在全部合成前置满足时，额外 `--execute-synthetic` 才能改变内置页码。真实/recording_relative/monotonic输入无法通过该执行选项。安装后同一入口为 `handshake-model-session`，不导入训练器或打开设备。

### 会话归档

目录仍是 `manifest.json`＋`events.jsonl`，流式预算、SHA/文件变更检查与 [只读观测回放](observation_replay.md) 共用。新的清单版本为 **model-session-replay-v1**，旧只读入口不会误读为无控制命令的旧格式。

每行字段由 `backend.interaction.model_replay.SessionItem` 定义：

- `at_ns`、`kind`、可选 `frame`（与原ReplayFrame相同）。
- kind支持start、stop、tick、frame、request_owner、surface_ack、heartbeat。
- start须给捕获的epoch；request_owner须给track_id与捕获的epoch；surface_ack/heartbeat须给捕获的epoch、revision；其他行禁止夹带这些控制字段。
- start是请求，可能因缺页面确认而拒绝；重复/被拒绝的请求保留结果，不当作已成功启用。
- stop独立于页面/模型确认；空时间段在最早预热/健康/页面/心跳截止点触发tick。

回放输出 `trace.jsonl` 与最后发布的 `report.json`，包含分类、策略候选、应用结果、代际、预热状态、控制拒绝与SHA。省略真实推理耗时来保持逻辑日志可比；这不测真实端到端延迟。surface_ack是记录的headless确认，不是Flutter绘制证据。影子模式不改变revision，使用来源于执行模式的页面确认轨迹时，必须保留/解释相应的拒绝，不能擅自改写为成功。

输入需授权，trace虽不保存图像，仍含派生分类与track信息，不应未经审查公开真实数据。

## 验证与仍待完成

新增30项专项测试通过；完整选择的非训练回归711项通过、4项训练用例deselect，另5个训练/梯度测试文件整文件排除。覆盖MLP/TCN真实前向、影子与合成执行、双向页码、全新32步预热、重新释放、独立停止/阻塞模型watchdog、旧结果与旧原始帧拒绝、重叠工作、源切换、校准库存、输入控制版本以及预热总截止不能续租。

```bash
.venv/bin/python -m pytest \
  tests/integration/test_model_session_bridge.py \
  tests/integration/test_model_session_replay.py
```

旧人工演示/旧手写分数合成宿主的默认行为保持不变，增加的保护性生命周期钩子用于共享实现，不复制第二套调度逻辑。原有70轮故障回归再次通过（约40.35秒，210次应用命令/重复拒绝，各23次失效注入，线程采样峰值4）；此前9000秒长测针对原提交，不能自动作为本轮新桥的长测证明。

离线sdist/wheel构建、锁文件/Ruff基础规则检查通过；wheel临时安装后，在仓库外用明确未训练的TCN包分别验证默认shadow只产生候选、显式合成执行改变页码，最终均停止。该安装借用当前环境已有依赖，不是新的依赖闭包/跨机认证。未改Flutter界面，仍复跑原有70项Flutter测试通过。

剩余主项：正式来源/方向语义审核与模型训练、真实验证集事件参数冻结、把本会话的owner/预热/停止接口接入统一原生相机演示页、隔离模型worker生命周期、受控真实手势执行与实机误触/延迟/权限验证。本轮没有开放真实手势控制，不把人工参数网络称为自主训练成果。
