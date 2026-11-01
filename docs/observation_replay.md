# T23：关键点文件的只读模型回放

本分支提供 `algorithm/replay.py`、`scripts/replay_observations.py` 和安装后的 `handshake-observation-replay`。从带外SHA可信回执验证输入与T16模型包，复用 `OnlineClassifier`、共享特征/32步窗口，实际运行已有MLP/TCN的**前向**。不训练、不访问相机、不调用调度器，不以回放开启真实控制。

## 输入与运行

输入目录只有 `manifest.json` 和 `events.jsonl`。协议定义为 `algorithm.replay.ReplayManifest/ReplayItem/ReplayFrame`：

- 清单含 `format_version: observation-replay-v1`、synthetic、review_reference、stream_id、timebase、events、events_bytes、events_sha256。
- 每行 `{at_ns, kind, frame}`，kind为start/stop/reset/tick/frame，非frame行的frame为null。
- frame载荷为width/height/transform/status/observations；status与observations直接复用共享 `FrameStatus`/`HandObservation`。完整整帧包含所有可见手，不从单轨记录擅自推断“只有一只手”。无手保留 `processed_no_hand` 和空observations，未处理/错误状态也保留。
- at_ns是该条目记录的到达/控制时间，源时间保留在status中；不能把两者混用。时间非递减，同时间依文件顺序；源帧ID与源时间在整个归档中严格递增。每归档只接一个stream/timebase，多录制分别运行，禁止拼接轨迹。
- 录制数据需事先获得使用授权并独立审核；SHA只能验证字节，不能证明许可或标注正确。可使用真实关键点填充本协议，本轮验证没有使用真实参与者资料。

```bash
.venv/bin/python scripts/replay_observations.py inspect \
  --input "$REPLAY_DIRECTORY" --sha256 "$TRUSTED_REPLAY_MANIFEST_SHA"

.venv/bin/python scripts/replay_observations.py run \
  --input "$REPLAY_DIRECTORY" --sha256 "$TRUSTED_REPLAY_MANIFEST_SHA" \
  --package "$MODEL_PACKAGE" --model-sha256 "$TRUSTED_MODEL_SHA" \
  --output /tmp/handshake-observation-replay-v1
```

合成输入或合成训练模型需要额外 `--allow-synthetic`。可选择同时传 `--calibration` 和 `--calibration-sha256`，按已校验包的验证库存绑定T15校准；不在回放数据上拟合或重新选择门限。没有校准时只输出raw logits，不把原始softmax叫校准概率。`inspect`/帮助不导入torch；`run`需要已有torch推理依赖但不读checkpoint/pickle。

## 行为与报告

完整输入先流式校验一遍，之后才加载模型/创建输出，第二遍才执行回放。最多200,000条、512MiB、每行256KiB、每帧16手；不一次加载全部坐标。错哈希、符号链接、重复JSON键、非有限值、错整帧身份、乱序或非法生命周期在校验阶段拒绝；运行中检测不可变输入变更。

start/reset创建新epoch和新窗口；stop清空分类器，停止期间的帧明确报告skipped_stopped而非删除。EOF清窗口并报告stopped。长无帧间隔在**虚拟时效截止点**插入input_stalled，而不是等下一帧到达才发现超时；超时也清历史，不能用断流前的窗口直接继续。迟到帧不能续租。

产物为 `predictions.jsonl` 与最后发布的 `report.json`。包含每帧state/reason、源身份、epoch、logits及可选校准概率/拒识状态；总帧数保留停止/拒绝帧，另统计预测类别。模型、输入、校准、实现和输出SHA关联。

这是**逻辑分类回放**，不是事件引擎/真实调度/真实GUI闭环，也不是正式分类准确率评估。输出排除真实前向耗时以保证可复现；虚拟时间不能冒充实际端到端延迟。正式事件闭环、倍速/逐帧界面和真实来源格式转换器仍需后续接入。

## 验证

新增19项测试（旧5类/15类、MLP/TCN、SHA/生命周期/超时/缺手/重置等），使用明确未训练的零权重包和程序生成关键点。不会调用trainer或反向传播。

```bash
.venv/bin/python -m pytest tests/integration/test_observation_replay.py
```

广域非训练回归626项通过、4项明确排除；继续排除 `test_training.py`、`test_mlp.py`、`test_tcn_causality.py`、`test_model_losses.py`、`test_model_bundle.py` 及两个数据集训练/恢复参数化用例族；未筛选的pytest仍包含训练，不应在本机执行。
