# T14/T17/T18 使用与模块边界

2026-10-29。现已实现 CPU 可复现训练器、交互状态机、通用映射与有界调度器，默认仍不注入系统操作。没有正式参与者数据、真实手势训练结果、桌面/手机产品界面或实机翻页。软件验证记录见 [本轮报告](verification_t14_t17_t18.md)。

当前基线已合入最新master的[公开数据优先路线](public_data_plan.md)与[MediaPipe兼容依赖/统一探针](perception_diagnostics.md)。本文软件仍是旧5类契约：公开15类需T07核验、统一协议迁移、T10划分后才能训练；不要求用户自行录制。公开版计划以UI/键盘启停，下面张掌/握拳流程仅为旧协议实现，不假称已支持来源类别。最新兼容验证见[master适配记录](verification_master_sync.md)。

## T14：从特征缓存训练

先按[特征说明](features_and_models.md)生成同一dataset/split/stats/config/stride的train与val缓存。T11 ObservationRecord仍是旧5类输入草案；公开来源适配器、15类契约升级与T09/T10审核尚未完成。可选自采的recording-v1也需显式转换；两条来源都必须加载对应版本的冻结清单，不能将15类直接伪标为5类。

```bash
.venv/bin/python -m backend.cli train \
  --train-cache data/features/v1/train --val-cache data/features/v1/val \
  --training-config configs/training/trainer.yaml \
  --output artifacts/runs/tcn-seed42
```

也可使用 `python -m algorithm.training.cli` 或 `python scripts/train.py`。训练依赖 `training` extra；查看采集、mock、交互模块不需要导入 PyTorch。训练预算默认 AdamW、lr=0.001、batch=64、80 epochs、patience=10、clip=1、CPU 单线程；模型结构仍采用 T12/T13 的固定配置及 dropout=0.2。切换 MLP 时复制训练 YAML，将 model_type 改成 mlp；同一数据/端点/预算用于公平对照。

训练入口拒绝 train/val 缓存哈希、统计、schema、类表、stride 不一致，以及可见 subject/recording 交叉。只传训练与验证缓存，不提供测试缓存参数；数据集整体指纹包含上游预登记划分，不意味着读取测试标签选模型。所有 class_valid 末端参与固定评价，包括 no_action 和尚未 ready 的可表示端点；ready 只另报覆盖率，不悄悄删去难例。T11 无法表示的初始无手帧仍在 inventory 中，不能宣称其分类已评估。

首版使用普通随机排列、无权重 CE、固定学习率，不同时叠加强重采样或复杂损失。每批记录监督数量、类别分布、loss、梯度范数和学习率；全掩码批次不更新优化器。非有限数/梯度使 run 失败并保留诊断。验证按固定顺序、eval/no_grad 遍历；五类固定分母，零 support 的 P/R/F1 为0并列出 absent_classes，Balanced Accuracy 也按固定五类平均。

每个新输出目录独占一个 run_id，生成 `config.yaml`、`epochs.jsonl`、`run_manifest.json`、`last.pt`、`best.pt`。best 选择使用验证 Macro-F1，超过 min_delta 才更新，同分保留更早 epoch；第一 epoch 必选。清单包括缓存/归一化/端点/类表/schema/实现哈希、架构和软件/CPU设置。失败运行不伪装成功。正式多种子实验仍需真实数据与训练预算登记。

```bash
# 在一个完整 epoch 后暂停；配置 max_epochs 不变。
.venv/bin/python -m backend.cli train \
  --train-cache data/features/v1/train --val-cache data/features/v1/val \
  --training-config configs/training/trainer.yaml \
  --output artifacts/runs/tcn-seed42 --stop-after-epoch 1

# 只恢复该目录自己的可信 last.pt；不覆盖其他实验。
.venv/bin/python -m backend.cli train \
  --train-cache data/features/v1/train --val-cache data/features/v1/val \
  --training-config configs/training/trainer.yaml \
  --output artifacts/runs/tcn-seed42 --resume artifacts/runs/tcn-seed42/last.pt
```

恢复校验配置、数据、代码、架构、软件和硬件类型；存储 optimizer、epoch/global_step、Python/NumPy/torch 与独立采样器 RNG。仅支持同环境、单进程、无 DataLoader worker 的完整 epoch 边界恢复；不承诺跨版本/设备逐步一致或任意 batch 中断恢复。恢复时以最后完整 checkpoint 重建训练曲线。checkpoint 使用 weights_only 加载；不接收不可信来源权重。此包仍不是 T16 发布包，也不包含校准或 frozen 事件参数。

可复跑训练健康诊断：

```bash
.venv/bin/python -m algorithm.training.health artifacts/training-health-new
```

它经过同一个 T11 缓存与 T14 训练器，比较可区分的人工类编码及打乱标签；两个“参与者”只是合成ID，验证是 development 健康检查，不是跨用户识别实验。原始缓存与权重留在受忽略 artifacts，公开报告只保存汇总。

## T17：交互状态机

推荐通过 `backend.interaction.runtime.InteractionRuntime` 使用。`ApplicationController` 持有单一上下文与串行闸门；平台/UI只调用 runtime 的 enable/stop/change_mode/change_model/health，接只读快照。不要把历史 `backend.mock.MockGestureEngine` 当成新实现，旧 T02 fixture 保留用于原契约回归。

状态为 DISABLED、IDLE、ARMING、ACTIVE、CANDIDATE、WAIT_RELEASE、COOLDOWN、SUSPENDED。初始禁用；显式启用后唯一稳定张掌才能获得 owner；先中性释放，再接普通挥动。握拳退出在等待释放和冷却时仍可成立。owner/模式/模型/挂起/停止变化升级 epoch、清候选；过期、乱序、跨源/epoch、错误模型均不补计保持时间。

`Evidence` 结合预测和同源原始几何。正式适配使用 `evidence_from_frame`，严格核对 stream/frame/source time/timebase 并检查整帧可见轨迹、关联歧义和当前质量。保守首版要求恰好一只可见手，多手/owner丢失/坏质量进入挂起，人工重新启用后再唤醒，不自动转移控制权。无手或 no_action 概率不构成释放：释放必须明确中性指形、区域、低速和持续时间。静态保持按源时间连续证据累计，超过100ms间隔中断。

挥动验证原始掌心位移、时长、纵向跨度和方向一致性，保留分类前的因果运动；分类首次在完成帧成立时不额外重复等待一个动作。首版最多保留256个点、最长1.5秒。一次提交后回程不能触发反方向，冷却结束也必须有中性释放。

默认 policy 要求校准概率；未完成 T15 时不会把任意神经 logits 当成概率。显式 `score_kind="rule_decision"` 只接受规则的一热决策分数，不声称已经校准。intent/phase 缺失保持 None；有 intent 分数时按 draft 门限检查，退出不被普通意图门限屏蔽。所有时间/门限仍为 draft，真实选择归 T07/T15/T28。

runtime 自带独立20ms时钟 watchdog。没有新预测、推理调用方阻塞或 UI 不刷新，仍检查 owner 超时与后端期限；停止路径不等待模型或绘制。它不等于已验证的 OS 全局急停键，平台生命周期和实机时延仍须 T20/T24 验收。合成时钟回放应禁用后台 watchdog，由宿主按同一时间轴调用 tick。

## T18：映射、调度与安全释放

`ActionMapper` 使用严格 actions 配置，将同一词元按模式映射成标准 ActionCommand；wake/exit 只产生控制事件。命令保留源事件、stream/epoch/target/TTL，不能排队后重启有效期。未知动作/参数由公共契约拒绝，没有脚本或 shell 执行入口。

| 类型 | 当前行为 |
| --- | --- |
| 翻页/媒体等离散命令 | event_id 与 command_id 双重去重；不自动重试 |
| 绝对位置/音量 | 有界FIFO、control_stream_id/update_seq 拒绝旧更新；尚无真实控制器 |
| 相对 scroll delta | 有界FIFO完整保留每个已接受 delta，队列满明确拒绝；不采用最新覆盖 |
| key/button-down、click | 调用前登记可能按下；异常/超时后仍释放，未确认释放不清账 |
| DryRunBackend | 记录拟执行，not_attempted，无注入/应用确认 |
| MockBackend | 仅内存副作用；可模拟调用前/后异常、阻塞、迟到副作用、释放失败 |
| 系统/外部应用后端 | 尚无，默认live仍拒绝；T19及后续实现 |

队列默认32，单调用 worker；提交闸门内复检停止、epoch、mode、能力、目标/焦点和时间。实际适配器的 capabilities/focus 探针必须非阻塞，当前只有内存实现；慢系统查询应由后续平台适配器异步更新快照。commit 是不可撤销的逻辑提交边界，停止后不再提交新调用；已经越过边界的外部调用可能晚完成，不能声称线程超时取消了系统调用。

独立 release worker 与普通调用隔离。停止立即设停止位、清队列、使旧 epoch 失效并请求释放；阻塞调用返回后再次请求释放。关闭时若尚有调用在途，保留迟到清理线程，主线程不无限 join。释放失败保持禁用，人工 stop 可重试释放；未确认安全静止禁止重新启用。safety 记录区分 stop_received、禁止新提交、release_finished，`safe_quiescent` 还要求无在途调用和可能按下项。重复 stop 幂等，不自动恢复。后台线程只是本机参考机制，不能保证强杀进程后释放外部设备。

诊断日志内存环最多2048条；可选外部日志sink经独立有界128项队列导出，抛错/阻塞不进入停止或释放路径，失败/队列满计数可见，关闭不无限等待；去重账本最多65536条，耗尽明确拒绝并需安全更换调度实例，不淘汰旧ID后允许重放。媒体/光标 schema/mock 可配置，但默认 allowlist/capability 只开放翻页；T21/T22的真实适配、速率积分及权限验收未交付。

```bash
.venv/bin/python -m backend.interaction.smoke --output artifacts/interaction-smoke-new.json
```

该演示执行真正的新状态机/映射/调度模块，合成轨迹依次唤醒、右挥、阻止左向回程、中性释放、左挥、握拳退出。只出现两个翻页拟执行结果，均 not_attempted；不是相机→训练模型→演示软件闭环。

## T04：统一隔离SDK诊断

```bash
# master固定的兼容依赖；完整感知基线使用Python3.12。
uv sync --locked --python 3.12 --extra dev --extra perception --extra training
.venv/bin/handshake-perception-probe \
  --model artifacts/perception/hand_landmarker.task \
  --model-sha256 fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1 \
  --timeout 30 --rss-limit-mib 768 --report artifacts/perception-probe-new.json
```

权威实现为 `backend/perception/probe.py`，同一有界子进程记录加载、三帧640×480黑图、关闭、期限/RSS/日志守护。旧 `python -m backend.perception.diagnostics --expected-sha256 ... --timeout-seconds ... --report ...` 与 `scripts/probe_perception.py` 保留参数和摘要兼容，但原生调用全部转发至统一实现，不再维护另一份SDK加载流程。

不访问相机，不自动下载或切换资产/依赖。SDK通过不代表真手/镜像/遮挡/交叉或GUI验收；SIGKILL本身也不能确定OOM。旧3.10/MediaPipe1.2.0的成功记录与master的3.12/0.10.21记录分别作为历史证据保留；实际当前环境与重新验证见[master适配报告](verification_master_sync.md)。
