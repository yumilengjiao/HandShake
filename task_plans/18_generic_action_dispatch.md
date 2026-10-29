# T18 实现通用命令映射、调度器与 mock 适配器

| 项目 | 约定 |
| --- | --- |
| 阶段 | S4 通用终端能力层 |
| 课程优先级 | P0 |
| 建议主责 | B（A/B 实名以总方案为准） |
| 前置任务 | T02, T05, T17 |
| 总方案依据 | 第 3.3、12、16.2～16.4 节；通用架构图 |
| 状态 | 模块完成；dry-run/mock映射、去重、有界队列、超时/迟到释放通过；真实后端待T19/T24 |

实施日期：2026-10-29。实际源码位于 `algorithm/training/`、`backend/interaction/`、`backend/actions/`（按本任务职责），使用与限制见 [说明](../docs/training_and_interaction.md)，证据见 [验证报告](../docs/verification_t14_t17_t18.md)。以下未验收项仍保留；规划路径以已实现职责目录为准。

**public-data-v1适配**：以[公开数据方案](../docs/public_data_plan.md)优先，不强制自采。现有模块只验证旧5类草案；15类数据须经T07来源/语义核验、统一契约迁移与T10划分后再接入。公开首版计划UI/键盘启停，不把Open twice/Grab伪映射为张掌/握拳。最新整体验证见[master适配记录](../docs/verification_master_sync.md)。

总约束和路径解释见 [任务导航](README.md)。下列代码、配置、命令和产物路径均为待实现约定，不代表目前已存在。新增细节属于总方案的实施起点；只有通过试采集和验证后才冻结，不得用最终测试结果反向挑选配置。

## 目标

建立可接翻页、音乐、音量、鼠标及未来应用的统一终端功能适配层。模型识别物理词元；模式映射为标准命令；实际平台后端只执行能力白名单，不在模型/状态机内部写快捷键。

## 命令与接口

```text
presentation: next_slide, previous_slide
media: media_play_pause, media_next, media_previous, adjust_volume(delta)
pointer: move_pointer(x,y), click(button), button_down/up, scroll(dx,dy)
safety: release_all
Backend.capabilities() -> set[str]
Backend.execute(ActionCommand) -> ActionResult
Backend.release_all() -> None
```

首版真实后端可只实现 presentation，其他能力仍有 schema/mock，不把未实现标支持。示例映射 `swipe_left+presentation→previous_slide`、`swipe_left+music→media_previous`、inactive→无普通命令。播放暂停姿态、音量模式由 T21 另行冻结。

## 实施步骤

1. 在 `configs/actions.yaml` 定义 gesture_id/mode→action、参数约束、后端能力要求和风险级别；未知配置直接错误，禁止任意 shell/脚本字符串。
2. 调度器单一所有者处理 ActionCommand，命令具备唯一 ID、源事件 ID、生成/过期时间、context_epoch、目标上下文。执行前在串行提交闸门内复检停止位、epoch、模式、能力、焦点和 TTL；不把检查与提交拆成可竞态的两段。已开始的外部系统调用未必能撤销，分别报告 stop_received、禁止后续提交、safe_quiescent/释放结果，不虚称任意瞬间都可撤回已注入事件。
3. 离散命令以 event_id/command_id 去重，翻页、切歌、播放暂停等非幂等命令不自动重试；uncertain 显式告警。连续更新用 control_stream_id/update_seq 拒绝旧更新，不把同一控制过程的所有新采样当重复离散事件。
4. 离散队列有界。绝对光标位置/绝对目标值可最新覆盖；音量/滚动优先传有时间戳和 TTL 的目标速率，由单一控制器按实际 dt 积分。已经计算出的相对 delta 不能直接最新覆盖，否则丢失控制量；若使用 delta 队列，应有界合并、限幅并禁止不确定重试。停止/失效时速率归零、清积分余量，不补发历史变化。
5. 发出 key/button-down 前先登记“可能按下”，确认按下后转为 held；返回超时/异常并不能证明未按下，仍纳入释放集合。click 的 finally 覆盖已确认及可能按下项，只有得到相应释放证据才清除；失败保持告警。release_all 是受限、可重入的安全动作，不被普通去重/能力失败挡住。
6. 后端调用必须限时并与控制/停止循环隔离，说明超时后底层调用是否仍可能完成。不能仅用线程 future 超时假装取消了系统调用；晚到按下/回执要再次触发安全释放或保持禁用。不能证明安全收尾的后端不启用持有类能力。DryRunBackend 只记录拟执行，MockBackend 能模拟异常前后已产生副作用、阻塞和晚到完成；不通过全局 monkeypatch 偷换 mock/live。
7. ActionResult 区分内部接收、系统注入和应用确认；不具备应用回执时不能把 injected=true 解释为用户看到了结果。

## 原生多端与远程执行边界

- ActionBackend 分为本应用内、系统/外部应用、T38 可选远程目标；每类执行前均经过调度，不因原生 UI 可调用 SDK 就允许绕过安全闸门。
- capabilities 按具体 OS/版本/权限/目标动态给出，缺失或撤销时拒绝。手机默认应用内命令；Android 敏感系统能力单独授权评估，iOS 不要求任意跨 App 注入，未知 OS 默认无外部能力。
- 平台 SDK 只在适配器加载，不让某个 OS 缺依赖导致训练/回放/mock 无法启动。T36 冻结边界，T37 用相同 fixture 验证移植后的去重、TTL、旧 epoch 拒绝与释放。
- T38 远程命令必须在执行端再校验授权/目标/新鲜度/租约；目标端 watchdog 不依赖手机在线。单调时钟不跨设备直接比较，标准命令不能包含任意 shell 或触摸脚本。

## 输出与验收

`actions/mapper.py`、`actions/dispatcher.py`、`actions/backends/dry_run.py`、`mock.py`、动作 schema 与能力矩阵。

- [x] 合成/mock验证：相同词元在演示与音乐模式生成不同标准命令，inactive 无命令。
- [x] 合成/mock验证：未实现能力清楚报 BACKEND_UNAVAILABLE，不静默当成功。
- [x] 合成/mock验证：重复、过期、旧 epoch、停止后的排队命令均不会注入。
- [x] 合成/mock验证：后端异常、失焦、重复stop、释放失败，以及外部日志抛错/阻塞均有故障注入记录；真实平台仍待T23/T24。

T19/T21/T22 仅添加具体能力后端；不重新实现识别、去重或权限逻辑。

## 前置任务文件

[T02 建立工程骨架、配置与公共数据契约](02_scaffold_and_contracts.md)；[T05 冻结手势词表、动作边界与标签规则](05_gesture_and_annotation_protocol.md)；[T17 实现唤醒、控制权、释放去重与独立停止状态机](17_interaction_state_machine.md)
