# T18 实现通用命令映射、调度器与 mock 适配器

| 项目 | 约定 |
| --- | --- |
| 阶段 | S4 通用终端能力层 |
| 课程优先级 | P0 |
| 建议主责 | B（A/B 实名以总方案为准） |
| 前置任务 | T02, T05, T17 |
| 总方案依据 | 第 3.3、12、16.2～16.4 节；通用架构图 |
| 状态 | 待实施；本文是规划，不代表已有代码、数据或结果 |

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
2. 调度器单一所有者处理 ActionCommand，命令具备唯一 ID、源事件 ID、生成/过期时间、context_epoch、允许的目标上下文。执行前重新检查停止位、epoch、有效模式、能力、焦点和 TTL。
3. 去重缓存同时覆盖重复投递 event_id/command_id；翻页、切歌、播放暂停等非幂等命令不自动重试。若注入结果不确定，记录 uncertain 等待下一次显式操作。
4. 离散队列有界，连续移动/音量用“最新值覆盖”而非无限追加；停止时清空普通队列并优先 release_all，不补发过期累计变化。
5. 调度器维护实际成功按下的键/按钮集合，事务式 click 用 finally 保护释放；释放失败显式告警。普通日志失败不可挡住清理，后端失败不能导致无界重试。
6. DryRunBackend 只记录拟执行命令；MockBackend 可注入 accepted/injected/timeout/uncertain/exception 场景。受支持平台后端拆小模块，不通过全局 monkeypatch 在 mock 与 live 之间偷偷切换。
7. ActionResult 区分内部接收、系统注入和应用确认；不具备应用回执时不能把 injected=true 解释为用户看到了结果。

## 输出与验收

`actions/mapper.py`、`actions/dispatcher.py`、`actions/backends/dry_run.py`、`mock.py`、动作 schema 与能力矩阵。

- [ ] 相同词元在演示与音乐模式生成不同标准命令，inactive 无命令。
- [ ] 未实现能力清楚报 BACKEND_UNAVAILABLE，不静默当成功。
- [ ] 重复、过期、旧 epoch、停止后的排队命令均不会注入。
- [ ] 后端异常、失焦、重复 stop、释放日志失败有故障注入记录。

T19/T21/T22 仅添加具体能力后端；不重新实现识别、去重或权限逻辑。

## 前置任务文件

[T02 建立工程骨架、配置与公共数据契约](02_scaffold_and_contracts.md)；[T05 冻结手势词表、动作边界与标签规则](05_gesture_and_annotation_protocol.md)；[T17 实现唤醒、控制权、释放去重与独立停止状态机](17_interaction_state_machine.md)
