# T05 标注指南

配套 `gesture-v1-draft`；规则与参数见 [手势协议](gesture_protocol.md)。当前只提供合成标注示例，真实标注与复核尚未开展。

## 按顺序标注

1. 查看原始未镜像视频及采集协议，记录 recording_id、匿名 subject_id、session_id、人工 hand_track_reference 与 annotator_id。人工轨迹引用不强依赖检测成功；不要把姓名/授权原件写入公开标签。
2. 使用 recording 起点相对整数 ns，区间 `[start_ns,end_ns)`。同一录制/轨迹/标注者的区间不重叠，可首尾相接。缺失不能跨录制拼接；采集器需另存 timebase 和源时间映射。
3. 先标原始准备/有效/释放/背景边界，再标可见词元。静态稳定起点即类别起点，达到保持时长才有参考事件；动态依据方向/位移证据。词元冲突采用动态优先规则。
4. 分类监督只使用有效动作或可确认背景；已回到中性的过渡可标 no_action。无法判定的准备/释放段 `gesture_label=null,class_valid=false`，不把不确定段塞入 no_action。
5. 单独给意图：明确请求1、普通自然动作0、不可确认ambiguous；自然左挥与指令左挥均标 swipe_left。intent_valid 仅对0/1为true。阶段不明时 phase=ambiguous、phase_valid=false，与类别/意图掩码独立。
6. 有意图不等于必有设备事件：保持不足、回程或未满足协议授权的动作无目标。只给有效区间中的参考事件赋 target_command 与 reference_event_time_ns，两者必须成对出现。自然动作或ambiguous意图不能有目标事件。
7. 在另一个上下文事件流记录 UI/键盘启停、模式、人工owner申请、暂停、停止与健康变化；不将其混入图像类别。T02 定义 ContextChange/HealthEvent，T06/T23 实施完整保存。
8. A/B 独立标同片，保存各自版本后复核。分歧记录原因与修订，不能用模型预测决定真值或覆盖旧标注而不留痕。

## 字段与验证

`shared/annotations.py` 定义 AnnotationInterval / AnnotationTimeline：

| 字段 | 规则 |
| --- | --- |
| recording/subject/session/hand_track/annotator | 非空匿名引用，同一录制不能变更subject/session |
| start_ns/end_ns | 整数非负ns，start < end |
| gesture_label/class_valid | 可知类别用词元与true；未知类别null与false |
| intent_label/intent_valid | 严格整数0/1与true；ambiguous与false |
| phase/phase_valid | background/preparation/active/release与true，或ambiguous与false |
| reference_event_time_ns/target_command | 同时有或同时null；有值要求有效、有意图、active，参考在 `[start,end)` 内 |
| quality_tags/environment_tags | 例如 occluded、unprocessed、return_motion、synthetic、near_negative；保留失败原始段 |
| schema/annotation_version/time_unit/timebase | 1.0.0 / gesture-v1-draft / ns / recording_relative |

阶段在首版以多段区间保存：一个动作可拆准备、active、release多行，同一动作通过 recording/track 与连续边界关联。active段可承载参考事件，准备/释放保留原边界及相应掩码。完整动作关联ID与会话NPZ格式由 T06 根据本契约补充，不能只保存成功检测帧。

首版方向目标校验：wake_open只允许request_control、exit_pose只允许exit_control、左挥允许previous_slide/previous_track、右挥允许next_slide/next_track。没有目标的自然动作仍有可见类别。窗口标签/采样从区间派生，T11 执行，真值和参考时刻不进入在线输入。

## 合成案例怎么读

`tests/fixtures/annotation_examples.json` 有5类、自然挥动、反向回程、短保持、中性停留和遮挡不确定。具体时间仅用于示范字段关系，不是从视频测得的边界；执行 `python -m pytest` 会检查合法性、相邻边界、重叠、意图/目标冲突及参考半开边界。

例如自然 swipe_left 与请求 swipe_left 只改intent/目标，词元不变；右挥后的可见左向回程仍为swipe_left、intent=0，无目标。张掌有效段从起点就标wake_open，reference在起点+600ms；保持仅100ms的张掌段保留wake_open但reference=null。

## 正式采集前的质量门槛

每类至少一组真实正反例获授权；同片由A/B独立标注，比较类别/意图一致率与阶段边界绝对偏差； unresolved 分歧修订协议后再采主数据。后续正式标注至少抽10%～20%双人复核，保留修订历史。尚无真实数据，因此上述工作记录为待 T07 验证。
