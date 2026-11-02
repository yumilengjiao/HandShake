# 公开数据来源

默认版本：**`ipn-hands-frames-v1`**，即 IPN HandS 的逐帧图片关键点与时间标签。**不下载视频，不需要摄像头，不执行数据包内脚本。**

## 下载与准备

```bash
uv run download
uv run prepare
uv run train --dry-run
```

来源与固定SHA在 [`sources.json`](sources.json)，项目审查清单在 [`shared/defaults/ipn_hands_release.json`](../shared/defaults/ipn_hands_release.json)。完整依据见[版本审查](../docs/ipn_hands_release.md)。

| 文件 | 用途 | 本次审查 |
| --- | --- | --- |
| [IPN_HandS_annotations.zip](https://drive.google.com/file/d/1HybcqZqnNKZGgJJXCMjj6tLFbpsNP_a1/view) | 15类、连续区间、帧数、官方train/test清单 | 82,274字节；采用明确的项目审查快照，旧发布方SHA差异保留 |
| [IPN_HandS_landmarks_MPHands.zip](https://drive.google.com/file/d/1Y-3_MUHJ6OYpv-BNof5vYDRR4YNgrGQ2/view) | 图片帧的MediaPipe Hands预测，21点xyz | 实际375,332,617字节，作者公布SHA一致；全部200个JSON、801,476帧已核查 |

大文件只保存在本机 `data/raw/ipn-hands/`，不提交Git。每帧0～2只手，未检出和多手情况不会被伪造为正常单手监督。GT/Holistic包不属于默认下载内容。

## 标注SHA差异如何处理

作者固定README公布：

```text
4686a2ebf70a6036166c261bc6992d050cfba08c3b18edcdd460b3fa9f30ec71
```

官方链接重复提供、本次项目审查采纳的快照：

```text
520dfbf3d52ecab6535d8841bba55e94d6b512596ed1cff4505d2c1345bcbaec
```

**这两个值仍不相同，不声称作者修复了校验或确认了原因。** 本次根据官方固定Git/下载链路、206个成员的逐项审查、匹配作者SHA的关键点包，以及独立Git/元数据的来源清单对照，采纳一个显式版本化的项目快照。

原始 `expected_sha256` 没有被删除或改写；`project_release` 引用另一个固定SHA的审查清单。仅这个精确版本可被默认流程采纳；其他文件、未来变化或无review引用的校验不符仍会拒绝。没有运行时跳过校验的选项。

## 默认划分与限制

默认按50个完整**来源采集组**固定划分：116个训练、32个验证、52个测试序列。保留官方测试集，只从官方训练组中固定选取验证组；不逐帧随机划分，同组四个序列不会跨集合。

来源采集键不是独立核实的参与者身份。模型/缓存会记录 `subject_identity_verified=false`；可以报告采集组留出结果，不能直接宣称已认证的跨人评估。原作者 `metadata.csv` 没有主体ID；本次只用它核对来源名称、集合及旧帧数，没有猜测人物身份。

保留HandS 15类词表和原640×480图片帧的30fps时间轴；不混用原IPN Hand缩放/去重后的320×240图片编号或原版14类时间标签。

## 官方资料与许可

- [作者说明：固定Git版本](https://github.com/GibranBenitez/IPN-hand/blob/093729579be671df0cc9dd8cbc58b9a342663800/README_HandS.md)
- [作者下载目录](https://drive.google.com/drive/folders/1eILZNuPRM53_Ige2jxGqXV3OdIeoMeIb)
- [原版元数据目录](https://drive.google.com/drive/folders/1-mihJEIFoNDpfo1puF8xAMJz6PGVKsBD)（这是文件夹，不是ZIP直链）
- [元数据CSV](https://drive.google.com/file/d/17d2lsFOJZ-R74PrkS83iO7xSV8ZDiwwT/view)（辅助审查，默认运行不需要额外下载）

数据与派生审查清单按 **CC BY 4.0** 保留归属。请引用 Benitez-Garcia 等人的 [IPN HandS论文](https://doi.org/10.3390/app15116321) 和 [IPN Hand论文](https://arxiv.org/abs/2005.02134)，并注明本项目的特征处理及分组修改。

数据可用于训练不等于正式训练成绩、真实操作意图或端到端控制安全已验证。其他平台/设备兼容性仍以实际测试为准。
