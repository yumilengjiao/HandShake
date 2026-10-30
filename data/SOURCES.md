# 公开数据下载入口（只提交链接，不提交数据）

核查日期：2026-10-29。用户选择 **public-data-v1：公开数据为主，不要求自行录制训练集**。机器可读来源、固定文档版本、许可证、预期SHA256及实际检查状态见 [`sources.json`](sources.json)。HTTP 200/网页能打开不等于数据已完整下载或可训练。

## 首选候选：IPN HandS，先用关键点，不必下载视频

- [作者说明（固定Git版本）](https://github.com/GibranBenitez/IPN-hand/blob/093729579be671df0cc9dd8cbc58b9a342663800/README_HandS.md)
- [作者Google Drive下载目录](https://drive.google.com/drive/folders/1eILZNuPRM53_Ige2jxGqXV3OdIeoMeIb?usp=sharing)
- [IPN HandS论文](https://doi.org/10.3390/app15116321)、[原IPN Hand论文](https://arxiv.org/abs/2005.02134)
- 作者明确将数据标为 **CC BY 4.0**，使用时保留来源、归属及两篇论文引用，注明预处理/重映射等修改。

| 文件 | 作者标称大小 | 用途 | 当前检查 |
| --- | --- | --- | --- |
| [IPN_HandS_annotations.zip](https://drive.google.com/file/d/1HybcqZqnNKZGgJJXCMjj6tLFbpsNP_a1/view?usp=drive_link) | <1 MB | 类别、连续区间标签、划分与帧计数；必需 | **SHA256与作者清单不一致，禁止直接导入** |
| [IPN_HandS_landmarks_MPHands.zip](https://drive.google.com/file/d/1Y-3_MUHJ6OYpv-BNof5vYDRR4YNgrGQ2/view?usp=drive_link) | 375 MB | MediaPipe Hands预测；优先训练输入候选 | 分享页HTTP 200，未下载大包 |
| [IPN_HandS_landmarks_GT.zip](https://drive.google.com/file/d/1pbCWxfzGZyexHe89lZtGgnHWfa958XJv/view?usp=drive_link) | 674 MB | 人工修正2D关键点的参考对照 | 可选，不是首轮必需 |
| [IPN_HandS_landmarks_MPHolistic.zip](https://drive.google.com/file/d/1hdtmr4JpK14rt1zMcbN39-oTedn2sBx1/view?usp=drive_link) | 1.1 GB | 另一种感知前端对照 | 可选，不必现在下载 |

覆盖200段原始视频，14类手势＋非手势；关键点为21点`[x,y,z]`，每帧0～2只手。官方列出148段训练/52段测试，但仍需审查参与者重叠，不能仅凭视频列表就宣称跨用户划分。GT只人工修正x/y，z不是人工验证的3D真值。

### 已发现的校验问题（不能隐瞒）

作者上述固定版本公布标注包SHA256：

```text
4686a2ebf70a6036166c261bc6992d050cfba08c3b18edcdd460b3fa9f30ec71
```

本次从官方链接两次取得相同的82,274字节ZIP，实际SHA256：

```text
520dfbf3d52ecab6535d8841bba55e94d6b512596ed1cff4505d2c1345bcbaec
```

ZIP的206个成员CRC通过，也能读取类别/划分文本；这**不能解释或消除SHA256不一致**。原因可能涉及上游文件版本变化，但当前未证实。文件仅留临时调查目录，未当成正式训练资料、未执行包内脚本。后续先核对作者更新/获得一致版本；不能把实际哈希替换为期望哈希后谎称通过。若暂不能解决，转原IPN Hand视频路线评估，不要求用户因此改为自采。

## 备用：原始IPN Hand RGB视频

- [官方网站与完整下载说明](https://gibranbenitez.github.io/IPN_Hand/)
- [原始MP4视频入口](https://drive.google.com/open?id=1O4Fn_jbAEcKIksXHmQIMcHTyaDw4wt9x)：作者标称合计 **4.6 GB**，5个独立包，每包40段视频。
- [原版标注入口](https://drive.google.com/open?id=1-mihJEIFoNDpfo1puF8xAMJz6PGVKsBD)：作者标称364 KB。本次分享页可打开，直接下载请求返回500，尚未取得有效包；不能当自动直链承诺可用。
- 数据/标注为CC BY 4.0；基准代码MIT不是数据许可的替代说明。

可用于用本项目固定版本MediaPipe重新提取骨架。原始IPN Hand与HandS的**类别、帧索引、时间轴不同**：原版320×240帧曾去近重复帧，HandS在原视频上用`ffmpeg fps=30`重新抽帧，区间1-based且含末帧。禁止混用两套标签或用帧序号猜真实时间。

## 静态增强备选：HaGRID（首轮不用下载）

[官方仓库](https://github.com/hukenovs/hagrid)有`palm`、`fist`等手型，但不是连续挥动/操作意图数据。当前HaGRIDv2完整包约1.5 TB，512px版也约119.4 GB，不适合作为本轮第一步。

**许可特别注意**：已读取作者[英文许可PDF](https://github.com/hukenovs/hagrid/blob/080e18917376ec935e453cd0e599c23478c7e98f/license/en_us.pdf)，其脚注明确写“不是Creative Commons许可”，不能只看README图标就标成标准CC BY-SA 4.0。采用前按实际自定义条款审核。

## 本地目录与Git规则

```text
data/
  README.md                     # Git跟踪
  SOURCES.md                    # Git跟踪：人工阅读入口
  sources.json                  # Git跟踪：链接、版本、许可、校验状态
  external/                     # Git忽略：下载包与解压数据
    IPN_HandS/
    IPN_Hand/
  ...                           # 其他真实数据仍忽略
```

下载时先确认可用空间、条款和来源；完成后核对`sources.json`中的预期值，再检查归档成员与完整性。不使用登录Cookie、临时签名下载地址或私人网盘作为提交的固定入口。Google Drive需要人工确认或限制流量时，保留官方分享页，不绕过访问限制。

**URL清单只解决“仓库不放大文件”**，不等于别人`git clone`后已经有数据、训练模型或能直接运行`train`。导入、版本化词表和小规模深度训练管线已实现，见 [操作指南](../docs/public_data_implementation.md)；官方数据核验、全量流式缓存与正式训练仍未完成。路线和验收见 [公开数据方案](../docs/public_data_plan.md)。本轮没有上传任何图片、视频、原始标签、权重或会话HTML。
