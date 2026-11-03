# 课程报告（2026-11-03）

- **可编辑 Word**：[HandShake课程报告.docx](HandShake课程报告.docx)。正文、表格均可编辑；图可替换。
- **提交用 PDF**：[HandShake课程报告.pdf](HandShake课程报告.pdf)，A4共4页、全部文字黑色、灰度图、嵌入中文字体。
- **文本源稿**：[HandShake课程报告.md](HandShake课程报告.md)。也可直接编辑。
- **真实指标及来源**：[results.json](results.json)，包含逐类Precision/Recall/F1、原始混淆计数、训练曲线、配置与哈希。
- 图片保留SVG和PNG两种格式。Word/PDF是分别生成的版本；**修改Word后，请用Word/WPS重新导出PDF**，原PDF不会随之自动更新。

## 提交前必填

1. 两人的真实姓名、学号与实际贡献，核对建议分工并补充提交/实验依据。
2. 训练总耗时（交付日志未记录）。
3. 如补做冻结后的测试集实验或真实相机实验，再填写对应留空项；不要用验证成绩替代最终测试，也不要编造成功率/延迟。

当前图表已用真实TCN产物填好，不是合成训练成绩。模型第2轮被选为最佳，第6轮F1略高但增加量小于`min_delta=0.0001`，因此未替换。总体Accuracy为78.85%，Macro-F1为46.52%，不能据此宣称现场翻页可靠。

## 本机产物位置

```text
artifacts/imports/training-252a1acda9380066/artifacts.zip
artifacts/runs/20261102T174832Z-tcn-af17e513/
artifacts/report-tools/  # 报告/出图/检查工具，仅本机，按要求不提交Git
```

压缩包及模型不提交普通Git。归档SHA256：
`252a1acda938006649f261eb5708257c76d8d002fe65cf43b38ad06e20fc4390`。
模型、分数的随包哈希提供完整性检查，不是独立第三方认证。

接收方本机数据准备回执与训练方不同，因此没有覆盖本机`latest.json`或伪造数据绑定。使用外部模型加载入口，不能强行使用`app --model`冒充本机训练。

```bash
uv run app \
  --package artifacts/runs/20261102T174832Z-tcn-af17e513/models/449a7e8ad1e7eb8e-afdb96f0 \
  --sha256 1c3e0253b5a6f543df3f7472a4a48ab9f6ebe35aa240807beb787cd76a84e2da \
  --calibration artifacts/runs/20261102T174832Z-tcn-af17e513/evaluations/20261102T180313Z-val-81509e0e/validation-calibration-local.json \
  --calibration-sha256 966d00a4b32fd3479158246b9f92e99d4fb396f1b0377bdfed0c49d7b4eefd0d \
  --shadow --verify-only
# 去掉 --verify-only 才启动界面；不会自动打开相机或执行应用动作。
```

本轮未运行训练、未打开真实摄像头、未读取测试集样本。工程回归数字引用11月2日记录；11月3日新增接收模型、离线分数重算、验证集校准及报告检查，见[验收回执](../verification/trained_handoff_2026-11-03.json)。

报告出稿后另增加了[实验性本地手势演示](../local_gesture_demo.md)。它仍未通过真实相机验收，因此没有将报告中的现场成功率/延迟留空项改成“通过”；也没有擅自覆盖你可能已经编辑的Word正文。
