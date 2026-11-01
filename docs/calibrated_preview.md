# T16/T20：验证集校准概率的只读原生预览

新增 **HSP4/v4** 二进制协议。v2相机、v3原始logit接口和现有人工翻页保持兼容；不把原v3的`calibrated: false`偷偷改义。v4在同帧分类旁增加独立`probability`对象，包含验证集温度、拒识阈值、draft状态、合成标记和校准SHA，始终 `actions_enabled: false`。

## 运行

```bash
.venv/bin/python scripts/run_native.py \
  --bundle frontend/native/build/linux/x64/release/bundle/handshake_stage \
  --perception-model "$PERCEPTION_ASSET" \
  --model-package "$MODEL_PACKAGE" --model-sha256 "$TRUSTED_MODEL_SHA" \
  --calibration "$CALIBRATION_JSON" --calibration-sha256 "$TRUSTED_CALIBRATION_SHA"
```

仍须在UI显式点击“打开相机预览”。启动器不继承shell中遗留的模型/校准资产选择；资产由本次显式参数选择。直接使用原生程序时，可配置对应`HANDSHAKE_*`变量；校准使用`HANDSHAKE_CALIBRATION`和`HANDSHAKE_CALIBRATION_SHA256`。

缺少配对参数直接拒绝。worker在创建相机或MediaPipe detector **之前**完成模型包与校准校验；通过 `load_for_bundle`绑定模型SHA、数据/划分、统计/配置、词表及模型选择时的验证库存。坏校准不回退成“已经校准”，也不自动联网下载。

## 显示与清除规则

- 原始网络logits仍单列；UI新增 `验证集温度校准 draft`、最高概率、温度、`分类通过阈值/分类拒识/拒识配置不可用`、合成校准警告和校准SHA片段。
- `accepted`只表示分类通过阈值，不是操作意图/用户授权，也不代表最终事件参数冻结。
- 校准不可行时可显示温度缩放概率，但拒识配置明确unavailable，不虚构门限。
- 同帧校验包含epoch/frame/source-time、模型版本、概率长度、归一化、概率与当前logits/温度的一致性及拒识结果。Python和Dart都验证；时间保持十进制大整数，不经double。
- 暖启动、缺手、过期、错误、停止后没有概率对象，不残留旧概率。回看仍只读，不能发出控制请求。
- 原模型包仍是`uncalibrated_preview_only`，不原位修改包内校准文件。`model_status=loaded_uncalibrated`指原始模型包/原始分类层；独立probability对象描述已应用的外部draft校准。

## 验证与边界

- 17项新增Python测试；624项广域非训练回归通过，4项训练用例及5个训练/梯度测试文件排除。
- Flutter 70项单元/widget测试通过；覆盖v2/v3兼容、v4解析、概率/拒识提示、停止清除、禁用动作和相机预览按钮。
- Linux原生进程集成通过：真实Flutter↔Python的HSP4连接、合成预览、回看、停止、坏资产在访问相机前拒绝及重连。使用刻意不存在的模型/校准/感知路径和 `HANDSHAKE_CAMERA_TEST=0`，没有真实摄像头访问。
- Linux Release构建、Flutter analyze通过。原生测试发现并修复了v4相机按钮版本判断遗漏。

测试与概率样例全为程序构造，不是正式模型校准效果。本机未运行神经网络训练，不开启手势控制。正式模型效果、连续流事件参数冻结、真实手势翻页和外部应用控制仍未验收。

协议契约：`shared/calibrated_preview_protocol.py`及`.schema.json`；跨语言样例：`frontend/native/test/fixtures/calibrated_preview_v4.bin`。文档：[T15校准](evaluation_and_calibration.md)、[原模型包](model_bundle_and_inference.md)。
