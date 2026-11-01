# T24：应用内停止、故障与资源长时诊断

`backend/safety_soak.py` 提供独立、明确合成的长时运行诊断，不开启相机、不读取真实数据、不导入或训练神经网络，也不注入OS输入。它使用实际T18调度工作线程和实际内置页码后端。

```bash
# 常规短检查（每轮含3个应用命令和3个重复事件拒绝）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m backend.safety_soak --cycles 70 --allow-synthetic \
  --output /tmp/handshake-soak-70

# 长时软件稳定性检查，需要显式授权且使用新输出目录
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m backend.safety_soak --seconds 9000 --allow-synthetic \
  --max-rss-mb 256 --max-growth-mb 128 --pace-ms 25 \
  --code-revision "$(git rev-parse HEAD)" --output /tmp/handshake-soak-9000
```

时长使用 `time.monotonic()`，不是把虚拟源时间当作真实经过时长。最大4小时；也可按最多100,000轮运行。25ms轮间节奏用于限制持续CPU占用，输入源时钟仍是虚拟时间。中断或异常写入interrupted/failed，不宣称通过。

## 实际检查

每轮显式启用/绑定owner，经过几何中性释放，执行右/右/左的预编写运动与保持/回程段；检查确切的应用页码（页边界不回绕）、重复事件被拒绝。轮流注入无预测时效截止和整帧多手冲突，确认独立watchdog/所有权策略撤销控制。每轮都显式停止并等待真实调度/释放线程确认安全静止。

- 停止→安全静止及虚拟源截止跳变→watchdog观测的耗时，用真实单调时钟测量。
- 使用固定直方图保存全部计时样本，p95为桶上界而非假装精确分位数。
- 当前RSS来自Linux `/proc/self/statm`；记录每轮结束时的RSS/Python线程采样峰值（不是连续OS高水位），设绝对RSS/增长预算和线程预算。
- 每32轮轮换宿主，既压测关闭/重建也避免把有限去重容量当成无限生命周期保证。
- `progress.json`每5秒原子更新，内存只保留最近360个资源采样；最终生成`report.json`和带SHA的`receipt.json`。
- 所有业务输入是手写logits和程序几何；surface ack为模拟宿主确认，不是Flutter绘制。没有真实用户误触率、真实识别准确率或实际摄像头→应用延迟的结论。

## 代码与短验证

9项自动测试覆盖限额/显式授权、三类故障、失败报告、实际墙钟时长与无Torch/相机导入。另运行70轮预检：3个宿主、210次应用内命令、210次重复拒绝、23次源时效故障、23次所有权故障，全部停止/释放确认通过；实测约30.17秒，RSS采样峰值约54.7MiB，Python线程采样峰值4。长期结果须以实际最终报告为准，不能用这个30秒探针代替数小时结论。

本整合分支同时包含三个独立任务分支，但不修改master：

- `feat/observation-model-replay`：关键点/校验模型只读回放。
- `feat/calibrated-readonly-preview`：v4原生draft概率显示。
- `feat/offline-handoff-kit`：白名单交接与完整性预检。

正式T24的真实相机/模型卡死、外部应用权限、温升与真实误触/时延、最终候选全链签认仍未完成。
