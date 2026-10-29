# MediaPipe 初始化兼容修复与验收（2026-10-29）

分支：`fix/capture-perception-validation`。接续 [T03/T04/T06 初次交付记录](verification_t03_t04_t06.md)，不覆盖旧测试、旧依赖或旧相机报告。

## 结论与边界

- **当前机器的模型启动阻塞已有可验证的兼容修复**：固定 MediaPipe **0.10.21**、opencv-contrib-python **4.11.0.86**，由依赖约束锁定 NumPy **1.26.4**，真实 Hand Landmarker 加载、三帧黑色 RGB 的 VIDEO 推理及关闭通过。
- **不是查明并修复了 SIGKILL 的最终来源**：MediaPipe 1.2.0 对照仍在加载阶段收到 SIGKILL；没有 OOM、超时或具体驱动归因的充分证据。不修改上游 SDK、不修改系统安全策略、不把新版自动回退为旧版或合成检测器。
- 新增 `backend/perception/probe.py`、安装入口 `handshake-perception-probe` 与薄入口 `scripts/perception_probe.py`。原生调用放独立子进程，父进程保留阶段、退出码/信号、超时、Linux RSS 采样及 cgroup 内存事件。
- **154项自动测试通过**（原131项＋23项探针回归），依赖检查、编译、sdist/wheel及独立wheel环境模型探针通过。
- 黑图返回无手仅证明模型确实运行；**不代表真手精度、左右手/镜像、遮挡、双手交叉或实际相机推理速度已经验收**。没有访问参与者资料，没有采集训练数据，没有训练模型，没有键鼠注入。

## 环境对照：先独立验证，再改锁文件

本次宿主：Linux 7.2.11-arch1-1 / x86_64 / CPython 3.12.14，Wayland。使用同一份本地官方 Hand Landmarker float16/v1资产，SHA256：

```text
fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
```

资产来源/许可边界见 [采集指南](collection_workflow.md)。探针不自动下载资产，不访问相机。

| 环境 | MediaPipe / NumPy / OpenCV发行版 | 实际结果 |
| --- | --- | --- |
| 原项目环境 | 1.2.0 / 2.5.4 / 5.0.0.93 | 约2.21秒后returncode=-9；本次监控未主动杀进程，无成功模型输出 |
| 独立兼容候选 | 0.10.21 / 1.26.4 / 4.11.0.86 | 真实加载→3帧黑图推理→关闭，returncode=0；原131项回归通过 |
| 独立1.2.0对照 | 1.2.0 / 1.26.4 / 4.11.0.86 | 同一旧版NumPy/OpenCV下仍SIGKILL，约2.71秒；尚未到model_ready |
| 更新后的项目锁环境 | 0.10.21 / 1.26.4 / 4.11.0.86 | 模型探针、真实SDK＋合成图像preview均退出0，全部154项回归通过 |
| 非源码目录独立wheel安装 | 0.10.21 / 1.26.4 / 4.11.0.86 | 从`/tmp`运行已安装探针，真实加载、推理、关闭通过 |

独立候选通过后，复制其发行依赖集合构造对照，再换为1.2.0及其所需certifi 2026.10.23；两个环境的`uv pip check`均通过。该对照说明**仅降NumPy/OpenCV不能解决当前故障**，不等于定位了上游的具体代码缺陷。

1.2.0正式对照采样峰值约182MiB，低于探针768MiB阈值；`monitor_reason=null`，cgroup各级OOM/OOM-kill计数未增加，内核日志未找到相关条目。RSS是离散采样，不保证捕获所有峰值；日志缺失也不能证明所有外部原因均不存在。GDB下仍终止于SIGKILL，未获得能归因的有效调用栈。首次GDB启动受宿主nushell参数解析影响，关闭GDB启动shell后才成功运行，未改变用户shell。

新锁会引入旧SDK所需JAX/protobuf等传递依赖，并改变NumPy/scipy等解析版本；没有安装第二个cv2提供包。**当前完整感知安装仅验证Python3.12/Linux x86_64**，请显式使用3.12；不能从核心`requires-python>=3.10`推断该感知wheel覆盖所有Python版本/ABI。training extra未在本轮重验，其他OS/手机与真实应用控制仍待各任务验收。

## 可复现诊断

```bash
uv sync --locked --python 3.12 --extra dev --extra perception
uv pip check
.venv/bin/handshake-perception-probe \
  --model artifacts/perception/hand_landmarker.task \
  --model-sha256 fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1 \
  --timeout 30 --rss-limit-mib 768 \
  --report artifacts/perception-probe-new.json

# 等价入口：已安装包或editable环境，支持不在源码目录运行
.venv/bin/python -m backend.perception.probe --model <本地资产>
.venv/bin/python scripts/perception_probe.py --model <本地资产>
```

- 报告独占创建，不覆盖旧证据；无效期限在创建文件前拒绝。
- 子进程明确记录`adapter_import → model_load → model_ready → inference ×3 → model_close → complete`。仅进程退出0且存在完成阶段才通过；哈希错误、异常、原生信号、缺成功输出均失败，不伪装无手或跳过。
- 成功退出0；失败、主动期限/资源中止、Ctrl-C退出2。报告区分子进程`returncode=-9/signal=SIGKILL`与监控自身的`timeout/rss_limit/log_limit`，**SIGKILL不等于OOM**。
- 默认30秒父进程期限，终止后最多等待5秒；Windows使用进程终止，POSIX使用新建子进程组。若仍无法回收，明确`shutdown_timeout`，不宣称安全清理完成。父进程自身被系统强杀时不保证完整报告。
- Linux RSS每约20ms采样，默认超过768MiB主动中止；不是硬内存限制，也不统计整个进程树内存。其他OS保留期限但RSS/cgroup标不可用，不伪造数字。
- 单路日志超过约1MiB主动中止；报告只保留末32KiB并标截断。临时日志随探针结束删除；不写图像/骨架坐标，不创建录制目录。
- 探针子进程取消全局`PYTHONPATH`，设置OMP/OPENBLAS线程环境为1；这不承诺限制SDK的所有线程。没有在正常capture路径偷偷改环境。
- **常规preview/collect仍在自身进程内加载SDK**。本工具是可复现预检，不是完整采集进程隔离/崩溃恢复，不用它背书T17/T24的停止安全。

## 相机稳定性补验

独立执行不加载模型、不保存画面/关键点的headless相机探针，只写统计元数据：

```bash
.venv/bin/handshake camera_probe --camera 0 --headless --duration 300 \
  --report artifacts/perception/camera-5min.json
.venv/bin/handshake camera_probe --camera 0 --headless --duration 5 \
  --report artifacts/perception/camera-reopen.json
```

300秒探针实际约 **300.280秒**，V4L2/RGB 640×480，**3003帧**，实测采集 **10.0146Hz**、消费 **9.9973Hz**。最新帧槽峰值1、消费者替换丢帧0、状态账本溢出0；3002帧进入消费者，停机尾部1帧保留为未处理。驱动漏帧未知，设备报告30FPS不代表实测30FPS。

3002个软件read完成→消费者年龄样本：p50 **914,615ns**、p95 **2,295,526ns**、max **6,317,967ns**，不是曝光端到端延迟。`error=null`、`shutdown_confirmed=true`；关闭后5秒重新打开也正常完成。无模型，3003帧全部`not_processed`，不是成功无手识别；`complete=false`表示未录制数据集，不是探针失败。

此项通过的是**5分钟headless采集稳定性和随后重开**，不替代人工窗口预览、设备拔插/占用或真手感知验收。

## 自动测试与打包

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  --junitxml=docs/verification/pytest_perception_fix.xml
.venv/bin/python -m compileall -q backend shared scripts
uv build --offline --out-dir /tmp/handshake-perception-fix-dist
```

新增测试覆盖正常/异常退出、真实子进程自发SIGKILL、超时回收、RSS监控原因、日志上限、启动失败、非法期限、缺资产/错误哈希报告、拒绝覆盖、完成协议、三帧黑图及推理异常时关闭。测试默认不导入真实SDK、不打开设备；POSIX信号用例在非POSIX明确跳过，不据此宣称Windows已验证。

[JUnit报告](verification/pytest_perception_fix.xml)仅去除hostname，其余结论保留；[机器可读摘要](verification/perception_fix_summary.json)分列独立候选、失败对照、最终锁环境、独立wheel和相机证据。详细原始日志留在Git忽略的`artifacts/perception/`，不公开会话HTML或机器私有路径。

## 剩余验收（由操作者执行）

1. 先运行模型探针，再打开`handshake preview --model <本地资产>`，核对真手左右、镜像开关、进出画、遮挡、双手交叉与实际处理率。
2. 人工关闭窗口、重复停止/重新打开，设备占用、权限拒绝/撤销与拔插；不自动恢复live。
3. 核对采集窗口提示、暂停仍采背景、取消、不完整恢复、骨架回放/区间编辑/导出。
4. 取得所有参与者明确授权后，才运行短collect并进行T07双人试标；不能照抄synthetic授权引用或把黑图/合成骨架当训练数据。

T03/T04/T06继续保留“待完整验收”，不因依赖兼容修复、154项自动测试或无手黑图推理就勾选全部完成。
