# yumilengjiao3 基于最新 master 的适配验证

日期：2026-10-29（Asia/Shanghai）。基线 `origin/master=62b2ee6`，原功能分支 `origin/yumilengjiao3=5a188bb`。本轮同步代码/目录/文档，不下载公开训练大包，不创建产品界面，不启用真实系统操作。

## 分支与目录处理

- 本地master原为9cdcbc7，与远端历史分叉。保留本地 `backup/master-before-sync-20261029` 和 `backup/yumilengjiao3-before-master-sync-20261029` 后，对齐origin/master并执行pull确认最新。
- yumilengjiao3合入该master，21个文件冲突已逐项处理；master的公开数据来源、任务T07/T08调整、感知修复、依赖/锁文件完整保留，T14训练器、T17状态机与T18调度保留。
- 当前说明、目录README、总方案及任务导航统一为public-data-v1。T06自采可选；T07改为公开来源/语义/完整性预检，T08下载/版本管理。公开数据标注SHA阻塞未被伪装为已解决。
- 现有模块仍是旧5类协议；15类IPN HandS导入、类表/契约/模型/事件映射迁移与UI/键盘启停尚未实现。Open twice/Grab未冒充静态张掌/握拳。新增回归确认类别14不能混入旧5类缓存训练。
- SDK原生探针仅由master的 `backend/perception/probe.py` 实现。旧 `backend/perception/diagnostics.py` 与 `scripts/probe_perception.py` 转发统一有界worker，并兼容原参数/摘要；canonical入口仍为 `handshake-perception-probe` / `scripts/perception_probe.py`。

## 新环境与软件验证

原项目 `.venv` 未改动。另建临时Python3.12环境，使用master的完整锁文件安装dev/perception/training：

```bash
UV_CACHE_DIR=/tmp/handshake-master-sync-cache \
UV_PYTHON_INSTALL_DIR=/tmp/handshake-master-sync-python \
UV_PROJECT_ENVIRONMENT=/tmp/handshake-master-sync-venv \
uv sync --locked --python 3.12 --extra dev --extra perception --extra training

env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/tmp/handshake-master-sync-venv/bin/python -m pytest \
  --junitxml=docs/verification/master_sync/pytest.xml
```

实际CPython3.12.14、NumPy1.26.4、MediaPipe0.10.21、OpenCV4.11.0.86、PyTorch2.15.0+cpu、pytest9.2.0。`uv pip check`检查52包兼容。原依赖环境与旧报告保留，不把旧2.14/Python3.10训练状态冒充新环境续训成果。

**431项测试通过，无跳过，19.71秒**，见[JUnit](verification/master_sync/pytest.xml)。包括旧训练/恢复/交互/调度回归、master的原生子进程守护，以及新增兼容转发、已有报告阻止启动、无效期限不建文件和公开类别误入旧缓存拒绝。MLP/TCN独立进程epoch恢复检查也在新依赖下通过。

离线构建sdist/wheel成功；wheel装入独立target后从 `/tmp` 以 `python -I` 导入安装版，禁止torch导入时交互演示仍产生2个拟翻页结果；确认旧诊断引用同一个canonical模块。解除阻止后，安装版训练器在合成缓存上完成2 epoch。包资源trainer.yaml可加载，摘要/包哈希见[安装检查](verification/master_sync/packaging.json)。这些仍不是15类公开数据实验或原生应用安装验收。

## 当前主机SDK预检的限制

使用相同官方模型资产、固定兼容依赖，运行统一探针；不打开相机、不记录像素或手部资料：

```bash
MPLCONFIGDIR=/tmp/handshake-master-sync-mpl \
/tmp/handshake-master-sync-venv/bin/python -m backend.perception.probe \
  --model artifacts/perception/hand_landmarker.task \
  --model-sha256 fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1 \
  --timeout 30 --rss-limit-mib 768 \
  --report artifacts/master-sync-perception-probe.json
```

第一次在 `model_load` 阶段触发30秒监控超时，未进入model_ready。随后单独导入MediaPipe约0.59秒成功；在此条件下再次完整探针仍在同阶段触发30秒监控超时。两次采样RSS峰值均低于768MiB限制。退出信号由探针自身的timeout守护触发，不能当作自发SIGKILL或OOM证据；没有静默降级、更换模型或反复无界重试。

完整记录见[首次SDK超时](verification/master_sync/perception_probe_initial.json)、[独立导入](verification/master_sync/perception_import.json)、[导入后SDK超时](verification/master_sync/perception_probe_after_import.json)。本机的初始化原因仍未确定；master中另一主机0.10.21成功及旧3.10/1.2.0成功记录均保留，互不覆盖。自动测试通过和安装版训练成功不代表当前主机真实感知已通过，更不代表真手/相机到应用闭环完成。

## 交接

后续按public-data-v1解决T07来源完整性/方向与类别，统一迁移协议，T10冻结划分后使用T14正式训练。T17/T18旧5类合成模块可作为安全参考；公开版启停迁移、T15/T16、T19和T36/T20仍是后续工作。当前主机感知加载超时单独保留待查，不修改master已验证的依赖约束或历史证据。

[本轮文件/基线清单](verification/master_sync/implementation_manifest.json)。
