# T01 → T02 → T05 实施与验证记录

> 历史证据说明：下文保留该轮 Web＋手机需求及软件验证结果，不代表最新产品形态。当前已改为 [native-v2 原生多端](native_architecture.md)，T20/T36/T37/T38 均待实施；此次文档修订不重跑软件验证、不更新历史 manifest/哈希或宣称多平台通过。

2026-10-29（Asia/Shanghai）。执行者：Codex；实际成员A/B由用户后续填写。本轮没有提交Git commit或外部发布；代码版本为当前HEAD加dirty标记，精确文件哈希见 [实现清单](verification/implementation_manifest.json)。

## 交付与结论

| 任务 | 本轮已实现 | 验证结论 | 剩余验收 |
| --- | --- | --- | --- |
| T01 | 范围/P0～P2、Web＋手机后续方向、课程差异表、建议分工、平台矩阵、评价口径与mock平台草案 | 文档与安全默认值可核查 | 用户补分工/通知/课程差异；演示软件/权限/焦点与独立停止实测 |
| T02 | Python单工程、CLI、八类数据契约、上下文/健康事件、六类配置、JSONL、mock链路、依赖锁、安装说明 | **81项测试通过**；全部依赖探针通过；sdist/wheel构建与独立环境CLI通过 | 实际相机、感知模型资产与演示目标尚未验证 |
| T05 | 5类词表、坐标示意、类别/意图/阶段/参考/释放规则、标注验证器、13段合成案例 | 合成标签/边界/掩码/自然回程与目标一致性通过 | T06/T07获授权真实正反例、A/B同片独立试标与最终冻结 |

三个任务在导航中保留“待验证”，并注明软件/草案已交付；没有将实机或真实数据退出条件虚标完成。合成资料不是实验结果，未声称真实识别、误触率或跨设备兼容。

## 实际验证命令与环境

Linux x86_64 / Python3.10.12；普通命令使用项目 `.venv`。机器和依赖证据见 [环境JSON](verification/environment_t02.json)，其中摄像头只列举节点，没有打开或采集。

```bash
uv lock
uv sync --locked --extra dev --extra perception --extra training
# 清理两个OpenCV提供包曾共用文件造成的损坏
uv sync --locked --extra dev --extra perception --extra training --reinstall-package opencv-contrib-python --offline

env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --junitxml=docs/verification/pytest_t02_t05.xml
uv pip check
env -u PYTHONPATH .venv/bin/python scripts/check_environment.py --timeout 45 --output docs/verification/environment_t02.json
.venv/bin/handshake dry_run --config-dir configs > docs/verification/synthetic_smoke.jsonl
uv build --offline --out-dir /tmp/handshake-t02-dist
```

依赖安装/正常宿主探针在允许的沙箱外执行，原因见 [依赖问题记录](dependencies.md)。最终pytest结果为 **81 passed，0 failed/0 errors/0 skipped**；[JUnit报告](verification/pytest_t02_t05.xml)去除了机器hostname字段，测试结果未改动。`uv pip check` 返回47个包兼容；报告未包含由ROS的PYTHONPATH引入的系统包。

所有八类契约均生成JSON Schema成功，默认配置的仓库路径与包内资源加载结果一致。无相机/模型依赖的mock链路输出8条可回读记录，包含启用、owner、合成FrameStatus、预测、事件、命令、mock结果和停止；[合成日志](verification/synthetic_smoke.jsonl)中 `synthetic=true`、`timebase=synthetic`、模型版本 `synthetic-script-v1`，实际注入为not_attempted，回执等级mock。

独立安装验证实际执行：

```bash
uv venv --python /usr/bin/python3 /tmp/handshake-t02-wheel-env
uv pip install --python /tmp/handshake-t02-wheel-env/bin/python /tmp/handshake-t02-dist/handshake_core-0.1.0-py3-none-any.whl pydantic==2.14.1 PyYAML==6.0.3 numpy==2.2.6
cd /tmp
env -u PYTHONPATH /tmp/handshake-t02-wheel-env/bin/handshake dry_run
```

从`/tmp`运行后断言 `backend.__file__` 位于新环境site-packages、默认配置可读、8条日志可校验且结果not_attempted/mock，全部通过。发行包与安装证据哈希见 [打包检查JSON](verification/packaging_t02.json)；临时wheel没有作为权重或真实数据入仓库。

## 覆盖重点与能力边界

- 严格ns整数/单位与schema、形状/非有限值、处理无手/未处理/错误区分、概率和可选头。
- 未知/重复YAML键、未知动作、白名单/能力、缺失映射、负时长和不支持的live配置。
- 切换启停/模式/模型/owner、挂起、重连后旧stream/epoch不能发事件；未来/过期预测、到期/重复命令拒绝。
- 连续命令必须带控制流ID和序号；uncertain注入不被压成false，注入确认不冒充应用响应。
- 同词元自然/指令不同意图；半开区间、相邻与重叠、短保持、回程、参考时间与目标冲突。
- 未实现collect/train/evaluate/replay/live返回可理解错误；合成日志可回读且不覆盖既有文件。

当前MockGestureEngine不实现唤醒/持有/回程释放状态机，MockDispatcher不证明并发停止/迟到副作用安全；这些须T17/T18/T23/T24完成。当前FeatureWindow来自零值fixture，不是T11特征工程。没有数据集、模型权重、训练/分类指标、真实翻页、HTTP服务或手机页面，因此相关哈希/性能/截图为不适用，未伪造。

## 下一步交接

该轮原定继续 T03/T04 → T06 → T07，Web/手机 UI 交后续 T20。现按 native-v2 更新为 T36 选型 → T20 原生桌面，T37 本机手机、T38 可选配对控制，见 [当前任务导航](../task_plans/README.md)。T05 仍需 T07 真实试标冻结；分工/课程日期由用户补充，历史验证不覆盖这些新增任务。
