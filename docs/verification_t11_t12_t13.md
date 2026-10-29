# T11 → T12/T13 实施与验证记录

2026-10-29（Asia/Shanghai），`handshake-core 0.2.0`；执行者Codex，真实A/B仍由用户填写。本轮代码未提交Git或发布。完整实现、配置、schema、类表和锁文件SHA256见 [实现清单](verification/t11_t12_t13/implementation_manifest.json)。

## 验收结论

| 任务 | 软件交付/结果 | 状态与后续 |
| --- | --- | --- |
| T11 | 94维共享几何、字段级train统计、真实dt/缺失/健康时钟、32帧窗口、草案Dataset、训练增强、哈希缓存与构建CLI | 合成/软件通过；真实观测与正式统计待T04/T07/T10，保留待验证 |
| T12 | B0有限历史规则、MLP94→128→64→5、统一预测器、draft模型配置、可过拟合和5类合成诊断 | 模块交付已完成；正式训练/校准/事件实验待T14/T15/T28 |
| T13 | 四个单卷积因果残差块、同一时刻通道LN、末帧分类、有效标签CE、forward_all诊断 | 结构/因果/梯度验收已完成，交T14训练 |

**309项测试通过，0 failures/errors/skipped**；[JUnit报告](verification/t11_t12_t13/pytest.xml)只去除hostname字段，不修改结果。全套包含原T02/T05回归和本轮类表、特征、统计、窗口、数据、缓存、配置、规则/神经网络及预测器测试。

## 实际命令与环境

Linux x86_64，CPython3.10.12，NumPy2.2.6，PyTorch2.14.1+cpu，pytest9.2.0。沿用已锁定基础/感知/CPU训练环境，无新服务依赖；`uv pip check`确认47个安装包兼容。

```bash
uv lock --offline
uv sync --locked --extra dev --extra perception --extra training --offline
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --junitxml=docs/verification/t11_t12_t13/pytest.xml
env -u PYTHONPATH .venv/bin/python -m algorithm.smoke --output docs/verification/t11_t12_t13/model_smoke.json
uv build --offline --out-dir /tmp/handshake-t11-dist
uv pip check
```

报告输出采用独占创建，复跑smoke须选择新路径；不覆盖历史报告。ROS环境/全局pytest插件隔离方式沿用T02，未修改系统Python。

## 特征与数据证据

[data_pipeline.json](verification/t11_t12_t13/data_pipeline.json)使用明确标记synthetic的3个匿名模拟用户、120条记录，预分配train/val/test各40条。三个集合各保留40个末端，32帧预热后9个ready，覆盖率9/40，warm-up末端未从分母删除。该数字是短合成流的软件诊断，不能用作真实可用率。

统计仅由train拟合；val/test缓存复用同一stats，NPZ/索引/统计重载通过，记录schema/classes/config/dataset/split/stats与字节哈希。测试还反事实修改test几何，train的mean/std/count逐位不变；跨用户原片/lineage、多人可见角色混用、错误dataset hash和非冻结真实manifest在拟合前被拒绝。

重点回归：

- local63＋global10＋mask21公式、平移/尺度归一化、宽高比与未镜像方向保留，输出有限；离线Dataset与在线逐帧使用同一函数、数值一致。
- 无手/质量失败/未处理分开；初始no-hand不建假轨迹；保留完整原始记录与不可用末端。导数失效字段不参与统计，几何无效标准化后置0，dt/missing固定缩放。
- 冷启动连续32有效、预热中缺失/大间隔重计，短失跟保留历史但禁当前执行；持续每帧缺失超过重置时长也要重新预热32帧。
- 处理时钟比源帧晚100ms时，下一条合法源帧不会因早于上一处理时钟被误拒；源顺序/健康与新鲜度仍检查。
- 健康查询不能倒退使旧特征重新变新鲜；延迟超过长断流阈值的样本不留预热历史，append可返回None表示已清空。
- 增强来源限定train，派生记录保留父哈希/seed/配置/lineage，左右翻转同步交换类别并重算速度，test用户不能借source_split声明进入train。
- 缓存拒绝覆盖，不使用pickle；字节篡改、错误schema/classes/依赖哈希拒绝。真实T06会话与T10审核仍未完成，当前JSON是T11草案适配。

## 网络与基线证据

| 项目 | 实际结果 |
| --- | --- |
| MLP参数量 | 20,741，输入[B,94]→[B,5] |
| TCN参数量 | **56,325**，输入[B,32,94]→[B,5] |
| TCN感受野 | **31**；四块d=1/2/4/8，每块一次k3卷积，左pad |
| 因果性 | 改变未来输入不改变过去logits；感受野最早影响与末端窗口重算一致 |
| 数值/训练接口 | CPU前后向有限，eval可复现，错误类表/维度/窗口拒绝；全无监督batch返回None供训练器跳步 |
| 统一输入 | B0/MLP/TCN接同一FeatureWindow和质量/ready/新鲜度/时钟门控；不重贴stream/epoch |
| 输出边界 | 原始logits或规则决策分数；intent/phase/校准概率不伪造；不发真实设备事件 |

[model_smoke.json](verification/t11_t12_t13/model_smoke.json)记录seed42、1个CPU线程、20个容易分离的合成编码张量、每模型100次Adam更新。MLP与TCN均可完全记忆这20个样本，梯度/loss有限；最终训练loss约0.000449/0.000546。模型报告逐类precision/recall/F1及混淆矩阵，但**全部是人造训练集诊断，不能作为真实识别或泛化成绩**。B0报告5个手工几何fixture的5类分项结果，也只证明规则语义按预期工作。

该次CPU batch1前向采样30次：MLP p50/p95约0.027/0.029ms，TCN约0.304/0.314ms。只含模型前向/输入检查，无相机、感知、网络、事件确认、设备注入或应用响应；不是端到端延迟。训练检查耗时约55/589ms，同样没有方法比较意义。正式预算与公平指标归T14/T15/T28。

## 安装包检查

sdist与wheel构建通过。将本地0.2.0 wheel临时安装到项目虚拟环境、移除editable映射后，用`python -I`并从源码目录外执行检查，实际模块路径位于site-packages。feature_schema.json、classes.json、mlp/tcn YAML均可加载，两个模型参数量与RF一致。另一次隔离模式检查将结果保存为 [packaging.json](verification/t11_t12_t13/packaging.json)，包含分发文件哈希。随后已恢复editable源码安装。

```bash
uv pip install --offline --no-deps --python .venv/bin/python --reinstall-package handshake-core /tmp/handshake-t11-dist/handshake_core-0.2.0-py3-none-any.whl
# 从/tmp以项目解释器 -I 导入schema/classes/model_config并核对参数量
uv sync --locked --extra dev --extra perception --extra training --offline --reinstall-package handshake-core
```

## 交接与未完成项

使用说明与API在 [特征与模型](features_and_models.md)。T11真实统计和正式数据导出等待T04/T07/T10；T12/T13的模块出口已经满足，可供T14共用训练器开发。当前无真实dataset版本/权重哈希，没有正式测试集选择、校准、事件指标、相机或Web/手机界面。`handshake train`仍明确提示T14尚未实现；合成smoke不保存checkpoint。分工、正式训练和后续Web仍按既有约定推进。
