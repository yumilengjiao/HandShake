# configs：公共配置与实验设置

> T02/T05 已提供 runtime、interaction、gestures、actions、training、privacy 和 platforms/mock 配置，通过符号链接使用 `shared/defaults/` 的同一来源。所有门限均为 draft 起点；包内默认值和 `--config-dir configs` 加载结果一致，尚无验证集选择的最优门限。

## 主责

- A：训练、模型、评估与特征参数。
- B：相机、运行状态机、平台能力、动作映射与界面设置。
- 两人共同审核词表、特征 schema、隐私、安全及跨模块参数。

## 后续配置划分

| 计划文件/子目录 | 内容 |
| --- | --- |
| `default.yaml` | 安全默认运行模式、相机探测和基础选项 |
| `gestures.yaml` | 词元 ID、定义、版本及阶段/释放协议 |
| `actions.yaml` | 模式到白名单动作的映射与能力要求 |
| `features/` | 特征顺序、坐标/时间约定、窗口和缺失策略 |
| `training/` | MLP/TCN 结构、采样、优化、种子及选择指标 |
| `evaluation/` | 数据版本引用、指标、事件匹配与搜索预算 |
| `platforms/` | 实测 OS/应用能力、权限与焦点规则 |

配置 schema 和校验代码属于 `shared/`，不是在 YAML 中写业务逻辑。

## 原生平台配置交接

T36/T37 的 profile 需区分 OS/版本/ABI、相机/感知/推理适配器、权限/生命周期策略、应用内与外部能力；默认无未知外部权限，运行时能力撤销立即禁用。不同平台共用手势/特征语义但不共用未经验证的快捷键或焦点策略。移动端配置存储/导出按应用沙箱实现，不依赖仓库符号链接或开发者绝对路径。

T38 的配对凭据留在平台安全存储，不入公共 YAML；会话授权和能力仍经执行端校验。新增字段在实施时同步 schema/版本与测试，此次不修改现有 mock 配置。

## 约束

1. 默认 dry-run；开启 live 必须经过后端明确授权和检查，不能仅靠改一行 YAML 绕过安全协议。
2. 未知字段、动作、错误单位、负时间、缺能力和版本不匹配应报错。
3. 方案门限标为起点，验证后冻结；特征统计和校准结果进入对应模型包，不随意覆盖通用配置。
4. 使用配置指定数据/产物根路径，不能写死某位组员的绝对路径。
5. 配置可以入 Git，但不得含姓名映射、密钥、个人档案、真实授权材料或敏感窗口信息；私有覆盖放 `private/`，环境凭据不入库。
6. 每次训练/发布保存最终合并配置与哈希，不能只记录原始默认文件。

相关任务：[T02](../task_plans/02_scaffold_and_contracts.md)、[T05](../task_plans/05_gesture_and_annotation_protocol.md)、[T14](../task_plans/14_training_and_experiment_tracking.md)、[T18](../task_plans/18_generic_action_dispatch.md)。

返回 [根目录说明](../README.md)。
