# shared：公共契约与共享基础模块

> T02/T05 已实现 `contracts.py`、`config.py`、`records.py`、`annotations.py` 与包内 draft 默认配置。契约/配置/合成标注校验已接入测试；共享特征和真实 NPZ 会话格式尚待 T11/T06。

## 主责与任务

- **A+B 共同维护**；A 主责共享特征，B 主责运行上下文需求，接口变更双方复核。
- 对应 T02 数据契约、T11 共享特征，以及 T06/T10 的记录格式/校验。
- 以 [T02](../task_plans/02_scaffold_and_contracts.md) 和 [T11](../task_plans/11_features_and_windows.md) 为详细规格。

## 后续内容

- `contracts`：FramePacket/FrameStatus、HandObservation、FeatureWindow、ModelPrediction、GestureEvent、ActionCommand/Result、UI 状态快照。
- `features/`：坐标约定、宽高比、归一化、运动特征、缺失掩码、时序窗口与 ready。
- `records/`：公共记录读写、schema、版本、哈希及结构验证；不包含采集设备或训练 Dataset。
- `config/`：公共配置类型、校验与版本兼容检查；配置实例放根 `configs/`。
- 时间、枚举、错误类型和纯函数工具，只在确实两端复用时加入。

上述名称为后续模块划分，不表示已有代码。

## 依赖与所有权

- 本目录不反向依赖 frontend/backend/algorithm，也不发设备操作。
- 尽量只依赖基础库及必要的数组/校验库，不为了公共字段导入 PyTorch、MediaPipe 或 UI 框架。
- `shared/` 定义状态对象/时钟接口，后端拥有运行生命周期和安全决策；训练优化由算法模块拥有。
- Python 特征保持单一实现，离线与在线不得各抄一套。T37 移动运行时优先绑定公共核心；必须移植时以同一版本化规范和 golden fixtures 检查特征、窗口、ready 和数值误差，不各端自由调参。
- 记录读写是通用格式能力；采集流程在 backend，训练采样/增强在 algorithm。

## 原生多端交接

T36 为跨语言壳确定 schema/绑定和 IPC/FFI，不等于所有端直接导入 Python。时间整数必须无损编码，平台回调时基显式转换；T38 跨设备不得直接比较单调时钟。平台能力、权限/生命周期和网络会话等增量字段需在实现时升级 schema 并同步校验，此次不改既有契约代码。

## 接口变更规则

1. 先说明字段、形状、单位、缺省值、错误行为及兼容性。
2. 时间用明确 ns/timebase，不能将模型完成时间当源帧时间。
3. stream/epoch 在工作提交时捕获并贯穿链路；None 不等于置信度 1。
4. 变更特征顺序、标签语义或归一化需升 schema，并处理模型/缓存兼容。
5. 同步双方调用方、根配置及 `tests/` 中合成测试，不仅修改一侧。

首次交接建议先用纯合成数据验证形状、序列化、因果性、缺失和跨版本拒绝，再接真实相机及模型。

返回 [根目录说明](../README.md)。
