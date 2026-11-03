# HandShake 命令与脚本参数参考

按2026-11-03工作区实际CLI解析器核对；包含本轮新增训练选项。旧文档中的计划入口不算已实现命令。

## 1. 先区分三种入口

```bash
uv run train --seed 42                     # 日常短命令
uv run handshake-model inspect --help      # 已注册高级命令
uv run python scripts/train.py --help      # 源码中的兼容脚本
```

`uv run`是环境/命令执行器，不是训练算法。`uv run train`等价于`uv run hs train`。参数放在命令之后；uv自身选项放在命令之前，例如`uv run --extra training handshake-model ...`。NVIDIA环境使用`--extra cuda`，不要同时指定互斥的training/cuda extras。

日常命令自动读取`handshake.toml`、本机覆盖`handshake.local.toml`；CLI参数只覆盖这次运行。高级工具通常不自动继承这套配置，需要手动指定资产。

## 2. 所有日常短命令

日常命令统一支持：`--workspace DIR`指定工作区；`--debug`显示异常堆栈；`-h/--help`查看帮助。无workspace时从当前目录向上寻找配置，也可通过`HANDSHAKE_WORKSPACE`指定。

| 命令 | 用途 | 专属参数 |
| --- | --- | --- |
| `uv run hs` | 显示总帮助，或通过子命令调用下列功能 | 子命令 |
| `uv run doctor` | 检查依赖、设备和数据状态；不训练、不开相机 | 无 |
| `uv run status` | 当前与doctor调用同一实现 | 无 |
| `uv run setup` | 准备项目虚拟环境 | `--gpu`安装CUDA训练依赖；`--camera`安装OpenCV/MediaPipe，不打开相机 |
| `uv run download` | 按来源清单下载并核对数据包，正确文件复用 | 无；路径来自配置 |
| `uv run prepare` | 校验/导入数据，构建冻结划分和train/val/test缓存 | 无；路径、stride等来自配置 |
| `uv run train` | 新建独立run并从头训练 | 见下一节 |
| `uv run resume` | 恢复最近run的完整epoch checkpoint | 无；不能传训练超参数改实验 |
| `uv run evaluate` | 评估最近模型；必要时先导出模型包 | `--split val/test`，默认val |
| `uv run calibrate` | 最近模型的验证集温度/拒识校准，不更新网络权重 | 无；只用val |
| `uv run export` | 导出最近已完成run的最佳模型为推理包 | 无 |
| `uv run app` | Linux原生预览；默认不开相机、不执行应用动作 | 见第4节 |
| `uv run check` | 非训练Python回归 | `--training`启用包含合成训练的完整回归；可用加速器测试也显式启用，无设备时skip |

`train`创建新的随机初始化模型；`resume`才是续训。默认在`artifacts/runs/`保存独立结果。短命令evaluate/calibrate/export依赖`latest.json`，不是自动逐一处理所有历史run。

## 3. `uv run train`全部参数

下表默认值是当前共享配置；本机配置可覆盖。未提供CLI参数不会覆盖配置。

| 参数 | 默认 | 含义/取值 |
| --- | --- | --- |
| `--model` | tcn | `tcn`因果时序模型；`mlp`末帧基线 |
| `--device` | auto | `auto/cpu/cuda/cuda:N/mps`；不可用的显式设备报错，不静默换设备 |
| `--epochs` | 80 | 最多训练轮数；可早停，不保证一定跑满 |
| `--batch-size` | 64 | 每批窗口末端数量；紧凑缓存最多1024 |
| `--learning-rate` | 0.001 | AdamW初始学习率，须为有限正数 |
| `--weight-decay` | 0.0001 | AdamW权重衰减，有限且不小于0 |
| `--patience` | 10 | 验证Macro-F1连续无足够改善的早停耐心轮数 |
| `--seed` | 42 | 初始化/随机采样种子；整数0到2^32−1 |
| `--sampling` | uniform | `uniform`每轮无放回遍历所有训练端点；`supervised_uniform`只遍历class_valid端点，各一次，不删窗口历史、不改验证分母 |
| `--class-weighting` | none | `none`普通掩码CE；`sqrt_inverse`使用训练标签频次的平方根逆频率权重，不改验证CE |
| `--scheduler` | constant | `constant`恒定学习率；`plateau`验证Macro-F1停滞时衰减 |
| `--lr-patience` | 2 | plateau容忍停滞轮数；默认连续3轮不足够改善后衰减，可为0 |
| `--lr-factor` | 0.5 | plateau每次乘的系数，必须在0和1之间 |
| `--min-learning-rate` | 0.000001 | plateau学习率下限，不大于初始学习率 |
| `--dry-run` | 关闭 | 只校验配置和train/val数据，不创建run、不导入训练器、不开始训练 |

`lr_*`参数在constant模式不产生调度。plateau要求`patience >= lr_patience + 2`，使首次衰减后至少有机会训练一轮。降低学习率不会重置早停计数。

以下训练设置没有对应短命令参数，写入配置的`[train]`：`threads=1`、`grad_clip_norm=1.0`、`min_delta=0.0001`、`runs="artifacts/runs"`、`auto_export=true`。threads是计算线程数，grad_clip_norm限制梯度范数，min_delta是验证选模/plateau的绝对改善门槛；auto_export控制完成后是否自动导出。优化器固定AdamW，checkpoint按全类验证Macro-F1选择。

```bash
uv run train --class-weighting sqrt_inverse --seed 42 --dry-run
# 确认预算和设备后，去掉dry-run才开始训练
uv run train --class-weighting sqrt_inverse --seed 42
uv run evaluate
```

每轮保存best.pt、last.pt、epochs.jsonl等。中断只恢复最后完整轮次；更改损失、采样、代码、数据或环境须开新run。比较实验时每次训练后先evaluate再开始下一次；最终模型和策略冻结之前不使用test选参数。

## 4. `uv run app`全部参数

| 参数 | 含义 |
| --- | --- |
| `--model` | 加载本机最近run模型，布尔开关，不接模型路径 |
| `--package DIR` | 指定别人交付的模型包，不要求本机训练数据 |
| `--sha256 HASH` | 外部模型checksums.json的可信SHA256；必须与package同时提供 |
| `--calibration FILE` | 外部模型匹配的验证校准文件 |
| `--calibration-sha256 HASH` | 校准可信SHA256，与calibration成对且须有package |
| `--shadow` | 启用影子owner界面，只显示候选，不执行动作 |
| `--allow-synthetic-model` | 只允许在显式shadow诊断中加载合成模型 |
| `--bundle FILE` | 使用已构建Linux原生可执行文件，无需Flutter SDK |
| `--verify-only` | 只校验资产，不启动UI或相机 |

model与package二选一；shadow/verify-only要求其中之一。摄像头编号、Flutter路径和感知模型来自配置`[app]`的camera/flutter/perception_model，不是短命令app的CLI选项。

## 5. 已注册高级工具：全部公开参数

本节只有各自`--help`是通用参数；**不通用支持日常命令的workspace/debug**。下面“必需”包括解析器要求及执行时要求。路径/哈希须来自可信回执，不能随便填以绕过验证。

通用含义：output是输出位置；input是输入归档/文件；package是推理模型目录；run是训练目录；cache是特征缓存；scores是前向分数归档。sha256一般对应该输入的可信文件/manifest哈希；model-sha256对应模型checksums.json；cache-index-sha256/index-sha256对应缓存回执内canonical index_sha256，不一定等于index.json文件字节哈希。calibration及calibration-sha256指定匹配校准文件与哈希。

### 5.1 `handshake-demo`：实验性内置幻灯片

参数：`--run DIR`必需；`--allow-local-demo`显式授权；`--verify-only`只验证；`--perception-model FILE`感知task模型；`--camera N`默认0；`--slides DIR`加载1到32张PNG/JPEG幻灯片。

启动UI须allow-local-demo，verify-only无需该授权。默认显示内置演示页；运行后还需键盘显式开启相机、控制与owner。只控制本程序的页码，不连接PowerPoint/WPS、不注入系统按键。

### 5.2 `handshake-data`：离线审核、导入、构建缓存

| 子命令 | 用途 | 全部参数 |
| --- | --- | --- |
| `audit-ipn` | 审核标注ZIP | 必需`--catalog --annotations`；可选`--report` |
| `import-ipn` | 标注/关键点转统一记录 | 必需`--catalog --annotations --landmarks --subjects --output`；可选`--report` |
| `build-cache` | 构建旧NPZ缓存 | 必需`--imported --output --split --dataset-version --review-reference`；可选`--stats --max-records --report` |
| `build-stream-cache` | 构建v3流式/紧凑缓存 | build-cache全部参数，另有`--max-groups --stride` |
| `inspect-cache` | 只读核对紧凑缓存 | 必需`--cache --index-sha256`；可选`--report` |

catalog是审查过的来源清单；annotations/landmarks是对应ZIP；subjects是真实审核主体计划，不猜身份。imported是统一记录目录，split为train/val/test，dataset-version标记数据版本，review-reference是真实审核依据。val/test的stats须复用训练集统计。max-records为审核条数预算：NPZ默认10000，stream默认2000000；max-groups默认50000，stride默认1。高级import-ipn需审核主体计划；默认采集组路线建议直接使用短命令prepare，不编造主体资料。

### 5.3 `handshake-model`：模型包

| 子命令 | 用途 | 全部参数 |
| --- | --- | --- |
| `export` | 显式导出指定run | 必需`--run --normalization --output`；可选`--allow-synthetic` |
| `inspect` | 加载并校验指定包 | 必需`--package --sha256`；可选`--threads`默认1 |
| `benchmark` | CPU前向性能测量，不训练 | inspect全部参数，另有`--iterations`默认100 |

normalization是该run的训练归一化stats.json；threads允许1到8；allow-synthetic只用于显式合成软件验证，不把结果当真实手势成绩。

### 5.4 `handshake-evaluate`：显式评估/校准/诊断

| 子命令 | 用途 | 全部参数 |
| --- | --- | --- |
| `score` | 指定模型在指定缓存上的CPU前向分数 | 必需`--package --model-sha256 --cache --cache-index-sha256 --split --output`；可选`--batch-size`默认256、`--threads`默认1、`--allow-synthetic` |
| `classify` | 分数归档生成分类指标 | 必需`--scores --sha256 --output`；可选`--calibration --calibration-sha256` |
| `diagnose` | 验证错误、类别/ready/历史长度等诊断 | classify全部参数，另有`--allow-synthetic` |
| `calibrate` | 只用验证分数拟合温度与拒识 | 必需`--scores --sha256 --output`；可选`--temperatures --thresholds --max-risk --min-coverage --min-accepted --allow-synthetic` |
| `events` | 独立事件/时间区间指标 | 必需`--input --sha256 --output` |

score的split是val/test。temperatures和thresholds各接一个或多个候选值；省略时使用实现内候选集合。max-risk默认0.1，约束接受样本的错误风险；min-coverage默认0.5，要求最低接受覆盖率；min-accepted默认20，要求最低接受数量。校准不重训网络；diagnose不是测试集调参入口。classification的校准路径/哈希须成对。输出不能覆盖既有产物。

### 5.5 回放工具

| 命令 | 用途 | 全部参数 |
| --- | --- | --- |
| `handshake-public-replay` | 手写合成分数/几何经过15类策略改变内置页码 | 必需`--output`及显式`--allow-synthetic` |
| `handshake-observation-replay inspect` | 校验观测归档 | 必需`--input --sha256` |
| `handshake-observation-replay run` | 归档经特征、网络及可选校准只读回放 | 必需`--input --sha256 --package --model-sha256 --output`；可选`--calibration --calibration-sha256 --allow-synthetic --threads`，threads默认1 |
| `handshake-model-session` | 模型回放接校准/owner/事件策略 | 必需`--input --sha256 --package --model-sha256 --calibration --calibration-sha256 --output`；可选`--execute-synthetic --allow-synthetic --threads`，threads默认1 |

execute-synthetic允许对程序内合成演示对象执行，不授权操作真实桌面；allow-synthetic只允许合成资产。以上不自动开相机，不做训练。

### 5.6 `handshake-perception-probe`：感知SDK进程探针

必需`--model FILE`；可选`--model-sha256 HASH`、`--timeout`默认30秒、`--rss-limit-mib`默认768MiB、`--report FILE`。加载真实MediaPipe task，在合成图像上检查初始化/调用，监控子进程；不是摄像头或真实手势成功率测试。

### 5.7 `handshake-safety-soak`：合成生命周期压力测试

必需`--output DIR`、显式`--allow-synthetic`，以及**二选一**`--seconds N`或`--cycles N`。可选`--max-rss-mb`默认256、`--max-growth-mb`默认128、`--pace-ms`默认25、`--code-revision`。

seconds是时长，cycles是循环次数；rss/growth是内存预算，pace-ms是循环节拍，code-revision记录版本。只验证合成软件生命周期，不代表真实相机长时稳定。

### 5.8 `handshake-handoff`：离线交付包

- `build`必需：`--output --core-wheel --lockfile --license-file --code-revision --review-reference`。可选：`--python-version --wheelhouse --native --model --model-sha256 --model-license --calibration --calibration-sha256`。
- `inspect`必需：`--input --sha256`。

core-wheel是项目wheel；lockfile是依赖锁；license-file是许可证；review-reference是审核依据；wheelhouse是离线依赖wheel目录；native是原生构建资产；model/model-license/calibration是附带模型、许可与校准。相互关联的文件/哈希须匹配，不自动安装、启动或训练。

### 5.9 `handshake`：旧采集/注释总入口

模式位置参数：`dry_run`默认、`preview`、`camera_probe`、`collect`、`inspect`、`recover`、`annotate`、`train`，另有解析器列出但**执行明确拒绝**的`evaluate/replay/live`。评估请用新的evaluate/handshake-evaluate，回放用专用工具。

除train转发外，全部参数分组如下：

| 参数组 | 参数、默认与含义 |
| --- | --- |
| 配置/合成日志 | `--config-dir`旧六份YAML配置目录；`--log`只用于dry_run合成JSONL |
| 视频来源 | `--source camera/video/synthetic`默认camera；`--camera`默认0；`--video`视频文件；`--video-clock fps/decoder`默认fps |
| 图像几何 | `--width`640；`--height`480；`--fps`30；`--rotation 0/90/180/270`默认0；`--no-mirror`关闭镜像预览 |
| 感知资产 | `--model`MediaPipe task而非手势网络；`--model-sha256`；`--model-source`默认user-supplied-local-asset；`--model-license`默认提示先核许可 |
| 时长/窗口 | `--duration`默认30秒；`--frames`默认90个合成帧；`--headless`不打开预览窗口；`--report`JSON报告 |
| 录制 | `--output`默认data/recordings；`--subject --session`匿名ID；`--consent-keypoints --consent-reference`授权关键点及匿名依据；`--save-raw --consent-raw`分别申请/授权原始RGB存储；`--prompt-seed`默认42 |
| 记录检查/注释 | `--recording`记录目录；`--allow-incomplete`只允许检查不完整记录；`--labels`手工注释JSON；`--edit-labels`已有注释修订；`--annotator`默认annotator-local；`--hand-reference`默认manual-hand-1 |

collect需要subject、session、consent-keypoints、consent-reference；真实输入需感知model；save-raw还需要单独consent-raw。source=video需video。inspect/recover/annotate需recording；headless注释需labels。默认公开数据训练不需要自录，这些不是train的前置步骤。

## 6. `scripts/*.py`兼容入口与独立脚本

调用语法为`uv run python scripts/文件名.py ...`。这些都是当前真实存在文件，不包括scripts/README.md中的未来计划文件。

| 脚本 | 对应入口/用途；参数 |
| --- | --- |
| `train.py` | 底层训练，见下面；**不是**日常train的同一参数接口 |
| `evaluate.py` | 等同handshake-evaluate；需要score/classify/calibrate/diagnose/events子命令，**不是**无参数日常evaluate |
| `build_handoff.py` | 等同handshake-handoff |
| `camera_probe.py` | 等同handshake camera_probe，参数见5.9 |
| `collect.py` | 等同handshake collect，参数见5.9 |
| `annotate.py` | 等同handshake annotate，参数见5.9 |
| `perception_probe.py` | 等同handshake-perception-probe |
| `probe_perception.py` | 旧感知探针接口：可选`--model --expected-sha256 --timeout-seconds`默认20秒，必需`--report`；新调用建议用perception_probe.py |
| `replay_public_policy.py` | 等同handshake-public-replay |
| `replay_observations.py` | 等同handshake-observation-replay |
| `replay_model_session.py` | 等同handshake-model-session |
| `build_features.py` | 草案ObservationRecord JSON生成旧NPZ：位置参数`records manifest output`；必需`--split train/val/test`；可选`--stats`，val/test实际要求训练stats |
| `check_environment.py` | 依赖导入/CPU运算子进程探针：可选`--output`、`--timeout`默认45秒；不开相机、不训练 |
| `check_native_shadow.py` | 临时未训练资产的Linux影子UI验证：可选`--flutter`默认FLUTTER_BIN或flutter、`--model-type mlp/tcn`默认mlp、`--timeout`默认180秒 |
| `probe_stream_cache.py` | 合成容量探针：必需`--output`；可选`--frames-per-video`默认10001，允许10001到25000；`--with-training`显式启用合成训练阶段，默认仅缓存 |
| `smoke_models.py` | 合成模型结构/记忆能力诊断，**会做小型神经训练**：可选`--output --seed`默认42、`--steps`默认100；不是正式数据训练 |
| `smoke_interaction.py` | 合成交互状态机冒烟，不注入真实动作：可选`--output` |
| `run_native.py` | 底层原生启动，参数见下面 |

### 底层 `scripts/train.py` / `handshake train`

必需：`--train-cache DIR --val-cache DIR --output DIR`。可选：`--training-config FILE`训练YAML、`--resume FILE`同run可信last.pt、`--stop-after-epoch N`在绝对第N轮后暂停，主要验证续训。无dry-run选项，不自动关联短命令latest，不自动导出。

训练YAML字段见`algorithm/defaults/trainer.yaml`：`format_version model_type optimizer learning_rate weight_decay batch_size max_epochs patience min_delta grad_clip_norm seed device threads sampling class_weighting scheduler lr_patience lr_factor min_learning_rate checkpoint_metric shuffle_labels_diagnostic`。字段含义与第3节对应，额外shuffle_labels_diagnostic是打乱训练标签的健康诊断，不用于正式成绩；format_version固定trainer-v1，checkpoint_metric固定val_macro_f1。**底层默认device是cpu**，不是短命令配置的auto；epochs在YAML中叫max_epochs，model叫model_type。

### 底层 `scripts/run_native.py`

全部参数：`--flutter`默认FLUTTER_BIN或flutter；`--bundle FILE`；`--camera N`默认0；`--perception-model FILE`；`--model-package DIR --model-sha256 HASH`；`--calibration FILE --calibration-sha256 HASH`；`--shadow-session`；`--allow-synthetic-model`。

flutter/bundle选择启动方式；传模型需成对model-package/model-sha256，并需感知perception-model；校准路径/哈希成对且需模型。shadow-session需匹配资产，合成模型只能用于shadow-session。仍默认停止、相机由UI显式打开。这里没有短命令app的verify-only，且模型路径选项名字不同。

## 7. 帮助与推荐工作流

```bash
uv run train --help
uv run handshake-evaluate score --help
uv run python scripts/train.py --help

# 普通训练：不需要先运行所有诊断脚本
uv run doctor
uv run train --dry-run
uv run train --class-weighting sqrt_inverse --seed 42
uv run evaluate
# 选择模型后才校准；意外中断可resume，不是每次都要resume
uv run calibrate
```

这些命令不是必须逐个运行的流水线。高级探针、回放、压力测试和交付工具按任务选择。基线/改进对照以及不能直接拿本机新run与旧缓存训练结果作严格归因的原因，见[训练改进说明](training_improvements.md)。
