# T29：白名单离线交接包与换机完整性预检

`backend/handoff.py` / `scripts/build_handoff.py` / 安装后 `handshake-handoff` 提供显式输入、不可覆盖的交接目录。当前目标配置为Linux/x86_64及指定Python主/次版本；不把源码zip或wheel叫作独立原生安装器。

## 构建

```bash
uv build --offline --out-dir /tmp/handshake-wheel
.venv/bin/python scripts/build_handoff.py build \
  --core-wheel /tmp/handshake-wheel/handshake_core-0.2.0-py3-none-any.whl \
  --lockfile uv.lock --license-file LICENSE \
  --code-revision "$(git rev-parse HEAD)" \
  --review-reference YOUR_PUBLIC_RELEASE_REVIEW \
  --output /tmp/handshake-handoff-v1
```

源码revision是构建者的声明；不能代替文件哈希或可信构建证明。请从已检查、已提交的代码构建wheel，再记录对应revision。结果返回 `manifest_sha256`，通过独立可信渠道交给接收方。

可选输入：

- `--wheelhouse DIRECTORY`：目录只允许wheel文件；逐个检查ZIP路径、重复项、加密/符号链接、大小和METADATA。不会自行联网下载依赖。
- `--native frontend/native/build/linux/x64/release/bundle`：只接受符合Linux Release布局的显式目录；必须有ELF主程序、libapp、Flutter库和ICU数据，其余仅接受已列出的运行时库、字体、着色器、图片和许可资产后缀。拒绝隐藏文件、会话HTML、原始CSV、密钥文件等，不扫描整个项目。
- `--model MODEL_DIRECTORY --model-sha256 TRUSTED_SHA --model-license LICENSE_REVIEW`：只复制T16固定清单，校验带外SHA和每个文件。模型格式/数值一致性仍须在接收端通过真正的模型加载器验证；这里**只验证字节完整性，不运行模型**。
- `--python-version 3.12`：指定接收端主/次版本。默认使用构建进程Python版本。

所有输入都只读。文件先拷贝、检查修改时间/大小/哈希、设置明确执行位，最后发布 `manifest.json`。单文件512MiB、总计2GiB、10,000文件，目录扫描及wheel展开也有预算。输出已存在就拒绝；中断目录无有效清单，不得当作完整发布。

## 接收方预检

```bash
.venv/bin/python scripts/build_handoff.py inspect \
  --input /tmp/handshake-handoff-v1 --sha256 TRUSTED_MANIFEST_SHA
```

预检验证完整文件库存、路径和权限、各文件SHA、wheel身份、模型原始SHA以及声明的目标OS/架构/Python版本；列出当前环境的核心依赖版本，但不导入torch、打开相机、运行安装器或启动原生程序。

**完整性通过不等于离线环境已准备好**：

- manifest固定记录 `dependency_resolution: not_verified`、`runtime_bundled: false`、`launch_verified: false`，禁止篡改为已验收。
- 未内置Python，也未自动收齐NumPy/Pydantic/PyYAML及传递依赖。给出的wheelhouse可能不完整，或平台/ABI不符。
- `uv.lock`只作为审查参考；任意提供的wheelhouse没有自动被证明完全复现该锁文件。正式发布还需锁定依赖解析和第二台机器安装验证。
- 先在新虚拟环境中运行 `pip install --dry-run --ignore-installed --no-index --find-links KIT/wheels CORE_WHEEL`。仅在离线解析计划经过审查后再显式安装。缺wheel是阻塞，不得静默联网。
- 原生GTK、图形栈等系统依赖、动态库兼容性、相机权限、签名/分发许可都没有被哈希检查验证。

交接包内 `OFFLINE_README.md` 给出安装前检查、无相机/无训练合成冒烟、清除遗留资产环境变量、显式启动、回退和删除范围。不会替用户安装或运行任何包内可执行文件。

## 隐私与授权

收集范围为显式提供的wheel、锁文件、许可证，以及可选已审查原生目录/模型包。路径写成相对路径，不记录开发者主目录。不会自动收集会话HTML、视频、训练原始数据或身份表。

白名单不是内容DLP或法律审查：wheel或模型元数据内部仍可能含私有内容，图片也无法靠扩展名判断隐私；`review_reference`和模型许可引用必须对应真实人工审查。第三方SDK/字体/模型/数据的授权不因装入交接包而自动扩大。

## 验证

29项新测试：完整性/不可覆盖、路径穿越、隐藏文件、坏wheel/重复成员、额外/缺失/变更文件、符号链接、错误执行位、重新哈希的私有文件、伪造运行声明、模型SHA和许可前置、平台不匹配、无安装/无torch导入。模型字节测试明确使用不可推理的格式样例，验证工具不会把“哈希正确”冒充可用模型。

```bash
.venv/bin/python -m pytest tests/unit/test_handoff.py
```

广域非训练回归636项通过，4项明确排除；仍排除全部训练/梯度用例。正式跨机安装、依赖锁完全复现、模型卡实测指标、签名安装器及T24/T28最终发布签认仍待验收。
