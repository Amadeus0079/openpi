# OpenPI 推理控制台

控制台现由仓库中的 `openpi.control` 管理，入口为 `python scripts/control_console.py`
或安装项目后的 `openpi-control`。无需 Node 服务或前端构建。原 YAM-2 工作台的三视角布局、
EE 动作曲线和关节计划曲线已迁入 `src/openpi/control/static/`；任务和模型不再使用机器上的日期部署目录。

当前内置后端为 **YAM-1 保存支持回放**：实际模型推理、双 base / 原生 mount 映射和离线 IK。
页面明确显示保存帧；没有实时相机、daemon、CAN、归零或运动发送入口。
原 YAM-2 现场控制服务不受本次安装影响，不能将此后端的回放验收解释为真机控制验收。

## 启动与选择权重

在完整 OpenPI 环境中运行：

```bash
python scripts/control_console.py
```

默认根目录为仓库的 `checkpoints/`，默认页面为 <http://127.0.0.1:8765/>。
`checkpoints` 可以整体软链到数据盘，内部训练目录也可以是软链。扫描去重并避免软链循环。
没有 checkpoint 时页面仍可启动，新权重落盘后会自动出现，也可点击“刷新 checkpoints”。

目录按 **训练任务 / 配方 → 数字步数** 分组，支持任意中间层级，例如：

```text
checkpoints/
  pi05_yam_hybrid/pour_water/1000/
  pi05_yam_hybrid/pour_water/4000/
  yam1-cross-embodiment-20261028/M0/4000/
  yam1-cross-embodiment-20261028/M1/4000/
```

下拉任务分别显示 `pi05_yam_hybrid/pour_water`、`yam1-cross-embodiment-20261028/M0` 等相对路径，
步数按数值从新到旧排列。这里的“任务”是训练目录；一个模型内 cup / tumbler 等保存输入在“目标支持示例”里另选。

发现 PyTorch `model.safetensors` 或 JAX `params`，并要求存在 `assets/`。
发布清单存在时会逐文件核对存在性、大小和目录步数，完成标记存在时也核对步数。
隐藏的 `.4000.download` 等临时目录不会列为可选权重。未完成目录显示在刷新按钮下方的诊断中。
列表里的“文件齐全”不表示推理已通过；加载时重新执行 SHA256、模型配置、资产及握手身份校验。
选择的文件若被替换或删除，必须重新选择和加载，不会继续使用旧服务假冒新模型。

其他 OpenPI 模型可以在同一页面发现、通过 `--config` 指定训练配置加载 WebSocket 服务，
但当前保存回放和 YAM-1 IK 后端只接受 `yam1-hybrid-inference-v1`，不把其他模型输出交给 YAM-1 适配器。

## YAM-1 配置

M0 示例；所有本机路径通过命令参数提供，不写入源码：

```bash
python scripts/control_console.py \
  --checkpoint-id yam1-cross-embodiment-20261028/M0/4000 \
  --output /path/to/data-disk/openpi-control \
  --ik-python /path/to/mink-environment/bin/python \
  --support-poses /path/to/approved-review-frames.json \
  --yam-owner /path/to/YAM/gello_software \
  --model-lock /path/to/shared-model.lock
```

- `--checkpoints`：可覆盖默认的仓库 `checkpoints`。
- `--ik-python`：使用独立的 MuJoCo/Mink 环境，避免改动推理环境。该环境需 NumPy、SciPy、MuJoCo、Mink 0.0.13。
- `--support-poses`：经确认的目标支持姿态 JSON，格式为 `frames` 列表，每帧含 `task` 和
  `joint_configuration.left_joint1…6/right_joint1…6`。当前关节由保存的 EE state 做离线 IK 重建，
  不是实时关节反馈。未提供时只做原生 mount 坐标映射，不伪造关节计划。
- `--yam-owner`：仅当用户选择“本机采集控制器限制”时，通过 AST 只读解析该机的速度规则；不导入或启动控制器。
- `--model-lock`：与其他离线验收器共用的 GPU 排他锁。模型加载和整个存活期间持锁，退出后释放。
- `--memory-fraction`：默认限制单模型 CUDA 分配为显存的 52%；不关闭其他用户的 GPU 进程。

操作顺序：选择训练任务和步数 → 应用选择 → 启动 server → 选择支持示例 → 只读推理。
支持每轮 1–20 次推理、采用 horizon 1–30、IK 前缀 1–H。模型始终输出 30×14，时间基准为 30 Hz。
默认使用 URDF 速度限制。显式选择采集速度模式后需要重新应用参数；IK 拒绝会保留原因，不放宽约束。

每次实际预测写入输出目录的 `runs/<run-id>/`，包含原始动作、加载身份、原生 EE 目标、
成功的关节计划或 IK 拒绝诊断。曲线只显示实际模型输出与已求解计划，没有虚构 ACK 或实测反馈。

新版 YAM-1 资产同时接入标准 `create_trained_policy`，所以 `scripts/serve_policy.py` 也能加载，
包括训练 Gemma 快照、SHA 绑定的 v3 标定和 domain-I/O 模型的 `yam1` 头路由。
组别和训练步数从清单读取，不把 M0–M3 或 4000 固定为选择列表。

## 连接与退出

HTTP 和模型 WebSocket 都仅监听回环地址。远程访问通过 SSH 转发，例如：

```bash
ssh -N -L 18765:127.0.0.1:8765 YAM-1
```

然后打开 <http://127.0.0.1:18765/>。“关闭模型”只停止本控制台创建的子进程。
最后一个页面离开或失联约 10 秒后会取消本轮并关闭所属模型，刷新页面可接续会话。
退出后端也清理子进程；Linux 另有父进程死亡信号保护。不读取历史 PID 文件来杀进程。

## 开发检查

```bash
PYTHONPATH=src pytest src/openpi/control --confcutdir=src/openpi/control
node --check src/openpi/control/static/workbench.js
```

这组测试覆盖目录分层、数字排序、软链循环、临时和不完整发布过滤、动态增加/删除、
选择失效、HTTP 会话与不可用操作、参数范围、模型锁取消及已退出子进程保护。
真实权重、CUDA 和离线 IK 需要另在目标机器通过页面验证。
