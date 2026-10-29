# OpenPI 推理控制台

控制台现由仓库中的 `openpi.control` 管理，入口为 `python scripts/control_console.py`
或安装项目后的 `openpi-control`。无需 Node 服务或前端构建。原 YAM-2 工作台的三视角布局、
EE 动作曲线和关节计划曲线已迁入 `src/openpi/control/static/`；任务和模型不再使用机器上的日期部署目录。

内置 YAM-1 现场模式与保存支持回放两种输入。现场模式包含三路实时相机、训练示范首帧叠图、
只读起点预检、慢速到示范起点和真机策略推理；保存回放用于不连接硬件的离线诊断。
现场接口复用已运行的本机 protocol-6 daemon，不启动、重启或归零 daemon。
安装后的功能可用性与测试范围分别记录；只读验收不代表运动或倒水成功率验收。

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
- `--yam-owner`：校验现有 daemon 的代码归属，并通过 AST 只读解析本机采集速度规则；不启动控制器。
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

## 实时叠图、回位与策略

在上述命令后追加本机参数即可启用现场模式（默认输入随之变为现场）：

```bash
  --live-profile /path/to/.config/yam/profiles/default.yaml \
  --client-root /path/to/YAM/gello_software \
  --recordings-root /path/to/TeleopData \
  --live-python /path/to/camera-mink-environment/bin/python
```

`--live-python` 使用独立 NumPy 2 环境，需本机相机 SDK、pyrealsense2、pyzmq、PyYAML、MCAP/PyAV、
msgpack、websockets、Pillow、SciPy、MuJoCo/Mink。模型服务仍使用 OpenPI 环境。
profile 指定回环 RPC、左右臂和三相机；相机序列号、尺寸和旋转与 checkpoint 内 YAM-1 标定核对。
所有预览和策略观测共享同一个 CameraRig，避免重复占用设备。

1. 在“观测与示范对齐”选择现场输入和训练示范，点击“加载示范”。参考仅来自当前 checkpoint
   的 `target_support`，按原始 MCAP SHA 校验；不会自动读取开发集或保留测试集。
2. 调整参考不透明度，核对三视角与物体布局。叠图来自原始首帧；回位目标来自同一示范的首条
   实测原生关节反馈，不使用由 EE 做 IK 重建的近似位置。
3. 点击“只读预检起点”检查当前反馈、daemon 限位和慢速关节衔接，显示预计耗时，不发送命令。
   确认场景后，“慢速到示范起点”才可用；该按钮会实际运动，臂关节目标速度不超过 0.15 rad/s。
   回位按已有标定 daemon 的原生关节限位校验；策略 IK 另保留 URDF 名义限位。
4. 启动所选 server 后，“只读推理”使用真实相机和关节、固定本轮锚点、输出 action 与 IK，
   全程不创建运动会话。现场只读单次决策用于诊断。
5. “开始真机推理”会实际运动。需先确认叠图场景，或明确勾选“跳过对齐，从当前姿态开始”。
   每轮从新鲜观测推理，执行所选前缀，记录实际 ACK 与关节反馈。最多 1200 轮，可随时停止。

模型固定输出 30×14；30 Hz 模型时间不变。目标更新频率支持 1/2/5/15/30/60/90/120 Hz：
低于 30 时延长各模型目标的保持时间，高于 30 时按原时间插值。RPC 指令流始终至少 30 Hz，
等待推理/IK 时重复最后已确认目标；不把未知 ACK 的命令重发。运动结束和异常退出都请求并确认 SAFE_HOLD。
回位计划 60 秒失效，反馈变化超过 0.02 rad 需重新预检。相机超过 1 秒或页面心跳超过 3 秒即中止运动。
停止按钮是软件停止并保持入口，不能替代硬件急停。

FK/IK 为数值模型，不包含碰撞检测。操作前需由现场人员核对空间与场景；功能恢复不等于现场运动验收。

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
实时路径另覆盖原始首关节/SHA绑定、仅使用支持集、八档时基、慢速回位限位、
只读路径不构造 CommandWriter、推理后重新读取 IK 种子、相机/页面过期及未知 ACK 不重发并保持。
真实权重、CUDA、相机和 IK 需要另在目标机器通过页面验证；运动测试需现场授权。
