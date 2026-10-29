# YAM-2 hybrid π0.5 迁移与验收记录（2026-10-19）

## 当前状态

- Server：`/home/ps/zichen/workplace/openpi`，`yam/hybrid`，基于官方 `215abfb217dbac7d5f1273282331b9b1866c0479`。
- Client：`/home/ps/zichen/workplace/YAM-Client`，`feat/pi-client`，基于 `82a5cb3`；已按用户要求从 `/home/ps/workspace/YAM-Client` 迁移。工作树与未提交改动保留，Git 工作树/real2sim 子模块元数据已修复，独立环境按锁文件重建并验证。
- 代码未提交/push。原 OCL、RPent、YAM/main 和昇腾训练源码未改写。vLLM 已正常停止。
- 三个 step 10000 checkpoint 已全部迁到 `/nas/data/yam-ckpts/hybrid_pi05/<dataset>/10000`，全部逐文件大小/SHA256 校验通过，使用暂存目录核验后发布，排除 optimizer/RNG。三份权重共 43,401,497,616 字节，另有完整资产、run、完成标记与清单。
- 当前 `yam_only` server 监听 `127.0.0.1:8001`。按用户在场授权，原 YAM/main daemon 已用原 profile 恢复，代码/配置指纹与先前一致；最终状态 `SAFE_HOLD`、hold_confirmed=true、硬件健康=true、session_id=null。
- 本轮仅执行 shadow，没有归零、没有发送策略运动指令；daemon 初始化沿用原有流程。三组现场 shadow 都在 IK 检查处拒绝，执行步数为 0，尚未达到可执行 15 步轨迹的现场验收标准。

本目录下现有 5 个 Git 仓库（robot_calibration、YAM-Client、其 real2sim 子模块、openpi、agent-as-policy）身份已设置为 `Zichen Zhang <chinazzcxhl@gmail.com>`。Client 使用工作树专属配置；原 YAM/main、YAM-v2 与全局 Git 身份未改变。

## 本轮实测结果

每模型对 checkpoint 自带回放观测做 3 次真实 CUDA/WebSocket 推理，并对保存的预测检查前 15 步；另各做 1 次现场 RGB+实测关节的 shadow。下列样本数用于接口和运动学验收，不是任务成功率。

| 模型 | SHA256/严格 CUDA 加载/有限 [30,14] | 回放真值 15 步 IK | 三次预测前缀 IK | 现场 shadow |
| --- | --- | --- | --- | --- |
| yam_only | 通过 | 通过 | 3/3 通过 | 右臂第 9 点拒绝，位置残差约 0.239 mm |
| yam_source_top10 | 通过 | 通过 | 0/3（限位处拒绝） | 左臂第 1 点拒绝，位置残差约 0.306 mm |
| yam_global_top80 | 通过 | 通过 | 3/3 通过 | 左臂第 5 点拒绝，位置残差约 0.531 mm |

三组 state 预处理与训练保存值的误差均为 0；动作反归一化最大误差分别约 4.94e-8、7.19e-8、7.29e-8。RGB、letterbox、camera mask、归一化后的 state mask 和 200 token 检查通过。

回放观测请求耗时（ms）：yam_only 413/119/117；source_top10 318/116/115；global_top80 126/118/118。第三组在此之前先执行了现场 shadow，因此第一项并非冷启动。现场 shadow 推理约 353/174/444 ms，IK 至拒绝约 38/36/39 ms。

相机预检取得五组同步三路 640×480 HWC uint8 RGB，最大时间偏差约 8.4 ms；serial、分辨率、fps 和 rotation 与训练资产一致，现场 left/right 映射为 left_wrist/right_wrist。

100 个随机姿态、近邻种子（σ=0.02 rad）的 IK 全部收敛，位置/旋转残差均低于 1e-4；中位 0.414 ms、P95 1.049 ms。此前 100 姿态的独立 URDF/MuJoCo FK 最大位置差为 4.4e-16 m。这些数值不构成碰撞或现场标定验收。

## 当前运动学阻碍

执行阈值保持位置 1e-4 m、旋转 1e-4 rad，使用 checkpoint URDF 的名义关节限位和 30 Hz 额定速度。整段 15 点必须全部通过才会提交。

三组现场拒绝均为当前连续解分支中的 joint2 下限：独立无约束数值解分别需要约 -0.00371、-0.00471、-0.00812 rad，而 URDF 下限为 0。独立有约束最小二乘保留相同位置残差。source_top10 的三个离线预测也在该下限处拒绝；将诊断迭代数从 200 增至 2000 后未改善。

原 daemon 的 joint2 下限为 -0.15 rad，来源是 `i2rt/robots/get_robot.py` 对 XML 限位统一增加 ±0.15 rad buffer，不是本轮实测验收的机械极限。没有据此放宽 IK 限位，也没有提高残差阈值来接受这些段。下一步需要先核对 joint2 实际零位与有效限位、确认部署起点，再决定限位/可达性处理及归零和短程运动验收。

标定资产仍保留 draft/candidate 状态。用户确认相机与基座未移动；本轮没有独立完成外参、碰撞或任务成功率验收。

## 资产、日志与复现

OBS 前缀：`obs://sai.liyl/zichen/openpi-hybrid-checkpoints/hybrid-three-20261018/`。早期桶满已由用户解决；发布脚本检查训练记录的关键源码 hash，上传源码快照和 inference-only 文件，清单最后发布。原训练文件不改写。

昇腾发布脚本（仓库外）：
`/inspire/sj-ssd3/project/robot-reasoning/public/zichen/workplace/hybrid-integration-artifacts/yam2-migration-20261019/ascend_publish.py`

YAM-2 server 的 `.migration-logs/` 保存下载/发布状态、源码对照、预处理、CUDA 推理、daemon 恢复和 shadow 日志。Client `outputs/pi-client/` 中，`replay-<dataset>-20261019`、`server-<dataset>-20261019`、`shadow-<dataset>-20261019` 保存逐模型的观测、预测、IK、失败诊断及最终 daemon 状态。shadow 中也在 IK 前保存原始 EE 输出和实测种子。

机器可读汇总：server 的 `docs/hybrid-validation-20261019.json`。完整权重 SHA256、数据 identity、耗时、通过数和产物路径均在其中；checkpoint 的 inference-manifest.json 是逐文件校验依据。

重复下载会核验已有目标，拒绝覆盖不同清单或损坏的正式目标：

```bash
cd /home/ps/zichen/workplace/openpi
.venv/bin/python scripts/fetch_hybrid_checkpoints.py
```

## Server

```bash
cd /home/ps/zichen/workplace/openpi
bash examples/yam/serve_hybrid.sh yam_only 8001
```

可选 yam_source_top10 或 yam_global_top80。切换前正常结束现有 server；每次只加载一个模型。沿用 `scripts/serve_policy.py`、配置 `pi05_yam_hybrid`，Torch CUDA，JAX CPU，关闭 torch.compile。

输入为三路 HWC uint8 RGB、state[22]、state_valid[22]、camera_valid[3]、prompt；输出未归一化相机锚点 actions[30,14]。reset 返回 `{"status":"ok"}`，锚点由 client 管理。握手绑定权重 SHA256、数据 identity、动作版本和维度。

训练模型、tokenizer、模型预处理、策略及 Transformers 补丁与源码快照一致；只对配置注册、部署资产验证、host/device 和 reset 做必要接入。server 启动验证完整清单、训练模型配置、统计和契约；tokenizer 只读 checkpoint 资产。独立 Python 3.11 环境使用 Torch 2.7.1+cu126、Transformers 4.53.2、NumPy 1.26.4；实际依赖见 `docs/hybrid-cuda-runtime.txt`。

## Client

```bash
cd /home/ps/zichen/workplace/YAM-Client/gello_software
scripts/pi_client --mode replay \
  --checkpoint /nas/data/yam-ckpts/hybrid_pi05/yam_only/10000
scripts/pi_client --mode shadow \
  --checkpoint /nas/data/yam-ckpts/hybrid_pi05/yam_only/10000
```

replay 不访问硬件或 server；shadow 只读现有 daemon 与相机，经过真实 server、坐标转换和 IK，不创建运动会话。默认任务文字 `pour water`。也可在该独立环境使用原生 `yam pi-client`。

真实 server 与回放预测检查：

```bash
.venv/bin/python scripts/check_pi_server.py --checkpoint CHECKPOINT --output NEW_OUTPUT_DIR
.venv/bin/python scripts/check_pi_predictions.py --checkpoint CHECKPOINT --predictions NEW_OUTPUT_DIR
```

server 的 `scripts/check_hybrid_preprocessing.py CHECKPOINT --output REPORT.json` 检查训练保存的预处理结果。使用 server 自己的 `.venv/bin/python`。

以下运动阶段尚未执行；本轮用户仅授权恢复原 daemon 和 shadow。完成限位/零位核对并取得后续现场运动授权后，先验收自动归零与结束保持：

```bash
scripts/pi_client --mode run --home-only --checkpoint CHECKPOINT
```

再验收单个 15 步 chunk，之后才增加轮长：

```bash
scripts/pi_client --mode run --execute-steps 15 --max-chunks 1 --checkpoint CHECKPOINT
```

run 将双臂六关节归零、夹爪打开到 1，经实测反馈检查后建立 episode 锚点；同一 episode 复用锚点。唯一命令线程 30 Hz 发送轨迹和最后已接受目标，保留 daemon 0.2 秒断流保护，不追赶迟到 tick；推理、IK、相机不发运动指令。未知 ACK 不重发，结束确认保持。归零、推理、IK、执行时长分别记录。

从 checkpoint URDF 构建六关节 MuJoCo 模型，末端为 gripper mount 上的 policy_ee_mount，不采用默认 grasp_site。仅通过原 YAM/main protocol 6 daemon 的 6001/6002 RPC 控制，client 不自动创建 daemon。

Client 独立环境：NumPy 2.2.6、SciPy 1.15.3、MuJoCo 3.10.0、Mink 0.0.13、qpsolvers 4.7.1、quadprog 0.1.13；uv.lock 锁定并通过离线安装及 57 包依赖检查。几何与 wire 代码保留来源和许可。

## 已运行代码检查

Client 新增 9 项测试与既有归零/RPC/控制/迁移回归合计 51 项通过；迁移路径及 IK 结构化错误记录更新后，新 9 项重新通过。Server 11 项 CPU 移植/策略/协议测试和 2 项 CUDA 小模型 FP32/BF16 对照通过；NPU 专用测试未在 YAM-2 执行。真实大模型结果见上表。两端依赖、lint 和 diff 空白检查通过。

尚未验收：三模型可执行的现场 15 步闭环、自动归零、策略运动、碰撞与外参、倒水任务成功率。


## 2026-10-29 核心推理源码归档

本次将 hybrid CUDA 推理实现、既有 CPU/CUDA/NPU 回归测试、权重获取与预处理检查脚本、启动入口及历史验收资料纳入 Git。上文的服务状态和未提交状态是 2026-10-19 的历史快照，不代表当前运行状态。

2026-10-29 在 YAM-2 现有环境重新运行四份 hybrid 回归测试：13 passed、5 skipped；跳过项均为需显式启用的原生 NPU 测试。本次没有重新加载大模型或运行机械臂。`.deployment-20261020/`、`.deployment-20261021/` 保留为本机部署工作区，其中的工作台源码、一次性操作脚本、备份、日志和运行产物不在本次核心推理提交范围内。
