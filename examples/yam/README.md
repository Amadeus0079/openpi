# YAM pouring: MCAP to LeRobot and Ascend FSDP fine-tuning

This example uses the task instruction:

> Pour the water from the right cup into the left cup.

Run all commands from the OpenPI repository root. Ascend commands must run outside
the Codex tool sandbox, where the eight NPU device nodes are visible.

```bash
bash scripts/ascend/setup_uv.sh
source scripts/ascend/openpi-ascend-env.sh
uv sync --locked --no-dev --group yam
python examples/yam/convert_mcap_to_lerobot.py
python examples/yam/compute_norm_stats.py
bash examples/yam/train_fsdp.sh
```

The converter recursively reads MCAP files under `../../datasets/yam` and writes
LeRobot v2.1 to `data/yam/pour_water`. Raw MCAP files are never modified or uploaded.
`--task.raw-dir`, `--task.dataset-dir`, and `--task.prompt` override the conversion
defaults. A changed dataset location must also be reflected in the training data
configuration. The local default config is `pi05_yam`.

The source format uses JSON envelopes and base64 H.264 access units. Conversion
uses outer `source_time_ns`, maintains one continuous decoder per camera and
episode, and samples at 30 Hz with the most recent preceding measurement.
Required streams are `/left/state`, `/right/state`, `/action/accepted`, and
`/cameras/{main,left,right}/rgb`. Requested actions are not used. The 14-dimensional
state/action order is left six joints, left gripper, right six joints, right
gripper. Native RGB resolution is preserved. The conversion report records
stream timing, sample age, repeated frames, camera gaps, accepted limited actions,
and source/output manifests. Partial conversions remain in `.incomplete` and are
never accepted for training; an explicit `--force-rebuild` can retry them.

Training maps the main camera to `base_0_rgb` and the two wrist cameras to their
matching OpenPI views. It subtracts the current joint positions from future
joint actions, preserving both absolute gripper values. No ALOHA calibration,
gripper geometry conversion, or joint sign changes are applied. Normalization
covers every converted frame and episode-clamped 16-step action chunk. The
normalization provenance must match the converted dataset before training starts.

`pi05_yam` uses the full pretrained Pi0.5 model, with no frozen parameters or LoRA:

| Setting | Value |
| --- | --- |
| Device/backend | 8 Ascend NPUs, HCCL |
| Sharding | PyTorch FSDP `FULL_SHARD` |
| Batch | 32 per rank, 256 global, one batch per optimizer step |
| Optimizer steps | 5000 |
| Action horizon / prompt length | 16 / 200, including discretized robot state |
| Precision | BF16 compute, FP32 master parameters and optimizer states |
| Memory | Gradient checkpointing in vision and joint Gemma computation |
| Learning rate | 100-step warmup, cosine decay from 2.5e-5 to 2.5e-6 |
| Checkpoint interval | 1000 steps and the final step |

The FSDP entrypoint is `scripts/train_pytorch_fsdp.py`. It wraps the whole model:
OpenPI's joint Gemma prefix/suffix computation calls decoder submodules directly,
so wrapping only `GemmaDecoderLayer` would bypass FSDP's parameter-gather hooks.
Mixed precision and checkpointing preserve parameter, gradient, and optimizer
sharding while keeping all parameters trainable.

Logs, run metadata, metrics, progress, and checkpoints are written under
`checkpoints/pi05_yam/yam_pour_water_fsdp_5k`. A checkpoint contains standard
`model.safetensors`, full optimizer state, per-rank RNG state, normalization assets,
and a completion marker published after every rank finishes saving.

Checkpoints store one canonical entry for shared embedding/output weights, so
OpenPI's standard safetensors loader can read them. Weight loading converts the
model to FP32 before reading the checkpoint, preserving master-weight precision
on resume. Resume with:

```bash
bash examples/yam/train_fsdp.sh --resume
```

Use a different `EXP_NAME` for a separate experiment. For a short end-to-end check
with the same eight devices and per-rank batch size, for example:

```bash
EXP_NAME=yam_fsdp_smoke bash examples/yam/train_fsdp.sh --num-train-steps 2 --save-interval 2
```

Conversion and YAM IO are adapted from
[OCL-openpi `feat/mcap`](https://github.com/YushunXiang/OCL-openpi/tree/feat/mcap),
reference commit `e517a3a223daf2d1683aa8c172cc3ed23473c7a3`.
