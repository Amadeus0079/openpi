# OpenPI Pi0.5 on Ascend CANN 9.0

This assembly is the OpenPI variant from `DrivingSDK/model_examples/Pi-0.5/openpi`, not the separate LeRobot Pi0.5 recipe. It pins OpenPI to commit `15a9616a00943ada6c20a0f158e3adb39df2ccac` and applies its NPU training, tokenizer, and LIBERO configuration changes. The data loader retains `lerobot.common.datasets`, which matches the LeRobot commit pinned in `pyproject.toml`; the recipe's newer `lerobot.datasets` import does not exist in that pinned version.

The local branch is `feat/ascend`. The environment is `workplace/openpi/.venv`, managed by uv with an independent Python 3.11 interpreter. It uses PyTorch `2.7.1+cpu`, torchvision `0.22.1+cpu`, and `torch_npu 2.7.1.post4` with CANN 9.0.0. JAX runs on CPU for preprocessing and checkpoint conversion; PyTorch runs training on Ascend. TorchCodec is pinned to `0.4.0` for [PyTorch 2.7 compatibility](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).

Install or rebuild from the repository root (`workplace/openpi`):

```bash
bash scripts/ascend/setup_uv.sh
source scripts/ascend/openpi-ascend-env.sh
```

`setup_uv.sh` installs from `uv.lock` and applies the required OpenPI Transformers replacements. It uses the project package indexes in `scripts/ascend/uv.toml` and copy installation mode so the replacements do not modify uv's shared package cache. Run this setup script again after reinstalling Transformers. Python installations and the uv cache are kept under `zichen/.cache/uv`, independently of Conda.

The environment script loads the shared runtime settings from `zichen/scripts/ascend_env.sh`, activates `.venv`, and exports the model paths below. CANN, the NPU driver, and system FFmpeg libraries remain system dependencies.

From the repository root (`workplace/openpi`):

```bash
source scripts/ascend/openpi-ascend-env.sh
scripts/ascend/prepare_pi05_weights.sh
export LIBERO_DATASET_REPO_ID=/absolute/path/to/libero_10_no_noops_1.0.0_lerobot
python scripts/compute_norm_stats.py --config-name pi05_libero
scripts/ascend/train_pi05_libero.sh
```

The raw JAX checkpoint, PaliGemma tokenizer, Hugging Face cache, and converted PyTorch checkpoint are all rooted at `/inspire/sj-ssd3/project/robot-reasoning/public/models/pi05_openpi`. `scripts/ascend/prepare_pi05_weights.sh` can only run after the raw checkpoint download completes.

For multiple NPUs, set `NPROC_PER_NODE` to the exposed device count before running the launcher. The tool sandbox hides `/dev/davinci*`; NPU validation and training must run outside that sandbox. Loading the shared Ascend runtime settings outside the sandbox exposes eight healthy 910B2C NPUs.

Check device access in the activated environment:

```bash
python -c 'import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count()); print((torch.ones(2, device="npu:0") + 1).cpu())'
python scripts/train_pytorch.py pi05_libero --help
```

Use `uv run --no-sync` if launching through uv after setup, so launch does not install additional development packages or reinstall the patched Transformers files. Dataset preparation and normalization statistics are required before a full LIBERO training run.

Validated on 2026-10-14 outside the tool sandbox: dependency consistency, byte-for-byte Transformers replacements, training and normalization CLI imports, local tokenizer loading, LeRobot video decoding through TorchCodec, and tensor calculations on all eight NPUs. A full Pi0.5 model loaded the local pretrained weights and completed a finite-loss forward pass, backward pass, and AdamW parameter update on one NPU with synthetic inputs (batch size 1, action horizon 2, prompt length 8, bfloat16, gradient checkpointing, sampling compilation disabled). This verifies the environment and model computation; a full LIBERO training run and distributed training have not been validated.
