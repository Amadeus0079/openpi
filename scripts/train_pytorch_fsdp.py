"""Native Ascend/HCCL FSDP full fine-tuning of OpenPI, with resumable checkpoints."""

import dataclasses
import datetime
import gc
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullOptimStateDictConfig
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
import torch_npu  # noqa: F401 -- registers the native npu device and HCCL backend

from openpi.data.yam_mcap import require_complete
from openpi.data.yam_spec import McapTaskSpec
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.shared import normalize
from openpi.training import config as configs
from openpi.training import data_loader
from openpi.training.pytorch_weights import load_fp32_weights
from openpi.training.pytorch_weights import save_fp32_weights
from openpi.training.pytorch_weights import shared_weight_aliases


def checkpoint_path(config):
    candidates = [p for p in config.checkpoint_dir.glob("[0-9]*") if (p / "complete.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No complete checkpoint in {config.checkpoint_dir}")
    return max(candidates, key=lambda p: int(p.name))


def save_checkpoint(model, optimizer, step, config, data_config, run):
    """All ranks participate in full-state collection; rank zero publishes atomically."""
    rank = dist.get_rank()
    destination = config.checkpoint_dir / str(step)
    staging = config.checkpoint_dir / f".incomplete-{step}"
    if rank == 0:
        staging.mkdir(exist_ok=False)
    dist.barrier()
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        weights = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
    torch.save(
        {
            "torch": torch.get_rng_state(),
            "npu": torch.npu.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        staging / f"rng-rank{rank}.pt",
    )
    if rank == 0:
        save_fp32_weights(weights, staging / "model.safetensors", run.get("weight_aliases", {}))
        torch.save(optimizer_state, staging / "optimizer.pt")
        normalize.save(staging / "assets" / data_config.asset_id, data_config.norm_stats)
        (staging / "run.json").write_text(json.dumps({**run, "step": step}, indent=2))
    del weights, optimizer_state
    dist.barrier()
    if rank == 0:
        (staging / "complete.json").write_text(json.dumps({"step": step, "world_size": dist.get_world_size()}))
        staging.rename(destination)
        (config.checkpoint_dir / "progress.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "target_steps": config.num_train_steps,
                    "checkpoint": str(destination),
                    "status": "complete" if step == config.num_train_steps else "running",
                },
                indent=2,
            )
        )
        print(f"CHECKPOINT step={step} path={destination}", flush=True)
    dist.barrier()
    gc.collect()
    torch.npu.empty_cache()


def train(config: configs.TrainConfig):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != config.fsdp_devices or config.batch_size % world_size:
        raise ValueError("Launch world size must equal fsdp_devices and divide the global batch size")
    if config.wandb_enabled or config.ema_decay is not None:
        raise ValueError("This full fine-tuning entrypoint requires wandb_enabled=False and ema_decay=None")
    # Limit JAX/OpenMP worker thread pools to the CPUs assigned to each rank.
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        subset = cpus[local_rank::world_size][:8]
        if subset:
            os.sched_setaffinity(0, subset)
    torch.set_num_threads(4)
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    dist.init_process_group("hccl", timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    try:
        task = McapTaskSpec()
        conversion = require_complete(task)
        provenance_path = config.assets_dirs / task.repo_id / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        if (
            provenance["dataset_fingerprint"] != conversion["dataset_fingerprint"]
            or provenance["action_horizon"] != config.model.action_horizon
        ):
            raise ValueError("Dataset/normalization provenance mismatch; recompute YAM normalization")
        restore = checkpoint_path(config) if config.resume else None
        if rank == 0:
            if not config.resume and config.checkpoint_dir.exists() and any(config.checkpoint_dir.iterdir()):
                raise FileExistsError(f"Experiment already exists: {config.checkpoint_dir}; use --resume")
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        print(f"RANK {rank}: building full Pi0.5 model", flush=True)
        model_config = dataclasses.replace(config.model, pytorch_compile_mode=None)
        base = PI0Pytorch(model_config)
        weight_path = (restore or Path(config.pytorch_weight_path)) / "model.safetensors"
        # FSDP flat buffers require one storage dtype. Master parameters/Adam states
        # stay FP32, while forward/backward matmuls use BF16 autocast.
        load_fp32_weights(base, weight_path)
        aliases = shared_weight_aliases(base)
        base.gradient_checkpointing_enable()
        base.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        parameter_count = sum(p.numel() for p in base.parameters())
        if not all(p.requires_grad for p in base.parameters()):
            raise ValueError("Full fine-tuning requires every model parameter to be trainable")
        # Gemma's joint prefix/suffix path calls decoder submodules directly.
        # Wrapping GemmaDecoderLayer would bypass FSDP pre-forward hooks. Root
        # FULL_SHARD keeps these manual calls within one valid FSDP forward.
        model = FSDP(
            base,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16
            ),
            use_orig_params=True,
            sync_module_states=True,
            limit_all_gathers=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr_schedule.peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        start_step = 0
        if restore:
            previous = json.loads((restore / "run.json").read_text())
            if (
                previous["dataset_fingerprint"] != conversion["dataset_fingerprint"]
                or previous["world_size"] != world_size
                or previous["global_batch_size"] != config.batch_size
                or previous.get("action_horizon") != config.model.action_horizon
                or previous.get("max_token_len") != config.model.max_token_len
                or previous.get("discrete_state_input") != config.model.discrete_state_input
            ):
                raise ValueError("Resume dataset/world size/batch size/model input differs from saved training")
            full_optimizer = (
                torch.load(restore / "optimizer.pt", map_location="cpu", weights_only=False) if rank == 0 else None
            )
            optimizer.load_state_dict(FSDP.scatter_full_optim_state_dict(full_optimizer, model, optim=optimizer))
            start_step = previous["step"]
        loader = data_loader.create_data_loader(config, framework="pytorch", shuffle=True)
        dataset_config = loader.data_config()
        batches = iter(loader)
        # Replaying data selection restores the exact DistributedSampler position.
        for _ in range(start_step):
            next(batches)
        torch.manual_seed(config.seed + rank)
        np.random.seed(config.seed + rank)
        random.seed(config.seed + rank)
        if restore:
            rng = torch.load(restore / f"rng-rank{rank}.pt", map_location="cpu", weights_only=False)
            torch.set_rng_state(rng["torch"])
            torch.npu.set_rng_state(rng["npu"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        run = {
            "config": config.name,
            "world_size": world_size,
            "global_batch_size": config.batch_size,
            "per_rank_batch_size": config.batch_size // world_size,
            "target_steps": config.num_train_steps,
            "sharding_strategy": "FULL_SHARD",
            "backend": "hccl",
            "precision": "bf16_with_fp32_master",
            "parameters": parameter_count,
            "trainable_parameters": parameter_count,
            "weight_aliases": aliases,
            "dataset_fingerprint": conversion["dataset_fingerprint"],
            "task_instruction": task.prompt,
            "action_horizon": config.model.action_horizon,
            "max_token_len": config.model.max_token_len,
            "discrete_state_input": config.model.discrete_state_input,
            "initial_weights": str(weight_path),
        }
        if rank == 0:
            (config.checkpoint_dir / "run.json").write_text(json.dumps(run, indent=2))
            print(json.dumps(run), flush=True)
        model.train()
        schedule = config.lr_schedule.create()
        for step in range(start_step + 1, config.num_train_steps + 1):
            started = time.monotonic()
            observation, actions = next(batches)
            if actions.shape[0] != run["per_rank_batch_size"]:
                raise ValueError(f"Incorrect per-rank batch: {actions.shape}")
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(device=device, dtype=torch.float32)
            lr = float(schedule(step - 1))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("npu", dtype=torch.bfloat16):
                loss = model(observation, actions).float().mean()
            averaged = loss.detach().clone()
            dist.all_reduce(averaged)
            averaged /= world_size
            if not torch.isfinite(averaged).item():
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            gradient_norm = model.clip_grad_norm_(config.optimizer.clip_gradient_norm)
            if not torch.isfinite(gradient_norm).item():
                raise FloatingPointError(f"Non-finite gradients at step {step}")
            optimizer.step()
            torch.npu.synchronize()
            if rank == 0 and (step % config.log_interval == 0 or step == 1):
                metric = {
                    "step": step,
                    "loss": averaged.item(),
                    "gradient_norm": gradient_norm.item(),
                    "lr": lr,
                    "seconds": time.monotonic() - started,
                    "per_rank_batch_size": actions.shape[0],
                    "global_batch_size": config.batch_size,
                    "peak_npu_gb": torch.npu.max_memory_allocated() / 1e9,
                }
                with (config.checkpoint_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metric) + "\n")
                (config.checkpoint_dir / "progress.json").write_text(
                    json.dumps(
                        {
                            **metric,
                            "target_steps": config.num_train_steps,
                            "status": "running",
                        },
                        indent=2,
                    )
                )
                print("TRAIN " + json.dumps(metric), flush=True)
            if step % config.save_interval == 0 or step == config.num_train_steps:
                save_checkpoint(model, optimizer, step, config, dataset_config, run)
        if rank == 0:
            print(f"TRAINING COMPLETE: {config.num_train_steps} optimizer steps", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    train(configs.cli())
