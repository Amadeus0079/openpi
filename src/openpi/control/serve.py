"""Owned localhost policy process for the control console."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_yam_hybrid")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.52)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from openpi.policies.hybrid_assets import sha
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training.config import get_config

    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.device)
    policy = create_trained_policy(get_config(args.config), args.checkpoint, pytorch_device=args.device)
    metadata = dict(policy.metadata, checkpoint=str(args.checkpoint), step=int(args.checkpoint.name))
    weights = args.checkpoint / "model.safetensors"
    if weights.is_file() and not metadata.get("model_sha256"):
        metadata["model_sha256"] = sha(weights)
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / "loaded.json").write_text(json.dumps(metadata, indent=2))
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
