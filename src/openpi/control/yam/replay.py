"""Offline YAM-1 mapping worker; compatible with a separate Mink environment."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.control.yam.mapping import IKFailure
from openpi.control.yam.mapping import Yam1ActionAdapter


def reconstruct(kinematics, target, seeds):
    ranked = sorted(
        seeds,
        key=lambda q: (
            np.linalg.norm(kinematics.fk(q)[:3, 3] - target[:3, 3])
            + 0.1 * Rotation.from_matrix(kinematics.fk(q)[:3, :3] @ target[:3, :3].T).magnitude()
        ),
    )
    failures = []
    for seed in ranked:
        try:
            return kinematics.solve(target, seed)[0]
        except IKFailure as error:
            failures.append(error.diagnostics)
    raise IKFailure("Cannot reconstruct saved current pose from support seeds", {"trials": failures})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-poses", type=Path)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--velocity-mode", choices=["urdf", "collection"], default="urdf")
    parser.add_argument("--yam-owner", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "hardware_access": False,
        "hardware_motion": False,
        "executable_command_sent": False,
        "geometry_status": "pending",
        "ik_status": "not_configured",
        "input_role": "saved_target_support",
        "starting_joints_source": "offline IK reconstruction; no live feedback",
    }
    adapter = Yam1ActionAdapter(args.checkpoint, args.output / "numerical-models")
    adapter.bridge.validate_policy_metadata(json.loads(args.metadata.read_text()))
    observations = json.loads((args.checkpoint / "validation/observations.json").read_text())
    sample = next(x for x in observations if x["task"] == args.sample)
    clips = json.loads((args.checkpoint / "assets/yam_hybrid/deployment/target/clip-geometry.json").read_text())[
        "clips"
    ]
    clip = next(x for x in clips if x["original_id"] == sample["original_id"])
    adapter.bridge.reset_recorded_clip(clip)
    actions = np.load(args.actions, allow_pickle=False)
    targets = adapter.bridge.decode(actions)
    np.savez_compressed(args.output / "ee-targets.npz", **targets)
    report.update(geometry_status="passed", model_identity_verified=True, sample=sample)
    if args.support_poses is not None:
        frames = json.loads(args.support_poses.read_text())["frames"]
        with np.load(args.checkpoint / f"validation/replay-{args.sample}.npz", allow_pickle=False) as saved:
            current = adapter.bridge.decode(np.tile(saved["state"][:14], (30, 1)))
            grippers = [float(saved["state"][6]), float(saved["state"][13])]
        try:
            joints = {}
            for side in ["left", "right"]:
                seeds = [
                    np.array([f["joint_configuration"][f"{side}_joint{i}"] for i in range(1, 7)])
                    for f in frames
                    if f["task"] == args.sample
                ]
                joints[side] = reconstruct(
                    adapter.left if side == "left" else adapter.right, current[side + "_base_T_native_mount"][0], seeds
                )
            feedback = np.r_[joints["left"], grippers[0], joints["right"], grippers[1]]
            plan = adapter.plan(
                actions, feedback, args.steps, velocity_mode=args.velocity_mode, yam_owner=args.yam_owner
            )
            np.savez_compressed(
                args.output / "joint-plan.npz",
                command_targets=plan["command_targets"],
                relative_time_s=plan["relative_time_s"],
            )
            report.update(
                ik_status="passed",
                points=plan["command_targets"].tolist(),
                velocity_source=plan["velocity_source"],
                steps=args.steps,
                diagnostics=plan["ik"],
            )
        except IKFailure as error:
            report.update(ik_status="rejected", reason=str(error), diagnostics=error.diagnostics)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
