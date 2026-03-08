"""
Export deployment config to the log directory.

This is a local replacement for `unitree_rl_lab.utils.export_deploy_cfg`.
It serializes key env parameters (action scale, joint order, decimation, etc.)
into a YAML file inside the experiment log directory for later use in sim2sim.
"""

from __future__ import annotations

import os

import yaml


def export_deploy_cfg(env, log_dir: str) -> None:
    """Export minimal deployment config needed for sim2sim and on-robot inference.

    Args:
        env: The unwrapped Isaac Lab environment (ManagerBasedRLEnv).
        log_dir: Path to the current experiment log directory.
    """
    deploy_cfg: dict = {}

    # --- Simulation timing ---
    try:
        deploy_cfg["sim_dt"] = float(env.cfg.sim.dt)
        deploy_cfg["decimation"] = int(env.cfg.decimation)
    except AttributeError:
        pass

    # --- Action joints (policy order) ---
    try:
        action_manager = env.action_manager
        action_terms = list(action_manager._terms.values())
        if action_terms:
            term = action_terms[0]
            if hasattr(term, "joint_names"):
                deploy_cfg["action_joints"] = list(term.joint_names)
            if hasattr(term, "scale"):
                scale = term.scale
                # scale may be a float or a tensor
                try:
                    deploy_cfg["action_scale"] = float(scale)
                except Exception:
                    deploy_cfg["action_scale"] = scale.tolist()
    except AttributeError:
        pass

    # --- Robot asset path ---
    try:
        asset_cfg = env.cfg.scene.robot
        if hasattr(asset_cfg, "spawn") and hasattr(asset_cfg.spawn, "asset_path"):
            deploy_cfg["asset_path"] = asset_cfg.spawn.asset_path
    except AttributeError:
        pass

    # Write to log directory
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "deploy_cfg.yaml")
    with open(out_path, "w") as f:
        yaml.dump(deploy_cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"[INFO] Deploy config exported to: {out_path}")
