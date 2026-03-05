#!/usr/bin/env python3
"""
Isaac Lab → MuJoCo  Sim2Sim Deployment
========================================
Deploy an Isaac Lab-trained RL locomotion policy into MuJoCo for
sim-to-sim validation.

Usage
-----
    # Basic — walk straight with default command (vx=0.5)
    python deploy.py --config configs/walk_straight.yaml

    # Override velocity command
    python deploy.py --config configs/walk_straight.yaml --cmd_vx 0.8 --cmd_wz 0.1

    # Use ONNX model instead of TorchScript
    python deploy.py --config configs/walk_straight.yaml \
        --policy ../../pretrained/walk_straight/exported/policy.onnx

Keyboard controls (during simulation)
--------------------------------------
    ↑ / ↓    Increase / decrease  lin_vel_x  by 0.1
    ← / →    Increase / decrease  ang_vel_z  by 0.1
    Space     Zero all velocity commands
    R         Reset robot to initial pose
    Q / Esc   Quit simulation

Architecture
------------
The code is intentionally modular so you can swap out components:

    ┌──────────┐     ┌──────────────────┐     ┌──────────┐
    │ MujocoEnv│────▶│ObservationBuilder│────▶│PolicyRun.│
    │  (state) │     │   (obs+history)  │     │ (action) │
    └────┬─────┘     └──────────────────┘     └────┬─────┘
         │  PD torque ◀── target positions ◀───────┘
         ▼
      physics step
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

# Ensure the sim2sim package is importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.mujoco_env import JointConfig, MujocoEnv
from core.observation import ObservationBuilder
from core.policy_runner import PolicyRunner


# ═══════════════════════════════════════════════════════════════════════════
# Configuration Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Load YAML config, resolving relative paths to absolute."""
    config_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths
    for key in ("policy_path", "xml_path"):
        path = cfg[key]
        if not os.path.isabs(path):
            cfg[key] = os.path.normpath(os.path.join(config_dir, path))

    return cfg


def build_joint_configs(pd_cfg: dict) -> list[JointConfig]:
    """Convert the ``pd_controller`` YAML section into JointConfig objects."""
    configs: list[JointConfig] = []
    for name, params in pd_cfg.items():
        configs.append(
            JointConfig(
                name=str(name),
                kp=float(params["kp"]),
                kd=float(params["kd"]),
                default_pos=float(params["default"]),
            )
        )
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── CLI arguments ─────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Isaac Lab → MuJoCo Sim2Sim Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the task YAML config file.")
    parser.add_argument("--policy", type=str, default=None,
                        help="Override policy model path (.pt or .onnx).")
    parser.add_argument("--xml", type=str, default=None,
                        help="Override MuJoCo scene XML path.")
    parser.add_argument("--cmd_vx", type=float, default=None,
                        help="Override linear velocity x command.")
    parser.add_argument("--cmd_vy", type=float, default=None,
                        help="Override linear velocity y command.")
    parser.add_argument("--cmd_wz", type=float, default=None,
                        help="Override angular velocity z command.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override simulation duration (seconds).")
    parser.add_argument("--no-keyboard", action="store_true",
                        help="Disable keyboard velocity control.")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    cfg = load_config(args.config)

    policy_path = args.policy or cfg["policy_path"]
    xml_path    = args.xml    or cfg["xml_path"]
    sim_dt      = cfg["sim_dt"]
    decimation  = cfg["decimation"]
    ctrl_dt     = sim_dt * decimation
    duration    = args.duration or cfg["duration"]

    cmd = np.array(cfg["command"], dtype=np.float32)
    if args.cmd_vx is not None:
        cmd[0] = args.cmd_vx
    if args.cmd_vy is not None:
        cmd[1] = args.cmd_vy
    if args.cmd_wz is not None:
        cmd[2] = args.cmd_wz

    obs_cfg    = cfg["observation"]
    act_cfg    = cfg["action"]
    action_scale   = act_cfg["scale"]
    action_joints  = act_cfg["joints"]
    num_actions    = len(action_joints)
    history_length = obs_cfg["history_length"]

    init_base_pos = tuple(cfg.get("init_base_pos", [0.0, 0.0, 0.66]))

    # ── Print banner ──────────────────────────────────────────────────────
    print("=" * 65)
    print("  X2Ultra Sim2Sim — Isaac Lab → MuJoCo")
    print("=" * 65)
    print(f"  Policy       : {policy_path}")
    print(f"  XML Scene    : {xml_path}")
    print(f"  Control Freq : {1.0 / ctrl_dt:.0f} Hz  (dt={ctrl_dt:.4f}s)")
    print(f"  Action Scale : {action_scale}")
    print(f"  Num Actions  : {num_actions}")
    print(f"  History Len  : {history_length}")
    obs_per_frame = 9 + 3 * num_actions
    print(f"  Obs Dims     : {obs_per_frame} × {history_length} = {obs_per_frame * history_length}")
    print(f"  Velocity Cmd : vx={cmd[0]:.2f}  vy={cmd[1]:.2f}  wz={cmd[2]:.2f}")
    print(f"  Duration     : {duration:.0f} s")
    print("=" * 65)

    # ── Create MuJoCo environment ─────────────────────────────────────────
    print("\n[1/3] Loading MuJoCo model …")
    joint_cfgs = build_joint_configs(cfg["pd_controller"])
    env = MujocoEnv(xml_path, sim_dt, joint_cfgs)
    env.print_joint_mapping()

    # ── Resolve action joint indices (policy order → MuJoCo ctrl order) ──
    action_ctrl_ids = env.get_joint_ctrl_indices(action_joints)
    action_defaults = np.array(
        [env.default_pos[idx] for idx in action_ctrl_ids], dtype=np.float32
    )
    print(f"\n  Action joints (policy order → ctrl index):")
    for i, (name, ctrl_idx) in enumerate(zip(action_joints, action_ctrl_ids)):
        print(f"    [{i:2d}] {name:35s} → ctrl[{ctrl_idx:2d}]  default={action_defaults[i]:.3f}")

    # ── Load policy ───────────────────────────────────────────────────────
    print(f"\n[2/3] Loading policy …")
    policy = PolicyRunner(policy_path, num_actions)

    # ── Create observation builder ────────────────────────────────────────
    print(f"\n[3/3] Initialising observation builder …")
    obs_builder = ObservationBuilder(
        num_actions=num_actions,
        history_length=history_length,
        ang_vel_scale=obs_cfg["ang_vel_scale"],
        dof_vel_scale=obs_cfg["dof_vel_scale"],
    )
    print(f"  Frame size : {obs_builder.frame_size}")
    print(f"  Total obs  : {obs_builder.obs_size}")

    # ── Initialise simulation state ───────────────────────────────────────
    env.set_initial_state(base_pos=init_base_pos)

    action = np.zeros(num_actions, dtype=np.float32)
    counter = 0

    # Pre-fill observation history with the initial standing state
    def read_policy_joints():
        """Read pos/vel for the 12 action joints in policy order."""
        all_q = env.get_qpos()
        all_dq = env.get_qvel()
        pos = np.array([all_q[idx]  for idx in action_ctrl_ids], dtype=np.float32)
        vel = np.array([all_dq[idx] for idx in action_ctrl_ids], dtype=np.float32)
        return pos, vel

    joint_pos, joint_vel = read_policy_joints()
    obs_builder.prefill(
        base_quat=env.get_base_quat(),
        base_ang_vel_body=env.get_base_ang_vel(),
        joint_pos=joint_pos,
        joint_default_pos=action_defaults,
        joint_vel=joint_vel,
        velocity_cmd=cmd,
    )

    # ── Keyboard callback ─────────────────────────────────────────────────
    should_reset = False

    def key_callback(keycode: int):
        nonlocal cmd, should_reset
        DELTA_V = 0.1
        if keycode == 265:    # ↑
            cmd[0] = min(cmd[0] + DELTA_V, 1.5)
        elif keycode == 264:  # ↓
            cmd[0] = max(cmd[0] - DELTA_V, -1.0)
        elif keycode == 262:  # → (turn right = negative wz)
            cmd[2] = max(cmd[2] - DELTA_V, -1.0)
        elif keycode == 263:  # ← (turn left  = positive wz)
            cmd[2] = min(cmd[2] + DELTA_V, 1.0)
        elif keycode == 32:   # Space — stop
            cmd[:] = 0.0
        elif keycode == 82:   # R — reset
            should_reset = True
        elif keycode in (81, 256):  # Q / Esc
            print("\n[INFO] Quit requested.")
            sys.exit(0)

        print(f"  CMD: vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}")

    # ── Main simulation loop ─────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Simulation started!  (Close window or press Q to quit)")
    print("  Keyboard:  ↑↓ = vx   ←→ = wz   Space = stop   R = reset")
    print("─" * 65 + "\n")

    viewer_kwargs = {"key_callback": key_callback} if not args.no_keyboard else {}

    with mujoco.viewer.launch_passive(env.model, env.data, **viewer_kwargs) as viewer:
        start_time = time.time()

        while viewer.is_running() and (time.time() - start_time) < duration:
            step_start = time.time()

            # Handle reset request
            if should_reset:
                env.set_initial_state(base_pos=init_base_pos)
                action[:] = 0.0
                obs_builder.reset()
                joint_pos, joint_vel = read_policy_joints()
                obs_builder.prefill(
                    base_quat=env.get_base_quat(),
                    base_ang_vel_body=env.get_base_ang_vel(),
                    joint_pos=joint_pos,
                    joint_default_pos=action_defaults,
                    joint_vel=joint_vel,
                    velocity_cmd=cmd,
                )
                counter = 0
                should_reset = False
                print("  [RESET] Robot reset to initial state.")

            # ---- PD torque → physics step ----
            env.step_pd()
            counter += 1

            # ---- Policy inference (every `decimation` steps) ----
            if counter % decimation == 0:
                # Read state
                joint_pos, joint_vel = read_policy_joints()
                base_quat = env.get_base_quat()
                base_ang_vel = env.get_base_ang_vel()

                # Build observation (with history)
                obs = obs_builder.build(
                    base_quat=base_quat,
                    base_ang_vel_body=base_ang_vel,
                    joint_pos=joint_pos,
                    joint_default_pos=action_defaults,
                    joint_vel=joint_vel,
                    velocity_cmd=cmd,
                    last_action=action,
                )

                # Policy forward pass
                action = policy.infer(obs)

                # Action → target joint positions
                target_pos = action * action_scale + action_defaults

                # Update MuJoCo targets (only for action joints)
                env.reset_targets()  # non-action joints → default
                env.set_targets(action_ctrl_ids, target_pos)

            # ---- Viewer sync ----
            viewer.sync()

            # ---- Real-time pacing ----
            elapsed = time.time() - step_start
            sleep_time = env.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\nSimulation finished!")


if __name__ == "__main__":
    main()
