"""
MuJoCo Environment Wrapper
===========================
Handles model loading, actuator ↔ joint index mapping, and PD torque control.

Key challenge: In MuJoCo, ``d.ctrl`` indices (actuator order) and ``d.qpos``
indices (joint order) are **not** the same when actuators reference joints that
appear in a different order in the kinematic tree.  This module resolves the
mapping automatically via ``model.actuator_trnid``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .math_utils import quat_rotate_inverse


@dataclass
class JointConfig:
    """Per-joint PD controller configuration.

    Attributes:
        name: Joint name as it appears in the MuJoCo XML.
        kp:   Proportional gain  (stiffness).
        kd:   Derivative gain    (damping).
        default_pos: Default / rest position in radians.
    """

    name: str
    kp: float
    kd: float
    default_pos: float


class MujocoEnv:
    """Thin wrapper around a MuJoCo model providing named-joint PD control.

    On construction the class:

    1. Loads the XML model and creates ``MjData``.
    2. Builds a **joint-name → ctrl-index** mapping so that all downstream
       code can refer to joints by name.
    3. Stores per-actuator PD gains and default positions.

    The :meth:`step_pd` method computes PD torques for *all* actuators and
    writes them to ``d.ctrl``, then steps physics once.

    Args:
        xml_path: Path to the MuJoCo scene XML.
        sim_dt: Desired physics timestep (overrides the value in XML).
        joint_configs: Per-joint configuration (PD gains + default position).
    """

    def __init__(
        self,
        xml_path: str,
        sim_dt: float,
        joint_configs: list[JointConfig],
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt

        self.nu = self.model.nu  # number of actuators

        # ── name → index look-ups ──────────────────────────────────────────
        self._joint_name_to_ctrl: dict[str, int] = {}
        self._ctrl_to_qpos = np.zeros(self.nu, dtype=int)
        self._ctrl_to_qvel = np.zeros(self.nu, dtype=int)

        for i in range(self.nu):
            jnt_id = self.model.actuator_trnid[i, 0]
            jnt_name = self.model.joint(jnt_id).name
            self._joint_name_to_ctrl[jnt_name] = i
            self._ctrl_to_qpos[i] = self.model.jnt_qposadr[jnt_id]
            self._ctrl_to_qvel[i] = self.model.jnt_dofadr[jnt_id]

        # ── PD gains & defaults (per actuator) ─────────────────────────────
        self.kp = np.zeros(self.nu, dtype=np.float32)
        self.kd = np.zeros(self.nu, dtype=np.float32)
        self.default_pos = np.zeros(self.nu, dtype=np.float32)

        for jcfg in joint_configs:
            idx = self._joint_name_to_ctrl.get(jcfg.name)
            if idx is not None:
                self.kp[idx] = jcfg.kp
                self.kd[idx] = jcfg.kd
                self.default_pos[idx] = jcfg.default_pos
            else:
                print(f"  [WARN] Joint '{jcfg.name}' not found in MuJoCo model — skipped")

        # ── current target position (init = default) ──────────────────────
        self.target_pos = self.default_pos.copy()

    # ================================================================== #
    # Joint queries (by ctrl index)
    # ================================================================== #

    def get_joint_ctrl_indices(self, joint_names: list[str]) -> list[int]:
        """Return the ``ctrl`` indices corresponding to *joint_names*.

        Raises ``KeyError`` if a name is not found.
        """
        indices: list[int] = []
        for name in joint_names:
            if name not in self._joint_name_to_ctrl:
                raise KeyError(
                    f"Joint '{name}' not found.  Available: "
                    f"{sorted(self._joint_name_to_ctrl.keys())}"
                )
            indices.append(self._joint_name_to_ctrl[name])
        return indices

    def get_joint_qpos_indices(self, joint_names: list[str]) -> list[int]:
        """Return the ``qpos`` indices corresponding to *joint_names*."""
        indices: list[int] = []
        for name in joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id == -1:
                raise KeyError(f"Joint '{name}' not found in MuJoCo model.")
            indices.append(self.model.jnt_qposadr[jnt_id])
        return indices

    def get_joint_qvel_indices(self, joint_names: list[str]) -> list[int]:
        """Return the ``qvel`` (dof) indices corresponding to *joint_names*."""
        indices: list[int] = []
        for name in joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id == -1:
                raise KeyError(f"Joint '{name}' not found in MuJoCo model.")
            indices.append(self.model.jnt_dofadr[jnt_id])
        return indices

    def get_qpos(self) -> np.ndarray:
        """Joint positions for all actuators (``nu``-dim, actuator order)."""
        return self.data.qpos[self._ctrl_to_qpos].copy()

    def get_qvel(self) -> np.ndarray:
        """Joint velocities for all actuators (``nu``-dim, actuator order)."""
        return self.data.qvel[self._ctrl_to_qvel].copy()

    def get_base_quat(self) -> np.ndarray:
        """Base quaternion ``[qw, qx, qy, qz]``."""
        return self.data.qpos[3:7].copy()

    def get_base_ang_vel(self) -> np.ndarray:
        """Base angular velocity in **body frame**.

        MuJoCo ≥ 3.0 stores free-joint angular velocity in the **world**
        frame (``qvel[3:6]``).  Isaac Lab's ``root_ang_vel_b`` is in body
        frame, so we apply ``R^T @ ω_world`` here.
        """
        omega_world = self.data.qvel[3:6].copy()
        quat = self.get_base_quat()
        return quat_rotate_inverse(quat, omega_world)

    def get_base_lin_vel(self) -> np.ndarray:
        """Base linear velocity in **world frame**."""
        return self.data.qvel[0:3].copy()

    def get_base_pos(self) -> np.ndarray:
        """Base position ``[x, y, z]`` in world frame."""
        return self.data.qpos[0:3].copy()

    # ================================================================== #
    # Control
    # ================================================================== #

    def set_targets(self, ctrl_indices: list[int], targets: np.ndarray):
        """Set target positions for specific actuators (by ctrl index)."""
        for i, idx in enumerate(ctrl_indices):
            self.target_pos[idx] = targets[i]

    def set_target_positions(self, ctrl_indices: list[int], target_positions: np.ndarray | list[float]):
        """Set target positions for the internal PD controller.
        
        注意：必须更新 self.target_pos 而不能直接赋值给 self.data.ctrl。
        因为目前模型全使用的是 <motor> (力矩控制) 驱动器，
        这里的期望是在底层的 step_pd 中通过 PD 公式转化为力矩后，再写入 self.data.ctrl。
        """
        for i, idx in enumerate(ctrl_indices):
            self.target_pos[idx] = target_positions[i]

    def reset_targets(self):
        """Reset all targets to their default positions."""
        self.target_pos[:] = self.default_pos

    def step_pd(self):
        """Compute PD torques → write to ``ctrl`` → step physics once."""
        q = self.data.qpos[self._ctrl_to_qpos]
        dq = self.data.qvel[self._ctrl_to_qvel]
        tau = self.kp * (self.target_pos - q) + self.kd * (0.0 - dq)
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

    # ================================================================== #
    # Initialisation helpers
    # ================================================================== #

    def set_initial_state(self, base_pos: tuple[float, ...] = (0.0, 0.0, 0.66)):
        """Reset the simulation to a standing pose.

        Sets the base position, identity quaternion, and all joint positions
        to their configured defaults.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = base_pos
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion

        # Write default joint positions through the actuator mapping
        for i in range(self.nu):
            self.data.qpos[self._ctrl_to_qpos[i]] = self.default_pos[i]

        self.target_pos[:] = self.default_pos
        mujoco.mj_forward(self.model, self.data)

    # ================================================================== #
    # Debug
    # ================================================================== #

    def print_joint_mapping(self):
        """Print actuator → joint mapping table for debugging."""
        print("\n  Actuator → Joint Mapping:")
        print(f"  {'ctrl':>5}  {'actuator':40s}  {'qpos':>5}  {'joint':30s}  {'kp':>7}  {'kd':>5}  {'def':>7}")
        print("  " + "-" * 105)
        for i in range(self.nu):
            act_name = self.model.actuator(i).name
            jnt_id = self.model.actuator_trnid[i, 0]
            jnt_name = self.model.joint(jnt_id).name
            print(
                f"  [{i:3d}]  {act_name:40s}  [{self._ctrl_to_qpos[i]:3d}]  {jnt_name:30s}"
                f"  {self.kp[i]:7.1f}  {self.kd[i]:5.1f}  {self.default_pos[i]:7.3f}"
            )
