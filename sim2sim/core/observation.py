"""
Observation Builder with Per-Term History Support
===================================================
Constructs the observation vector matching Isaac Lab's ``ObservationManager``
with per-term history stacking (``CircularBuffer``).

**Critical**: Isaac Lab stacks history **per observation term**, NOT per frame.
Each term has its own circular buffer of ``history_length`` entries. The final
observation is built by flattening each term's history (oldest → newest) and
then concatenating all terms.

Observation terms (matching Isaac Lab's ``PolicyCfg`` order):

=====  =================  ====  =========================
Index  Name               Dims  Notes
=====  =================  ====  =========================
  0    base_ang_vel         3   body-frame, × ang_vel_scale
  1    projected_gravity    3   body-frame unit vector
  2    velocity_commands    3   [vx, vy, wz]
  3    joint_pos_rel       12   current − default
  4    joint_vel_rel       12   × dof_vel_scale
  5    last_action         12   raw action from prev step
=====  =================  ====  =========================

With ``history_length = 5`` the layout is (for 12 action joints)::

    [ang_vel_t-4(3) .. ang_vel_t(3),     # 3×5 = 15
     gravity_t-4(3) .. gravity_t(3),     # 3×5 = 15
     cmd_t-4(3)     .. cmd_t(3),         # 3×5 = 15
     pos_rel_t-4(12).. pos_rel_t(12),   # 12×5 = 60
     vel_rel_t-4(12).. vel_rel_t(12),   # 12×5 = 60
     action_t-4(12) .. action_t(12)]    # 12×5 = 60
                                         total = 225
"""

from __future__ import annotations

from collections import OrderedDict, deque

import numpy as np

from .math_utils import get_projected_gravity


class ObservationBuilder:
    """Build policy observations with per-term history stacking.

    Matches Isaac Lab's ``ObservationManager`` behaviour:
    - Each observation term has its own ``deque`` (equivalent to
      Isaac Lab's ``CircularBuffer``).
    - Within each term the history is flattened oldest → newest.
    - All term histories are concatenated in definition order.

    This is **different** from per-frame stacking (which would interleave
    terms across frames).  Using the wrong layout will completely break
    the deployed policy.

    Args:
        num_actions:    Number of action dimensions (= controlled joints).
        history_length: How many timesteps of history per term (1 = no history).
        ang_vel_scale:  Multiplier for angular velocity (matches training).
        dof_vel_scale:  Multiplier for joint velocity   (matches training).
    """

    # Observation term names in Isaac Lab PolicyCfg order.
    TERM_NAMES: tuple[str, ...] = (
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    )

    def __init__(
        self,
        num_actions: int = 12,
        history_length: int = 1,
        ang_vel_scale: float = 0.2,
        dof_vel_scale: float = 0.05,
    ):
        self.num_actions = num_actions
        self.history_length = history_length
        self.ang_vel_scale = ang_vel_scale
        self.dof_vel_scale = dof_vel_scale

        # Per-term dimensions  (order must match TERM_NAMES)
        self._term_dims: OrderedDict[str, int] = OrderedDict([
            ("base_ang_vel",       3),
            ("projected_gravity",  3),
            ("velocity_commands",  3),
            ("joint_pos_rel",      num_actions),
            ("joint_vel_rel",      num_actions),
            ("last_action",        num_actions),
        ])

        # 3 + 3 + 3 + N + N + N  where N = num_actions
        self.frame_size = sum(self._term_dims.values())
        # Total obs = sum of (term_dim × history_length) for every term
        self.obs_size = self.frame_size * history_length

        # Per-term history buffers  {name: deque of np arrays}
        self._term_histories: OrderedDict[str, deque] = OrderedDict()
        self.reset()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self):
        """Clear all per-term history buffers (fill with zeros)."""
        self._term_histories.clear()
        for name, dim in self._term_dims.items():
            buf: deque[np.ndarray] = deque(maxlen=self.history_length)
            for _ in range(self.history_length):
                buf.append(np.zeros(dim, dtype=np.float32))
            self._term_histories[name] = buf

    def build(
        self,
        base_quat: np.ndarray,
        base_ang_vel_body: np.ndarray,
        joint_pos: np.ndarray,
        joint_default_pos: np.ndarray,
        joint_vel: np.ndarray,
        velocity_cmd: np.ndarray,
        last_action: np.ndarray,
    ) -> np.ndarray:
        """Compute current-step terms, push to per-term history, return full obs.

        All joint arrays must be in **policy order** (the 12 controlled
        joints as listed in the config YAML).

        Args:
            base_quat:          [qw, qx, qy, qz]  MuJoCo convention.
            base_ang_vel_body:  Angular velocity already in body frame (3,).
            joint_pos:          Current leg-joint positions  (num_actions,).
            joint_default_pos:  Default / rest positions     (num_actions,).
            joint_vel:          Current leg-joint velocities  (num_actions,).
            velocity_cmd:       Velocity command [vx, vy, wz] (3,).
            last_action:        Previous raw action output    (num_actions,).

        Returns:
            Full observation ``(obs_size,)`` — per-term history, then terms
            concatenated.
        """
        # Compute each observation term
        gravity = get_projected_gravity(base_quat)

        current_terms: OrderedDict[str, np.ndarray] = OrderedDict([
            ("base_ang_vel",      (base_ang_vel_body * self.ang_vel_scale).astype(np.float32)),
            ("projected_gravity", gravity),
            ("velocity_commands", velocity_cmd.copy().astype(np.float32)),
            ("joint_pos_rel",     (joint_pos - joint_default_pos).astype(np.float32)),
            ("joint_vel_rel",     (joint_vel * self.dof_vel_scale).astype(np.float32)),
            ("last_action",       last_action.copy().astype(np.float32)),
        ])

        # Push each term into its own history buffer
        for name, value in current_terms.items():
            self._term_histories[name].append(value)

        # Build output: for each term, flatten its history (oldest → newest),
        # then concatenate all terms.
        parts: list[np.ndarray] = []
        for name in self.TERM_NAMES:
            # deque oldest at index 0, newest at index -1
            term_flat = np.concatenate(list(self._term_histories[name]))
            parts.append(term_flat)

        return np.concatenate(parts)

    def prefill(
        self,
        base_quat: np.ndarray,
        base_ang_vel_body: np.ndarray,
        joint_pos: np.ndarray,
        joint_default_pos: np.ndarray,
        joint_vel: np.ndarray,
        velocity_cmd: np.ndarray,
    ):
        """Fill every per-term history slot with the initial observation.

        Matches Isaac Lab's ``CircularBuffer`` first-push behaviour: on the
        very first ``append``, *all* history slots are filled with the same
        data so the policy sees a consistent (not zero-padded) buffer from
        the start.
        """
        last_action = np.zeros(self.num_actions, dtype=np.float32)
        gravity = get_projected_gravity(base_quat)

        init_terms: OrderedDict[str, np.ndarray] = OrderedDict([
            ("base_ang_vel",      (base_ang_vel_body * self.ang_vel_scale).astype(np.float32)),
            ("projected_gravity", gravity),
            ("velocity_commands", velocity_cmd.copy().astype(np.float32)),
            ("joint_pos_rel",     (joint_pos - joint_default_pos).astype(np.float32)),
            ("joint_vel_rel",     (joint_vel * self.dof_vel_scale).astype(np.float32)),
            ("last_action",       last_action),
        ])

        self._term_histories.clear()
        for name, dim in self._term_dims.items():
            buf: deque[np.ndarray] = deque(maxlen=self.history_length)
            for _ in range(self.history_length):
                buf.append(init_terms[name].copy())
            self._term_histories[name] = buf
