"""Clock-Controlled Residual Joint Position Action for HRP-5P.

网络输出 13 维动作向量：
    a_t[0:12] : 下肢 12 个关节的目标位置残差 (Position Residuals)
    a_t[12]   : 时钟相位偏移量 (Phase offset, a_delta_phi)

控制数据流:
    phi_{t+1} = (phi_t + clip(a_delta_phi, -5, 5) + 1) % L   (L=80 对应 2s 周期)
    q_des = a_t[0:12] * scale + q_nominal
"""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ClockJointPositionAction(JointAction):
    """动态时钟控制的关节位置 Action。

    网络直接输出 13 维动作：
        - 前 12 维: 关节位置残差 (叠加到标称位置 q_nominal 上)
        - 第 13 维: 时钟相位偏移量 a_delta_phi (用于动态调节步频)

    时钟更新规则 (每个 env step 即每个策略步 ~25ms 更新一次):
        phi_{t+1} = (phi_t + clip(a_delta_phi, -5, 5) + 1) % L
    其中 L=80 对应 2s 步态周期 (在 40Hz 策略频率下).

    关键属性:
        _phi   : [num_envs]  当前时钟计数 phi ∈ [0, L)
        _L     : int         周期总步长 (default=80)
        _a_delta_phi_range : tuple  相位偏移裁剪范围 (default=(-5, 5))
    """

    cfg: ClockJointPositionActionCfg

    def __init__(self, cfg: ClockJointPositionActionCfg, env: ManagerBasedRLEnv):
        # 父类 JointAction 会根据 joint_names 解析关节 ID (12 个腿部关节)
        super().__init__(cfg, env)

        # 标称关节位置 (q_nominal)：来自 URDF 的 default_joint_pos
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # 时钟参数
        self._L: int = cfg.clock_period_steps              # 周期步长 L=80
        self._delta_phi_min: float = cfg.delta_phi_range[0]  # 裁剪下界
        self._delta_phi_max: float = cfg.delta_phi_range[1]  # 裁剪上界

        # 时钟状态: [num_envs], phi ∈ [0, L), 初始化为 0
        self._phi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # 缓存最新的相位偏移供外部读取 (观测函数使用)
        self._last_delta_phi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """策略网络输出维度 = 12 关节残差 + 1 相位偏移 = 13."""
        return len(self._joint_ids) + 1

    @property
    def phi(self) -> torch.Tensor:
        """当前时钟相位 phi: [num_envs], ∈ [0, L)."""
        return self._phi

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None):
        """重置指定环境的时钟相位和 action 缓冲区。"""
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._phi[env_ids] = 0.0
        self._last_delta_phi[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Core Action Processing (called once per policy step)
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        """处理 13 维网络输出。

        Args:
            actions: [num_envs, 13]
                actions[:, 0:12] — 关节位置残差
                actions[:, 12]  — 时钟相位偏移量 a_delta_phi
        """
        self._raw_actions[:] = actions[:, : self.action_dim]

        # 分离关节残差与相位偏移
        a_residual = actions[:, :12]           # [num_envs, 12]
        a_delta_phi = actions[:, 12]           # [num_envs]

        # 裁剪相位偏移并缓存
        a_delta_phi_clipped = torch.clamp(a_delta_phi, self._delta_phi_min, self._delta_phi_max)
        self._last_delta_phi = a_delta_phi_clipped

        # 更新时钟相位:  phi_{t+1} = (phi_t + clip(a_delta_phi, -5, 5) + 1) % L
        self._phi = (self._phi + a_delta_phi_clipped + 1.0) % self._L

        # 计算目标关节位置: q_des = residual * scale + q_nominal
        self._processed_actions = a_residual * self._scale + self._offset

    # ------------------------------------------------------------------
    # Apply Actions (called once per sim step, i.e. decimation times)
    # ------------------------------------------------------------------

    def apply_actions(self):
        """将计算好的目标关节位置发送到底层物理仿真。

        策略以 40 Hz 运行，底层仿真以更高频率 (1/sim.dt) 运行。
        在每个 sim step 中，我们保持目标位置不变 (ZOH — Zero-Order Hold)，
        由底层 PD 控制器将实际关节驱动到目标位置。
        """
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


# ==============================================================================
# 配置类
# ==============================================================================

@configclass
class ClockJointPositionActionCfg(JointActionCfg):
    """动态时钟控制关节位置 Action 的配置。

    Attributes:
        use_default_offset: 是否以 URDF 默认关节位置作为残差中心。
        clock_period_steps: 时钟周期总步长 L。
            在 40 Hz 策略下，L=80 对应 2s 步态周期。
        delta_phi_range: 相位偏移 a_delta_phi 的裁剪范围 (步长单位)。
            默认 (-5, 5)，即最大 ±5 步 = ±125ms 的时间偏移。
    """

    class_type: type[ActionTerm] = ClockJointPositionAction

    # 是否以 default_joint_pos 为偏置中心
    use_default_offset: bool = True

    # 步态时钟周期步长  L = ceil(T_gait / T_policy) = ceil(2.0 / 0.025) = 80
    clock_period_steps: int = 80

    # 相位偏移裁剪范围 (步长单位)
    delta_phi_range: tuple[float, float] = (-5.0, 5.0)
