"""CPG (Central Pattern Generator) + Residual Joint Position Action.

将神经网络输出视为步态残差，叠加到 CPG 生成的周期性信号上：
    output_rad = default_rad + network_rad + CPG_rad

其中:
    - default_rad : 关节默认站立位姿（来自 URDF default_joint_pos）
    - network_rad : RL 策略网络输出 × scale（残差修正）
    - CPG_rad     : 中枢模式发生器产生的周期性摆动信号
"""

from __future__ import annotations

import math
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CPGJointPositionAction(JointAction):
    """关节位置 Action = default_offset + network_residual + CPG_signal.

    CPG 使用简单的相位振荡器，为每条腿的每个关节生成正弦/余弦周期信号。
    不同腿之间通过 phase_offset 实现交替步态。
    网络只需要学习"残差"，大幅降低学习难度。

    关键属性:
        _cpg_phase : [num_envs, num_legs]  当前相位 ∈ [0, 2π)
        _cpg_signal: [num_envs, num_joints] CPG 产生的关节角度偏移
    """

    cfg: CPGJointPositionActionCfg

    def __init__(self, cfg: CPGJointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # --- 使用 default_joint_pos 作为中心位姿 ---
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # --- CPG 参数 ---
        self._num_legs = len(cfg.cpg_phase_offsets)
        self._joints_per_leg = len(self._joint_ids) // self._num_legs
        assert len(self._joint_ids) % self._num_legs == 0, (
            f"关节数 {len(self._joint_ids)} 不能被腿数 {self._num_legs} 整除"
        )

        # 每条腿的相位偏移 [num_legs]
        self._leg_phase_offsets = torch.tensor(
            cfg.cpg_phase_offsets, dtype=torch.float32, device=self.device
        )

        # 每个关节的振幅 [num_joints]  (与 _joint_ids 一一对应)
        if isinstance(cfg.cpg_amplitudes, (int, float)):
            self._amplitudes = torch.full(
                (len(self._joint_ids),), cfg.cpg_amplitudes, dtype=torch.float32, device=self.device
            )
        elif isinstance(cfg.cpg_amplitudes, list):
            assert len(cfg.cpg_amplitudes) == len(self._joint_ids), (
                f"cpg_amplitudes 长度 {len(cfg.cpg_amplitudes)} != 关节数 {len(self._joint_ids)}"
            )
            self._amplitudes = torch.tensor(cfg.cpg_amplitudes, dtype=torch.float32, device=self.device)
        else:
            # dict: 按正则匹配
            self._amplitudes = torch.zeros(len(self._joint_ids), dtype=torch.float32, device=self.device)
            for pattern, value in cfg.cpg_amplitudes.items():
                indices, _, _ = self._asset.find_joints(pattern)
                # 只保留在 _joint_ids 中的那些
                for idx in indices:
                    if idx in self._joint_ids:
                        local_idx = list(self._joint_ids).index(idx)
                        self._amplitudes[local_idx] = value

        # CPG 全局相位: [num_envs, num_legs], 初始化为 leg_phase_offsets
        self._cpg_phase = self._leg_phase_offsets.unsqueeze(0).expand(self.num_envs, -1).clone()

        # CPG 输出缓冲区: [num_envs, num_joints]
        self._cpg_signal = torch.zeros(self.num_envs, len(self._joint_ids), dtype=torch.float32, device=self.device)

        # 仿真时间步长 (sim dt, 不是 env dt)
        self._sim_dt = env.sim.cfg.dt

    @property
    def action_dim(self) -> int:
        """网络输出维度 = 关节数（残差）"""
        return len(self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        """重置被选中环境的 CPG 相位和 action 缓冲区。"""
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # 重置相位到初始偏移
        self._cpg_phase[env_ids] = self._leg_phase_offsets.unsqueeze(0).expand(len(env_ids), -1).clone()
        self._cpg_signal[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        """每个 env step 调用一次：处理网络输出（残差部分）。

        processed_actions = network_output * scale + default_offset
        （CPG 信号在 apply_actions 中按 sim_dt 实时叠加）
        """
        self._raw_actions[:] = actions
        # 仅做仿射变换 (scale + offset)，CPG 在 apply_actions 中叠加
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def apply_actions(self):
        """每个 sim step 调用一次：推进 CPG 相位并叠加到关节目标。

        最终目标:
            joint_target = default_offset + network_residual * scale + CPG_signal
                         = self._processed_actions + self._cpg_signal
        """
        # --- 推进 CPG 相位 ---
        # dφ = 2π * frequency * dt
        dphi = 2.0 * math.pi * self.cfg.cpg_frequency * self._sim_dt
        self._cpg_phase += dphi
        # 保持在 [0, 2π) 范围内，防止数值溢出
        self._cpg_phase = self._cpg_phase % (2.0 * math.pi)

        # --- 生成 CPG 关节信号 ---
        # 对每条腿，用 sin(phase) 乘以各关节振幅
        for leg_idx in range(self._num_legs):
            start = leg_idx * self._joints_per_leg
            end = start + self._joints_per_leg
            # sin 振荡: [num_envs, 1] × [joints_per_leg] → [num_envs, joints_per_leg]
            sin_signal = torch.sin(self._cpg_phase[:, leg_idx]).unsqueeze(-1)
            self._cpg_signal[:, start:end] = sin_signal * self._amplitudes[start:end].unsqueeze(0)

        # --- 最终叠加: processed_actions 已包含 (default + network*scale)，再加 CPG ---
        joint_targets = self._processed_actions + self._cpg_signal

        # --- 发送关节位置目标 ---
        self._asset.set_joint_position_target(joint_targets, joint_ids=self._joint_ids)


# ==============================================================================
# 配置类
# ==============================================================================

@configclass
class CPGJointPositionActionCfg(JointActionCfg):
    """CPG + 残差关节位置控制的配置。

    Attributes:
        use_default_offset: 是否使用 URDF 中的默认关节位置作为中心位姿。
        cpg_frequency:  CPG 振荡器频率 (Hz)。例如 1.25 Hz = 0.8s 周期，
                        与步态奖励的 period=0.8 一致。
        cpg_phase_offsets: 每条腿的初始相位偏移 (rad)。
                          例如 [0.0, π] 表示两条腿交替半周期（trot 步态）。
        cpg_amplitudes: 每个关节的 CPG 振幅 (rad)。
                        可以是 float（所有关节相同）、list（逐关节指定）
                        或 dict（按正则匹配关节名）。
    """

    class_type: type[ActionTerm] = CPGJointPositionAction

    # 是否用 default_joint_pos 作为 offset
    use_default_offset: bool = True

    # CPG 振荡器频率 (Hz): 1/period
    # 默认 1.25 Hz → 周期 0.8s，与 gait reward 的 period=0.8 一致
    cpg_frequency: float = 1.25

    # 每条腿的初始相位偏移 (rad)
    # [左腿, 右腿] — 差 π 表示交替步态 (trot)
    cpg_phase_offsets: list[float] = MISSING

    # 各关节 CPG 振幅 (rad)
    # float: 全部相同; list: 逐关节; dict: 按正则匹配
    cpg_amplitudes: float | list[float] | dict[str, float] = 0.1
