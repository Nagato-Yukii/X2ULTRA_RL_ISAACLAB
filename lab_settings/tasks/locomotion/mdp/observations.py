from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


# ==============================================================================
# HRP-5P 专用观测函数
# ==============================================================================

def root_roll_pitch(env: ManagerBasedRLEnv) -> torch.Tensor:
    """从投影重力向量中提取躯干 Roll 和 Pitch 角。

    忽略 Yaw 角以保证航向不变性。投影重力 g_b = R^T @ g_w，
    其中 g_w = [0, 0, -1]（归一化）。

    Roll  ≈ atan2(g_b_y, -g_b_z)  (绕 X 轴倾斜)
    Pitch ≈ atan2(-g_b_x, -g_b_z) (绕 Y 轴倾斜)

    Returns:
        Tensor [num_envs, 2]: [roll, pitch]  (单位: rad)
    """
    # projected_gravity_b: [num_envs, 3], 已归一化的重力在机体系的投影
    g_b = env.scene["robot"].data.projected_gravity_b  # (gx, gy, gz)

    roll = torch.atan2(g_b[:, 1], -g_b[:, 2])   # 绕 X 轴
    pitch = torch.atan2(-g_b[:, 0], -g_b[:, 2]) # 绕 Y 轴
    return torch.stack([roll, pitch], dim=-1)


def motion_mode_onehot(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """将速度指令映射为 3 类 One-Hot 运动模式编码。

    模式判断逻辑（基于 base_velocity 指令幅值）:
        Forward  [1, 0, 0]: abs(lin_vel_x) > lin_threshold
        Inplace  [0, 1, 0]: NOT Forward  AND  abs(ang_vel_z) > ang_threshold
        Standing [0, 0, 1]: 其余（所有速度指令接近零）

    Args:
        command_name: 速度指令的名称，默认 "base_velocity"。

    Returns:
        Tensor [num_envs, 3]: one-hot 编码 [Forward, Inplace, Standing]
    """
    cmd = env.command_manager.get_command(command_name)  # [num_envs, 3]: (vx, vy, wz)

    lin_vel_x = cmd[:, 0]
    ang_vel_z = cmd[:, 2]

    LIN_THRESHOLD = 0.1   # m/s，超过则认为是前进模式
    ANG_THRESHOLD = 0.05  # rad/s，超过则认为是原地踏步模式

    is_forward = torch.abs(lin_vel_x) > LIN_THRESHOLD
    is_inplace = (~is_forward) & (torch.abs(ang_vel_z) > ANG_THRESHOLD)
    is_standing = ~(is_forward | is_inplace)

    # [num_envs, 3]: [Forward, Inplace, Standing]
    onehot = torch.stack([is_forward.float(), is_inplace.float(), is_standing.float()], dim=-1)
    return onehot


def mode_reference(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """返回与运动模式对应的标量参考值。

    - Forward 模式: 前向目标线速度 lin_vel_x
    - Inplace 模式: 目标转向角速度 ang_vel_z
    - Standing 模式: 0.0

    Returns:
        Tensor [num_envs, 1]: 标量参考值
    """
    cmd = env.command_manager.get_command(command_name)  # [num_envs, 3]

    lin_vel_x = cmd[:, 0]
    ang_vel_z = cmd[:, 2]

    LIN_THRESHOLD = 0.1
    ANG_THRESHOLD = 0.05

    is_forward = torch.abs(lin_vel_x) > LIN_THRESHOLD
    is_inplace = (~is_forward) & (torch.abs(ang_vel_z) > ANG_THRESHOLD)

    # Forward → 使用 lin_vel_x；Inplace → 使用 ang_vel_z；Standing → 0
    ref = torch.where(is_forward, lin_vel_x, torch.where(is_inplace, ang_vel_z, torch.zeros_like(lin_vel_x)))
    return ref.unsqueeze(-1)  # [num_envs, 1]


def hrp5p_clock_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """读取时钟 Action 中的当前相位 phi，返回 sin/cos 特征。

    时钟相位 phi ∈ [0, L)，L=80 对应 2s 步态周期 @40Hz。
    该观测为步态奖励 (feet_gait) 提供节律感知信号。

    Returns:
        Tensor [num_envs, 2]: [sin(2π·phi/L), cos(2π·phi/L)]
    """
    # 从 action_manager 中找到 ClockJointPositionAction 并读取 phi
    # action_manager.terms 是一个 OrderedDict[str, ActionTerm]
    clock_action = None
    for term in env.action_manager._terms.values():
        # 动态类型检查，避免硬依赖类名
        if hasattr(term, "phi") and hasattr(term, "_L"):
            clock_action = term
            break

    if clock_action is None:
        # 回退：返回全零（不中断训练，仅作安全保障）
        return torch.zeros(env.num_envs, 2, device=env.device)

    phi = clock_action.phi              # [num_envs]
    L = float(clock_action._L)         # 80.0

    angle = 2.0 * torch.pi * phi / L
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # [num_envs, 2]
