from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING: # 只在IDE检查类型，不影响运行
    from isaaclab.envs import ManagerBasedRLEnv

# 课程设置

def terrain_levels_stair(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_level: int = 1,
    distance_threshold: float = 4.0,
) -> torch.Tensor:

    """
    楼梯地形课程：
    与 terrain_levels_vel 的区别：
    1. 升级阈值更低（2m 而非 terrain_size/2=4m），适配低速走楼梯
    2. 有最低级别下限（min_level=1），保证始终有楼梯暴露，避免卡在纯平地
    3. 降级条件更宽松：只在走得特别少（<0.5m）时才降级
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain

    # 计算本 episode 内变化的欧氏距离 |x - x0| + |y - y0| # 只取前两维,+ |z - z0|这段不取
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )

    move_up = distance > distance_threshold
    move_down = (distance < 0.5) & (~move_up)

    terrain.update_env_origins(env_ids, move_up, move_down)

    # 强制最低级别为 min_level，保证始终有楼梯暴露
    below_min = terrain.terrain_levels < min_level
    if below_min.any():
        terrain.terrain_levels[below_min] = min_level
        terrain.env_origins[below_min] = terrain.terrain_origins[
            terrain.terrain_levels[below_min], terrain.terrain_types[below_min]
        ]

    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:

    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
