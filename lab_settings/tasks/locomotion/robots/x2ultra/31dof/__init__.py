import gymnasium as gym

# rsl_rl_cfg_entry_point 指向本项目内的 BasePPORunnerCfg（已从 unitree_rl_lab 解耦）
# 路径格式：Python 模块路径:类名，由 isaaclab_tasks hydra 系统动态加载
_PPO_CFG = "locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"

gym.register(
    id="Zhiyuan-X2Ultra-31dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

gym.register(
    id="Zhiyuan-X2Ultra-31dof-Stair",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_env_cfg:StairEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stair_env_cfg:StairPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

gym.register(
    id="Zhiyuan-X2Ultra-31dof-Velocity-CPG",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_wCPG_env_cfg:CPGRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_wCPG_env_cfg:CPGRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

gym.register(
    id="Zhiyuan-X2Ultra-31dof-Stair-CPG",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_wCPG_env_cfg:StairCPGEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stair_wCPG_env_cfg:StairCPGPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)
