# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg): #基类直接用rslrl的ppo配置
    num_steps_per_env = 24 # Horizon
    max_iterations = 50000 # total_steps ≈ num_steps_per_env * max_iterations
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class StairPPORunnerCfg(BasePPORunnerCfg):
    """楼梯任务的 PPO 配置。

    盲走策略：观测维度与 velocity 任务相同，网络结构不变。
    依靠 history_length=5 的时序信息隐式推断地形。
    """

    pass

# 假如有新的任务，继承BasePPORunnerCfg，然后修改需要修改的参数即可
