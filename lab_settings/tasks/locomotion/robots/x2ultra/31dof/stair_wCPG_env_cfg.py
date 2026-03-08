"""
Stair-Climbing Task with CPG-based Action
==========================================
在 stair_env_cfg.py 基础上引入 CPG（中枢模式发生器）:

1. 楼梯任务的 CPG 特点：
   - 更慢的频率（1.0 Hz，周期 1.0s）：楼梯需要更稳定、更慢的步态
   - 更大的振幅：需要更高的抬腿动作来跨越台阶
   - 更强的膝关节弯曲：knee 振幅增大到 0.85 rad (49°)

2. 与 velocity CPG 的区别：
   - velocity: 1.67 Hz, knee=0.70 rad (快速平地行走)
   - stair: 1.0 Hz, knee=0.85 rad (慢速稳定爬楼)

3. 设计理念：
   - CPG 提供稳定的周期性抬腿动作
   - 网络学习如何适应不同高度的台阶（通过残差修正）
   - 历史观测帮助网络感知地形变化
"""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from locomotion import mdp

# 复用 stair 任务的配置
from .stair_env_cfg import (
    LEG_JOINT_NAMES,
    StairSceneCfg,
    StairCommandsCfg,
    StairRewardsCfg,
    StairTerminationsCfg,
    StairCurriculumCfg,
    EventCfg,
)

# 导入 CPG Action
from locomotion.mdp.actions.cpg_action import CPGJointPositionActionCfg


# ==============================================================================
# CPG Action 配置（针对楼梯优化）
# ==============================================================================

@configclass
class StairCPGActionsCfg:
    """楼梯任务的 CPG + 残差配置。

    楼梯 CPG 参数（与 velocity 的区别）:
        cpg_frequency = 1.0 Hz → 周期 1.0s（velocity: 1.67 Hz, 0.6s）
            - 楼梯需要更慢、更稳定的步态
            - 给足时间让脚完全离地、跨越台阶、稳定着地

        cpg_amplitudes（更大的摆动幅度）:
            - hip_pitch: ±0.45 rad (±26°) - velocity: 0.35 rad
              更大的髋关节前摆，帮助跨越台阶

            - knee: ±0.85 rad (±49°) - velocity: 0.70 rad
              显著增大膝关节弯曲，抬高小腿以跨越 18cm 台阶

            - ankle_pitch: ±0.30 rad (±17°) - velocity: 0.25 rad
              增大踝关节补偿，适应台阶高度变化

        scale = 0.30 → 残差修正能力（velocity: 0.25）
            - 楼梯高度变化大（3-18cm），需要更大的残差修正范围
            - 网络输出 [-1, 1] 映射到 [-17°, 17°]

    设计思路：
        - CPG 提供"标准爬楼步态"（慢速、高抬腿）
        - 网络通过残差适应不同台阶高度
        - 历史观测（5帧）帮助网络预判台阶变化
    """

    JointPositionAction = CPGJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.30,  # 增大残差修正能力（楼梯高度变化大）
        use_default_offset=True,
        # --- CPG 参数（楼梯专用）---
        cpg_frequency=1.0,  # Hz, 周期 = 1.0s（慢速稳定步态）
        cpg_phase_offsets=[0.0, math.pi],  # 左右腿交替
        # 12个关节的 CPG 振幅（针对楼梯优化）:
        cpg_amplitudes=[
            0.45, 0.45,   # hip_pitch: ±26° (大幅前摆跨越台阶)
            0.0,  0.0,    # hip_roll: 侧向由网络控制
            0.0,  0.0,    # hip_yaw: 旋转由网络控制
            0.85, 0.85,   # knee: ±49° (大幅弯曲抬高小腿)
            0.30, 0.30,   # ankle_pitch: ±17° (适应台阶高度)
            0.0,  0.0,    # ankle_roll: 侧向由网络控制
        ],
    )


# ==============================================================================
# 观测配置（增加 CPG 相位）
# ==============================================================================

@configclass
class StairCPGObservationsCfg:
    """楼梯盲走观测 + CPG 相位信号。

    与 stair 原版的区别：增加 cpg_phase (2维)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor 观测 = 原版 6 项 + CPG 相位 (2维)。"""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )
        last_action = ObsTerm(func=mdp.last_action)

        # === 新增: CPG 相位观测 ===
        # period=1.0 与 CPG 频率 1.0Hz 一致
        cpg_phase = ObsTerm(func=mdp.gait_phase, params={"period": 1.0})

        def __post_init__(self):
            self.history_length = 5  # 保持历史观测，帮助感知地形
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 观测 = 原版 + CPG 相位。"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )
        last_action = ObsTerm(func=mdp.last_action)
        cpg_phase = ObsTerm(func=mdp.gait_phase, params={"period": 1.0})

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ==============================================================================
# 主配置: CPG 版楼梯环境
# ==============================================================================

@configclass
class StairCPGEnvCfg(ManagerBasedRLEnvCfg):
    """CPG + 残差的楼梯攀爬环境。

    与原版 StairEnvCfg 的区别:
        - actions → StairCPGActionsCfg (CPG + 残差)
        - observations → StairCPGObservationsCfg (增加 CPG 相位)
    """

    # Scene settings — 复用原版
    scene: StairSceneCfg = StairSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: StairCPGObservationsCfg = StairCPGObservationsCfg()
    actions: StairCPGActionsCfg = StairCPGActionsCfg()  # ← 核心变化
    commands: StairCommandsCfg = StairCommandsCfg()
    # MDP settings — 复用原版
    rewards: StairRewardsCfg = StairRewardsCfg()
    terminations: StairTerminationsCfg = StairTerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: StairCurriculumCfg = StairCurriculumCfg()

    def __post_init__(self):
        """Post initialization — 与原版完全一致。"""
        self.decimation = 10
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class StairCPGPlayEnvCfg(StairCPGEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
