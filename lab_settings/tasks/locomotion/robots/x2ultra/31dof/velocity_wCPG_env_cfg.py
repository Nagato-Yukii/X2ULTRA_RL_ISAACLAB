"""Velocity tracking environment with CPG-based residual action.

基于 velocity_env_cfg.py，将 Action 从纯 JointPositionAction 替换为
CPGJointPositionAction，使得 RL 策略只需学习步态残差：

    joint_target = default_pos + network_residual × scale + CPG_signal

与原版的区别：
    1. ActionsCfg 使用 CPGJointPositionActionCfg 代替 JointPositionActionCfg
    2. 观测中新增 cpg_phase（sin/cos 相位信号），供策略感知当前步态阶段
    3. 其余 Scene / Event / Reward / Termination / Curriculum 完全复用原版
"""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from locomotion import mdp

# 复用原版中的场景、事件、奖励等配置（相对导入，同一个包内）
from .velocity_env_cfg import (
    LEG_JOINT_NAMES,
    RobotSceneCfg,
    EventCfg,
    CommandsCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumCfg,
)

# 导入 CPG Action
from locomotion.mdp.actions.cpg_action import CPGJointPositionActionCfg


# ==============================================================================
# 唯一变化: ActionsCfg — 使用 CPG + 残差
# ==============================================================================

@configclass
class CPGActionsCfg:
    """使用 CPG + 残差的 Action 配置。

    CPG 参数说明（参考 Unity X02randAgent_v1.cs 调整）:
        cpg_frequency = 1.67 Hz → 周期 0.6s，与 Unity 训练一致
        cpg_phase_offsets = [0.0, π] → 左右腿交替半周期 (trot 步态)
        cpg_amplitudes: 各关节的 CPG 摆动振幅 (rad)
            - hip_pitch: ±0.35 rad (±20°) - Unity: 10-40°，取中值30°的一半
            - knee: ±0.70 rad (±40°) - Unity 使用2倍振幅，且膝关节需要更大摆动
            - ankle_pitch: ±0.25 rad (±14°) - Unity: 10-40°，取中值
            - hip_roll / hip_yaw / ankle_roll: 0.0 (侧向/旋转由网络残差控制)
        scale = 0.25 → 网络残差的缩放因子，增大以匹配 Unity 的修正能力

    注意: Unity v1 中 knee 使用反向且2倍振幅 (-2*(dh*uf + d0))，
          这里通过增大振幅来模拟，但方向保持一致（Isaac Lab 的 sin 波自然包含正负）
    """

    JointPositionAction = CPGJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,  # 增大残差修正能力（Unity v1: 60°，这里约14°）
        use_default_offset=True,
        # --- CPG 参数（基于 Unity v1 调整）---
        cpg_frequency=1.67,  # Hz, 周期 = 1/1.67 ≈ 0.6s (Unity: T1=30, dt=0.01, period=0.6s)
        cpg_phase_offsets=[0.0, math.pi],  # 左腿 0, 右腿 π (交替步态)
        # 12个关节的 CPG 振幅 (按 LEG_JOINT_NAMES 展开顺序):
        #   左hip_pitch, 右hip_pitch,
        #   左hip_roll,  右hip_roll,
        #   左hip_yaw,   右hip_yaw,
        #   左knee,      右knee,
        #   左ankle_pitch, 右ankle_pitch,
        #   左ankle_roll, 右ankle_roll
        cpg_amplitudes=[
            0.35, 0.35,   # hip_pitch: ±20° (Unity: 10-40°)
            0.0,  0.0,    # hip_roll: 侧向由网络控制
            0.0,  0.0,    # hip_yaw: 旋转由网络控制
            0.70, 0.70,   # knee: ±40° (Unity: 2倍振幅，20-80°)
            0.25, 0.25,   # ankle_pitch: ±14° (Unity: 10-40°)
            0.0,  0.0,    # ankle_roll: 侧向由网络控制
        ],
    )


# ==============================================================================
# 观测: 在原版基础上增加 CPG 相位观测
# ==============================================================================

@configclass
class CPGObservationsCfg:
    """在原版观测基础上增加 gait_phase 信号。

    gait_phase 提供 [sin(φ), cos(φ)]，让策略知道当前处于步态的哪个阶段，
    从而输出更准确的残差修正。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy 观测 = 原版 6 项 + CPG 相位 (2维)。"""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )
        last_action = ObsTerm(func=mdp.last_action)

        # === 新增: CPG 相位观测 ===
        # gait_phase 返回 [sin(φ), cos(φ)]，维度=2
        # period=0.6 与 CPG 频率 1.67Hz 一致（参考 Unity v1）
        cpg_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.6})

        def __post_init__(self):
            self.history_length = 5
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
            func=mdp.joint_vel_rel, scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )
        last_action = ObsTerm(func=mdp.last_action)
        cpg_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.6})

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ==============================================================================
# 主配置: CPG 版环境
# ==============================================================================

@configclass
class CPGRobotEnvCfg(ManagerBasedRLEnvCfg):
    """CPG + 残差 的 velocity-tracking 环境。

    与原版 RobotEnvCfg 的区别:
        - actions → CPGActionsCfg (CPG + 残差)
        - observations → CPGObservationsCfg (增加 CPG 相位)
    """

    # Scene settings — 复用原版
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: CPGObservationsCfg = CPGObservationsCfg()
    actions: CPGActionsCfg = CPGActionsCfg()  # ← 核心变化
    commands: CommandsCfg = CommandsCfg()
    # MDP settings — 复用原版
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

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
class CPGRobotPlayEnvCfg(CPGRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
