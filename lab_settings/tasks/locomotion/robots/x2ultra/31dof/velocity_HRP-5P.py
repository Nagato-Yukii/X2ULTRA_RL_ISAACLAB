"""Velocity tracking environment for HRP-5P with dynamic clock-based gait control.

架构规范:
    - 算法框架: PPO (复用 BasePPORunnerCfg)
    - 控制频率: 40 Hz (sim.dt=0.0025s, decimation=10)
    - 步态周期: 2s → L=80 步 @40Hz
    - 观测空间: 35维 × 5帧历史 = 175维输入
    - 动作空间: 13维 (12 关节残差 + 1 时钟相位偏移)

与基线 velocity_env_cfg.py 的区别:
    1. ActionsCfg  → ClockJointPositionActionCfg (13维输出，含动态时钟)
    2. ObservationsCfg → 35维自定义观测 (含运动模式 one-hot、时钟相位 sin/cos)
    3. sim.dt=0.0025s + decimation=10 → 精确 40Hz 策略频率
    4. 场景/事件/奖励/终止/课程设置完全复用原版
"""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from locomotion import mdp

# 复用原版场景、事件、奖励、终止、课程配置（同一包内相对导入）
from .velocity_env_cfg import (
    LEG_JOINT_NAMES,
    RobotSceneCfg,
    EventCfg,
    CommandsCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumCfg,
)

# 导入动态时钟 Action
from locomotion.mdp.actions.hrp5p_clock_action import ClockJointPositionActionCfg


# ==============================================================================
# 动作配置: 13 维动态时钟控制
# ==============================================================================

@configclass
class HRP5PActionsCfg:
    """HRP-5P 下肢控制的动作配置。

    网络输出 13 维动作:
        a_t[0:12] — 下肢 12 个关节的目标位置残差 (Position Residuals)
        a_t[12]   — 时钟相位偏移量 a_delta_phi (动态调节步频)

    关节映射 (每条腿 6 DOF, LEG_JOINT_NAMES 展开顺序):
        Hip Pitch, Hip Roll, Hip Yaw, Knee Pitch, Ankle Pitch, Ankle Roll
        (左腿 0-5, 右腿 6-11)

    控制数据流 (在 process_actions / apply_actions 中实现):
        phi_{t+1} = (phi_t + clip(a_delta_phi, -5, 5) + 1) % 80
        q_des     = a_t[0:12] * scale + q_nominal

    注: scale=0.25 ≈ ±14.3° 残差范围，足以修正步态细节而不破坏稳定性。
    """

    JointPositionAction = ClockJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,                        # 残差缩放因子 (rad)
        use_default_offset=True,           # 以 URDF 标称位置为零位基准
        clock_period_steps=80,             # L=80 步 × 25ms = 2s 步态周期
        delta_phi_range=(-5.0, 5.0),       # 最大 ±5 步 = ±125ms 时间偏移
    )


# ==============================================================================
# 观测配置: 35 维 × 5 帧历史
# ==============================================================================

@configclass
class HRP5PObservationsCfg:
    """HRP-5P 观测配置。

    Policy 观测向量 (35 维, 严格按以下顺序拼接):
        1. root_roll_pitch     [2]  — 躯干 Roll, Pitch（忽略 Yaw）
        2. base_ang_vel        [3]  — 躯干角速度 (IMU 局部系，×0.2)
        3. joint_pos_rel       [12] — 下肢 12 关节位置 (相对标称位置)
        4. joint_vel_rel       [12] — 下肢 12 关节速度 (×0.05)
        5. motion_mode_onehot  [3]  — 运动模式 one-hot [Forward, Inplace, Standing]
        6. mode_reference      [1]  — 模式参考值 (前向速度 or 转向角速度)
        7. hrp5p_clock_phase   [2]  — 步态时钟 [sin(2π·phi/80), cos(2π·phi/80)]
        合计: 2+3+12+12+3+1+2 = 35 维

    Critic 额外拥有特权信息 base_lin_vel [3]，总计 38 维。

    历史堆叠: history_length=5 (当前帧 + 4 历史帧)，实际网络输入 = 35 × 5 = 175 维。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy 观测组: 35 维标准观测，含噪声，含历史堆叠。"""

        # 1. 躯干姿态: Roll, Pitch (2维)
        root_roll_pitch = ObsTerm(
            func=mdp.root_roll_pitch,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # 2. 躯干角速度 (3维, IMU 局部系)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        # 3. 下肢关节位置: 12维
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )

        # 4. 下肢关节速度: 12维
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )

        # 5. 运动模式 One-Hot 指令: [Forward, Inplace, Standing] (3维)
        motion_mode_onehot = ObsTerm(
            func=mdp.motion_mode_onehot,
            params={"command_name": "base_velocity"},
        )

        # 6. 模式参考值: 标量 (1维) — 行走→lin_vel_x, 踏步→ang_vel_z
        mode_reference = ObsTerm(
            func=mdp.mode_reference,
            params={"command_name": "base_velocity"},
        )

        # 7. 步态时钟相位: [sin, cos] (2维)
        hrp5p_clock_phase = ObsTerm(
            func=mdp.hrp5p_clock_phase,
        )

        def __post_init__(self):
            self.history_length = 5          # 当前帧 + 4 历史帧
            self.enable_corruption = True    # 训练时启用观测噪声
            self.concatenate_terms = True    # 拼接为单一向量

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 观测组: 38 维 (Policy 35维 + 特权 base_lin_vel 3维)。

        Critic 拥有真实线速度信息 (无噪声)，用于更准确的价值估计。
        """

        # 特权信息: 真实底盘线速度 (3维)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # === 以下与 PolicyCfg 完全一致（无噪声版）===

        root_roll_pitch = ObsTerm(func=mdp.root_roll_pitch)

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)

        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )

        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )

        motion_mode_onehot = ObsTerm(
            func=mdp.motion_mode_onehot,
            params={"command_name": "base_velocity"},
        )

        mode_reference = ObsTerm(
            func=mdp.mode_reference,
            params={"command_name": "base_velocity"},
        )

        hrp5p_clock_phase = ObsTerm(func=mdp.hrp5p_clock_phase)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ==============================================================================
# 主环境配置
# ==============================================================================

@configclass
class HRP5PEnvCfg(ManagerBasedRLEnvCfg):
    """HRP-5P 速度跟踪环境 (动态时钟步态控制)。

    控制规格:
        策略频率: 40 Hz  (sim.dt=0.0025s × decimation=10)
        步态周期: 2.0s   (L=80 步 @40Hz)
        动作维度: 13     (12 关节残差 + 1 相位偏移)
        观测维度: 35     (×5帧历史 = 175维)

    场景/事件/奖励/终止/课程: 完全复用基线 velocity_env_cfg.py
    """

    # 场景设置 — 复用原版
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP 核心设置
    observations: HRP5PObservationsCfg = HRP5PObservationsCfg()
    actions: HRP5PActionsCfg = HRP5PActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP 辅助设置 — 复用原版
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后初始化: 设定精确 40Hz 策略频率。

        控制频率计算:
            sim.dt    = 0.0025 s  → 仿真频率 = 400 Hz
            decimation = 10       → 策略频率 = 400 / 10 = 40 Hz ✓
            env.step_dt = sim.dt × decimation = 0.025 s (25ms per policy step)
            步态周期 T = L × step_dt = 80 × 0.025 = 2.0 s ✓

        与基线的区别: 基线使用 sim.dt=0.002s + decimation=10 → 50Hz
        """
        # --- 通用设置 ---
        self.decimation = 10
        self.episode_length_s = 20.0

        # --- 仿真参数 ---
        self.sim.dt = 0.0025                                        # 400Hz 仿真
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # --- 传感器更新周期 ---
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # --- 地形课程 ---
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class HRP5PPlayEnvCfg(HRP5PEnvCfg):
    """HRP-5P 的推理/可视化环境配置 (少量环境，最大速度范围)。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # 推理时使用最大速度范围以展示全部能力
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
