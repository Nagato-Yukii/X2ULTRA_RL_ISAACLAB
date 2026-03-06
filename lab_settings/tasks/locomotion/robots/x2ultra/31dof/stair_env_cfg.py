"""
Stair-Climbing Task Environment Configuration
================================================
在 velocity-tracking 任务基础上扩展:
1. 楼梯地形 + 课程化难度（step_height 从 3cm → 18cm）
2. 盲走策略（不使用 height_scan），依靠本体感觉 + 历史观测隐式感知地形
3. 调整后的奖励：去掉绝对 base_height，放宽 orientation 惩罚，
   提高 foot_clearance 目标高度
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import ZHIYUAN_X2Ultra_31DOF_CFG as ROBOT_CFG
from .... import mdp

# ── 沿用 velocity 任务的共享配置 ──────────────────────────
from .velocity_env_cfg import (
    LEG_JOINT_NAMES,
    EventCfg,       # 物理材质随机化 / 质量随机化 / 推力扰动 / 关节重置
    CommandsCfg,     # 速度指令 + 指令课程
    ActionsCfg,      # 12-DoF 下肢位置控制
)

# ===========================================================================
#                              地形配置
# ===========================================================================
STAIRS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,            # 10 个难度级别（行）
    num_cols=20,            # 20 列，按 proportion 分配给不同子地形
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        # 20% 平地：最简单难度下几乎全是平地，保证 early-stage 稳定
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),
        # 40% 金字塔楼梯（上坡 → 中心平台 → 下坡）
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.03, 0.18),   # difficulty 线性插值 3cm → 18cm
            step_width=0.30,                   # 每级台阶深度 30cm
            platform_width=1.5,                # 中心平台宽 1.5m
            border_width=0.0,
            holes=False,
        ),
        # 40% 倒金字塔楼梯（下坡 → 中心凹坑 → 上坡）
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.03, 0.18),
            step_width=0.30,
            platform_width=1.5,
            border_width=0.0,
            holes=False,
        ),
    },
)


# ===========================================================================
#                              场景
# ===========================================================================
@configclass
class StairSceneCfg(InteractiveSceneCfg):
    """楼梯场景：替换地形为课程化楼梯，其余与 velocity 任务相同。"""

    # ── ground terrain ──
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIRS_TERRAIN_CFG,
        max_init_terrain_level=5,   # 训练开始时最高分配到第 5 级（共 10 级）
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # ── robot ──
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ── sensors ──
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 17×11=187 points
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # ── lights ──
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ===========================================================================
#                           观测（盲走）
# ===========================================================================
@configclass
class StairObservationsCfg:
    """盲走观测：与 velocity 任务相同，依靠 history_length=5 隐式感知地形。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor 观测：6 项本体感觉，不含地形感知。"""

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

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 观测：含 privileged base_lin_vel。"""

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

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ===========================================================================
#                           奖励（针对楼梯调整）
# ===========================================================================
@configclass
class StairRewardsCfg:
    """
    与 velocity 任务的主要差异:
    - 移除 base_height 惩罚（在楼梯上绝对高度会变化）
    - 降低 flat_orientation_l2 权重（上楼梯时轻微前倾是正常的）
    - 提高 foot_clearance 目标高度（脚需要抬高才能跨上台阶）
    """

    # ── task: 速度跟踪 ──
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.5)

    # ── base penalties ──
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # ── robot orientation（降低权重 -5 → -2，楼梯上轻微前倾属正常） ──
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # NOTE: 不使用绝对高度惩罚 base_height，因为在楼梯上绝对高度不固定

    # ── feet ──
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_body_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": -0.48,  # 体坐标系：站立脚 Z ≈ -0.63，抬高 15cm → -0.48
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # ── contacts ──
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


# ===========================================================================
#                           终止条件
# ===========================================================================
@configclass
class StairTerminationsCfg:
    """与 velocity 任务相同——绝对高度兜底 + 姿态翻倒检测。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 绝对高度 < 0.15m 时终止（楼梯上的正常高度远高于此）
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.15})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})


# ===========================================================================
#                           课程
# ===========================================================================
@configclass
class StairCurriculumCfg:
    """
    课程学习:
    - terrain_levels: 根据行走距离在地形难度级别间升降
      （走得远 → 升级到更高楼梯；走不远 → 降级到更矮楼梯）
    - lin_vel_cmd_levels: 根据速度跟踪奖励逐步扩大指令速度范围
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


# ===========================================================================
#                      主环境配置 & Play 配置
# ===========================================================================
@configclass
class StairEnvCfg(ManagerBasedRLEnvCfg):
    """楼梯攀爬训练环境。"""

    # Scene
    scene: StairSceneCfg = StairSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic
    observations: StairObservationsCfg = StairObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP
    rewards: StairRewardsCfg = StairRewardsCfg()
    terminations: StairTerminationsCfg = StairTerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: StairCurriculumCfg = StairCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # 如果配置了 terrain_levels 课程，则自动打开地形生成器的 curriculum 模式
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class StairPlayEnvCfg(StairEnvCfg):
    """推理/可视化配置：少量环境 + 完整难度范围 + 最大速度指令。"""

    def __post_init__(self):
        super().__post_init__()
        # 较少环境数量便于可视化
        self.scene.num_envs = 32
        # 保留多种难度以观察不同楼梯高度下的表现
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        # 使用完整指令范围
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
