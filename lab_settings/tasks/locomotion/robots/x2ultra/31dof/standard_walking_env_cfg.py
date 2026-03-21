"""Standard Walking — 全向行走任务（更新 PD 增益与默认姿态）

与 Velocity 任务完全相同，仅替换：
  - default_dof_pos：髋部 / 膝盖 / 踝关节更深蹲姿态
  - actuator stiffness / damping：按部署实测参数重新标定
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from assets.robots.config_setting import ZHIYUAN_X2Ultra_31DOF_CFG
from .velocity_env_cfg import RobotEnvCfg, RobotPlayEnvCfg

# ─── 新机器人配置：只改 init_state 和 actuators ─────────────────────────────
ZHIYUAN_X2Ultra_31DOF_STANDARD_WALKING_CFG = ZHIYUAN_X2Ultra_31DOF_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.66),
        joint_pos={
            # LEG — 更深蹲站立姿态
            ".*hip_pitch_joint":   -0.2480,
            ".*knee_joint":         0.5303,
            ".*ankle_pitch_joint": -0.2823,
            # 其余关节保持零位
            ".*hip_roll_joint":     0.0,
            ".*hip_yaw_joint":      0.0,
            ".*ankle_roll.*":       0.0,
            # 腰部
            "waist_yaw_.*":         0.0,
            "waist_pitch_.*":       0.0,
            "waist_roll_.*":        0.0,
            # 手臂（非动作关节）
            ".*shoulder_pitch_.*":  0.0,
            ".*elbow_.*":          -0.9,
            "left_shoulder_roll_joint":  0.2,
            "right_shoulder_roll_joint": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 髋部 + 膝盖
        "legs_hip_knee": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_.*",
                ".*_hip_yaw_.*",
                ".*_hip_roll_.*",
                ".*_knee_.*",
            ],
            effort_limit_sim=120,
            velocity_limit_sim=11.936,
            stiffness={
                ".*_hip_pitch_.*": 120.0,
                ".*_hip_roll_.*":  120.0,
                ".*_hip_yaw_.*":   120.0,
                ".*_knee_.*":      150.0,
            },
            damping={
                ".*_hip_pitch_.*": 5.0,
                ".*_hip_roll_.*":  5.0,
                ".*_hip_yaw_.*":   5.0,
                ".*_knee_.*":      5.0,
            },
            armature=0.01,
        ),
        # 踝关节
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_.*",
                ".*_ankle_roll.*",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_.*": 36,
                ".*_ankle_roll.*":   24,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_.*": 13.088,
                ".*_ankle_roll.*":   15.077,
            },
            stiffness={
                ".*_ankle_pitch_.*": 40.0,
                ".*_ankle_roll.*":   30.0,
            },
            damping={
                ".*_ankle_pitch_.*": 3.0,
                ".*_ankle_roll.*":   2.0,
            },
            armature=0.01,
        ),
        # 腰部（非动作关节，维持零位）
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_.*",
                "waist_pitch_.*",
                "waist_roll_.*",
            ],
            effort_limit_sim={
                "waist_yaw_.*":   120,
                "waist_pitch_.*":  48,
                "waist_roll_.*":   48,
            },
            velocity_limit_sim=13.088,
            stiffness={
                "waist_yaw_.*":   160.0,
                "waist_pitch_.*":  80.0,
                "waist_roll_.*":   80.0,
            },
            damping={
                "waist_yaw_.*":   5.0,
                "waist_pitch_.*": 5.0,
                "waist_roll_.*":  5.0,
            },
            armature=0.01,
        ),
        # 手臂（非动作关节）
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_.*",
                ".*_shoulder_roll_.*",
                ".*_shoulder_yaw.*",
                ".*_elbow_.*",
                ".*_wrist_yaw_.*",
                ".*_wrist_pitch_.*",
                ".*_wrist_roll_.*",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_.*": 36,
                ".*_shoulder_roll_.*":  36,
                ".*_shoulder_yaw.*":    24,
                ".*_elbow_.*":          24,
                ".*_wrist_yaw_.*":      24,
                ".*_wrist_pitch_.*":     4.8,
                ".*_wrist_roll_.*":      4.8,
            },
            velocity_limit_sim=15.077,
            stiffness={
                ".*_shoulder_pitch_.*": 80.0,
                ".*_shoulder_roll_.*":  40.0,
                ".*_shoulder_yaw.*":    40.0,
                ".*_elbow_.*":          40.0,
                ".*_wrist_.*":          40.0,
            },
            damping={
                ".*_shoulder_pitch_.*": 4.0,
                ".*_shoulder_roll_.*":  1.0,
                ".*_shoulder_yaw.*":    1.0,
                ".*_elbow_.*":          1.0,
                ".*_wrist_.*":          1.0,
            },
            armature=0.01,
        ),
        # 头部（非动作关节）
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_yaw_joint",
                "head_pitch_joint",
            ],
            effort_limit_sim={
                "head_yaw_joint":   2.6,
                "head_pitch_joint": 0.6,
            },
            velocity_limit_sim=6.28,
            stiffness=10.0,
            damping=1.0,
            armature=0.01,
        ),
    },
)


# ─── 训练环境配置 ─────────────────────────────────────────────────────────────
@configclass
class StandardWalkingEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ZHIYUAN_X2Ultra_31DOF_STANDARD_WALKING_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )


@configclass
class StandardWalkingPlayEnvCfg(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ZHIYUAN_X2Ultra_31DOF_STANDARD_WALKING_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
