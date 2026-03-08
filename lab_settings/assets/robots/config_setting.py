# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from . import actuators as unitree_actuators

UNITREE_MODEL_DIR = "/home/suzumiyaharuhi/X2Ultra_RL_IsaacLab/robot_model"#"path/to/unitree_model"  # Replace with the actual path to your unitree_model directory
UNITREE_ROS_DIR = "/home/suzumiyaharuhi/X2Ultra_RL_IsaacLab/robot_ros"#"path/to/unitree_ros"  # Replace with the actual path to your unitree_ros package

@configclass
class X2UltraArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9

@configclass
class X2UltraUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )

@configclass
class X2UltraUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True # 用胶囊体进行碰撞仿真
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    def replace_asset(self, meshes_dir, urdf_path):
        """Replace the asset with a temporary copy to avoid modifying the original asset.

        When need to change the collisions, place the modified URDF file separately in this repository,
        and let `meshes_dir` be provided by `unitree_ros`.
        This function will auto construct a complete `robot_description` file structure in the `/tmp` directory.
        Note: The mesh references inside the URDF should be in the same directory level as the URDF itself.
        """
        tmp_meshes_dir = "/tmp/IsaacLab/unitree_rl_lab/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/unitree_rl_lab", exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = "/tmp/IsaacLab/unitree_rl_lab/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)

""" Configuration for robots."""
ZHIYUAN_X2Ultra_31DOF_CFG = X2UltraArticulationCfg(
    # 使用简化碰撞的 URDF，避免复杂 mesh 碰撞导致仿真不稳定
    spawn=X2UltraUrdfFileCfg(
        asset_path=f"{UNITREE_ROS_DIR}/robots/x2ultra_description/x2_ultra_simple_collision.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.66),
        joint_pos={
            # LEG — 微曲站立姿态
            ".*hip_pitch_joint": -0.15,
            ".*knee_joint": 0.30,
            ".*ankle_pitch_joint": -0.15,
            # ARM
            ".*shoulder_pitch_joint": 0.3,
            ".*elbow_joint": -0.9,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 大关节：髋部 + 膝盖 + 腰部yaw
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
                ".*_hip_pitch_.*": 200.0,
                ".*_hip_roll_.*": 200.0,
                ".*_hip_yaw_.*": 150.0,
                ".*_knee_.*": 200.0,
            },
            damping={
                ".*_hip_pitch_.*": 8.0,
                ".*_hip_roll_.*": 8.0,
                ".*_hip_yaw_.*": 6.0,
                ".*_knee_.*": 8.0,
            },
            armature=0.01,
        ),
        # 踝关节 — 平衡最关键
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_.*",
                ".*_ankle_roll.*",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_.*": 36,
                ".*_ankle_roll.*": 24,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_.*": 13.088,
                ".*_ankle_roll.*": 15.077,
            },
            stiffness={
                ".*_ankle_pitch_.*": 60.0,
                ".*_ankle_roll.*": 60.0,
            },
            damping={
                ".*_ankle_pitch_.*": 5.0,
                ".*_ankle_roll.*": 5.0,
            },
            armature=0.01,
        ),
        # 腰部
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_.*",
                "waist_pitch_.*",
                "waist_roll_.*",
            ],
            effort_limit_sim={
                "waist_yaw_.*": 120,
                "waist_pitch_.*": 48,
                "waist_roll_.*": 48,
            },
            velocity_limit_sim=13.088,
            stiffness={
                "waist_yaw_.*": 200.0,
                "waist_pitch_.*": 100.0,
                "waist_roll_.*": 100.0,
            },
            damping={
                "waist_yaw_.*": 8.0,
                "waist_pitch_.*": 5.0,
                "waist_roll_.*": 5.0,
            },
            armature=0.01,
        ),
        # 手臂
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
                ".*_shoulder_roll_.*": 36,
                ".*_shoulder_yaw.*": 24,
                ".*_elbow_.*": 24,
                ".*_wrist_yaw_.*": 24,
                ".*_wrist_pitch_.*": 4.8,
                ".*_wrist_roll_.*": 4.8,
            },
            velocity_limit_sim=15.077,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        # 头部
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_yaw_joint",
                "head_pitch_joint",
            ],
            effort_limit_sim={
                "head_yaw_joint": 2.6,
                "head_pitch_joint": 0.6,
            },
            velocity_limit_sim=6.28,
            stiffness=10.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    #joint_sdk_names任意顺序补充
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        "head_yaw_joint",
        "head_pitch_joint",
    ],

)