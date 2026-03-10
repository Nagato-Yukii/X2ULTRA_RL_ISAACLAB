#!/usr/bin/env python3

"""
HRP-5P Sim2Sim Deployment — Isaac Lab → MuJoCo
===============================================
将基于 Isaac Lab / RSL_RL 训练的 HRP-5P 策略网络部署到 MuJoCo 进行
Sim-to-Sim 仿真验证。

观测空间 (35维 × history_length=5 = 175维输入):
  [1] Roll, Pitch                          (2D) ← projected_gravity_b
  [2] base_ang_vel_body × 0.2             (3D) ← IMU 机体系角速度
  [3] joint_pos_rel (12条腿关节)           (12D) ← q - q_nominal
  [4] joint_vel_rel × 0.05 (12条腿关节)   (12D) ← q_dot
  [5] motion_mode_onehot                  (3D) ← 键盘 1/2/3 切换
  [6] mode_reference                      (1D) ← 速度标量指令
  [7] clock_phase [sin, cos]              (2D) ← phi ∈ [0,80)

动作空间 (网络输出 13维，但推理时只取前12维关节残差):
  a_t[0:12] → q_des = a_t[0:12] × scale + q_nominal (下肢12关节)
  a_t[12]   → 时钟相位偏移 Δφ (在此 sim2sim 中仍作为输入反馈)
              注：在 Isaac Lab 环境中 phi 由 ClockJointPositionAction 管理，
              在 sim2sim 中我们在主控制循环中手动维护 phi。

键盘控制:
  1       → Standing  模式: mode = [0, 0, 1]
  2       → Inplace   模式: mode = [0, 1, 0]
  3       → Forward   模式: mode = [1, 0, 0]
  ↑ / ↓  → 前向目标线速度 ±0.1 m/s  (Forward 模式下生效)
  ← / →  → 目标转向角速度 ±0.1 rad/s (Inplace 模式下生效)
  Space   → 重置速度指令为 0
  R       → 重置机器人到初始姿态
  Q / Esc → 退出仿真

Usage:
  python sim2sim_hrp5p.py --policy path/to/policy.pt --xml path/to/scene.xml
  python sim2sim_hrp5p.py --policy path/to/policy.onnx --xml path/to/scene.xml

"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

# ── 使 core 包可被导入（以脚本方式直接运行时）──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.math_utils import get_projected_gravity, quat_rotate_inverse
from core.mujoco_env import JointConfig, MujocoEnv
from core.policy_runner import PolicyRunner


# ═══════════════════════════════════════════════════════════════════════════════
# 硬编码的任务常量（与 velocity_HRP-5P.py 严格一致）
# ═══════════════════════════════════════════════════════════════════════════════

# 40Hz 控制频率：sim_dt=0.0025s × decimation=10 → step_dt=0.025s
SIM_DT        = 0.0025    # 仿真物理步长 (s)
DECIMATION    = 10        # 策略每隔多少个物理步推理一次

# 步态时钟参数
CLOCK_L       = 80        # 时钟周期步长 L (= 2s ÷ 0.025s/step)
DELTA_PHI_MIN = -5.0      # φ 偏移量裁剪下界 (步长)
DELTA_PHI_MAX =  5.0      # φ 偏移量裁剪上界 (步长)

# 动作缩放 (与 ClockJointPositionActionCfg.scale 一致)
ACTION_SCALE  = 0.25

# 观测缩放系数
ANG_VEL_SCALE = 0.2       # 躯干角速度缩放
JOINT_VEL_SCALE = 0.05    # 关节速度缩放

# 历史长度 (当前帧 + 4 历史帧，与 history_length=5 一致)
HISTORY_LENGTH = 5
FRAME_DIM      = 35       # 单帧观测维度

# ── Per-term 观测定义（与 HRP5PObservationsCfg.PolicyCfg 严格一致）─────────────
# Isaac Lab 在设置 group-level history_length 时，为每个 term 各自独立创建
# CircularBuffer，因此最终 175 维向量的布局是 per-term 堆叠，而非 per-frame。
TERM_NAMES = [
    "root_roll_pitch",      # 2D
    "base_ang_vel",         # 3D  (×ANG_VEL_SCALE)
    "joint_pos_rel",        # 12D
    "joint_vel_rel",        # 12D (×JOINT_VEL_SCALE)
    "motion_mode_onehot",   # 3D
    "mode_reference",       # 1D
    "hrp5p_clock_phase",    # 2D
]
TERM_DIMS = [2, 3, 12, 12, 3, 1, 2]   # 合计 35

# 速度指令平滑
VEL_DELTA     = 0.1       # 每次按键的速度增量 (m/s 或 rad/s)
VX_MAX        = 1.5       # 前向速度上限 (m/s)
VX_MIN        = -0.5      # 前向速度下限
WZ_MAX        = 1.0       # 转向角速度上限 (rad/s)
WZ_MIN        = -1.0      # 转向角速度下限

# 模式阈值（与 observations.py motion_mode_onehot 一致）
LIN_THRESHOLD = 0.1       # m/s，超过则认为 Forward 模式
ANG_THRESHOLD = 0.05      # rad/s，超过则认为 Inplace 模式

# ── 下肢 12 关节名称（顺序必须与 Isaac Lab LEG_JOINT_NAMES regex 展开后一致）──────
# Isaac Lab 用 regex [".*_hip_pitch_joint", ".*_hip_roll_joint", ...] 展开时，
# 每个 pattern 先匹配 left 再匹配 right（按 URDF Articulation 内部 DOF 排序），
# 因此展开结果是：按关节类型分组，同类型内左腿在前、右腿在后。
#
# 展开顺序（与策略网络 action 向量一一对应）:
#   idx 0  : left_hip_pitch_joint
#   idx 1  : right_hip_pitch_joint
#   idx 2  : left_hip_roll_joint
#   idx 3  : right_hip_roll_joint
#   idx 4  : left_hip_yaw_joint
#   idx 5  : right_hip_yaw_joint
#   idx 6  : left_knee_joint
#   idx 7  : right_knee_joint
#   idx 8  : left_ankle_pitch_joint
#   idx 9  : right_ankle_pitch_joint
#   idx 10 : left_ankle_roll_joint
#   idx 11 : right_ankle_roll_joint
LEG_JOINT_NAMES_DEFAULT = [
    "left_hip_pitch_joint",    # 0  — hip pitch 左
    "right_hip_pitch_joint",   # 1  — hip pitch 右
    "left_hip_roll_joint",     # 2  — hip roll  左
    "right_hip_roll_joint",    # 3  — hip roll  右
    "left_hip_yaw_joint",      # 4  — hip yaw   左
    "right_hip_yaw_joint",     # 5  — hip yaw   右
    "left_knee_joint",         # 6  — knee      左
    "right_knee_joint",        # 7  — knee      右
    "left_ankle_pitch_joint",  # 8  — ankle pitch 左
    "right_ankle_pitch_joint", # 9  — ankle pitch 右
    "left_ankle_roll_joint",   # 10 — ankle roll  左
    "right_ankle_roll_joint",  # 11 — ankle roll  右
]

# ── 全身关节的默认标称位置 (q_nominal)，单位 rad ──────────────────────────────────
# 根据 config_setting.py 的 init_state:
#   腿部: hip_pitch:-0.15,  knee: 0.30,  ankle_pitch:-0.15
#   手臂: shoulder_pitch: 0.3, elbow: -0.9
#         left_shoulder_roll: 0.2, right_shoulder_roll: -0.2
ALL_JOINT_DEFAULTS_DEFAULT = {
    # ------ 腿部 12 关节 ------
    "left_hip_pitch_joint":   -0.15,
    "right_hip_pitch_joint":  -0.15,
    "left_hip_roll_joint":     0.0,
    "right_hip_roll_joint":    0.0,
    "left_hip_yaw_joint":      0.0,
    "right_hip_yaw_joint":     0.0,
    "left_knee_joint":         0.30,
    "right_knee_joint":        0.30,
    "left_ankle_pitch_joint": -0.15,
    "right_ankle_pitch_joint":-0.15,
    "left_ankle_roll_joint":   0.0,
    "right_ankle_roll_joint":  0.0,
    # ------ 腰部 3 关节 ------
    "waist_yaw_joint":         0.0,
    "waist_pitch_joint":       0.0,
    "waist_roll_joint":        0.0,
    # ------ 手臂 14 关节 ------
    "left_shoulder_pitch_joint":  0.3,
    "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint":   0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_shoulder_yaw_joint":    0.0,
    "right_shoulder_yaw_joint":   0.0,
    "left_elbow_joint":          -0.9,
    "right_elbow_joint":         -0.9,
    "left_wrist_roll_joint":      0.0,
    "right_wrist_roll_joint":     0.0,
    "left_wrist_pitch_joint":     0.0,
    "right_wrist_pitch_joint":    0.0,
    "left_wrist_yaw_joint":       0.0,
    "right_wrist_yaw_joint":      0.0,
    # ------ 头部 2 关节 ------
    "head_yaw_joint":             0.0,
    "head_pitch_joint":           0.0,
}

# ── 全身关节默认 PD 增益（与 config_setting.py 训练配置完全一致）──────────────────────
ALL_PD_DEFAULT = {
    # ------ 腿部 ------
    "left_hip_pitch_joint":   {"kp": 200.0, "kd": 8.0},
    "right_hip_pitch_joint":  {"kp": 200.0, "kd": 8.0},
    "left_hip_roll_joint":    {"kp": 200.0, "kd": 8.0},
    "right_hip_roll_joint":   {"kp": 200.0, "kd": 8.0},
    "left_hip_yaw_joint":     {"kp": 150.0, "kd": 6.0},
    "right_hip_yaw_joint":    {"kp": 150.0, "kd": 6.0},
    "left_knee_joint":        {"kp": 200.0, "kd": 8.0},
    "right_knee_joint":       {"kp": 200.0, "kd": 8.0},
    "left_ankle_pitch_joint": {"kp": 60.0,  "kd": 5.0},
    "right_ankle_pitch_joint":{"kp": 60.0,  "kd": 5.0},
    "left_ankle_roll_joint":  {"kp": 60.0,  "kd": 5.0},
    "right_ankle_roll_joint": {"kp": 60.0,  "kd": 5.0},
    # ------ 腰部 ------
    "waist_yaw_joint":        {"kp": 200.0, "kd": 8.0},
    "waist_pitch_joint":      {"kp": 100.0, "kd": 5.0},
    "waist_roll_joint":       {"kp": 100.0, "kd": 5.0},
    # ------ 手臂 ------
    "left_shoulder_pitch_joint":  {"kp": 40.0, "kd": 2.0},
    "right_shoulder_pitch_joint": {"kp": 40.0, "kd": 2.0},
    "left_shoulder_roll_joint":   {"kp": 40.0, "kd": 2.0},
    "right_shoulder_roll_joint":  {"kp": 40.0, "kd": 2.0},
    "left_shoulder_yaw_joint":    {"kp": 40.0, "kd": 2.0},
    "right_shoulder_yaw_joint":   {"kp": 40.0, "kd": 2.0},
    "left_elbow_joint":           {"kp": 40.0, "kd": 2.0},
    "right_elbow_joint":          {"kp": 40.0, "kd": 2.0},
    "left_wrist_roll_joint":      {"kp": 40.0, "kd": 2.0},
    "right_wrist_roll_joint":     {"kp": 40.0, "kd": 2.0},
    "left_wrist_pitch_joint":     {"kp": 40.0, "kd": 2.0},
    "right_wrist_pitch_joint":    {"kp": 40.0, "kd": 2.0},
    "left_wrist_yaw_joint":       {"kp": 40.0, "kd": 2.0},
    "right_wrist_yaw_joint":      {"kp": 40.0, "kd": 2.0},
    # ------ 头部 ------
    "head_yaw_joint":             {"kp": 10.0, "kd": 1.0},
    "head_pitch_joint":           {"kp": 10.0, "kd": 1.0},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 观测拼接辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def compute_roll_pitch(quat: np.ndarray) -> np.ndarray:
    """从四元数计算躯干 Roll 和 Pitch，忽略 Yaw（航向不变性）。

    基于 projected_gravity_b = R^T @ [0,0,-1] 推算，
    与 Isaac Lab observations.py::root_roll_pitch 等价。

    Args:
        quat: MuJoCo 格式四元数 [qw, qx, qy, qz]

    Returns:
        np.ndarray shape (2,): [roll, pitch]  (单位 rad)
    """
    g_b = get_projected_gravity(quat)  # [gx, gy, gz] in body frame

    # roll  ≈ atan2(g_b_y, -g_b_z)   (绕 X 轴倾斜)
    # pitch ≈ atan2(-g_b_x, -g_b_z)  (绕 Y 轴倾斜)
    roll  = math.atan2(g_b[1], -g_b[2])
    pitch = math.atan2(-g_b[0], -g_b[2])
    return np.array([roll, pitch], dtype=np.float32)


def compute_clock_phase(phi: float) -> np.ndarray:
    """将时钟计数 phi 转换为正余弦特征。

    与 Isaac Lab observations.py::hrp5p_clock_phase 等价。

    Args:
        phi: 当前时钟计数，ϕ ∈ [0, CLOCK_L)

    Returns:
        np.ndarray shape (2,): [sin(2π·phi/L), cos(2π·phi/L)]
    """
    angle = 2.0 * math.pi * phi / CLOCK_L
    return np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)

def print_obs_debug(frame: np.ndarray, step: int = 0):
    """格式化打印单帧 35 维观测，用于与 Isaac Lab 零误差对齐比对。"""
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    print(f"\n[DEBUG] ── 第 {step} 物理步的观测帧 (35维) 详细拆解 ──")
    print(f"  [0:2]   Roll, Pitch (2D)     = {frame[0:2]}")
    print(f"  [2:5]   Base Ang Vel (3D)    = {frame[2:5]}  (机体系, 已乘 {ANG_VEL_SCALE})")
    print(f"  [5:17]  Leg Pos Res (12D)    = {frame[5:17]}  (q - q_nom, 未缩放)")
    print(f"  [17:29] Leg Vel (12D)        = {frame[17:29]}  (q_dot, 已乘 {JOINT_VEL_SCALE})")
    print(f"  [29:32] Motion Mode (3D)     = {frame[29:32]}")
    print(f"  [32]    Mode Ref (1D)        = [{frame[32]:.4f}]")
    print(f"  [33:35] Clock [sin, cos] (2D)= {frame[33:35]}")
    print("──────────────────────────────────────────────────────────\n")
    np.set_printoptions() # 恢复默认


def print_full_obs_debug(obs: np.ndarray, step: int = 0):
    """格式化打印 per-term 堆叠的 175 维完整观测，用于与 Isaac Lab 第 0 步精准对比。

    布局 (Isaac Lab CircularBuffer per-term stacking):
        [term_0_oldest, ..., term_0_newest,   ← root_roll_pitch: 2×5=10
         term_1_oldest, ..., term_1_newest,   ← base_ang_vel:    3×5=15
         term_2_oldest, ..., term_2_newest,   ← joint_pos_rel:  12×5=60
         term_3_oldest, ..., term_3_newest,   ← joint_vel_rel:  12×5=60
         term_4_oldest, ..., term_4_newest,   ← motion_mode:     3×5=15
         term_5_oldest, ..., term_5_newest,   ← mode_reference:  1×5=5
         term_6_oldest, ..., term_6_newest]   ← clock_phase:     2×5=10
                                               total = 175
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    print(f"\n{'='*80}")
    print(f"[DEBUG] 第 {step} 步 — 完整 {obs.shape[0]} 维观测 (per-term × {HISTORY_LENGTH} history)")
    print(f"{'='*80}")

    offset = 0
    for name, dim in zip(TERM_NAMES, TERM_DIMS):
        total = dim * HISTORY_LENGTH
        term_data = obs[offset:offset + total]
        print(f"\n  [{offset}:{offset+total}] {name} ({dim}D × {HISTORY_LENGTH} = {total})")
        for h in range(HISTORY_LENGTH):
            age = HISTORY_LENGTH - 1 - h   # t-4, t-3, t-2, t-1, t
            start = h * dim
            end   = start + dim
            label = f"t-{age}" if age > 0 else "t  "
            print(f"    {label}: {term_data[start:end]}")
        offset += total

    print(f"\n  Total consumed: {offset} dims (expected {sum(d * HISTORY_LENGTH for d in TERM_DIMS)})")
    print(f"{'='*80}\n")
    np.set_printoptions()



def build_obs_frame(
    quat:          np.ndarray,   # [qw, qx, qy, qz]
    ang_vel_body:  np.ndarray,   # [wx, wy, wz] in body frame
    joint_pos:     np.ndarray,   # [12] 关节位置
    joint_vel:     np.ndarray,   # [12] 关节速度
    joint_nominal: np.ndarray,   # [12] 标称位置
    motion_mode:   np.ndarray,   # [3]  One-Hot [Forward, Inplace, Standing]
    mode_ref:      float,        # 标量速度参考值
    phi:           float,        # 时钟计数
) -> np.ndarray:
    """拼接单帧 35 维观测向量。

    严格按照 velocity_HRP-5P.py HRP5PObservationsCfg.PolicyCfg 的顺序：
      idx  0- 1: Roll, Pitch                         (2D)
      idx  2- 4: ang_vel_body × ANG_VEL_SCALE        (3D)
      idx  5-16: joint_pos_rel = q - q_nominal       (12D)
      idx 17-28: joint_vel × JOINT_VEL_SCALE         (12D)
      idx 29-31: motion_mode one-hot                 (3D)
      idx 32   : mode_reference scalar               (1D)
      idx 33-34: [sin(2π·phi/L), cos(2π·phi/L)]     (2D)

    Returns:
        np.ndarray shape (35,)
    """
    roll_pitch    = compute_roll_pitch(quat)                         # (2,)
    ang_vel_scaled = ang_vel_body * ANG_VEL_SCALE                   # (3,)
    pos_rel       = joint_pos - joint_nominal                        # (12,)
    vel_scaled    = joint_vel * JOINT_VEL_SCALE                      # (12,)
    clock_feat    = compute_clock_phase(phi)                         # (2,)
    mode_ref_arr  = np.array([mode_ref], dtype=np.float32)          # (1,)

    frame = np.concatenate([
        roll_pitch,       # [0:2]   Roll, Pitch
        ang_vel_scaled,   # [2:5]   躯干角速度 (×0.2)
        pos_rel,          # [5:17]  下肢关节位置残差
        vel_scaled,       # [17:29] 下肢关节速度 (×0.05)
        motion_mode,      # [29:32] 运动模式 One-Hot
        mode_ref_arr,     # [32]    模式参考值标量
        clock_feat,       # [33:35] 步态时钟 [sin, cos]
    ])
    assert frame.shape == (FRAME_DIM,), f"Frame dim error: {frame.shape}"
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Term 历史缓冲区（匹配 Isaac Lab CircularBuffer 行为）
# ═══════════════════════════════════════════════════════════════════════════════

class PerTermHistoryBuffer:
    """Per-term 独立历史缓冲区，严格匹配 Isaac Lab 的 CircularBuffer 行为。

    Isaac Lab 在设置 group-level ``history_length`` 时，会为每个 observation
    term 独立创建一个 ``CircularBuffer(max_len=history_length)``。
    最终观测向量的拼接方式是::

        concat([
            term_0_oldest, ..., term_0_newest,   # dim_0 × H
            term_1_oldest, ..., term_1_newest,   # dim_1 × H
            ...                                   # ...
        ])

    **与旧版 HistoryBuffer 的关键差异：**

    1. **Per-term vs per-frame**:
       旧版将 35 维整帧拼好后入 buffer → [frame_t, frame_{t-1}, ...]。
       新版每个 term 各自独立维护 deque → per-term 拼接。
       两者的 175 维元素排列完全不同。

    2. **时间顺序**:
       旧版 newest→oldest；新版 oldest→newest（与 CircularBuffer.buffer 一致）。
    """

    def __init__(self, term_dims: list[int], history_length: int = HISTORY_LENGTH):
        self.term_dims = term_dims
        self.history_length = history_length
        self.num_terms = len(term_dims)
        self.frame_dim = sum(term_dims)
        self.obs_size = self.frame_dim * history_length
        self._bufs: list[deque] = [
            deque(maxlen=history_length) for _ in range(self.num_terms)
        ]

    def reset(self, init_values: list[np.ndarray] | None = None):
        """用给定的初始值预填充所有 term 的历史缓冲区。

        Args:
            init_values: 长度为 num_terms 的列表，每个元素为该 term 的初始值。
                         若为 None，则全部填零。
        """
        for i in range(self.num_terms):
            self._bufs[i].clear()
            if init_values is not None:
                v = init_values[i].astype(np.float32).copy()
            else:
                v = np.zeros(self.term_dims[i], dtype=np.float32)
            for _ in range(self.history_length):
                self._bufs[i].append(v.copy())

    def push(self, term_values: list[np.ndarray]):
        """将各 term 的当前值分别压入各自的历史缓冲区。

        Args:
            term_values: 长度为 num_terms 的列表，term_values[i] 对应第 i 个 term。
        """
        assert len(term_values) == self.num_terms
        for i, v in enumerate(term_values):
            self._bufs[i].append(v.astype(np.float32))

    def get_obs(self) -> np.ndarray:
        """返回 per-term 堆叠的完整观测向量。

        每个 term 内部: oldest → newest (与 Isaac Lab CircularBuffer.buffer 一致)
        term 之间: 按定义顺序 (TERM_NAMES) 拼接

        总维度 = sum(term_dim_i) × history_length = 35 × 5 = 175.
        """
        parts: list[np.ndarray] = []
        for buf in self._bufs:
            # deque 内部: index 0 = oldest, index -1 = newest
            # 直接按 deque 顺序拼接 → oldest→newest  ✅
            parts.append(np.concatenate(list(buf), axis=0))
        return np.concatenate(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 参数解析
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HRP-5P Isaac Lab → MuJoCo Sim2Sim 验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--policy", type=str, required=True,
                        help="Policy 模型路径 (.pt TorchScript 或 .onnx)")
    parser.add_argument("--xml",    type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "robot", "X2_URDF", "scene.xml"
                        ),
                        help="MuJoCo 场景 XML 路径 (默认: sim2sim/robot/X2_URDF/scene.xml)")
    parser.add_argument("--duration", type=float, default=600.0,
                        help="最大仿真时长 (秒), 默认 600 s")
    parser.add_argument("--base-height", type=float, default=0.68,
                        help="初始站立高度 (m), 默认 0.68 (与 x2_ultra.xml body pos 一致)")
    parser.add_argument("--no-keyboard", action="store_true",
                        help="禁用键盘控制")
    parser.add_argument("--mode", type=str, default="standing",
                        choices=["standing", "inplace", "forward"],
                        help="初始运动模式 (默认 standing)")
    parser.add_argument("--vx", type=float, default=0.0,
                        help="初始前向速度指令 (m/s)")
    parser.add_argument("--wz", type=float, default=0.0,
                        help="初始转向角速度指令 (rad/s)")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Banner ─────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  X2Ultra HRP-5P Sim2Sim — Isaac Lab → MuJoCo")
    print("=" * 65)
    print(f"  Policy       : {args.policy}")
    print(f"  XML Scene    : {args.xml}")
    print(f"  Control Freq : {1.0 / (SIM_DT * DECIMATION):.0f} Hz  "
          f"(sim_dt={SIM_DT}s  decimation={DECIMATION})")
    print(f"  Clock Period : L={CLOCK_L} steps × {SIM_DT * DECIMATION * 1000:.0f}ms = "
          f"{CLOCK_L * SIM_DT * DECIMATION:.1f}s gait cycle")
    print(f"  Action Scale : {ACTION_SCALE}")
    print(f"  Obs Dims     : {FRAME_DIM} × {HISTORY_LENGTH} = {FRAME_DIM * HISTORY_LENGTH}")
    print(f"  Num Actions  : 12 (lower limb residuals)")
    print("=" * 65)

    # ── 控制状态 (mutable, 由键盘回调修改) ────────────────────────────────
    #   motion_mode: [Forward, Inplace, Standing] one-hot
    mode_map = {
        "standing": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "inplace":  np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "forward":  np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }
    motion_mode  = mode_map[args.mode].copy()
    mode_ref     = np.array([args.vx if args.mode == "forward" else args.wz],
                             dtype=np.float32)
    should_reset = [False]   # 用列表包装以便在嵌套函数中修改

    # 步态时钟: phi ∈ [0, CLOCK_L) 整数计数，每个控制步 +1 (+ Δφ 偏移)
    phi = 0.0

    # ── 构建 MuJoCo 环境 ───────────────────────────────────────────────────
    print("\n[1/3] 加载 MuJoCo 模型 …")
    # 3. 构造全身 31 个关节的 PD 及初始位置
    joint_configs = []
    for name, pd in ALL_PD_DEFAULT.items():
        nom = ALL_JOINT_DEFAULTS_DEFAULT.get(name, 0.0)
        joint_configs.append(JointConfig(name, pd["kp"], pd["kd"], nom))

    # 初始化 MuJoCo Wrapper (此时会将全身 31 个关节设为指定 PD 与标称位)
    env = MujocoEnv(args.xml, SIM_DT, joint_configs)
    env.print_joint_mapping()

    # 获取 12 个腿部控制关节的 ID 映射 (用于计算策略网络 Action 残差)
    action_ctrl_ids = env.get_joint_ctrl_indices(LEG_JOINT_NAMES_DEFAULT)
    # 获取需要放入 policy obs 的 12 个关节位置/速度传感器 ID
    action_qpos_ids = env.get_joint_qpos_indices(LEG_JOINT_NAMES_DEFAULT)
    action_qvel_ids = env.get_joint_qvel_indices(LEG_JOINT_NAMES_DEFAULT)

    print("\n  [全身有 31 个关节参与 PD 控制保障刚性]")
    print(f"  [Policy将覆盖控制其中 {len(action_ctrl_ids)} 个下肢关节]")
    # 标称关节位置 q_nominal (MuJoCo default_pos 顺序)
    # This will be calculated inside the loop now, based on ALL_JOINT_DEFAULTS_DEFAULT
    # action_nominal = np.array(
    #     [env.default_pos[idx] for idx in action_ctrl_ids], dtype=np.float32
    # )
    print(f"\n  腿部关节 (policy 顺序 → ctrl index → qpos/qvel addr):")
    for i, (name, ctrl_idx, qpos_idx, qvel_idx) in enumerate(
            zip(LEG_JOINT_NAMES_DEFAULT, action_ctrl_ids, action_qpos_ids, action_qvel_ids)):
        print(f"    [{i:2d}] {name:35s} → ctrl[{ctrl_idx:2d}]  "
              f"qpos_addr={qpos_idx:2d}  qvel_addr={qvel_idx:2d}  "
              f"nominal={ALL_JOINT_DEFAULTS_DEFAULT[name]:.3f}")
    # ── 索引诊断: 验证 BUG #1 修复 ──────────────────────────────────────
    if action_ctrl_ids != action_qpos_ids:
        print(f"\n  ⚠ [DIAG] ctrl_ids ≠ qpos_ids — 已修复的索引空间混淆:")
        print(f"    ctrl_ids  = {action_ctrl_ids}")
        print(f"    qpos_ids  = {action_qpos_ids}")
        print(f"    qvel_ids  = {action_qvel_ids}")
        print(f"    旧代码用 qpos_ids 索引 ctrl-ordered 数组 → 读取错误关节数据！")
        print(f"    新代码直接从 data.qpos[qpos_ids] 读取 → 正确。")

    # ── 加载策略网络 ───────────────────────────────────────────────────────
    # NOTE: num_actions=13 告知 PolicyRunner 输出维度（含时钟偏移维）
    #       但 sim2sim 中只使用前 12 维关节残差；第 13 维用于更新 phi
    print(f"\n[2/3] 加载策略网络 (期望输出 13 维) …")
    policy = PolicyRunner(args.policy, num_actions=13)

    # ── 历史缓冲区 (per-term 独立堆叠，匹配 Isaac Lab CircularBuffer) ─────
    print(f"\n[3/3] 初始化 per-term 观测历史缓冲区 …")
    history = PerTermHistoryBuffer(term_dims=TERM_DIMS, history_length=HISTORY_LENGTH)

    # ── 辅助函数: 读取腿部关节状态 ────────────────────────────────────────
    def read_leg_joints() -> tuple[np.ndarray, np.ndarray]:
        """读取 12 个腿部关节的位置和速度（policy 顺序）。

        ⚠ 修复说明 (BUG #1 — 索引空间混淆):
           旧代码用 env.get_qpos()（返回按 ctrl 索引排列的数组）配合
           action_qpos_ids（raw qpos 地址，含自由基底偏移 7）做索引，
           导致读到错误关节的数据（例如 all_q[7] 实际读取的是 ctrl=7
           的关节，而非 qpos_addr=7 的关节）。

           修复: 直接从 env.data.qpos / env.data.qvel 用 raw 地址读取。
        """
        pos = np.array([env.data.qpos[idx] for idx in action_qpos_ids], dtype=np.float32)
        vel = np.array([env.data.qvel[idx] for idx in action_qvel_ids], dtype=np.float32)
        return pos, vel

    # ── 初始化仿真状态 ────────────────────────────────────────────────────
    env.set_initial_state(base_pos=(0.0, 0.0, args.base_height))

    # 用初始站立状态预填历史缓冲区 (per-term)
    j_pos, j_vel = read_leg_joints()
    # 12 个腿部关节的标称零位
    initial_action_nominal = np.array(
        [ALL_JOINT_DEFAULTS_DEFAULT[name] for name in LEG_JOINT_NAMES_DEFAULT], dtype=np.float32
    )
    init_terms = [
        compute_roll_pitch(env.get_base_quat()),                                          # [0] 2D
        (env.get_base_ang_vel() * ANG_VEL_SCALE).astype(np.float32),                     # [1] 3D
        (j_pos - initial_action_nominal).astype(np.float32),                              # [2] 12D
        (j_vel * JOINT_VEL_SCALE).astype(np.float32),                                    # [3] 12D
        motion_mode.astype(np.float32),                                                   # [4] 3D
        np.array([float(mode_ref[0])], dtype=np.float32),                                 # [5] 1D
        compute_clock_phase(phi),                                                          # [6] 2D
    ]
    history.reset(init_terms)

    # 初始动作 (位置残差，全零意味着保持标称位置)
    action_residual = np.zeros(12, dtype=np.float32)  # 仅12维关节残差
    sim_step_count  = 0                                # 物理步计数

    # ── 键盘回调 ──────────────────────────────────────────────────────────
    def key_callback(keycode: int):
        """MuJoCo viewer 的按键回调（在 viewer 线程中调用）。

        键码参考 (GLFW keycodes):
          49=1, 50=2, 51=3
          265=↑, 264=↓, 262=→, 263=←
          32=Space, 82=R, 81=Q, 256=Esc
        """
        nonlocal motion_mode, mode_ref

        # ── 模式切换 ────────────────────────────────────────────────────
        if keycode == 49:   # 1 → Standing
            motion_mode = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            mode_ref[0]  = 0.0
            print("  [MODE] Standing  [0, 0, 1]")

        elif keycode == 50:  # 2 → Inplace (原地踏步)
            motion_mode = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            mode_ref[0]  = 0.0
            print("  [MODE] Inplace   [0, 1, 0]")

        elif keycode == 51:  # 3 → Forward (向前行走)
            motion_mode = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            mode_ref[0]  = 0.3   # 进入 Forward 模式时给一个默认初速
            print(f"  [MODE] Forward   [1, 0, 0]  vx={mode_ref[0]:+.2f}")

        # ── Forward 模式: ↑↓ 调节前向速度 ──────────────────────────────
        elif keycode == 265:  # ↑  增加前向速度
            if motion_mode[0] > 0.5:  # 当前为 Forward 模式
                mode_ref[0] = float(np.clip(mode_ref[0] + VEL_DELTA, VX_MIN, VX_MAX))
                print(f"  [CMD] vx = {mode_ref[0]:+.2f} m/s")
            else:
                print("  [WARN] ↑ 仅在 Forward(3) 模式下生效")

        elif keycode == 264:  # ↓  减小前向速度
            if motion_mode[0] > 0.5:
                mode_ref[0] = float(np.clip(mode_ref[0] - VEL_DELTA, VX_MIN, VX_MAX))
                print(f"  [CMD] vx = {mode_ref[0]:+.2f} m/s")
            else:
                print("  [WARN] ↓ 仅在 Forward(3) 模式下生效")

        # ── Inplace 模式: ←→ 调节转向角速度 ────────────────────────────
        elif keycode == 263:  # ←  增加转向角速度（左转 = +wz）
            if motion_mode[1] > 0.5:  # 当前为 Inplace 模式
                mode_ref[0] = float(np.clip(mode_ref[0] + VEL_DELTA, WZ_MIN, WZ_MAX))
                print(f"  [CMD] wz = {mode_ref[0]:+.2f} rad/s")
            else:
                print("  [WARN] ← 仅在 Inplace(2) 模式下生效")

        elif keycode == 262:  # →  减小转向角速度（右转 = -wz）
            if motion_mode[1] > 0.5:
                mode_ref[0] = float(np.clip(mode_ref[0] - VEL_DELTA, WZ_MIN, WZ_MAX))
                print(f"  [CMD] wz = {mode_ref[0]:+.2f} rad/s")
            else:
                print("  [WARN] → 仅在 Inplace(2) 模式下生效")

        # ── 辅助控制 ────────────────────────────────────────────────────
        elif keycode == 32:   # Space — 清零速度指令
            mode_ref[0] = 0.0
            print("  [CMD] 速度指令归零")

        elif keycode == 82:   # R — 重置
            should_reset[0] = True
            print("  [RESET] 请求重置机器人…")

        elif keycode in (81, 256):  # Q / Esc — 退出
            print("\n  [INFO] 退出请求，正在关闭仿真…")
            sys.exit(0)

    # ── 主仿真循环 ────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  仿真开始！(关闭窗口 / 按 Q 退出)")
    print("  键盘:  1=站立  2=原地踏步  3=前进  ↑↓=速度  ←→=转向  Space=归零  R=重置")
    print("─" * 65 + "\n")

    viewer_kwargs = {"key_callback": key_callback} if not args.no_keyboard else {}

    with mujoco.viewer.launch_passive(env.model, env.data, **viewer_kwargs) as viewer:
        wall_start = time.time()

        while viewer.is_running() and (time.time() - wall_start) < args.duration:
            step_start = time.time()

            # ── 处理重置请求 ─────────────────────────────────────────────
            if should_reset[0]:
                env.set_initial_state(base_pos=(0.0, 0.0, args.base_height))
                action_residual[:] = 0.0
                phi = 0.0
                j_pos, j_vel = read_leg_joints()
                reset_terms = [
                    compute_roll_pitch(env.get_base_quat()),
                    (env.get_base_ang_vel() * ANG_VEL_SCALE).astype(np.float32),
                    (j_pos - initial_action_nominal).astype(np.float32),
                    (j_vel * JOINT_VEL_SCALE).astype(np.float32),
                    motion_mode.astype(np.float32),
                    np.array([float(mode_ref[0])], dtype=np.float32),
                    compute_clock_phase(phi),
                ]
                history.reset(reset_terms)
                sim_step_count  = 0
                should_reset[0] = False
                print("  [RESET] 已重置到初始站立状态  phi=0")

            # ── PD 扭矩 → 物理步 ─────────────────────────────────────────
            env.step_pd()
            sim_step_count += 1

            # --- 策略执行 (Policy Step) ---
            if sim_step_count % DECIMATION == 0:
                obs = history.get_obs()

                # [DEBUG 注入] 仅在第一帧推理前，打印完整 175 维 per-term 观测
                if sim_step_count == DECIMATION:
                    print_full_obs_debug(obs, step=sim_step_count)
                    # 同时也打印最新帧的 35 维分块（方便快速查看当前值）
                    newest_frame = np.concatenate([
                        obs[TERM_DIMS[0]*(HISTORY_LENGTH-1) : TERM_DIMS[0]*HISTORY_LENGTH],         # roll_pitch newest
                        obs[sum(TERM_DIMS[:1])*HISTORY_LENGTH + TERM_DIMS[1]*(HISTORY_LENGTH-1) :
                            sum(TERM_DIMS[:1])*HISTORY_LENGTH + TERM_DIMS[1]*HISTORY_LENGTH],       # ang_vel newest
                        obs[sum(TERM_DIMS[:2])*HISTORY_LENGTH + TERM_DIMS[2]*(HISTORY_LENGTH-1) :
                            sum(TERM_DIMS[:2])*HISTORY_LENGTH + TERM_DIMS[2]*HISTORY_LENGTH],       # pos_rel newest
                        obs[sum(TERM_DIMS[:3])*HISTORY_LENGTH + TERM_DIMS[3]*(HISTORY_LENGTH-1) :
                            sum(TERM_DIMS[:3])*HISTORY_LENGTH + TERM_DIMS[3]*HISTORY_LENGTH],       # vel_rel newest
                        obs[sum(TERM_DIMS[:4])*HISTORY_LENGTH + TERM_DIMS[4]*(HISTORY_LENGTH-1) :
                            sum(TERM_DIMS[:4])*HISTORY_LENGTH + TERM_DIMS[4]*HISTORY_LENGTH],       # mode newest
                        obs[sum(TERM_DIMS[:5])*HISTORY_LENGTH + TERM_DIMS[5]*(HISTORY_LENGTH-1) :
                            sum(TERM_DIMS[:5])*HISTORY_LENGTH + TERM_DIMS[5]*HISTORY_LENGTH],       # ref newest
                        obs[sum(TERM_DIMS[:6])*HISTORY_LENGTH + TERM_DIMS[6]*(HISTORY_LENGTH-1) :
                            sum(TERM_DIMS[:6])*HISTORY_LENGTH + TERM_DIMS[6]*HISTORY_LENGTH],       # clock newest
                    ])
                    print_obs_debug(newest_frame, step=sim_step_count)
                    print(f"  [DEBUG] 送入网络的 Tensor 总维度: {obs.shape}")

                # 获取 13 维动作: 12关节残差 + 1相位偏移
                action = policy.infer(obs) # Changed from policy.step to policy.infer to match original

                # --- Target Pos: 根据策略输出修改下肢关节设定 ---
                action_residual = action[:12]   # [12]
                action_delta_phi = action[12]   # [1]

                # 取出 12 个下肢关节的标称零位（依网络输出的LEG_JOINT_NAMES_DEFAULT排序）
                action_nominal = np.array([ALL_JOINT_DEFAULTS_DEFAULT[name] for name in LEG_JOINT_NAMES_DEFAULT])
                
                # 更新 12个腿部目标点（其余上肢关节将自动锁定在标称点）
                target_pos = action_residual * ACTION_SCALE + action_nominal
                env.set_target_positions(action_ctrl_ids, target_pos)
                
                # (f) 更新时钟相位
                #     phi_{t+1} = (phi_t + clip(a_delta_phi, -5, 5) + 1) % L
                #     clip 范围与 ClockJointPositionAction 一致
                delta_phi_clipped = float(np.clip(action_delta_phi, DELTA_PHI_MIN, DELTA_PHI_MAX))
                phi = (phi + delta_phi_clipped + 1.0) % CLOCK_L

                # (b) 计算各 term 当前值并分别压入 per-term 历史缓冲区
                j_pos, j_vel = read_leg_joints()  # 重新读取执行动作后的状态
                quat         = env.get_base_quat()
                ang_vel_body = env.get_base_ang_vel()
                current_terms = [
                    compute_roll_pitch(quat),                                          # [0] 2D
                    (ang_vel_body * ANG_VEL_SCALE).astype(np.float32),                # [1] 3D
                    (j_pos - action_nominal).astype(np.float32),                       # [2] 12D
                    (j_vel * JOINT_VEL_SCALE).astype(np.float32),                     # [3] 12D
                    motion_mode.astype(np.float32),                                    # [4] 3D
                    np.array([float(mode_ref[0])], dtype=np.float32),                  # [5] 1D
                    compute_clock_phase(phi),                                           # [6] 2D
                ]
                # (c) 将各 term 压入各自的历史缓冲区
                history.push(current_terms)


            # ── Viewer 同步 ─────────────────────────────────────────────
            viewer.sync()

            # ── 实时节拍控制 ─────────────────────────────────────────────
            elapsed    = time.time() - step_start
            sleep_time = SIM_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\n仿真结束。")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
