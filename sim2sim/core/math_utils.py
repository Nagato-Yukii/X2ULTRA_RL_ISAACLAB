"""
Math utilities for Sim2Sim deployment.

Provides quaternion operations and gravity projection used in observation
construction. All quaternions follow MuJoCo convention: [qw, qx, qy, qz]. 
所有四元数遵循MuJoCo约定[qw, qx, qy, qz]
"""

import numpy as np


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into body frame.

    Mujoco世界坐标系是右手坐标系ENU,载体坐标系也是右手坐标系
    记Mujoco世界坐标系为N,载体坐标系为B

    [x_b, y_b, z_b]^T = R^b_n * [x_n, y_n, z_n]^T

    R^b_n = R_y(\gamma) * R_x(\beta) * R_z(-\alpha)

        [ cos \gamma ,  0   , -sin \gamma]      [ 1 ,      0        ,      0     ]      [ cos(-\alpha)  ,  sin(-\alpha) ,      0     ]
    =   |       0    ,  1   ,      0     |  *   | 0 ,   cos \beta   ,  sin \beta |  *   | -sin(-\alpha) ,  cos(-\alpha) ,      0     |
        [ sin \gamma ,  0   , cos \gamma ]      [ 0 ,  -sin \beta   ,  cos \beta ]      [       0       ,      0        ,      1     ]

        [ (cos \alpha)*(cos \beta) + (sin \alpha)*(sin \beta)*(sin \gamma)  ,  (-sin \alpha)*(cos \gamma)+(cos \alpha)*(sin \beta)*(sin \gamma)   , (-cos \beta)*(sin \gamma)] 
    =   |                          (sin \alpha)*(cos \beta)                 ,                        (cos \alpha)*(cos \beta)                     ,          sin \beta       | 
        [ (sin \gamma)*(cos \alpha) - (sin \alpha*)(sin \beta)*(cos \gamma) ,  (-sin \alpha)*(sin \gamma)-(cos \alpha)*(sin \beta)*(cos \gamma)   , (cos \beta)*(cos \gamma) ]  

        [ r00 ,  r01   , r02]
    =   | r10 ,  r11   , r12|
        [ r20 ,  r21   , r22]

    在N坐标系下G_N被表示为[0,0,-g]^T,计算在载体坐标系下的重力表示为G_B = R^b_n * G_N = [r02 * -g , r12 * -g , r22 * -g]
    (这里的 r00~r22 的值是由四元数公式直接得出的，等价于欧拉角算出来的三角函数结果)
    
    Computes v_body = R^T @ v_world, where R is the rotation matrix
    constructed from the quaternion.

    Args:
        quat: Quaternion [qw, qx, qy, qz] (MuJoCo convention).
        vec:  Vector in world frame [x, y, z].

    Returns:
        Vector in body frame [x, y, z].
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]

    # R^T columns (rows of R)
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy + qw * qz)
    r02 = 2 * (qx * qz - qw * qy)
    r10 = 2 * (qx * qy - qw * qz)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz + qw * qx)
    r20 = 2 * (qx * qz + qw * qy)
    r21 = 2 * (qy * qz - qw * qx)
    r22 = 1 - 2 * (qx * qx + qy * qy)

    vx = r00 * vec[0] + r01 * vec[1] + r02 * vec[2]
    vy = r10 * vec[0] + r11 * vec[1] + r12 * vec[2]
    vz = r20 * vec[0] + r21 * vec[1] + r22 * vec[2]

    return np.array([vx, vy, vz], dtype=np.float32)


def get_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Compute gravity direction in body frame.

    Both MuJoCo and Isaac Lab use Z-up world frame, so gravity = [0, 0, -1].
    This function computes R^T @ [0, 0, -1].

    Matches Isaac Lab's ``asset.data.projected_gravity_b``.

    Args:
        quat: Quaternion [qw, qx, qy, qz].

    Returns:
        Gravity direction in body frame, unit vector [gx, gy, gz].
    """
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
