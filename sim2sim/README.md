# Sim2Sim — Isaac Lab → MuJoCo Deployment

将 Isaac Lab 训练的强化学习策略部署到 MuJoCo 进行 sim-to-sim 验证。

## 目录结构

```
sim2sim/
├── deploy.py              # 主入口脚本
├── README.md
├── core/                  # 可复用核心模块
│   ├── __init__.py
│   ├── math_utils.py      # 四元数旋转、重力投影
│   ├── policy_runner.py   # 策略加载 (.pt / .onnx)
│   ├── mujoco_env.py      # MuJoCo 环境封装 + PD 控制 + 关节映射
│   └── observation.py     # 观测构建 + 历史帧堆叠
├── configs/               # 任务配置 (YAML)
│   └── walk_straight.yaml # 直行任务配置
├── robot/                 # MuJoCo 机器人模型
│   └── X2_URDF/
│       ├── scene.xml
│       └── x2_ultra.xml
└── walk_straight/         # (旧版代码, 仅供参考)
```

## 快速开始

### 1. 安装依赖

```bash
pip install mujoco numpy pyyaml torch  # 或用 onnxruntime 替代 torch
```

### 2. 运行

```bash
cd sim2sim

# 默认配置 (vx=0.5 直行)
python deploy.py --config configs/walk_straight.yaml

# 自定义速度指令
python deploy.py --config configs/walk_straight.yaml --cmd_vx 0.8 --cmd_wz 0.2

# 使用 ONNX 模型
python deploy.py --config configs/walk_straight.yaml \
    --policy ../pretrained/walk_straight/exported/policy.onnx
```

### 3. 键盘控制

| 按键 | 功能 |
|------|------|
| ↑ / ↓ | 增减前进速度 `lin_vel_x` |
| ← / → | 增减转向速度 `ang_vel_z` |
| Space | 停止 (速度归零) |
| R | 重置机器人姿态 |
| Q / Esc | 退出 |

## 与训练环境的关键对齐

### 观测空间 (Observation)

每帧 45 维, 堆叠 5 帧 → 总共 **225 维**:

| 偏移 | 名称 | 维度 | 说明 |
|------|------|------|------|
| 0 | `base_ang_vel` | 3 | 体坐标系角速度, ×0.2 |
| 3 | `projected_gravity` | 3 | 体坐标系重力方向 |
| 6 | `velocity_commands` | 3 | 速度指令 [vx, vy, wz] |
| 9 | `joint_pos_rel` | 12 | 关节位置 − 默认位置 |
| 21 | `joint_vel_rel` | 12 | 关节速度, ×0.05 |
| 33 | `last_action` | 12 | 上一步的原始动作 |

### 动作空间 (Action)

12 维连续动作, 控制下肢 12 个关节:

```
target_joint_pos = action × 0.15 + default_joint_pos
```

### 关节顺序 (Policy Order)

由 Isaac Lab 的正则表达式匹配决定 (每个 pattern 先 left 后 right):

```
[0]  left_hip_pitch      [1]  right_hip_pitch
[2]  left_hip_roll       [3]  right_hip_roll
[4]  left_hip_yaw        [5]  right_hip_yaw
[6]  left_knee           [7]  right_knee
[8]  left_ankle_pitch    [9]  right_ankle_pitch
[10] left_ankle_roll     [11] right_ankle_roll
```

### MuJoCo 角速度

MuJoCo 的 `qvel[3:6]` (free joint) 默认为**体坐标系角速度**, 与 Isaac Lab
的 `root_ang_vel_b` 一致, 无需额外旋转。

## 扩展新任务

1. **复制** `configs/walk_straight.yaml` 为 `configs/your_task.yaml`
2. **修改** `policy_path` 指向新模型
3. 如果观测/动作结构不同, 修改:
   - `observation` 部分 (history_length, scales)
   - `action.joints` (关节名称和顺序)
   - 可能需要扩展 `core/observation.py` 添加新的观测项
4. 如果 PD 参数不同, 修改 `pd_controller` 部分
5. 运行: `python deploy.py --config configs/your_task.yaml`

## 扩展新机器人

1. 将新机器人的 MuJoCo XML + mesh 放入 `robot/` 目录
2. 创建新的 YAML 配置, 更新:
   - `xml_path`
   - `pd_controller` (所有关节)
   - `action.joints` (被策略控制的关节)
3. 确保 YAML 中的关节名称与 MuJoCo XML 中的 `<joint name="...">` 一致

## 模块说明

- **`core/mujoco_env.py`** — 通过 `actuator_trnid` 自动建立 actuator ↔ qpos/qvel
  映射, 解决 MuJoCo 中 ctrl 索引与 joint 索引不一致的问题。
- **`core/observation.py`** — 使用 `deque` 维护 FIFO 历史缓冲区, 支持任意
  `history_length`。初始化时可用 `prefill()` 填充避免零值冷启动。
- **`core/policy_runner.py`** — 统一的推理接口, 自动检测 `.pt` / `.onnx` 后端。
- **`core/math_utils.py`** — 纯 numpy 实现的四元数旋转, 无外部依赖。
