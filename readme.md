# X2Ultra RL IsaacLab

<div align="center">

**基于 Isaac Lab 的 X2Ultra 31-DoF 人形机器人强化学习训练框架**

*本项目由 `unitree_rl_lab` 派生改编，已完全解耦，无需安装原始仓库。*

</div>

---

## 📋 目录

- [项目简介](#-项目简介)
- [前置依赖](#-前置依赖)
- [快速开始](#-快速开始)
- [目录结构](#-目录结构)
- [训练](#-训练)
- [Sim2Sim 验证](#-sim2sim-验证)
- [常见问题](#-常见问题)

---

## 🤖 项目简介

本仓库针对 **X2Ultra 31-DoF 人形机器人** 实现了完整的强化学习训练管线，包含：

| 模块 | 说明 |
|------|------|
| **速度跟踪（Velocity）** | 训练机器人跟随给定速度命令行走 |
| **楼梯攀爬（Stair）** | 盲走策略，依靠时序观测隐式推断地形 |
| **CPG 步态（CPG 变体）** | 在以上两任务基础上叠加中央模式发生器辅助步态 |
| **Sim2Sim (MuJoCo)** | 将 Isaac Lab 训练的策略迁移到 MuJoCo 验证泛化性 |

---

## ⚙️ 前置依赖

> [!IMPORTANT]
> Isaac Lab 需要 **NVIDIA GPU + CUDA 11.8+** 环境，且必须单独安装。
> 请先完成以下依赖安装，再执行快速开始脚本。

### 必须提前安装

| 依赖 | 推荐版本 | 官方文档 |
|------|---------|---------|
| CUDA Toolkit | ≥ 11.8 | [NVIDIA 官网](https://developer.nvidia.com/cuda-downloads) |
| Isaac Sim | 4.x | [安装指南](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) |
| Isaac Lab | 最新版 | [Isaac Lab 安装](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) |
| Conda / Miniconda | 任意版本 | [Conda 官网](https://docs.conda.io/en/latest/miniconda.html) |

---

## 🚀 快速开始

```bash
# 第一步：克隆仓库
git clone <your-repo-url> X2Ultra_RL_IsaacLab
cd X2Ultra_RL_IsaacLab

# 第二步：运行安装脚本（自动创建 Conda 环境 + 注册 Python 路径）
bash setup.sh

# 第三步：激活环境
conda activate x2ultra_rl_env

# 第四步：开始训练（需要 GPU + Isaac Lab 环境）
cd scripts/rsl_rl
python train.py --task Zhiyuan-X2Ultra-31dof-Velocity-CPG
```

> [!NOTE]
> `setup.sh` 是幂等的，多次运行不会重复安装。

---

## 📁 目录结构

```
X2Ultra_RL_IsaacLab/
├── lab_settings/                  # 核心训练配置（Isaac Lab 扩展包）
│   ├── assets/robots/
│   │   ├── config_setting.py      # 机器人实体配置（URDF 路径、关节参数）
│   │   └── actuators.py           # 执行器模型（电机 PD 参数）
│   └── tasks/locomotion/
│       ├── agents/
│       │   └── rsl_rl_ppo_cfg.py  # PPO 超参数配置（BasePPORunnerCfg）
│       ├── mdp/                   # 奖励、观测、动作、课程定义
│       │   └── actions/
│       │       └── cpg_action.py  # CPG 步态辅助动作项
│       └── robots/x2ultra/31dof/
│           ├── __init__.py        # Gym 环境注册
│           ├── velocity_env_cfg.py
│           ├── velocity_wCPG_env_cfg.py
│           ├── stair_env_cfg.py
│           └── stair_wCPG_env_cfg.py
│
├── robot_ros/                     # 机器人模型资产（标准目录，无 Git 链接）
│   └── robots/x2ultra_description/
│       └── x2_ultra_simple_collision.urdf  # 训练用简化碰撞 URDF
│
├── scripts/                       # 训练与播放脚本
│   ├── rsl_rl/
│   │   ├── train.py               # 训练入口
│   │   ├── play.py                # 推理/录制入口
│   │   └── cli_args.py            # 命令行参数工具
│   ├── utils/                     # 本地工具函数（已从 unitree_rl_lab 解耦）
│   │   ├── export_deploy_cfg.py   # 导出部署配置 YAML
│   │   └── parser_cfg.py          # 解析环境配置
│   └── list_envs.py               # 列出所有已注册环境
│
├── sim2sim/                       # Isaac Lab → MuJoCo Sim2Sim 验证
│   ├── deploy.py                  # 部署主程序
│   ├── configs/                   # 各任务 YAML 配置
│   └── core/                     # MuJoCo 环境、观测构建、策略运行器
│
├── pretrained/                    # 预训练模型
├── logs/                          # 训练日志（自动生成）
├── environment.yml                # Conda 环境定义
├── setup.sh                       # 一键安装脚本
└── README.md
```

---

## 🏋️ 训练

```bash
cd scripts/rsl_rl

# 列出所有可用任务
python ../list_envs.py

# 速度跟踪 + CPG 步态（推荐入门）
python train.py --task Zhiyuan-X2Ultra-31dof-Velocity-CPG --num_envs 4096

# 楼梯攀爬 + CPG
python train.py --task Zhiyuan-X2Ultra-31dof-Stair-CPG --num_envs 2048

# 从检查点恢复训练
python train.py --task Zhiyuan-X2Ultra-31dof-Velocity-CPG --resume

# 播放已训练策略
python play.py --task Zhiyuan-X2Ultra-31dof-Velocity-CPG --num_envs 32
```

训练日志保存在：`logs/rsl_rl/<task_name>/<timestamp>/`

---

## 🔄 Sim2Sim 验证

在 MuJoCo 中验证 Isaac Lab 训练的策略（**无需 GPU，仅需 CPU**）：

```bash
cd sim2sim

# 基础直行验证
python deploy.py --config configs/walk_straight.yaml

# 自定义速度指令
python deploy.py --config configs/walk_straight.yaml --cmd_vx 0.8 --cmd_wz 0.1

# 使用 ONNX 模型（更快推理）
python deploy.py --config configs/walk_straight.yaml \
    --policy ../pretrained/walk_straight/exported/policy.onnx
```

**键盘控制：**

| 按键 | 功能 |
|------|------|
| `↑ / ↓` | 增加 / 减小 线速度 Vx |
| `← / →` | 增加 / 减小 角速度 Wz |
| `Space` | 归零所有速度指令 |
| `R` | 重置机器人姿态 |
| `Q / Esc` | 退出仿真 |

---

## ❓ 常见问题

**Q: 运行 `train.py` 时报 `ModuleNotFoundError: No module named 'isaaclab'`？**
> A: Isaac Lab 未正确安装，或当前 Python 环境不是 Isaac Sim 内置的 Python。请参考 [Isaac Lab 安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)。

**Q: `setup.sh` 报错 `conda: command not found`？**
> A: 请先安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 并将其添加到 PATH。

**Q: URDF 找不到 / 机器人模型加载失败？**
> A: 确认 `robot_ros/robots/x2ultra_description/x2_ultra_simple_collision.urdf` 文件存在。该文件应已包含在仓库中。

**Q: 如何修改 PPO 超参数？**
> A: 编辑 `lab_settings/tasks/locomotion/agents/rsl_rl_ppo_cfg.py` 中的 `BasePPORunnerCfg`。

---

## 📄 许可证

本项目基于 BSD-3-Clause 协议，详见 [LICENCE](LICENCE) 文件。
原始 `unitree_rl_lab` 及 Isaac Lab 相关代码保留原有版权声明。
