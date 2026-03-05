# X2Ultra RL IsaacLab

本项目是基于 `unitree_rl_lab` 的改编版本，旨在将其适配于 X2Ultra 机器人的强化学习训练。我们对原项目中的机器人配置、导入流程和相关参数进行了修改，使其能够直接在 Isaac Lab 环境中用于训练 X2Ultra。

## 目录结构

项目的核心文件结构如下，以便快速了解代码组织：

X2Ultra_RL_IsaacLab/
├── lab_settings/
│ ├── assets/robots/ # 机器人引用及配置
│ │ ├── actuators.py # 对机器人电机参数进行精确建模
│ │ └── config_settings.py # 机器人通用配置
│ ├── tasks/ # 定义训练任务与奖励函数
│ └── agents/ # 训练策略配置 (如 PPO)
│
├── robot_model/ # 机器人模型文件 (如 URDF)
│
├── robot_ros/ # 机器人 ROS 相关功能包
│
├── logs/ # 训练过程中的运行日志
│
├── outputs/ # 保存训练结果 (模型权重、测试数据等)
│
├── scripts/ # 启动训练或测试的脚本
│
├── LICENCE # 项目许可证
├── NOTICE
└── readme.md # 项目说明文档

本项目由unitree_rl_lab改编，导入和配置unitree机器人的配置被修改，旨在复用于训练x2ultra
