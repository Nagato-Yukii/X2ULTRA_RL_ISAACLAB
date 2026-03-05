文件结构:

X2Ultra_RL_IsaacLab/
│
├── lab_settings/
│   ├── assets/robots/        # 机器人引用及配置
│   │            ├── actuators.py   #对机器人电机参数进行精确建模
│   │            ├── config_settings.py     #
│   │
│   │
│   ├── tasks/          # 任务与奖励
│   └── agents/         # 训练配置 (PPO)
│
├── robot_model/        # 机器人资产
|
├── robot_ros/          # 机器人资产
│
├── logs/               # 运行日志
│
├── outputs/            # 训练结果
│
└── scripts/            # 启动脚本

本项目由unitree_rl_lab改编，导入和配置unitree机器人的配置被修改，旨在复用于训练x2ultra
