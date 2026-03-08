from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg): # 继承自官方均匀速度指令配置类，并重写limit_ranges
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING    # 新增一个自定义变量，必填
