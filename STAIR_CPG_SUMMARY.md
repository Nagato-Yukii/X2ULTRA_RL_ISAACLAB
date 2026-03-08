# Stair Environment 修正与 CPG 版本总结

## 完成的工作

### 1. 修正了原 stair_env_cfg.py 的问题

#### 修正 1: 步态周期
```python
# 修正前
"period": 0.8,  # velocity 任务的周期

# 修正后
"period": 1.0,  # 楼梯专用周期，更慢更稳定
```

#### 修正 2: 抬腿高度
```python
# 修正前
"target_height": -0.48,  # 抬高 15cm

# 修正后
"target_height": -0.43,  # 抬高 20cm，适应 18cm 台阶
```

#### 修正 3: 课程升级阈值
```python
# 修正前
"distance_threshold": 2.0,  # 20s 内走 2m 即可升级

# 修正后
"distance_threshold": 3.0,  # 提高到 3m，确保学会稳定爬楼
```

### 2. 创建了 stair_wCPG_env_cfg.py

新文件路径：`lab_settings/tasks/locomotion/robots/x2ultra/31dof/stair_wCPG_env_cfg.py`

#### CPG 参数（针对楼梯优化）

| 参数 | velocity CPG | **stair CPG** | 设计理由 |
|------|-------------|--------------|---------|
| **频率** | 1.67 Hz (0.6s) | **1.0 Hz (1.0s)** | 楼梯需要更慢、更稳定的步态 |
| **hip_pitch** | 0.35 rad (20°) | **0.45 rad (26°)** | 更大前摆以跨越台阶 |
| **knee** | 0.70 rad (40°) | **0.85 rad (49°)** | 显著增大弯曲以抬高小腿 |
| **ankle_pitch** | 0.25 rad (14°) | **0.30 rad (17°)** | 适应台阶高度变化 |
| **残差 scale** | 0.25 rad (14°) | **0.30 rad (17°)** | 更大修正范围（台阶高度变化大） |
| **相位周期** | 0.6s | **1.0s** | 与 CPG 频率一致 |

#### 设计思路

```
最终关节角度 = default_pos + CPG_signal + network_residual × 0.30

CPG 部分（主导）:
  - 提供"标准爬楼步态"：慢速（1.0s）、高抬腿（knee 49°）
  - 左右腿交替（相位差 π）
  - 周期性稳定

残差部分（修正）:
  - 网络学习适应不同台阶高度（3-18cm）
  - 通过历史观测（5帧）预判台阶变化
  - 修正范围 ±17°，足够应对高度变化
```

### 3. 注册了新任务

在 `__init__.py` 中新增：

```python
gym.register(
    id="Zhiyuan-X2Ultra-31dof-Stair-CPG",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_wCPG_env_cfg:StairCPGEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stair_wCPG_env_cfg:StairCPGPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
```

## 现在可用的任务

| 任务 ID | 描述 | CPG | 推荐用途 |
|---------|------|-----|---------|
| `Zhiyuan-X2Ultra-31dof-Velocity` | 平地速度跟踪 | ❌ | 基础训练 |
| `Zhiyuan-X2Ultra-31dof-Velocity-CPG` | 平地速度跟踪 + CPG | ✅ | 快速收敛的平地训练 |
| `Zhiyuan-X2Ultra-31dof-Stair` | 楼梯攀爬（已修正） | ❌ | 纯残差学习爬楼 |
| `Zhiyuan-X2Ultra-31dof-Stair-CPG` | 楼梯攀爬 + CPG | ✅ | **推荐：稳定爬楼训练** |

## 使用建议

### 方案 1: 从零开始训练 Stair-CPG

```bash
# 直接训练 CPG 版本
python scripts/train.py --task Zhiyuan-X2Ultra-31dof-Stair-CPG --num_envs 4096
```

**优点**：
- CPG 提供强先验，收敛更快
- 步态更稳定

**缺点**：
- 受 CPG 约束，灵活性略低

### 方案 2: 从 Velocity 模型迁移（推荐）

```bash
# 使用 velocity 训练 10000 iterations 的模型作为初始化
python scripts/train.py --task Zhiyuan-X2Ultra-31dof-Stair-CPG \
    --load_run logs/rsl_rl/zhiyuan_x2ultra_31dof_velocity/your_run_name \
    --checkpoint model_10000.pt
```

**优点**：
- 利用已有的平地行走能力
- 只需学习适应台阶的部分
- 训练时间更短

**注意**：
- velocity 模型是纯残差（无 CPG）
- 迁移到 CPG 版本时，网络需要适应 CPG 信号
- 建议降低初始学习率（0.5x）

### 方案 3: 修正后的原版 Stair（不推荐）

```bash
# 使用修正后的原版（无 CPG）
python scripts/train.py --task Zhiyuan-X2Ultra-31dof-Stair --num_envs 4096
```

**优点**：
- 最大灵活性
- 可以学习任意步态

**缺点**：
- 训练时间长
- 可能不稳定

## 训练监控指标

### 关键指标

1. **terrain_level**：当前难度级别（0-9）
   - 目标：稳定升级到 level 7-9（15-18cm 台阶）

2. **track_lin_vel_xy**：速度跟踪奖励
   - 目标：> 0.8

3. **feet_clearance**：抬腿高度奖励
   - 目标：> 0.5（说明能抬高到 20cm）

4. **gait**：步态奖励
   - 目标：> 0.3（说明左右腿交替良好）

5. **episode_length**：episode 长度
   - 目标：接近 20s（1000 steps）

### 调试建议

#### 如果步态过快/不稳定
```python
# 在 stair_wCPG_env_cfg.py 中调整
cpg_frequency = 0.8  # 降低频率（原 1.0）
```

#### 如果抬腿不够高
```python
# 增大 knee 振幅
cpg_amplitudes = [
    0.45, 0.45,   # hip_pitch
    0.0,  0.0,    # hip_roll
    0.0,  0.0,    # hip_yaw
    0.95, 0.95,   # knee: 增大到 54° (原 49°)
    0.30, 0.30,   # ankle_pitch
    0.0,  0.0,    # ankle_roll
]
```

#### 如果网络输出饱和
```python
# 增大残差 scale
scale = 0.35  # 增大到 20° (原 17°)
```

## CPG vs 纯残差对比

### 楼梯任务的特殊性

楼梯任务比平地行走复杂得多：

| 挑战 | 纯残差 | CPG + 残差 |
|------|--------|-----------|
| **学习周期性步态** | 从零学习 | CPG 提供 |
| **适应台阶高度** | 完全靠网络 | CPG 基础 + 网络微调 |
| **稳定性** | 依赖训练 | CPG 保证基础稳定 |
| **训练时间** | 长（20k+ iters） | 短（10k iters） |
| **灵活性** | 高 | 中 |

### 为什么 CPG 更适合楼梯

1. **周期性先验**：爬楼梯本质是周期性动作，CPG 天然匹配
2. **高抬腿动作**：CPG 的大振幅（knee 49°）提供基础抬腿能力
3. **稳定性**：CPG 保证基本步态，网络只需微调
4. **收敛速度**：减少学习空间，加快收敛

## 下一步

1. **训练 Stair-CPG**：
   ```bash
   python scripts/train.py --task Zhiyuan-X2Ultra-31dof-Stair-CPG
   ```

2. **监控训练**：
   ```bash
   tensorboard --logdir logs/rsl_rl/zhiyuan_x2ultra_31dof_stair_cpg
   ```

3. **测试模型**：
   ```bash
   python scripts/play.py --task Zhiyuan-X2Ultra-31dof-Stair-CPG \
       --checkpoint logs/.../model_XXXX.pt
   ```

4. **如果效果不佳**：
   - 检查 CPG 参数（频率、振幅）
   - 调整奖励权重
   - 增加训练时间

## 文件清单

修改/创建的文件：

1. ✅ `stair_env_cfg.py` - 修正了 3 个问题
2. ✅ `stair_wCPG_env_cfg.py` - 新建 CPG 版本
3. ✅ `__init__.py` - 注册新任务
4. ✅ `velocity_wCPG_env_cfg.py` - 之前已调整参数
5. ✅ `STAIR_ISSUES_AND_FIXES.md` - 问题分析文档
6. ✅ `STAIR_CPG_SUMMARY.md` - 本总结文档
