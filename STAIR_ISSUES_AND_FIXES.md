# Stair Environment 问题分析与修正建议

## 发现的问题

### 1. 奖励函数的问题

#### 问题 1.1: gait 奖励周期不匹配
```python
# stair_env_cfg.py 第 271 行
gait = RewTerm(
    func=mdp.feet_gait,
    weight=0.5,
    params={
        "period": 0.8,  # ❌ 问题：使用 velocity 任务的周期
        ...
    },
)
```

**问题**：楼梯任务应该使用更慢的步态周期（1.0s），但这里沿用了 velocity 的 0.8s。

**影响**：
- 步态奖励鼓励机器人快速行走（0.8s 周期）
- 但楼梯需要慢速稳定步态（1.0s 周期）
- 导致机器人在楼梯上步态不稳定

**修正**：
```python
gait = RewTerm(
    func=mdp.feet_gait,
    weight=0.5,
    params={
        "period": 1.0,  # ✅ 楼梯专用周期
        "offset": [0.0, 0.5],
        "threshold": 0.55,
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    },
)
```

#### 问题 1.2: feet_clearance 目标高度可能不足
```python
# stair_env_cfg.py 第 292 行
feet_clearance = RewTerm(
    func=mdp.foot_clearance_body_reward,
    weight=1.0,
    params={
        "target_height": -0.48,  # ⚠️ 抬高 15cm
        ...
    },
)
```

**分析**：
- 站立脚 Z ≈ -0.63m（体坐标系）
- 目标高度 -0.48m → 抬高 15cm
- 最高台阶 18cm，15cm 的抬腿高度**勉强够用**

**建议**：
- 对于 18cm 台阶，建议抬高到 20cm 以上
- 修正为 `target_height: -0.43`（抬高 20cm）

#### 问题 1.3: 缺少对台阶适应的奖励
当前奖励函数主要是惩罚项，缺少对"成功跨越台阶"的正向激励。

**建议增加**：
- 前进距离奖励（鼓励向前移动）
- 或者基于 terrain_level 的进度奖励

### 2. 课程学习的问题

#### 问题 2.1: 升级阈值可能过于宽松
```python
# stair_env_cfg.py 第 341 行
terrain_levels = CurrTerm(
    func=mdp.terrain_levels_stair,
    params={"min_level": 1, "distance_threshold": 2.0},
)
```

**分析**：
- 20s episode，2m 升级阈值
- 平均速度只需 0.1 m/s 就能升级
- 可能导致机器人还没学会稳定爬楼就升级到更难的台阶

**建议**：
- 增加到 3.0m 或 4.0m
- 或者增加额外的成功率要求

### 3. 观测空间的问题

#### 问题 3.1: 缺少相位信息
原版 stair 没有相位观测，网络不知道当前处于步态的哪个阶段。

**影响**：
- 网络难以学习周期性步态
- 需要更长时间才能收敛

**修正**：CPG 版本已经添加了 cpg_phase 观测

### 4. 动作空间的问题

#### 问题 4.1: 纯残差学习难度大
楼梯任务比平地行走复杂得多：
- 需要更高的抬腿动作
- 需要适应不同高度的台阶（3-18cm）
- 需要更稳定的步态

纯残差学习（无 CPG）需要网络从零学习这些复杂动作。

**修正**：使用 CPG 版本（stair_wCPG_env_cfg.py）

### 5. 终止条件的问题

#### 问题 5.1: 高度终止阈值可能过低
```python
# stair_env_cfg.py 第 317 行
base_height = DoneTerm(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.15}
)
```

**分析**：
- 在最高台阶（18cm）上，机器人基座高度会显著升高
- 0.15m 的阈值主要是防止摔倒
- 这个设置是合理的

## 修正优先级

### 高优先级（必须修正）
1. ✅ **gait 周期**：从 0.8s 改为 1.0s
2. ✅ **使用 CPG**：创建 stair_wCPG_env_cfg.py

### 中优先级（建议修正）
3. **feet_clearance 高度**：从 -0.48 改为 -0.43
4. **课程升级阈值**：从 2.0m 增加到 3.0m

### 低优先级（可选优化）
5. 增加前进距离奖励
6. 调整奖励权重

## CPG 版本的优势

stair_wCPG_env_cfg.py 针对楼梯任务优化：

| 参数 | velocity CPG | stair CPG | 原因 |
|------|-------------|-----------|------|
| **频率** | 1.67 Hz (0.6s) | 1.0 Hz (1.0s) | 楼梯需要更慢更稳定 |
| **hip_pitch** | 0.35 rad (20°) | 0.45 rad (26°) | 更大前摆跨越台阶 |
| **knee** | 0.70 rad (40°) | 0.85 rad (49°) | 更大弯曲抬高小腿 |
| **ankle_pitch** | 0.25 rad (14°) | 0.30 rad (17°) | 适应台阶高度变化 |
| **scale** | 0.25 rad (14°) | 0.30 rad (17°) | 更大残差修正范围 |

## 使用建议

1. **从 velocity 模型迁移训练**：
   ```bash
   # 使用 velocity 训练的模型作为初始化
   python scripts/train.py --task Zhiyuan-X2Ultra-31dof-Stair-CPG \
       --load_run velocity_model_path --checkpoint model_10000.pt
   ```

2. **监控指标**：
   - terrain_level 升级速度
   - 各难度级别的成功率
   - feet_clearance 奖励值

3. **调试建议**：
   - 先在低难度（level 0-2）测试 CPG 参数
   - 观察机器人是否能稳定跨越 3-6cm 台阶
   - 如果步态过快/过慢，调整 cpg_frequency
   - 如果抬腿不够高，增大 knee 振幅
