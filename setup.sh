#!/usr/bin/env bash
# =============================================================================
#  setup.sh — X2Ultra_RL_IsaacLab 一键安装脚本
#  用途：为新用户在全新机器上快速搭建完整的训练环境
#  使用方法：bash setup.sh
# =============================================================================

set -euo pipefail   # 任何命令失败立即退出

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="x2ultra_rl_env"

echo "============================================================"
echo "  X2Ultra RL IsaacLab — 安装脚本"
echo "  仓库根目录: ${REPO_ROOT}"
echo "============================================================"

# ────────────────────────────────────────────────────────────────
# Step 1: 检查 Isaac Lab 是否已安装
# ────────────────────────────────────────────────────────────────
echo ""
echo "[Step 1/4] 检查 Isaac Lab 依赖..."
if ! python -c "import isaaclab" 2>/dev/null; then
    echo "⚠️  警告: 未检测到 isaaclab 包。"
    echo ""
    echo "  Isaac Lab 需要单独安装，请参考官方文档："
    echo "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html"
    echo ""
    echo "  典型安装步骤（以 pip 安装的 Isaac Lab 为例）："
    echo "    ./isaaclab.sh -p -m pip install -e ."
    echo ""
    echo "  如果 Isaac Lab 尚未安装，后续步骤可能会失败。"
    echo "  继续安装其余依赖... (Ctrl+C 可中断)"
    sleep 3
fi

# ────────────────────────────────────────────────────────────────
# Step 2: 创建或更新 Conda 环境
# ────────────────────────────────────────────────────────────────
echo ""
echo "[Step 2/4] 创建 Conda 环境: ${ENV_NAME}..."

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  ✔ 环境 '${ENV_NAME}' 已存在，跳过创建。"
    echo "  💡 若要强制重建，请先运行: conda env remove -n ${ENV_NAME}"
else
    conda env create -f "${REPO_ROOT}/environment.yml"
    echo "  ✔ Conda 环境创建完成。"
fi

# ────────────────────────────────────────────────────────────────
# Step 3: 将 lab_settings 注册为可导入包（通过 .pth 文件）
# ────────────────────────────────────────────────────────────────
echo ""
echo "[Step 3/4] 将 lab_settings/ 添加到 Python 路径..."

# 获取 conda 环境的 site-packages 目录
SITE_PKGS=$(conda run -n "${ENV_NAME}" python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)

if [ -n "${SITE_PKGS}" ]; then
    PTH_FILE="${SITE_PKGS}/x2ultra_rl_lab_settings.pth"
    echo "${REPO_ROOT}/lab_settings" > "${PTH_FILE}"
    echo "${REPO_ROOT}/scripts" >> "${PTH_FILE}"
    echo "  ✔ 已创建 .pth 文件: ${PTH_FILE}"
    echo "  (包含路径: lab_settings/ 和 scripts/)"
else
    echo "  ⚠️  无法自动获取 site-packages 路径，请手动设置 PYTHONPATH："
    echo "  export PYTHONPATH=\"${REPO_ROOT}/lab_settings:${REPO_ROOT}/scripts:\$PYTHONPATH\""
fi

# ────────────────────────────────────────────────────────────────
# Step 4: 安装 rsl-rl-lib（若未安装）
# ────────────────────────────────────────────────────────────────
echo ""
echo "[Step 4/4] 检查 rsl-rl-lib..."

if conda run -n "${ENV_NAME}" python -c "import rsl_rl" 2>/dev/null; then
    INSTALLED_VER=$(conda run -n "${ENV_NAME}" python -c "import importlib.metadata; print(importlib.metadata.version('rsl-rl-lib'))" 2>/dev/null || echo "未知")
    echo "  ✔ rsl-rl-lib 已安装 (版本: ${INSTALLED_VER})，跳过。"
else
    echo "  rsl-rl-lib 未安装，正在安装..."
    conda run -n "${ENV_NAME}" pip install rsl-rl-lib
    echo "  ✔ rsl-rl-lib 安装完成。"
fi

# ────────────────────────────────────────────────────────────────
# 完成
# ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ✅ 安装完成！"
echo ""
echo "  激活环境后即可开始训练："
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  训练命令（在 scripts/rsl_rl/ 目录下执行）："
echo "    cd ${REPO_ROOT}/scripts/rsl_rl"
echo "    python train.py --task Zhiyuan-X2Ultra-31dof-Velocity-CPG"
echo ""
echo "  Sim2Sim 验证："
echo "    cd ${REPO_ROOT}/sim2sim"
echo "    python deploy.py --config configs/walk_straight.yaml"
echo "============================================================"
