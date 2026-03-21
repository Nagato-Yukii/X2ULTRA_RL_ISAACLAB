"""
Policy Runner — unified inference interface for .pt / .onnx models.

Supports three backends:
  1. TorchScript (.pt via torch.jit.save)
  2. RSL-RL checkpoint (.pt with 'model_state_dict' key, saved by rsl_rl OnPolicyRunner)
  3. ONNX Runtime (.onnx)
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def _build_mlp_from_state_dict(
    state_dict: dict,
    prefix: str = "actor",
    activation: str = "elu",
) -> "nn.Sequential":
    """从 RSL-RL checkpoint 的 state_dict 重建 actor MLP。

    RSL-RL 保存的 actor 权重命名规则为 ``actor.0.weight``, ``actor.2.weight``...
    （偶数索引为 Linear，奇数索引为激活函数，因此从 0,2,4... 的 weight 层推断结构）。

    Args:
        state_dict: checkpoint['model_state_dict']
        prefix:     权重前缀，默认 "actor"
        activation: 激活函数名称，默认 "elu"

    Returns:
        重建好的 nn.Sequential，可直接 load_state_dict。
    """
    ACT_MAP = {
        "elu":   nn.ELU,
        "relu":  nn.ReLU,
        "tanh":  nn.Tanh,
        "selu":  nn.SELU,
    }
    act_cls = ACT_MAP.get(activation.lower(), nn.ELU)

    # 收集所有 Linear 层索引（key 形如 "actor.0.weight"）
    linear_indices = sorted(
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith(f"{prefix}.") and k.endswith(".weight")
    )

    layers: list[nn.Module] = []
    for idx in linear_indices:
        w = state_dict[f"{prefix}.{idx}.weight"]
        out_features, in_features = w.shape
        layers.append(nn.Linear(in_features, out_features))
        # 最后一层不加激活函数
        if idx != linear_indices[-1]:
            layers.append(act_cls())

    return nn.Sequential(*layers)


class PolicyRunner:
    """Unified policy inference wrapper.

    Loads a trained policy from:
      - ``.pt`` TorchScript (via ``torch.jit.save``)
      - ``.pt`` RSL-RL checkpoint (dict with ``model_state_dict`` key)
      - ``.onnx`` ONNX Runtime

    Args:
        path: Path to the model file.
        num_actions: Expected action dimensionality.
        activation: Actor MLP 的激活函数（仅 RSL-RL checkpoint 模式需要），默认 "elu"。
    """

    def __init__(self, path: str, num_actions: int, activation: str = "elu"):
        self.path = path
        self.num_actions = num_actions

        if path.endswith(".onnx"):
            if not HAS_ORT:
                raise RuntimeError("ONNX backend requires onnxruntime: pip install onnxruntime")
            self.backend = "onnx"
            self.session = ort.InferenceSession(path)
            self._print_onnx_info()

        elif path.endswith(".pt"):
            if not HAS_TORCH:
                raise RuntimeError("Torch backend requires PyTorch: pip install torch")
            # 尝试判断是 TorchScript 还是 RSL-RL checkpoint
            raw = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(raw, dict) and "model_state_dict" in raw:
                # ── RSL-RL checkpoint 格式 ──────────────────────────────
                self.backend = "torch"
                sd = raw["model_state_dict"]
                self.model = _build_mlp_from_state_dict(sd, prefix="actor", activation=activation)
                # 只加载 actor 部分的权重（去掉 "actor." 前缀）
                actor_sd = {
                    k[len("actor."):]: v
                    for k, v in sd.items()
                    if k.startswith("actor.")
                }
                self.model.load_state_dict(actor_sd)
                self.model.eval()
                iter_n = raw.get("iter", "?")
                print(f"  [Torch] Loaded RSL-RL checkpoint (iter={iter_n}) from {path}")
                print(f"  [Torch] Actor MLP: {self.model}")
            else:
                # ── TorchScript 格式 ─────────────────────────────────────
                self.backend = "torch"
                self.model = torch.jit.load(path, map_location="cpu")
                self.model.eval()
                print(f"  [Torch] Loaded TorchScript model from {path}")

        else:
            raise ValueError(f"Unsupported model format: {path}  (only .pt / .onnx)")

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run forward pass.

        Args:
            obs: Observation vector (1-D float32 numpy array).

        Returns:
            Action vector (1-D float32 numpy array, length ``num_actions``).
        """
        if self.backend == "onnx":
            return self._infer_onnx(obs)
        else:
            return self._infer_torch(obs)

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _infer_torch(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs.reshape(1, -1).astype(np.float32))
        with torch.no_grad():
            action = self.model(obs_tensor)
        return action.squeeze().cpu().numpy().astype(np.float32)[: self.num_actions]

    def _infer_onnx(self, obs: np.ndarray) -> np.ndarray:
        obs_input = obs.reshape(1, -1).astype(np.float32)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: obs_input})

        # Find the output whose last dim matches num_actions
        for out in outputs:
            arr = np.array(out).squeeze()
            if arr.ndim >= 1 and arr.shape[-1] >= self.num_actions:
                return arr.flatten()[: self.num_actions].astype(np.float32)

        raise RuntimeError(
            f"Cannot extract {self.num_actions}-dim action from ONNX outputs "
            f"(shapes: {[np.array(o).shape for o in outputs]})"
        )

    def _print_onnx_info(self):
        print(f"  [ONNX] Loaded from {self.path}")
        print("  [ONNX] Inputs:")
        for inp in self.session.get_inputs():
            print(f"    {inp.name}: shape={inp.shape}, type={inp.type}")
        print("  [ONNX] Outputs:")
        for out in self.session.get_outputs():
            print(f"    {out.name}: shape={out.shape}, type={out.type}")
