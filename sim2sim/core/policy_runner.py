"""
Policy Runner — unified inference interface for .pt / .onnx models.

Supports both TorchScript (.pt) and ONNX Runtime (.onnx) backends.
Falls back gracefully when one backend is unavailable.
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class PolicyRunner:
    """Unified policy inference wrapper.

    Loads a trained policy from a ``.pt`` (TorchScript) or ``.onnx`` file and
    provides a single :meth:`infer` method that accepts a 1-D numpy observation
    and returns a 1-D numpy action.

    Args:
        path: Path to the model file (``.pt`` or ``.onnx``).
        num_actions: Expected action dimensionality.
    """

    def __init__(self, path: str, num_actions: int):
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
