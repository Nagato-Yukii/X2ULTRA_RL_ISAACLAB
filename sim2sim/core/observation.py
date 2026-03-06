"""
Observation Builder with Registry & Per-Term History
======================================================
Constructs the observation vector matching Isaac Lab's ``ObservationManager``
with per-term history stacking (``CircularBuffer``).

**Critical**: Isaac Lab stacks history **per observation term**, NOT per frame.
Each term has its own circular buffer of ``history_length`` entries. The final
observation is built by flattening each term's history (oldest → newest) and
then concatenating all terms.

Extensibility
-------------
To add a new observation term for a new task:

1. Register it with ``@register_obs_term``::

       @register_obs_term("height_scan", dim_hint=187)
       def _height_scan(state: dict) -> np.ndarray:
           return state["height_scan"].astype(np.float32)

2. Add it to your YAML config::

       observation:
         history_length: 5
         terms:
           - name: height_scan
             scale: 1.0

3. Provide the data in the state dict inside ``deploy.py``::

       state["height_scan"] = my_raycast_function(env)

Built-in terms
--------------
======  =================  ===========  ============================
Index   Name               Dims         Notes
======  =================  ===========  ============================
  0     base_ang_vel        3           body-frame (scale from YAML)
  1     projected_gravity   3           body-frame unit vector
  2     velocity_commands   3           [vx, vy, wz]
  3     joint_pos_rel       num_actions current − default
  4     joint_vel_rel       num_actions (scale from YAML)
  5     last_action         num_actions raw action from prev step
  6     base_lin_vel        3           world-frame linear velocity
======  =================  ===========  ============================

Example layout with ``history_length = 5`` and 12 action joints::

    [ang_vel_t-4(3) .. ang_vel_t(3),     # 3×5 = 15
     gravity_t-4(3) .. gravity_t(3),     # 3×5 = 15
     cmd_t-4(3)     .. cmd_t(3),         # 3×5 = 15
     pos_rel_t-4(12).. pos_rel_t(12),   # 12×5 = 60
     vel_rel_t-4(12).. vel_rel_t(12),   # 12×5 = 60
     action_t-4(12) .. action_t(12)]    # 12×5 = 60
                                         total = 225
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .math_utils import get_projected_gravity


# ═══════════════════════════════════════════════════════════════════════════
# Term Registry
# ═══════════════════════════════════════════════════════════════════════════

ObsTermFn = Callable[[dict[str, np.ndarray]], np.ndarray]

_TERM_REGISTRY: dict[str, ObsTermFn] = {}

# Dimension hints: int = fixed, "num_actions" = resolved at build time
_TERM_DIM_HINTS: dict[str, int | str] = {}


def register_obs_term(name: str, dim_hint: int | str | None = None):
    """Decorator to register an observation term compute function.

    The decorated function receives a *state dict* (str → np.ndarray) and
    must return a 1-D float32 array.  The ``dim_hint`` lets the builder
    know the output size at construction time so it can pre-allocate
    history buffers.

    Args:
        name:     Unique term name (referenced in YAML configs).
        dim_hint: Output dimension — an ``int`` for fixed sizes (e.g. 3),
                  or the string ``"num_actions"`` for action-dependent sizes.
                  May also be ``None`` if the user always specifies ``dim``
                  explicitly in the YAML.

    Example::

        @register_obs_term("height_scan", dim_hint=187)
        def _height_scan(state: dict) -> np.ndarray:
            return state["height_scan"].astype(np.float32)
    """

    def decorator(fn: ObsTermFn) -> ObsTermFn:
        _TERM_REGISTRY[name] = fn
        if dim_hint is not None:
            _TERM_DIM_HINTS[name] = dim_hint
        return fn

    return decorator


def get_registered_terms() -> list[str]:
    """Return names of all registered observation terms."""
    return list(_TERM_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Built-in Observation Terms
# ═══════════════════════════════════════════════════════════════════════════


@register_obs_term("base_ang_vel", dim_hint=3)
def _base_ang_vel(state: dict) -> np.ndarray:
    """Base angular velocity in body frame (3,)."""
    return state["base_ang_vel_body"].copy().astype(np.float32)


@register_obs_term("projected_gravity", dim_hint=3)
def _projected_gravity(state: dict) -> np.ndarray:
    """Gravity direction in body frame, unit vector (3,)."""
    return get_projected_gravity(state["base_quat"])


@register_obs_term("velocity_commands", dim_hint=3)
def _velocity_commands(state: dict) -> np.ndarray:
    """Velocity command [vx, vy, wz] (3,)."""
    return state["velocity_cmd"].copy().astype(np.float32)


@register_obs_term("joint_pos_rel", dim_hint="num_actions")
def _joint_pos_rel(state: dict) -> np.ndarray:
    """Relative joint positions: current − default (num_actions,)."""
    return (state["joint_pos"] - state["joint_default_pos"]).astype(np.float32)


@register_obs_term("joint_vel_rel", dim_hint="num_actions")
def _joint_vel_rel(state: dict) -> np.ndarray:
    """Joint velocities (num_actions,).  Scale is applied by the builder."""
    return state["joint_vel"].copy().astype(np.float32)


@register_obs_term("last_action", dim_hint="num_actions")
def _last_action(state: dict) -> np.ndarray:
    """Previous policy action output (num_actions,)."""
    return state["last_action"].copy().astype(np.float32)


@register_obs_term("base_lin_vel", dim_hint=3)
def _base_lin_vel(state: dict) -> np.ndarray:
    """Base linear velocity in world frame (3,)."""
    return state["base_lin_vel"].copy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Term Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ObsTermCfg:
    """Configuration for a single observation term.

    Attributes:
        name:           Registered term name (must appear in the registry).
        scale:          Scalar or per-element scale applied after compute.
        history_length: Per-term history length.  ``0`` means "use the
                        group-level default".
        dim:            Explicit output dimension.  Overrides ``dim_hint``
                        from the registry — useful for custom terms.
    """

    name: str
    scale: float | list[float] = 1.0
    history_length: int = 0
    dim: int | None = None


def _resolve_dim(cfg: ObsTermCfg, num_actions: int) -> int:
    """Determine the output dimension of an observation term."""
    if cfg.dim is not None:
        return cfg.dim
    hint = _TERM_DIM_HINTS.get(cfg.name)
    if hint is None:
        raise ValueError(
            f"Term '{cfg.name}' has unknown dimension.  "
            f"Specify 'dim' in the YAML or use dim_hint in @register_obs_term.  "
            f"Registered terms: {get_registered_terms()}"
        )
    return num_actions if hint == "num_actions" else int(hint)


# ═══════════════════════════════════════════════════════════════════════════
# Observation Builder
# ═══════════════════════════════════════════════════════════════════════════


class ObservationBuilder:
    """Data-driven observation builder with per-term history stacking.

    Reads a list of :class:`ObsTermCfg` to decide *which* observation terms
    to include and in *which* order.  Each term is computed via its
    registered function from the term registry, then scaled and pushed
    into a per-term FIFO history buffer.

    This design means **adding a new task with different observations
    requires only a new YAML config** — no Python changes needed (unless
    the task introduces an entirely new observation term, in which case a
    single ``@register_obs_term`` decorator suffices).

    Args:
        terms:                  Ordered list of term configurations.
        default_history_length: Fallback history length for terms that do
                                not specify one (i.e. ``history_length=0``).
        num_actions:            Number of action dimensions.
    """

    def __init__(
        self,
        terms: list[ObsTermCfg],
        default_history_length: int = 1,
        num_actions: int = 12,
    ):
        self.terms = terms
        self.num_actions = num_actions

        # ── Resolve dims, scales, history lengths ─────────────────────────
        self._term_dims: OrderedDict[str, int] = OrderedDict()
        self._term_scales: OrderedDict[str, np.ndarray | float] = OrderedDict()
        self._term_hist_lens: OrderedDict[str, int] = OrderedDict()

        for tcfg in terms:
            if tcfg.name not in _TERM_REGISTRY:
                raise ValueError(
                    f"Unknown observation term '{tcfg.name}'.  "
                    f"Available: {get_registered_terms()}"
                )
            dim = _resolve_dim(tcfg, num_actions)
            hl = tcfg.history_length if tcfg.history_length > 0 else default_history_length

            self._term_dims[tcfg.name] = dim
            self._term_hist_lens[tcfg.name] = hl

            # Resolve scale (scalar or per-element array)
            if isinstance(tcfg.scale, list):
                s = np.array(tcfg.scale, dtype=np.float32)
                if len(s) != dim:
                    raise ValueError(
                        f"Scale length {len(s)} != dim {dim} for term '{tcfg.name}'"
                    )
                self._term_scales[tcfg.name] = s
            else:
                self._term_scales[tcfg.name] = float(tcfg.scale)

        self.frame_size = sum(self._term_dims.values())
        self.obs_size = sum(
            dim * self._term_hist_lens[name]
            for name, dim in self._term_dims.items()
        )

        # Per-term history buffers
        self._term_histories: OrderedDict[str, deque] = OrderedDict()
        self.reset()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self):
        """Clear all per-term history buffers (fill with zeros)."""
        self._term_histories.clear()
        for name, dim in self._term_dims.items():
            hl = self._term_hist_lens[name]
            buf: deque[np.ndarray] = deque(maxlen=hl)
            for _ in range(hl):
                buf.append(np.zeros(dim, dtype=np.float32))
            self._term_histories[name] = buf

    def build(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """Compute terms from *state*, push to history, return full obs.

        Args:
            state: Dictionary mapping state keys to numpy arrays.  Must
                   contain all keys required by the configured terms.
                   Common keys: ``base_quat``, ``base_ang_vel_body``,
                   ``joint_pos``, ``joint_default_pos``, ``joint_vel``,
                   ``velocity_cmd``, ``last_action``.

        Returns:
            Full observation ``(obs_size,)`` — per-term history flattened
            then concatenated across terms.
        """
        term_values = self._compute_and_scale(state)

        for name, value in term_values.items():
            self._term_histories[name].append(value)

        parts: list[np.ndarray] = []
        for tcfg in self.terms:
            parts.append(np.concatenate(list(self._term_histories[tcfg.name])))

        return np.concatenate(parts)

    def prefill(self, state: dict[str, np.ndarray]):
        """Fill every history slot with the initial observation.

        Matches Isaac Lab's ``CircularBuffer`` first-push behaviour.
        """
        term_values = self._compute_and_scale(state)

        self._term_histories.clear()
        for tcfg in self.terms:
            hl = self._term_hist_lens[tcfg.name]
            buf: deque[np.ndarray] = deque(maxlen=hl)
            for _ in range(hl):
                buf.append(term_values[tcfg.name].copy())
            self._term_histories[tcfg.name] = buf

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _compute_and_scale(
        self, state: dict[str, np.ndarray]
    ) -> OrderedDict[str, np.ndarray]:
        """Call each term's registered function and apply its scale."""
        result: OrderedDict[str, np.ndarray] = OrderedDict()
        for tcfg in self.terms:
            fn = _TERM_REGISTRY[tcfg.name]
            value = fn(state)
            scale = self._term_scales[tcfg.name]
            if isinstance(scale, np.ndarray):
                value = value * scale
            elif scale != 1.0:
                value = value * scale
            result[tcfg.name] = value.astype(np.float32)
        return result
