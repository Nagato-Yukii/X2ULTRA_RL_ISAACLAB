"""
Parse environment config from task registry.

This is a local replacement for `unitree_rl_lab.utils.parser_cfg.parse_env_cfg`.
It uses the standard isaaclab_tasks API which is available when isaaclab is installed.
"""

from __future__ import annotations

from isaaclab_tasks.utils import parse_env_cfg  # noqa: F401 — re-export for compatibility
