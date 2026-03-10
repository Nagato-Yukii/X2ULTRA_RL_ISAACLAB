"""
Parse environment config from task registry.

Re-exports the project-level parse_env_cfg from lab_settings/utils/parser_cfg.py,
which extends the official isaaclab_tasks version with the extra `entry_point_key`
argument required by play.py.

Uses importlib.util to load by absolute path to avoid a circular import that would
occur if we simply did `from utils.parser_cfg import ...` (both files share the
same module name "utils.parser_cfg" from their respective root paths).
"""

from __future__ import annotations

import importlib.util
import pathlib

# Resolve the absolute path to lab_settings/utils/parser_cfg.py
_lab_parser_path = pathlib.Path(__file__).parent.parent.parent / "lab_settings" / "utils" / "parser_cfg.py"

# Load the module under a unique alias to avoid name collision
_spec = importlib.util.spec_from_file_location("lab_settings.utils.parser_cfg", _lab_parser_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export parse_env_cfg (supports entry_point_key)
parse_env_cfg = _mod.parse_env_cfg
