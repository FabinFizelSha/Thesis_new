"""Locate this workspace's root directory at runtime.

Every rsg Python file lives at <workspace_root>/src/rsg/..., and colcon's
--symlink-install preserves that: Path(__file__).resolve() inside an
installed/symlinked module resolves through the symlink back to the real
source-tree file (verified: install/rsg/lib/python3.12/site-packages/nodes/
phase1.py is a symlink to src/rsg/nodes/phase1.py). So the workspace root
can always be found by walking up from this file's own resolved location --
no hardcoded absolute path and no per-machine environment variable needed.
Cloning the workspace to any path, on any machine, under any username, just
works.

Two path conventions in config files (YAML, and the few Python literals that
aren't YAML-driven) are supported:
  * "~/..."                  -> expanded against the current user's home.
  * "${WORKSPACE_ROOT}/..."  -> expanded against get_workspace_root().
Both are handled by expand_path_value(); expand_paths_in_yaml() applies it
recursively to an entire loaded YAML document in one call, so individual
config-loading call sites never need to remember to expand anything.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_ROOT_MARKER = ("src", "rsg")


def get_workspace_root() -> Path:
    """Return the workspace root (the directory containing src/rsg/...)."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if candidate.joinpath(*_ROOT_MARKER).is_dir():
            return candidate
    raise RuntimeError(
        f"Could not locate the workspace root by walking up from {here}; "
        f"expected an ancestor directory containing {'/'.join(_ROOT_MARKER)}."
    )


def workspace_path(*parts: str) -> Path:
    """Join path components onto the workspace root, e.g. workspace_path('memory', 'tracker')."""
    return get_workspace_root().joinpath(*parts)


def expand_path_value(value: Any) -> Any:
    """Expand '~' and '${WORKSPACE_ROOT}' in a single string; pass through anything else unchanged."""
    if not isinstance(value, str) or not value:
        return value
    expanded = os.path.expanduser(value)
    if "${WORKSPACE_ROOT}" in expanded:
        expanded = expanded.replace("${WORKSPACE_ROOT}", str(get_workspace_root()))
    return expanded


def expand_paths_in_yaml(obj: Any) -> Any:
    """Recursively expand '~' and '${WORKSPACE_ROOT}' in every string value of a loaded YAML document.

    Safe to apply universally: strings that don't start with '~' or contain
    '${WORKSPACE_ROOT}' pass through os.path.expanduser unchanged, so this
    never touches non-path config values (topic names, thresholds, etc.).
    """
    if isinstance(obj, dict):
        return {key: expand_paths_in_yaml(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand_paths_in_yaml(value) for value in obj]
    return expand_path_value(obj)
