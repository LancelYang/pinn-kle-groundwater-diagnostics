"""Project path helpers.

All experiment configs use paths relative to the project root, not relative
to ``src/`` or the current shell directory.
"""

import re
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = PROJECT_ROOT.name
ROOT_RELATIVE_DIRS = {
    "configs",
    "data",
    "docs",
    "figures",
    "outputs",
    "scripts",
    "src",
    "PROGRESS",
}


def _suffix_after_project_root(parts):
    """Return the path suffix after the project directory name, if present."""
    for i, part in enumerate(parts):
        if part == PROJECT_NAME:
            return parts[i + 1 :]
    return None


def resolve_project_path(path_like):
    """Return an absolute path, resolving project-relative paths robustly.

    Historical experiment JSON files may contain Windows absolute paths even
    when the analysis is being run on macOS/Linux. If such a path includes the
    project directory name, map its suffix back onto the current checkout.
    """
    if path_like is None:
        return None

    raw = str(path_like)
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        win_path = PureWindowsPath(raw)
        suffix = _suffix_after_project_root(win_path.parts)
        if suffix is not None:
            return PROJECT_ROOT.joinpath(*suffix)

    normalized = raw.replace("\\", "/")
    first = normalized.split("/", 1)[0]
    if first in ROOT_RELATIVE_DIRS:
        return PROJECT_ROOT / normalized

    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
