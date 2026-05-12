"""Memory path resolution.

This implements memory path resolution from paths.ts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def get_memory_base() -> Path:
    """Get the base directory for memory storage.

    Priority:
    1. CLAUDE_CODE_REMOTE_MEMORY_DIR environment variable
    2. ~/.claude directory

    Returns:
        Path to memory base directory
    """
    remote_dir = os.environ.get("CLAUDE_CODE_REMOTE_MEMORY_DIR")
    if remote_dir:
        return Path(remote_dir)

    # Default to ~/.claude
    home = Path.home()
    return home / ".claude"


def get_project_slug(cwd: Optional[str] = None) -> str:
    """Get a sanitized project slug from the working directory.

    Args:
        cwd: Working directory (defaults to current)

    Returns:
        Sanitized project slug
    """
    cwd = cwd or os.getcwd()
    path = Path(cwd).resolve()

    # Try to get git root
    git_root = _get_git_root(path)
    if git_root:
        # Sanitize: replace non-alphanumeric with dashes
        slug = "".join(c if c.isalnum() else "-" for c in git_root.name)
        return slug.strip("-") or "default"

    # Fallback to directory name
    slug = "".join(c if c.isalnum() else "-" for c in path.name)
    return slug.strip("-") or "default"


def _get_git_root(path: Path) -> Optional[Path]:
    """Find git root directory.

    Args:
        path: Starting path

    Returns:
        Git root or None
    """
    current = path
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def get_auto_mem_path(cwd: Optional[str] = None) -> str:
    """Get the path to the auto memory directory.

    Priority:
    1. CLAUDE_COWORK_MEMORY_PATH_OVERRIDE
    2. settings.json autoMemoryDirectory (not implemented here)
    3. <memoryBase>/projects/<project-slug>/memory/

    Args:
        cwd: Working directory (defaults to current)

    Returns:
        Path to auto memory directory
    """
    # Check for override
    override = os.environ.get("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE")
    if override:
        return override

    # Default path
    memory_base = get_memory_base()
    project_slug = get_project_slug(cwd)
    return str(memory_base / "projects" / project_slug / "memory")


def is_auto_memory_enabled() -> bool:
    """Check if auto memory is enabled.

    Priority:
    1. CLAUDE_CODE_DISABLE_AUTO_MEMORY - disable if truthy
    2. CLAUDE_CODE_SIMPLE - disable if truthy
    3. Remote mode without persistent directory - disable
    4. settings.autoMemoryEnabled - check setting
    5. Default: enabled

    Returns:
        True if auto memory is enabled
    """
    # Check disable env var
    if _is_env_truthy("CLAUDE_CODE_DISABLE_AUTO_MEMORY"):
        return False

    # Check simple mode
    if _is_env_truthy("CLAUDE_CODE_SIMPLE"):
        return False

    # TODO: Check settings.json autoMemoryEnabled setting
    # For now, default to enabled
    return True


def _is_env_truthy(name: str) -> bool:
    """Check if an environment variable is truthy.

    Args:
        name: Environment variable name

    Returns:
        True if truthy
    """
    value = os.environ.get(name, "").lower()
    return value in ("1", "true", "yes", "on")


def get_agent_memory_base() -> Path:
    """Get the base directory for agent memory.

    Returns:
        Path to agent memory base
    """
    return get_memory_base() / "agent-memory"


def get_session_memory_dir(cwd: Optional[str] = None) -> Path:
    """Get the directory for session memory.

    Args:
        cwd: Working directory (defaults to current)

    Returns:
        Path to session memory directory
    """
    memory_base = get_memory_base()
    project_slug = get_project_slug(cwd)
    return memory_base / "projects" / project_slug / "session-memory"


def get_session_memory_path(cwd: Optional[str] = None) -> Path:
    """Get the path to the session memory file.

    Args:
        cwd: Working directory (defaults to current)

    Returns:
        Path to session memory file
    """
    return get_session_memory_dir(cwd) / "memory.md"


def is_auto_mem_path(file_path: str, cwd: Optional[str] = None) -> bool:
    """Check if a path is within the auto memory directory.

    Args:
        file_path: Path to check
        cwd: Working directory (defaults to current)

    Returns:
        True if path is within auto memory directory
    """
    memory_dir = get_auto_mem_path(cwd)
    return str(file_path).startswith(memory_dir)