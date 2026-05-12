"""Unified debug logging for teammate operations.

Writes debug output to both console and log file when enabled.

IMPORTANT: This module uses dynamic environment variable checking to avoid
the "import timing" problem where modules are imported before env vars are set.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# Global debug environment variable name
DEBUG_ENV_VAR = "CLAUDE_CODE_DEBUG_TEAMMATE"


def _is_debug_enabled() -> bool:
    """Check if debug is enabled via environment variable.

    Called dynamically at each log invocation to avoid caching issues
    when modules are imported before env vars are set.
    """
    return os.environ.get(DEBUG_ENV_VAR, "").lower() in ("1", "true", "yes")


# Log file path - ~/.claude/logs/teammate_debug.log
def _get_log_file_path() -> Path:
    """Get the log file path."""
    base_dir = Path.home() / ".claude" / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "teammate_debug.log"


# Global log file handle (lazy initialization)
_log_file: Optional[object] = None
_log_file_path: Optional[Path] = None


def _get_log_file():
    """Get or create the log file handle."""
    global _log_file, _log_file_path

    if _log_file is None:
        _log_file_path = _get_log_file_path()
        try:
            _log_file = open(_log_file_path, "a", encoding="utf-8")
        except Exception:
            _log_file = None

    return _log_file


def debug_log(tag: str, msg: str, enabled: bool = True) -> None:
    """Write debug message to log file only (not console).

    Writing to console interferes with Rich Live display and user input.
    All debug output goes to ~/.claude/logs/teammate_debug.log instead.

    Uses dynamic environment variable checking to handle import timing issues:
    - If enabled=True (caller explicitly enabled), always log
    - If enabled=False (caller's static check failed), check global env var
    - This ensures logs appear even when modules were imported before env var was set

    Args:
        tag: Module tag (e.g., "[IN_PROCESS_RUNNER]")
        msg: Message to write
        enabled: Whether debug is enabled from caller's perspective (static check)
    """
    # Dynamic check: if caller's static check failed, check global env var
    # This solves the "import timing" problem where modules cache DEBUG_* = False
    # at import time, before the env var is actually set
    if not enabled and not _is_debug_enabled():
        return

    # Format message with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    full_msg = f"{timestamp} {tag} {msg}"

    # Write to log file only (not console - avoids interfering with REPL)
    try:
        log_file = _get_log_file()
        if log_file:
            log_file.write(full_msg + "\n")
            log_file.flush()  # Ensure immediate write
    except Exception:
        pass  # Ignore file write errors


def close_log_file() -> None:
    """Close the log file handle."""
    global _log_file

    if _log_file:
        try:
            _log_file.close()
        except Exception:
            pass
        _log_file = None


# Convenience function for creating module-specific debug functions
def create_debug_print(tag: str, debug_flag: bool) -> callable:
    """Create a module-specific _debug_print function.

    Note: Even if debug_flag=False (due to import timing), the dynamic
    env var check in debug_log() will still enable logging when the
    environment variable is set.

    Args:
        tag: Module tag (e.g., "[IN_PROCESS_RUNNER]")
        debug_flag: The DEBUG_* flag for this module (may be False due to import timing)

    Returns:
        A _debug_print function for this module
    """
    def _debug_print(msg: str) -> None:
        debug_log(tag, msg, debug_flag)

    return _debug_print


__all__ = [
    "debug_log",
    "create_debug_print",
    "close_log_file",
    "DEBUG_ENV_VAR",
    "_is_debug_enabled",
]