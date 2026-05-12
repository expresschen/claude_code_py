"""In-process teammate spawning utilities.

Core spawn function is in task/manager.py - this module provides
helper functions and data structures.

Ported from: src/utils/swarm/spawnInProcess.ts
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

# Debug flag - controlled by environment variable CLAUDE_CODE_DEBUG_TEAMMATE
DEBUG_SPAWN_IN_PROCESS = os.environ.get("CLAUDE_CODE_DEBUG_TEAMMATE", "").lower() in ("1", "true", "yes")

# Import unified debug logger
from claude_code_py.utils.debug_log import debug_log

def _debug_print(msg: str) -> None:
    """Print debug message to console and log file."""
    debug_log("[SPAWN_IN_PROCESS]", msg, DEBUG_SPAWN_IN_PROCESS)

if TYPE_CHECKING:
    from claude_code_py.utils.abort_controller import AbortController

logger = logging.getLogger(__name__)


# =============================================================================
# Spinner Verbs
# =============================================================================

_SPINNER_VERBS = [
    "working",
    "thinking",
    "processing",
    "analyzing",
    "exploring",
    "searching",
    "building",
    "implementing",
    "reviewing",
    "validating",
]

_COMPLETION_VERBS = [
    "completed",
    "finished",
    "done",
    "resolved",
    "implemented",
    "reviewed",
    "validated",
    "analyzed",
    "built",
    "processed",
]


def _get_random_spinner_verb() -> str:
    """Get a random spinner verb."""
    return secrets.choice(_SPINNER_VERBS)


def _get_random_completion_verb() -> str:
    """Get a random completion verb."""
    return secrets.choice(_COMPLETION_VERBS)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SpawnContext:
    """Minimal context for spawning an in-process teammate."""

    set_app_state: Callable[[Callable], None]
    tool_use_id: Optional[str] = None


# =============================================================================
# ID Generation
# =============================================================================


def generate_task_id(task_type: str = "in_process_teammate") -> str:
    """Generate a unique task ID."""
    task_id = f"{task_type}-{int(time.time() * 1000)}-{secrets.token_hex(4)}"
    _debug_print(f"generate_task_id: '{task_id}' (type='{task_type}')")
    return task_id


def format_agent_id(agent_name: str, team_name: str) -> str:
    """Format an agent ID."""
    safe_name = agent_name.replace("@", "-")
    agent_id = f"{safe_name}@{team_name}"
    _debug_print(f"format_agent_id: '{agent_id}' (name='{agent_name}', team='{team_name}')")
    return agent_id


# =============================================================================
# Session ID
# =============================================================================

_CURRENT_SESSION_ID: Optional[str] = None


def set_session_id(session_id: str) -> None:
    """Set the current session ID."""
    global _CURRENT_SESSION_ID
    _CURRENT_SESSION_ID = session_id
    _debug_print(f"set_session_id: '{session_id}'")


def get_session_id() -> Optional[str]:
    """Get the current session ID."""
    session_id = _CURRENT_SESSION_ID
    _debug_print(f"get_session_id: '{session_id}'")
    return session_id


def generate_session_id() -> str:
    """Generate a new session ID."""
    session_id = f"session-{int(time.time() * 1000)}-{secrets.token_hex(8)}"
    _debug_print(f"generate_session_id: '{session_id}'")
    return session_id


# =============================================================================
# Cleanup Registry
# =============================================================================

_CLEANUP_HANDLERS: List[Callable] = []


def register_cleanup(handler: Callable) -> Callable[[], None]:
    """Register a cleanup handler."""
    _CLEANUP_HANDLERS.append(handler)

    def unregister() -> None:
        try:
            _CLEANUP_HANDLERS.remove(handler)
        except ValueError:
            pass

    return unregister


async def run_cleanup_handlers() -> None:
    """Run all registered cleanup handlers."""
    import asyncio

    for handler in _CLEANUP_HANDLERS:
        try:
            result = handler()
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.debug(f"Cleanup handler error: {e}")


# =============================================================================
# Task Registration (delegated to task/manager)
# =============================================================================

# Note: spawn_in_process_teammate is in task/manager.py
# Use: from claude_code_py.task.manager import spawn_in_process_teammate


STOPPED_DISPLAY_MS = 2000


__all__ = [
    "SpawnContext",
    "generate_task_id",
    "format_agent_id",
    "set_session_id",
    "get_session_id",
    "generate_session_id",
    "register_cleanup",
    "run_cleanup_handlers",
    "_get_random_spinner_verb",
    "_get_random_completion_verb",
    "STOPPED_DISPLAY_MS",
    # spawn_in_process_teammate is in task/manager.py
]