"""Teammate Context - Runtime context for in-process teammates.

This provides contextvars-based context for in-process teammates,
enabling concurrent teammate execution without global state conflicts.

Python 使用 contextvars.ContextVar 替代 TypeScript 的 AsyncLocalStorage。
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

# Debug flag - controlled by environment variable CLAUDE_CODE_DEBUG_TEAMMATE
DEBUG_CONTEXT = os.environ.get("CLAUDE_CODE_DEBUG_TEAMMATE", "").lower() in ("1", "true", "yes")

# Import unified debug logger
from claude_code_py.utils.debug_log import debug_log

def _debug_print(msg: str) -> None:
    """Print debug message to console and log file."""
    debug_log("[TEAMMATE_CONTEXT]", msg, DEBUG_CONTEXT)


@dataclass
class TeammateContext:
    """Runtime context for in-process teammates.

    Stored in ContextVar for concurrent access.
    """

    agent_id: str  # Full agent ID, e.g., "researcher@my-team"
    agent_name: str  # Display name, e.g., "researcher"
    team_name: str  # Team name this teammate belongs to
    parent_session_id: str  # Leader's session ID (for transcript correlation)
    abort_controller: Any  # AbortController for lifecycle management (linked to parent)
    is_in_process: bool = True  # Discriminator - always true for in-process teammates
    color: Optional[str] = None  # UI color assigned to this teammate
    plan_mode_required: bool = False  # Whether teammate must enter plan mode before implementing


# ContextVar for teammate context (替代 AsyncLocalStorage)
_TEAMMATE_CONTEXT: ContextVar[Optional[TeammateContext]] = ContextVar(
    "teammate_context", default=None
)


def get_teammate_context() -> Optional[TeammateContext]:
    """Get the current in-process teammate context, if running as one.

    Returns:
        TeammateContext or None if not running within an in-process teammate
    """
    ctx = _TEAMMATE_CONTEXT.get()
    if ctx:
        _debug_print(f"get_teammate_context: Found context for '{ctx.agent_id}'")
    else:
        _debug_print(f"get_teammate_context: No context (not in teammate)")
    return ctx


def run_with_teammate_context(context: TeammateContext, fn: Callable[[], T]) -> T:
    """Run a function with teammate context set.

    Used when spawning an in-process teammate to establish its execution context.

    Args:
        context: The teammate context to set
        fn: The function to run with the context

    Returns:
        The return value of fn
    """
    _debug_print(f"run_with_teammate_context: Setting context for '{context.agent_id}'")
    token = _TEAMMATE_CONTEXT.set(context)
    try:
        _debug_print(f"   → Running function...")
        result = fn()
        _debug_print(f"   ✅ Function completed")
        return result
    finally:
        _TEAMMATE_CONTEXT.reset(token)
        _debug_print(f"   ✅ Context reset")


async def run_with_teammate_context_async(
    context: TeammateContext,
    fn: Callable[[], T],
) -> T:
    """Run an async function with teammate context set.

    Args:
        context: The teammate context to set
        fn: The async function to run with the context

    Returns:
        The return value of fn
    """
    _debug_print(f"run_with_teammate_context_async: Setting context for '{context.agent_id}'")
    token = _TEAMMATE_CONTEXT.set(context)
    try:
        _debug_print(f"   → Running async function...")
        result = await fn()
        _debug_print(f"   ✅ Async function completed")
        return result
    finally:
        _TEAMMATE_CONTEXT.reset(token)
        _debug_print(f"   ✅ Context reset")


def is_in_process_teammate() -> bool:
    """Check if current execution is within an in-process teammate.

    This is faster than get_teammate_context() is not None for simple checks.

    Returns:
        True if running as an in-process teammate
    """
    result = _TEAMMATE_CONTEXT.get() is not None
    _debug_print(f"is_in_process_teammate: {result}")
    return result


def create_teammate_context(
    agent_id: str,
    agent_name: str,
    team_name: str,
    parent_session_id: str,
    abort_controller: Any,
    color: Optional[str] = None,
    plan_mode_required: bool = False,
) -> TeammateContext:
    """Create a TeammateContext from spawn configuration.

    The abortController is passed in by the caller. For in-process teammates,
    this is typically an independent controller (not linked to parent) so teammates
    continue running when the leader's query is interrupted.

    Args:
        agent_id: Full agent ID (e.g., "researcher@my-team")
        agent_name: Display name
        team_name: Team name
        parent_session_id: Parent session ID
        abort_controller: AbortController for lifecycle
        color: Optional UI color
        plan_mode_required: Whether plan mode is required

    Returns:
        A complete TeammateContext with is_in_process: True
    """
    _debug_print(f"create_teammate_context:")
    _debug_print(f"   agent_id: '{agent_id}'")
    _debug_print(f"   agent_name: '{agent_name}'")
    _debug_print(f"   team_name: '{team_name}'")
    _debug_print(f"   parent_session_id: '{parent_session_id}'")
    _debug_print(f"   color: '{color}'")
    _debug_print(f"   plan_mode_required: {plan_mode_required}")

    context = TeammateContext(
        agent_id=agent_id,
        agent_name=agent_name,
        team_name=team_name,
        parent_session_id=parent_session_id,
        abort_controller=abort_controller,
        color=color,
        plan_mode_required=plan_mode_required,
        is_in_process=True,
    )

    _debug_print(f"✅ TeammateContext created")
    return context


# =============================================================================
# Agent Identity Helpers
# =============================================================================


def format_agent_id(agent_name: str, team_name: str) -> str:
    """Format an agent ID from name and team.

    Args:
        agent_name: Agent display name
        team_name: Team name

    Returns:
        Full agent ID (e.g., "researcher@my-team")
    """
    return f"{agent_name}@{team_name}"


def parse_agent_id(agent_id: str) -> tuple[str, str]:
    """Parse an agent ID into name and team.

    Args:
        agent_id: Full agent ID (e.g., "researcher@my-team")

    Returns:
        Tuple of (agent_name, team_name)
    """
    if "@" in agent_id:
        parts = agent_id.split("@", 1)
        return parts[0], parts[1]
    return agent_id, "default"


def get_current_agent_id() -> Optional[str]:
    """Get the current agent ID from context.

    Returns:
        Agent ID or None if not in teammate context
    """
    ctx = get_teammate_context()
    if ctx:
        return ctx.agent_id
    return None


def get_current_agent_name() -> Optional[str]:
    """Get the current agent name from context.

    Returns:
        Agent name or None if not in teammate context
    """
    ctx = get_teammate_context()
    if ctx:
        return ctx.agent_name
    return None


def get_current_team_name() -> Optional[str]:
    """Get the current team name from context.

    Returns:
        Team name or None if not in teammate context
    """
    ctx = get_teammate_context()
    if ctx:
        return ctx.team_name
    return None


def get_current_parent_session_id() -> Optional[str]:
    """Get the parent session ID from context.

    Returns:
        Parent session ID or None if not in teammate context
    """
    ctx = get_teammate_context()
    if ctx:
        return ctx.parent_session_id
    return None


def get_current_teammate_color() -> Optional[str]:
    """Get the current teammate color from context.

    Returns:
        Teammate color or None if not in teammate context
    """
    ctx = get_teammate_context()
    if ctx:
        return ctx.color
    return None


def is_plan_mode_required() -> bool:
    """Check if plan mode is required for current teammate.

    Returns:
        True if plan mode required, False otherwise
    """
    ctx = get_teammate_context()
    if ctx:
        return ctx.plan_mode_required
    return False


# =============================================================================
# Dynamic Team Context (for runtime team joining)
# =============================================================================

# Dynamic context for tmux teammates (via CLI args)
_DYNAMIC_TEAM_CONTEXT: Optional[dict] = None


def set_dynamic_team_context(context: Optional[dict]) -> None:
    """Set dynamic team context (called when joining a team at runtime).

    Used for tmux teammates that receive their identity via CLI arguments.

    Args:
        context: Dict with agentId, agentName, teamName, color, planModeRequired, parentSessionId
    """
    global _DYNAMIC_TEAM_CONTEXT
    _DYNAMIC_TEAM_CONTEXT = context


def clear_dynamic_team_context() -> None:
    """Clear dynamic team context (called when leaving a team)."""
    global _DYNAMIC_TEAM_CONTEXT
    _DYNAMIC_TEAM_CONTEXT = None


def get_dynamic_team_context() -> Optional[dict]:
    """Get the current dynamic team context."""
    return _DYNAMIC_TEAM_CONTEXT


# =============================================================================
# Team Lead Identity
# =============================================================================

TEAM_LEAD_NAME = "team-lead"


def is_team_lead(team_context: Optional[dict] = None) -> bool:
    """Check if current execution is for the team lead.

    A session is considered team lead if:
    1. A team context exists with leadAgentId, AND
    2. Either our agentId matches leadAgentId, OR
    3. We have no agentId set (original session that created the team)

    Args:
        team_context: Optional team context from AppState

    Returns:
        True if this session is the team lead
    """
    ctx = get_teammate_context()
    if ctx:
        # In-process teammates are never leads
        return False

    # Check dynamic context
    dynamic_ctx = get_dynamic_team_context()
    if not dynamic_ctx:
        # No teammate context at all - this is the main session
        return True if team_context else False

    # Check if our agentId matches leadAgentId
    my_agent_id = dynamic_ctx.get("agentId")
    if team_context and team_context.get("leadAgentId"):
        lead_agent_id = team_context.get("leadAgentId")
        if my_agent_id == lead_agent_id:
            return True
        # Backwards compat: if no agentId set, this is the lead
        if not my_agent_id:
            return True

    return False


def is_teammate() -> bool:
    """Check if current execution is for a teammate (not lead).

    Returns:
        True if in teammate context
    """
    # In-process teammates
    if is_in_process_teammate():
        return True
    # Dynamic context teammates (tmux via CLI args)
    dynamic_ctx = get_dynamic_team_context()
    return bool(dynamic_ctx and dynamic_ctx.get("agentId") and dynamic_ctx.get("teamName"))


def get_parent_session_id() -> Optional[str]:
    """Get the parent session ID for this teammate.

    Priority: in-process context > dynamic context.

    Returns:
        Parent session ID or None
    """
    ctx = get_teammate_context()
    if ctx:
        return ctx.parent_session_id
    dynamic_ctx = get_dynamic_team_context()
    if dynamic_ctx:
        return dynamic_ctx.get("parentSessionId")
    return None


__all__ = [
    "TeammateContext",
    "TEAM_LEAD_NAME",
    "get_teammate_context",
    "run_with_teammate_context",
    "run_with_teammate_context_async",
    "is_in_process_teammate",
    "create_teammate_context",
    "format_agent_id",
    "parse_agent_id",
    "get_current_agent_id",
    "get_current_agent_name",
    "get_current_team_name",
    "get_current_parent_session_id",
    "get_current_teammate_color",
    "is_plan_mode_required",
    "set_dynamic_team_context",
    "clear_dynamic_team_context",
    "get_dynamic_team_context",
    "is_team_lead",
    "is_teammate",
    "get_parent_session_id",
]