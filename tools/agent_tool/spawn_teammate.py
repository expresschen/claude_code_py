"""Teammate spawning logic for Agent tool.

This module provides the spawn_teammate function that handles
spawning in-process teammates when team_name and name parameters
are provided to the Agent tool.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Optional, TYPE_CHECKING

# Debug flag - controlled by environment variable CLAUDE_CODE_DEBUG_TEAMMATE
DEBUG_SPAWN = os.environ.get("CLAUDE_CODE_DEBUG_TEAMMATE", "").lower() in ("1", "true", "yes")

# Import unified debug logger
from claude_code_py.utils.debug_log import debug_log

def _debug_print(msg: str) -> None:
    """Print debug message to console and log file."""
    debug_log("[SPAWN_TEAMMATE]", msg, DEBUG_SPAWN)

if TYPE_CHECKING:
    from claude_code_py.tool.context import ToolUseContext

logger = logging.getLogger(__name__)


# =============================================================================
# Color Assignment
# =============================================================================

_TEAM_COLORS = ["red", "green", "yellow", "purple", "orange", "cyan", "magenta"]

# Track assigned colors by agent_id
_ASSIGNED_COLORS: dict[str, str] = {}


def _assign_color(agent_id: str) -> str:
    """Assign a color to a teammate.

    Uses a deterministic rotation based on the agent_id to ensure
    consistent color assignment across teammates.

    Args:
        agent_id: Agent ID to assign color to

    Returns:
        Color string (e.g., "red", "green")
    """
    # Check if already assigned
    if agent_id in _ASSIGNED_COLORS:
        return _ASSIGNED_COLORS[agent_id]

    # Count existing assignments to pick next color
    num_assigned = len(_ASSIGNED_COLORS)
    color = _TEAM_COLORS[num_assigned % len(_TEAM_COLORS)]

    # Store assignment
    _ASSIGNED_COLORS[agent_id] = color
    return color


# =============================================================================
# Input/Output Types
# =============================================================================


@dataclass
class SpawnTeammateInput:
    """Input for spawning a teammate."""

    name: str  # Display name for the teammate
    prompt: str  # Initial prompt for the teammate
    description: Optional[str] = None  # Optional task description
    team_name: Optional[str] = None  # Team name (required for spawn)
    model: Optional[str] = None  # Optional model override
    plan_mode_required: bool = False  # Whether plan mode is required
    agent_type: Optional[str] = None  # Optional agent type specifier
    tool_use_id: Optional[str] = None  # Tool use ID for correlation
    color: Optional[str] = None  # Optional UI color


@dataclass
class SpawnTeammateOutput:
    """Output from spawning a teammate."""

    success: bool
    agent_id: str  # Full agent ID (e.g., "worker@my-team")
    task_id: Optional[str] = None  # Task ID if spawned successfully
    error: Optional[str] = None  # Error message if failed


# =============================================================================
# Spawn Function
# =============================================================================


async def spawn_teammate(
    input: SpawnTeammateInput,
    context: "ToolUseContext",
) -> SpawnTeammateOutput:
    """Spawn an in-process teammate.

    This function:
    1. Checks if agent teams feature is enabled
    2. Checks if already a teammate (nesting prevention)
    3. Checks if already an in-process teammate
    4. Assigns color
    5. Gets parent session ID
    6. Gets cwd from context
    7. Builds SpawnTeammateConfig
    8. Calls spawn_in_process_teammate_v2
    9. Adds member to team file
    10. Adds teammate to AppState teamContext
    11. Returns SpawnTeammateOutput

    Args:
        input: Spawn configuration
        context: Tool use context

    Returns:
        SpawnTeammateOutput with spawn result
    """
    from claude_code_py.utils.swarm.constants import is_agent_teams_enabled
    from claude_code_py.utils.teammate_context import (
        is_teammate,
        is_in_process_teammate,
    )
    from claude_code_py.utils.team.team_file import (
        add_member_to_team,
        TeamMember,
        BackendType,
        format_agent_id,
    )
    from claude_code_py.task.manager import (
        SpawnTeammateConfig,
        spawn_in_process_teammate_v2,
    )
    from claude_code_py.utils.swarm.spawn_in_process import get_session_id
    from claude_code_py.state.team_state import add_teammate_to_context

    # Step a: Check if agent teams is enabled
    if not is_agent_teams_enabled():
        _debug_print("❌ Agent teams feature is NOT enabled")
        _debug_print("   Tip: Set CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 to enable")
        return SpawnTeammateOutput(
            success=False,
            agent_id=format_agent_id(input.name, input.team_name or "default"),
            error="Agent teams feature is not enabled. Set CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 to enable.",
        )

    _debug_print(f"✅ Starting spawn process")
    _debug_print(f"   name='{input.name}'")
    _debug_print(f"   team_name='{input.team_name}'")
    _debug_print(f"   prompt='{input.prompt[:100]}{'...' if len(input.prompt) > 100 else ''}'")
    _debug_print(f"   model='{input.model}'")
    _debug_print(f"   plan_mode_required={input.plan_mode_required}")
    _debug_print(f"   agent_type='{input.agent_type}'")

    # Step b: Check if already a teammate (nesting prevention)
    if is_teammate():
        _debug_print("❌ Cannot spawn: already running as a teammate")
        return SpawnTeammateOutput(
            success=False,
            agent_id=format_agent_id(input.name, input.team_name or "default"),
            error="Cannot spawn teammate from within a teammate context (nesting not allowed).",
        )
    _debug_print("✅ Nesting check passed (not already a teammate)")

    # Step c: Check if already an in-process teammate
    if is_in_process_teammate():
        _debug_print("❌ Cannot spawn: already running as in-process teammate")
        return SpawnTeammateOutput(
            success=False,
            agent_id=format_agent_id(input.name, input.team_name or "default"),
            error="Cannot spawn in-process teammate from within an in-process teammate.",
        )
    _debug_print("✅ In-process check passed")

    # Step d: Assign color
    agent_id = format_agent_id(input.name, input.team_name)
    color = input.color or _assign_color(agent_id)
    _debug_print(f"✅ Color assigned: color='{color}' for agent_id='{agent_id}'")

    # Step e: Get parent session ID
    parent_session_id = get_session_id() or "default"
    _debug_print(f"✅ Parent session ID: '{parent_session_id}'")

    # Step f: Get cwd from context
    try:
        cwd = context.get_cwd()
        _debug_print(f"✅ Working directory from context: '{cwd}'")
    except (AttributeError, TypeError) as e:
        _debug_print(f"⚠️ Could not get cwd from context: {e}")
        # Fallback: try to get from options directly
        try:
            cwd = context.options.cwd if hasattr(context, 'options') and hasattr(context.options, 'cwd') else "."
        except Exception:
            cwd = "."
        _debug_print(f"   Using fallback cwd: '{cwd}'")

    # Step g: Build SpawnTeammateConfig
    spawn_config = SpawnTeammateConfig(
        name=input.name,
        team_name=input.team_name,
        prompt=input.prompt,
        description=input.description,
        model=input.model,
        color=color,
        plan_mode_required=input.plan_mode_required,
        parent_session_id=parent_session_id,
        tool_use_id=input.tool_use_id,
        agent_type=input.agent_type,
        cwd=cwd,
    )
    _debug_print("✅ SpawnTeammateConfig built")

    # Get AppState callbacks from context
    set_app_state = context.set_app_state
    get_app_state = context.get_app_state
    _debug_print(f"✅ AppState callbacks: set_app_state={set_app_state is not None}, get_app_state={get_app_state is not None}")

    # Step h: Call spawn_in_process_teammate_v2
    try:
        _debug_print("→ Calling spawn_in_process_teammate_v2...")
        spawn_result = await spawn_in_process_teammate_v2(
            config=spawn_config,
            set_app_state=set_app_state,
            get_app_state=get_app_state,
        )

        _debug_print(f"← spawn_in_process_teammate_v2 returned:")
        _debug_print(f"   success={spawn_result.success}")
        _debug_print(f"   agent_id='{spawn_result.agent_id}'")
        _debug_print(f"   task_id='{spawn_result.task_id}'")
        if spawn_result.error:
            _debug_print(f"   error='{spawn_result.error}'")

        if not spawn_result.success:
            _debug_print("❌ Spawn failed!")
            return SpawnTeammateOutput(
                success=False,
                agent_id=agent_id,
                error=spawn_result.error or "Failed to spawn teammate",
            )

        # Step i: Add member to team file
        _debug_print("→ Adding member to team file...")
        team_member = TeamMember(
            agent_id=agent_id,
            name=input.name,
            agent_type=input.agent_type,
            model=input.model,
            prompt=input.prompt,
            color=color,
            plan_mode_required=input.plan_mode_required,
            joined_at=int(time.time() * 1000),
            cwd=cwd,
            backend_type=BackendType.IN_PROCESS,
            is_active=True,
            mode="default",
        )

        add_member_to_team(input.team_name, team_member)
        _debug_print(f"✅ Member added to team file: team='{input.team_name}'")
        logger.debug(f"Added member {agent_id} to team file {input.team_name}")

        # Step j: Add teammate to AppState teamContext
        if set_app_state and get_app_state:
            _debug_print("→ Adding teammate to AppState teamContext...")
            current_state = get_app_state()
            team_context = current_state.team_context

            updated_team_context = add_teammate_to_context(
                team_context=team_context,
                agent_id=agent_id,
                name=input.name,
                agent_type=input.agent_type or "default",
                color=color,
                cwd=cwd,
            )

            set_app_state(lambda prev: replace(
                prev,
                team_context=updated_team_context,
            ))
            _debug_print(f"✅ Teammate added to AppState teamContext")
            logger.debug(f"Added teammate {agent_id} to AppState teamContext")

        # Step k: Return SpawnTeammateOutput
        _debug_print("🎉 SPAWN SUCCESSFUL!")
        _debug_print(f"   agent_id='{agent_id}'")
        _debug_print(f"   task_id='{spawn_result.task_id}'")
        return SpawnTeammateOutput(
            success=True,
            agent_id=agent_id,
            task_id=spawn_result.task_id,
        )

    except Exception as e:
        _debug_print(f"❌ EXCEPTION during spawn: {type(e).__name__}: {e}")
        _debug_print(f"   This is an error that needs investigation")
        logger.error(f"Failed to spawn teammate: {e}")
        return SpawnTeammateOutput(
            success=False,
            agent_id=agent_id,
            error=str(e),
        )


__all__ = [
    "SpawnTeammateInput",
    "SpawnTeammateOutput",
    "spawn_teammate",
    "_TEAM_COLORS",
    "_assign_color",
]