"""Team context state management.

Manages the leader's team context in AppState.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, Callable

from claude_code_py.utils.team.team_file import (
    TeamFile,
    TeamMember,
    BackendType,
    format_agent_id,
)
from claude_code_py.utils.swarm.constants import TEAM_LEAD_NAME


def create_team_context(
    team_name: str,
    team_file_path: str,
    lead_agent_id: str,
    cwd: str,
    agent_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Create initial team context for AppState.

    Args:
        team_name: Team name
        team_file_path: Path to team config file
        lead_agent_id: Leader's agent ID
        cwd: Working directory
        agent_type: Optional agent type

    Returns:
        Team context dict
    """
    return {
        "teamName": team_name,
        "teamFilePath": team_file_path,
        "leadAgentId": lead_agent_id,
        "teammates": {
            lead_agent_id: {
                "name": TEAM_LEAD_NAME,
                "agentType": agent_type or TEAM_LEAD_NAME,
                "color": "blue",
                "cwd": cwd,
                "spawnedAt": int(time.time() * 1000),
            }
        },
    }


def add_teammate_to_context(
    team_context: Dict[str, Any],
    agent_id: str,
    name: str,
    agent_type: str,
    color: str,
    cwd: str,
) -> Dict[str, Any]:
    """Add a teammate to team context.

    Args:
        team_context: Current team context
        agent_id: New teammate's agent ID
        name: Teammate name
        agent_type: Agent type
        color: Assigned color
        cwd: Working directory

    Returns:
        Updated team context
    """
    teammates = team_context.get("teammates", {})
    teammates[agent_id] = {
        "name": name,
        "agentType": agent_type,
        "color": color,
        "cwd": cwd,
        "spawnedAt": int(time.time() * 1000),
    }
    return {**team_context, "teammates": teammates}


def remove_teammate_from_context(
    team_context: Dict[str, Any],
    agent_id: str,
) -> Dict[str, Any]:
    """Remove a teammate from team context.

    Args:
        team_context: Current team context
        agent_id: Agent ID to remove

    Returns:
        Updated team context
    """
    teammates = team_context.get("teammates", {})
    if agent_id in teammates:
        del teammates[agent_id]
    return {**team_context, "teammates": teammates}


def get_leader_team_name(team_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """Get the team name for the leader.

    Args:
        team_context: Team context from AppState

    Returns:
        Team name or None
    """
    if not team_context:
        return None
    return team_context.get("teamName")


# Note: set_leader_team_name is defined in utils/task/file_storage.py
# as a module-level variable, matching TypeScript tasks.ts implementation.
# Do not duplicate here.


__all__ = [
    "create_team_context",
    "add_teammate_to_context",
    "remove_teammate_from_context",
    "get_leader_team_name",
]