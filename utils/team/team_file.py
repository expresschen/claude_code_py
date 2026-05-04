"""Team file data structures and CRUD operations.

Ported from: src/utils/swarm/teamHelpers.ts
"""

from __future__ import annotations

import json
import os
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

TEAM_LEAD_NAME = "team-lead"


# =============================================================================
# Backend Types
# =============================================================================

class BackendType(str, Enum):
    """Backend type for teammate execution."""
    TMUX = "tmux"
    ITERM2 = "iterm2"
    IN_PROCESS = "in-process"


def is_pane_backend(backend_type: Union[BackendType, str]) -> bool:
    """Check if backend is a pane-based backend (tmux/iTerm2)."""
    return backend_type in (BackendType.TMUX, BackendType.ITERM2, "tmux", "iterm2")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TeamAllowedPath:
    """A path that all teammates can edit without asking permission."""
    path: str  # Directory path (absolute)
    tool_name: str  # Tool this applies to (e.g., "Edit", "Write")
    added_by: str  # Agent name who added this rule
    added_at: int  # Unix timestamp (milliseconds)


@dataclass
class TeamMember:
    """A member of a team."""
    agent_id: str  # Full agent ID: "name@team"
    name: str  # Display name
    agent_type: Optional[str] = None
    model: Optional[str] = None
    prompt: Optional[str] = None
    color: Optional[str] = None  # UI color (e.g., 'red', 'blue')
    plan_mode_required: bool = False
    joined_at: int = 0  # Unix timestamp (milliseconds)
    tmux_pane_id: str = ""  # Pane ID for pane-based backends
    cwd: str = ""  # Working directory
    worktree_path: Optional[str] = None  # Git worktree path if isolated
    session_id: Optional[str] = None  # Session UUID
    subscriptions: List[str] = field(default_factory=list)
    backend_type: BackendType = BackendType.IN_PROCESS
    is_active: bool = True  # False when idle
    mode: str = "default"  # Permission mode

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "agentId": self.agent_id,
            "name": self.name,
            "agentType": self.agent_type,
            "model": self.model,
            "prompt": self.prompt,
            "color": self.color,
            "planModeRequired": self.plan_mode_required,
            "joinedAt": self.joined_at,
            "tmuxPaneId": self.tmux_pane_id,
            "cwd": self.cwd,
            "worktreePath": self.worktree_path,
            "sessionId": self.session_id,
            "subscriptions": self.subscriptions,
            "backendType": self.backend_type.value,
            "isActive": self.is_active,
            "mode": self.mode,
        }


@dataclass
class TeamFile:
    """Team configuration file."""
    name: str
    created_at: int  # Unix timestamp (milliseconds)
    lead_agent_id: str  # Leader's agent ID
    lead_session_id: Optional[str] = None  # Leader's session UUID
    description: Optional[str] = None
    hidden_pane_ids: List[str] = field(default_factory=list)
    team_allowed_paths: List[TeamAllowedPath] = field(default_factory=list)
    members: List[TeamMember] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "name": self.name,
            "createdAt": self.created_at,
            "leadAgentId": self.lead_agent_id,
            "leadSessionId": self.lead_session_id,
            "description": self.description,
            "hiddenPaneIds": self.hidden_pane_ids,
            "teamAllowedPaths": [
                {
                    "path": p.path,
                    "toolName": p.tool_name,
                    "addedBy": p.added_by,
                    "addedAt": p.added_at,
                }
                for p in self.team_allowed_paths
            ],
            "members": [m.to_dict() for m in self.members],
        }


# =============================================================================
# Path Utilities
# =============================================================================

def sanitize_team_name(name: str) -> str:
    """Sanitize a team name for filesystem use.

    Replaces all non-alphanumeric characters with hyphens and lowercases.
    """
    return "".join(c if c.isalnum() else "-" for c in name).lower()


def sanitize_agent_name(name: str) -> str:
    """Sanitize an agent name for deterministic agent IDs.

    Replaces @ with - to prevent ambiguity in agentName@teamName format.
    """
    return name.replace("@", "-")


def get_claude_config_home() -> Path:
    """Get the Claude config home directory."""
    config_home = os.environ.get("CLAUDE_CONFIG_HOME")
    if config_home:
        return Path(config_home)
    return Path.home() / ".claude"


def get_teams_dir() -> Path:
    """Get the teams directory: ~/.claude/teams"""
    return get_claude_config_home() / "teams"


def get_team_dir(team_name: str) -> Path:
    """Get the directory for a specific team."""
    return get_teams_dir() / sanitize_team_name(team_name)


def get_team_file_path(team_name: str) -> Path:
    """Get the path to a team's config.json file."""
    return get_team_dir(team_name) / "config.json"


def ensure_team_dir(team_name: str) -> Path:
    """Ensure the team directory exists."""
    dir_path = get_team_dir(team_name)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


# =============================================================================
# CRUD Operations (Sync)
# =============================================================================

def read_team_file(team_name: str) -> Optional[TeamFile]:
    """Read a team file synchronously.

    Args:
        team_name: Team name

    Returns:
        TeamFile or None if not found
    """
    path = get_team_file_path(team_name)
    try:
        content = path.read_text()
        data = json.loads(content)
        return dict_to_team_file(data)
    except FileNotFoundError:
        logger.debug(f"Team file not found for {team_name}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse team file for {team_name}: {e}")
        return None


def write_team_file(team_name: str, team_file: TeamFile) -> None:
    """Write a team file synchronously.

    Args:
        team_name: Team name
        team_file: TeamFile to write
    """
    ensure_team_dir(team_name)
    path = get_team_file_path(team_name)
    path.write_text(json.dumps(team_file.to_dict(), indent=2))


def add_member_to_team(team_name: str, member: TeamMember) -> bool:
    """Add a member to a team.

    Args:
        team_name: Team name
        member: TeamMember to add

    Returns:
        True if added, False if team not found
    """
    team_file = read_team_file(team_name)
    if not team_file:
        logger.warning(f"Cannot add member: team {team_name} not found")
        return False

    # Check if member already exists
    existing = next((m for m in team_file.members if m.agent_id == member.agent_id), None)
    if existing:
        # Update existing member
        team_file.members = [
            m if m.agent_id != member.agent_id else member
            for m in team_file.members
        ]
    else:
        team_file.members.append(member)

    write_team_file(team_name, team_file)
    logger.debug(f"Added member {member.agent_id} to team {team_name}")
    return True


def remove_member_by_agent_id(team_name: str, agent_id: str) -> bool:
    """Remove a member from a team by agent ID.

    Args:
        team_name: Team name
        agent_id: Agent ID to remove (e.g., "researcher@team")

    Returns:
        True if removed, False if not found
    """
    team_file = read_team_file(team_name)
    if not team_file:
        return False

    original_len = len(team_file.members)
    team_file.members = [m for m in team_file.members if m.agent_id != agent_id]

    if len(team_file.members) == original_len:
        logger.debug(f"Member {agent_id} not found in team {team_name}")
        return False

    write_team_file(team_name, team_file)
    logger.debug(f"Removed member {agent_id} from team {team_name}")
    return True


def remove_member_from_team(team_name: str, tmux_pane_id: str) -> bool:
    """Remove a member from a team by tmux pane ID.

    Args:
        team_name: Team name
        tmux_pane_id: Pane ID to remove

    Returns:
        True if removed, False if not found
    """
    team_file = read_team_file(team_name)
    if not team_file:
        return False

    original_len = len(team_file.members)
    team_file.members = [m for m in team_file.members if m.tmux_pane_id != tmux_pane_id]

    # Also remove from hidden_pane_ids
    if tmux_pane_id in team_file.hidden_pane_ids:
        team_file.hidden_pane_ids.remove(tmux_pane_id)

    if len(team_file.members) == original_len:
        return False

    write_team_file(team_name, team_file)
    logger.debug(f"Removed member with pane {tmux_pane_id} from team {team_name}")
    return True


def set_member_mode(team_name: str, member_name: str, mode: str) -> bool:
    """Set a team member's permission mode.

    Args:
        team_name: Team name
        member_name: Member name
        mode: Permission mode value

    Returns:
        True if set, False if not found
    """
    team_file = read_team_file(team_name)
    if not team_file:
        return False

    member = next((m for m in team_file.members if m.name == member_name), None)
    if not member:
        logger.debug(f"Cannot set mode: member {member_name} not found in team {team_name}")
        return False

    if member.mode == mode:
        return True

    member.mode = mode
    write_team_file(team_name, team_file)
    logger.debug(f"Set member {member_name} in team {team_name} to mode: {mode}")
    return True


def set_member_active(team_name: str, member_name: str, is_active: bool) -> bool:
    """Set a team member's active status.

    Args:
        team_name: Team name
        member_name: Member name
        is_active: True if active, False if idle

    Returns:
        True if set, False if not found
    """
    team_file = read_team_file(team_name)
    if not team_file:
        return False

    member = next((m for m in team_file.members if m.name == member_name), None)
    if not member:
        logger.debug(f"Cannot set active: member {member_name} not found in team {team_name}")
        return False

    if member.is_active == is_active:
        return True

    member.is_active = is_active
    write_team_file(team_name, team_file)
    logger.debug(f"Set member {member_name} to {'active' if is_active else 'idle'}")
    return True


# =============================================================================
# CRUD Operations (Async)
# =============================================================================

async def read_team_file_async(team_name: str) -> Optional[TeamFile]:
    """Read a team file asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, read_team_file, team_name)


async def write_team_file_async(team_name: str, team_file: TeamFile) -> None:
    """Write a team file asynchronously."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, write_team_file, team_name, team_file)


async def set_member_active_async(team_name: str, member_name: str, is_active: bool) -> bool:
    """Set member active status asynchronously."""
    team_file = await read_team_file_async(team_name)
    if not team_file:
        logger.debug(f"Cannot set active: team {team_name} not found")
        return False

    member = next((m for m in team_file.members if m.name == member_name), None)
    if not member:
        logger.debug(f"Cannot set active: member {member_name} not found in team {team_name}")
        return False

    if member.is_active == is_active:
        return True

    member.is_active = is_active
    await write_team_file_async(team_name, team_file)
    return True


# =============================================================================
# Deserialization
# =============================================================================

def dict_to_team_member(data: Dict[str, Any]) -> TeamMember:
    """Convert a dict to a TeamMember."""
    backend_type_str = data.get("backendType", "in-process")
    try:
        backend_type = BackendType(backend_type_str)
    except ValueError:
        backend_type = BackendType.IN_PROCESS

    return TeamMember(
        agent_id=data.get("agentId", ""),
        name=data.get("name", ""),
        agent_type=data.get("agentType"),
        model=data.get("model"),
        prompt=data.get("prompt"),
        color=data.get("color"),
        plan_mode_required=data.get("planModeRequired", False),
        joined_at=data.get("joinedAt", 0),
        tmux_pane_id=data.get("tmuxPaneId", ""),
        cwd=data.get("cwd", ""),
        worktree_path=data.get("worktreePath"),
        session_id=data.get("sessionId"),
        subscriptions=data.get("subscriptions", []),
        backend_type=backend_type,
        is_active=data.get("isActive", True),
        mode=data.get("mode", "default"),
    )


def dict_to_team_file(data: Dict[str, Any]) -> TeamFile:
    """Convert a dict to a TeamFile."""
    allowed_paths = [
        TeamAllowedPath(
            path=p.get("path", ""),
            tool_name=p.get("toolName", ""),
            added_by=p.get("addedBy", ""),
            added_at=p.get("addedAt", 0),
        )
        for p in data.get("teamAllowedPaths", [])
    ]

    members = [dict_to_team_member(m) for m in data.get("members", [])]

    return TeamFile(
        name=data.get("name", ""),
        created_at=data.get("createdAt", 0),
        lead_agent_id=data.get("leadAgentId", ""),
        lead_session_id=data.get("leadSessionId"),
        description=data.get("description"),
        hidden_pane_ids=data.get("hiddenPaneIds", []),
        team_allowed_paths=allowed_paths,
        members=members,
    )


# =============================================================================
# Format Agent ID
# =============================================================================

def format_agent_id(agent_name: str, team_name: str) -> str:
    """Format an agent ID as "name@team".

    Args:
        agent_name: Agent display name
        team_name: Team name

    Returns:
        Full agent ID
    """
    safe_name = sanitize_agent_name(agent_name)
    safe_team = sanitize_team_name(team_name)
    return f"{safe_name}@{safe_team}"


__all__ = [
    "TEAM_LEAD_NAME",
    "BackendType",
    "is_pane_backend",
    "TeamAllowedPath",
    "TeamMember",
    "TeamFile",
    "sanitize_team_name",
    "sanitize_agent_name",
    "get_teams_dir",
    "get_team_dir",
    "get_team_file_path",
    "ensure_team_dir",
    "read_team_file",
    "write_team_file",
    "add_member_to_team",
    "remove_member_by_agent_id",
    "remove_member_from_team",
    "set_member_mode",
    "set_member_active",
    "read_team_file_async",
    "write_team_file_async",
    "set_member_active_async",
    "dict_to_team_member",
    "dict_to_team_file",
    "format_agent_id",
]