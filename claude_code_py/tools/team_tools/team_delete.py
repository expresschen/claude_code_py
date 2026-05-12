"""TeamDelete Tool - Disband a swarm team and clean up.

Cleans up team directories, task directories, worktrees, and clears team context.

Environment variable: CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult
from claude_code_py.utils.team.team_file import (
    read_team_file,
    get_team_dir,
    sanitize_team_name,
)
from claude_code_py.utils.swarm.constants import TEAM_LEAD_NAME
from claude_code_py.utils.task.file_storage import get_tasks_dir, clear_leader_team_name
from claude_code_py.tools.team_tools.team_create import (
    is_agent_teams_enabled,
    clear_teammate_colors,
    stop_active_inbox_poller,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Input/Output Types
# =============================================================================


class TeamDeleteInput(BaseModel):
    """Input for TeamDelete tool."""

    # No input required - uses current team context


class TeamDeleteOutput(BaseModel):
    """Output for TeamDelete tool."""

    success: bool
    message: str
    team_name: Optional[str] = None


# =============================================================================
# Cleanup Functions
# =============================================================================


async def destroy_worktree(worktree_path: str) -> None:
    """Destroy a git worktree at the given path.

    First attempts to use `git worktree remove`, then falls back to rm -rf.
    Safe to call on non-existent paths.

    Ported from TypeScript teamHelpers.ts destroyWorktree().
    """
    worktree_path_obj = Path(worktree_path)
    if not worktree_path_obj.exists():
        return

    # Read the .git file in the worktree to find the main repo
    git_file_path = worktree_path_obj / ".git"
    main_repo_path: Optional[str] = None

    try:
        git_file_content = git_file_path.read_text().strip()
        # The .git file contains: gitdir: /path/to/repo/.git/worktrees/worktree-name
        import re
        match = re.match(r"^gitdir:\s*(.+)$", git_file_content)
        if match and match.group(1):
            # Go up 2 levels from .git/worktrees/name to get to .git
            worktree_git_dir = match.group(1)
            main_git_dir = str(Path(worktree_git_dir).parent.parent)
            # Get the repo root (parent of .git)
            main_repo_path = str(Path(main_git_dir).parent)
    except Exception:
        # Ignore errors reading .git file
        pass

    # Try to remove using git worktree remove command
    if main_repo_path:
        try:
            result = subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_path],
                cwd=main_repo_path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.debug(f"Removed worktree via git: {worktree_path}")
                return

            # Check if the error is "not a working tree" (already removed)
            if "not a working tree" in result.stderr:
                logger.debug(f"Worktree already removed: {worktree_path}")
                return

            logger.debug(f"git worktree remove failed, falling back to rm: {result.stderr}")
        except Exception as e:
            logger.debug(f"git worktree remove failed: {e}")

    # Fallback: manually remove the directory
    try:
        shutil.rmtree(worktree_path)
        logger.debug(f"Removed worktree directory manually: {worktree_path}")
    except Exception as e:
        logger.warning(f"Failed to remove worktree {worktree_path}: {e}")


def cleanup_team_directories(team_name: str) -> None:
    """Clean up team and task directories.

    Also cleans up git worktrees created for teammates.
    Ported from TypeScript teamHelpers.ts cleanupTeamDirectories().
    """
    sanitized_name = sanitize_team_name(team_name)

    # Read team file to get worktree paths BEFORE deleting the team directory
    team_file = read_team_file(team_name)
    worktree_paths: list[str] = []
    if team_file:
        for member in team_file.members:
            if member.worktree_path:
                worktree_paths.append(member.worktree_path)

    # Clean up worktrees first (async in sync context)
    if worktree_paths:
        loop = asyncio.new_event_loop()
        try:
            for worktree_path in worktree_paths:
                loop.run_until_complete(destroy_worktree(worktree_path))
        finally:
            loop.close()

    # Clean team directory
    team_dir = get_team_dir(team_name)
    if team_dir.exists():
        try:
            shutil.rmtree(team_dir)
            logger.debug(f"Cleaned up team directory: {team_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up team directory {team_dir}: {e}")

    # Clean task directory
    tasks_dir = get_tasks_dir(sanitized_name)
    if tasks_dir.exists():
        try:
            shutil.rmtree(tasks_dir)
            logger.debug(f"Cleaned up tasks directory: {tasks_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up tasks directory {tasks_dir}: {e}")


# =============================================================================
# TeamDelete Tool
# =============================================================================


class TeamDeleteTool(Tool[TeamDeleteInput, TeamDeleteOutput, Dict[str, Any]]):
    """Tool to disband a team and clean up."""

    name = "TeamDelete"
    description = "Clean up team and task directories when the swarm is complete"
    input_schema = TeamDeleteInput

    def is_read_only(self, args: TeamDeleteInput) -> bool:
        return False

    def is_concurrency_safe(self, args: TeamDeleteInput) -> bool:
        return False

    def is_enabled(self) -> bool:
        """Only enabled when experimental flag is set."""
        return is_agent_teams_enabled()

    async def prompt(self, options: Dict[str, Any]) -> str:
        """Get tool prompt."""
        return """# TeamDelete

Remove team and task directories when the swarm work is complete.

This operation:
- Removes the team directory (`~/.claude/teams/{team-name}/`)
- Removes the task directory (`~/.claude/tasks/{team-name}/`)
- Clears team context from the current session

**IMPORTANT**: TeamDelete will fail if the team still has active members. Gracefully terminate teammates first, then call TeamDelete after all teammates have shut down.

Use this when all teammates have finished their work and you want to clean up the team resources. The team name is automatically determined from the current session's team context."""

    async def call(
        self,
        args: TeamDeleteInput,
        context: "ToolUseContext",
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Any = None,
    ) -> TeamDeleteOutput:
        """Delete the current team."""
        set_app_state = context.set_app_state
        get_app_state = context.get_app_state

        app_state = get_app_state()
        team_context = app_state.team_context
        team_name = team_context.get("teamName") if team_context else None

        if team_name:
            # Read team config to check for active members
            team_file = read_team_file(team_name)

            if team_file:
                # Filter out the team lead - only count non-lead members
                non_lead_members = [
                    m for m in team_file.members
                    if m.name != TEAM_LEAD_NAME
                ]

                # Check for active members
                active_members = [
                    m for m in non_lead_members
                    if m.is_active
                ]

                if len(active_members) > 0:
                    member_names = ", ".join(m.name for m in active_members)
                    return ToolResult(data=TeamDeleteOutput(
                        success=False,
                        message=f"Cannot cleanup team with {len(active_members)} active member(s): {member_names}. Use requestShutdown to gracefully terminate teammates first.",
                        team_name=team_name,
                    ))

            # Clean up directories (including worktrees)
            cleanup_team_directories(team_name)

            # Stop the inbox poller
            stop_active_inbox_poller()

            # Clear leader team name so getTaskListId() falls back to session ID
            clear_leader_team_name()

            # Clear color assignments so new teams start fresh
            clear_teammate_colors()

        # Clear team context from AppState
        set_app_state(lambda prev: replace(
            prev,
            team_context=None,
            inbox={"messages": []},
        ))

        if team_name:
            message = f"Cleaned up directories and worktrees for team '{team_name}'"
        else:
            message = "No team name found, nothing to clean up"

        return ToolResult(data=TeamDeleteOutput(
            success=True,
            message=message,
            team_name=team_name,
        ))


__all__ = [
    "TeamDeleteTool",
    "TeamDeleteInput",
    "TeamDeleteOutput",
    "cleanup_team_directories",
    "is_agent_teams_enabled",
]