"""Team cleanup helpers for session management.

Handles cleanup of team directories, worktrees, and session tracking.

Ported from: src/utils/swarm/teamHelpers.ts
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Set, Optional, List
import logging

from .team_file import (
    read_team_file,
    sanitize_team_name,
    BackendType,
    is_pane_backend,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Session Cleanup Tracking
# =============================================================================

# Teams created this session that should be cleaned up on exit
_session_created_teams: Set[str] = set()


def register_team_for_session_cleanup(team_name: str) -> None:
    """Mark a team as created this session for cleanup on exit.

    Call this right after creating a team file.
    TeamDelete should call unregister_team_for_session_cleanup
    to prevent double-cleanup.

    Args:
        team_name: Team name to track
    """
    _session_created_teams.add(team_name)
    logger.debug(f"Registered team {team_name} for session cleanup")


def unregister_team_for_session_cleanup(team_name: str) -> None:
    """Remove a team from session cleanup tracking.

    Called after explicit TeamDelete to prevent double-cleanup.

    Args:
        team_name: Team name to remove
    """
    _session_created_teams.discard(team_name)
    logger.debug(f"Unregistered team {team_name} from session cleanup")


def get_session_created_teams() -> Set[str]:
    """Get all teams created this session."""
    return _session_created_teams.copy()


# =============================================================================
# Worktree Cleanup
# =============================================================================

async def destroy_worktree(worktree_path: str) -> None:
    """Destroy a git worktree at the given path.

    First attempts git worktree remove, then falls back to rm -rf.

    Args:
        worktree_path: Path to worktree directory
    """
    if not Path(worktree_path).exists():
        return

    # Read .git file to find main repo
    git_file_path = Path(worktree_path) / ".git"
    main_repo_path: Optional[str] = None

    try:
        git_content = git_file_path.read_text().strip()
        # Format: gitdir: /path/to/repo/.git/worktrees/worktree-name
        if git_content.startswith("gitdir:"):
            worktree_git_dir = git_content[7:].strip()
            # Go up 2 levels to get to .git
            main_git_dir = str(Path(worktree_git_dir).parent.parent)
            main_repo_path = str(Path(main_git_dir).parent)
    except Exception:
        pass

    # Try git worktree remove
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

            # Check if "not a working tree" error
            if "not a working tree" in result.stderr:
                logger.debug(f"Worktree already removed: {worktree_path}")
                return

            logger.debug(f"git worktree remove failed, falling back to rm: {result.stderr}")
        except Exception as e:
            logger.debug(f"git worktree remove error: {e}")

    # Fallback: manual removal
    try:
        shutil.rmtree(worktree_path)
        logger.debug(f"Removed worktree directory manually: {worktree_path}")
    except Exception as e:
        logger.warning(f"Failed to remove worktree {worktree_path}: {e}")


# =============================================================================
# Team Directory Cleanup
# =============================================================================

def cleanup_team_directories_sync(team_name: str) -> None:
    """Clean up team and task directories for a terminated team.

    Also cleans up git worktrees created for teammates.

    Args:
        team_name: Team name
    """
    sanitized = sanitize_team_name(team_name)

    # Read team file BEFORE deleting to get worktree paths
    team_file = read_team_file(team_name)
    worktree_paths: List[str] = []

    if team_file:
        for member in team_file.members:
            if member.worktree_path:
                worktree_paths.append(member.worktree_path)

    # Clean up worktrees first
    for worktree_path in worktree_paths:
        try:
            # Sync version - just use shutil
            if Path(worktree_path).exists():
                shutil.rmtree(worktree_path)
                logger.debug(f"Removed worktree: {worktree_path}")
        except Exception as e:
            logger.warning(f"Failed to remove worktree {worktree_path}: {e}")

    # Clean up team directory (~/.claude/teams/{team-name}/)
    from .team_file import get_team_dir

    team_dir = get_team_dir(team_name)
    try:
        shutil.rmtree(team_dir)
        logger.debug(f"Cleaned up team directory: {team_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up team directory {team_dir}: {e}")

    # Clean up tasks directory (~/.claude/tasks/{team-name}/)
    from claude_code_py.utils.task.file_storage import get_tasks_dir, notify_tasks_updated

    tasks_dir = get_tasks_dir(sanitized)
    try:
        shutil.rmtree(tasks_dir)
        logger.debug(f"Cleaned up tasks directory: {tasks_dir}")
        notify_tasks_updated()
    except Exception as e:
        logger.warning(f"Failed to clean up tasks directory {tasks_dir}: {e}")


async def cleanup_team_directories(team_name: str) -> None:
    """Clean up team directories asynchronously.

    Args:
        team_name: Team name
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, cleanup_team_directories_sync, team_name)


# =============================================================================
# Pane Cleanup (for orphaned teammates)
# =============================================================================

async def kill_orphaned_teammate_panes(team_name: str) -> None:
    """Kill orphaned pane-based teammates for a team.

    Called from cleanup_session_teams on ungraceful exit.

    Args:
        team_name: Team name
    """
    team_file = read_team_file(team_name)
    if not team_file:
        return

    # Filter for pane-backed members
    pane_members = [
        m for m in team_file.members
        if m.name != "team-lead"
        and m.tmux_pane_id
        and m.backend_type
        and is_pane_backend(m.backend_type)
    ]

    if not pane_members:
        return

    for member in pane_members:
        if not member.tmux_pane_id:
            continue

        try:
            # Kill tmux pane
            subprocess.run(
                ["tmux", "kill-pane", "-t", member.tmux_pane_id],
                capture_output=True,
            )
            logger.debug(f"Killed pane {member.tmux_pane_id} for {member.name}")
        except Exception as e:
            logger.debug(f"Failed to kill pane {member.tmux_pane_id}: {e}")


# =============================================================================
# Session Cleanup
# =============================================================================

async def cleanup_session_teams() -> None:
    """Clean up all teams created this session.

    Called on SIGINT/SIGTERM for graceful shutdown.
    """
    teams = list(_session_created_teams)
    if not teams:
        return

    logger.debug(f"cleanup_session_teams: removing {len(teams)} orphan team(s)")

    # Kill panes first
    for team_name in teams:
        try:
            await kill_orphaned_teammate_panes(team_name)
        except Exception as e:
            logger.debug(f"Error killing panes for {team_name}: {e}")

    # Then clean directories
    for team_name in teams:
        try:
            await cleanup_team_directories(team_name)
        except Exception as e:
            logger.debug(f"Error cleaning up {team_name}: {e}")

    # Clear tracking
    _session_created_teams.clear()


__all__ = [
    "register_team_for_session_cleanup",
    "unregister_team_for_session_cleanup",
    "get_session_created_teams",
    "destroy_worktree",
    "cleanup_team_directories",
    "cleanup_team_directories_sync",
    "kill_orphaned_teammate_panes",
    "cleanup_session_teams",
]