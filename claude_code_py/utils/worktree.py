"""Worktree utilities for git worktree isolation.

This implements git worktree creation and management for session isolation.
Reference: TypeScript worktree.ts
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from claude_code_py.storage.session import (
    WorktreeSession,
    save_worktree_state,
    clear_worktree_state,
)


# =============================================================================
# Constants
# =============================================================================

VALID_WORKTREE_SLUG_SEGMENT = re.compile(r"^[-a-zA-Z0-9._]+$")
MAX_WORKTREE_SLUG_LENGTH = 64

# Ephemeral worktree patterns for cleanup
EPHEMERAL_WORKTREE_PATTERNS = [
    re.compile(r"^agent-a[0-9a-f]{7}$"),
    re.compile(r"^wf_[0-9a-f]{8}-[0-9a-f]{3}-\d+$"),
    re.compile(r"^wf-\d+$"),
    re.compile(r"^bridge-[A-Za-z0-9_]+(-[A-Za-z0-9_]+)*$"),
]


# =============================================================================
# Types
# =============================================================================


@dataclass
class WorktreeCreateResult:
    """Result of worktree creation."""

    worktree_path: str
    worktree_branch: str
    head_commit: str
    existed: bool
    base_branch: Optional[str] = None


# =============================================================================
# Validation
# =============================================================================


def validate_worktree_slug(slug: str) -> None:
    """Validate a worktree slug to prevent path traversal and directory escape.

    The slug is joined into `.claude/worktrees/<slug>` via path joining, which
    normalizes `..` segments. Forward slashes are allowed for nesting.

    Args:
        slug: Worktree slug to validate

    Raises:
        ValueError: If slug is invalid
    """
    if len(slug) > MAX_WORKTREE_SLUG_LENGTH:
        raise ValueError(
            f"Invalid worktree name: must be {MAX_WORKTREE_SLUG_LENGTH} characters "
            f"or fewer (got {len(slug)})"
        )

    for segment in slug.split("/"):
        if segment == "." or segment == "..":
            raise ValueError(
                f"Invalid worktree name '{slug}': must not contain '.' or '..' path segments"
            )
        if not VALID_WORKTREE_SLUG_SEGMENT.match(segment):
            raise ValueError(
                f"Invalid worktree name '{slug}': each '/'-separated segment must "
                f"be non-empty and contain only letters, digits, dots, underscores, "
                f"and dashes"
            )


# =============================================================================
# Git Helpers
# =============================================================================


def _run_git(
    args: list[str],
    cwd: Optional[str] = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run a git command.

    Args:
        args: Git arguments
        cwd: Working directory
        check: Whether to raise on error
        capture_output: Whether to capture stdout/stderr

    Returns:
        CompletedProcess result
    """
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
    }

    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=True,
        env=env,
    )


def find_git_root(path: Optional[str] = None) -> Optional[str]:
    """Find git root directory.

    Args:
        path: Starting path (defaults to cwd)

    Returns:
        Git root path or None
    """
    path = path or os.getcwd()
    current = Path(path).resolve()

    while current != current.parent:
        if (current / ".git").exists():
            return str(current)
        current = current.parent

    return None


def find_canonical_git_root(path: Optional[str] = None) -> Optional[str]:
    """Find canonical git root (resolves through worktrees).

    Args:
        path: Starting path (defaults to cwd)

    Returns:
        Canonical git root or None
    """
    path = path or os.getcwd()
    git_root = find_git_root(path)

    if not git_root:
        return None

    # Check if we're in a worktree
    git_dir = Path(git_root) / ".git"
    if git_dir.is_file():
        # This is a worktree, read the pointer
        try:
            content = git_dir.read_text()
            # Worktree .git file contains: gitdir: /path/to/main/.git/worktrees/name
            if content.startswith("gitdir:"):
                main_git_dir = Path(content.split(":")[1].strip()).parent.parent
                return str(main_git_dir)
        except Exception:
            pass

    return git_root


def get_current_branch(cwd: Optional[str] = None) -> Optional[str]:
    """Get current git branch.

    Args:
        cwd: Working directory

    Returns:
        Branch name or None
    """
    try:
        result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_default_branch(cwd: Optional[str] = None) -> str:
    """Get default branch (main or master).

    Args:
        cwd: Working directory

    Returns:
        Default branch name
    """
    # Try origin/main
    try:
        result = _run_git(
            ["rev-parse", "--abbrev-ref", "origin/HEAD"],
            cwd=cwd,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().replace("origin/", "")
    except Exception:
        pass

    # Fall back to main/master
    try:
        result = _run_git(["rev-parse", "--verify", "main"], cwd=cwd, check=False)
        if result.returncode == 0:
            return "main"
    except Exception:
        pass

    return "master"


def get_head_commit(cwd: Optional[str] = None) -> Optional[str]:
    """Get HEAD commit SHA.

    Args:
        cwd: Working directory

    Returns:
        Commit SHA or None
    """
    try:
        result = _run_git(["rev-parse", "HEAD"], cwd=cwd, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# =============================================================================
# Worktree Operations
# =============================================================================


def worktrees_dir(repo_root: str) -> str:
    """Get the worktrees directory path.

    Args:
        repo_root: Repository root path

    Returns:
        Path to .claude/worktrees directory
    """
    return str(Path(repo_root) / ".claude" / "worktrees")


def flatten_slug(slug: str) -> str:
    """Flatten nested slugs for branch names.

    Args:
        slug: Original slug

    Returns:
        Flattened slug
    """
    return slug.replace("/", "+")


def worktree_branch_name(slug: str) -> str:
    """Generate a worktree branch name.

    Args:
        slug: Worktree slug

    Returns:
        Branch name (e.g., "worktree-my-feature")
    """
    return f"worktree-{flatten_slug(slug)}"


def worktree_path_for(repo_root: str, slug: str) -> str:
    """Get the worktree path for a slug.

    Args:
        repo_root: Repository root
        slug: Worktree slug

    Returns:
        Path to worktree directory
    """
    return str(Path(worktrees_dir(repo_root)) / flatten_slug(slug))


def _check_worktree_exists(worktree_path: str) -> Optional[str]:
    """Check if a worktree already exists and get its HEAD.

    Args:
        worktree_path: Worktree directory path

    Returns:
        HEAD commit SHA if exists, None otherwise
    """
    path = Path(worktree_path)
    if not path.exists():
        return None

    # Check if it's a valid git worktree
    git_file = path / ".git"
    if git_file.is_file():
        try:
            return get_head_commit(worktree_path)
        except Exception:
            pass

    return None


async def get_or_create_worktree(
    repo_root: str,
    slug: str,
    options: Optional[dict[str, Any]] = None,
) -> WorktreeCreateResult:
    """Create a new git worktree or resume existing one.

    Args:
        repo_root: Repository root path
        slug: Worktree slug
        options: Optional options (pr_number, etc.)

    Returns:
        WorktreeCreateResult with path and branch info
    """
    validate_worktree_slug(slug)

    worktree_path = worktree_path_for(repo_root, slug)
    worktree_branch = worktree_branch_name(slug)

    # Fast resume path: check if worktree exists
    existing_head = _check_worktree_exists(worktree_path)
    if existing_head:
        return WorktreeCreateResult(
            worktree_path=worktree_path,
            worktree_branch=worktree_branch,
            head_commit=existing_head,
            existed=True,
        )

    # Create new worktree
    worktrees_path = Path(worktrees_dir(repo_root))
    worktrees_path.mkdir(parents=True, exist_ok=True)

    # Get base branch
    base_branch = get_default_branch(repo_root)

    # Fetch origin
    try:
        _run_git(["fetch", "origin", base_branch], cwd=repo_root, check=False)
    except Exception:
        pass

    # Get base commit
    base_sha = get_head_commit(repo_root) or ""

    # Create worktree
    result = _run_git(
        ["worktree", "add", "-B", worktree_branch, worktree_path, base_branch],
        cwd=repo_root,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to create worktree: {result.stderr}")

    return WorktreeCreateResult(
        worktree_path=worktree_path,
        worktree_branch=worktree_branch,
        head_commit=base_sha,
        base_branch=base_branch,
        existed=False,
    )


async def perform_post_creation_setup(
    repo_root: str,
    worktree_path: str,
) -> None:
    """Perform post-creation setup for a worktree.

    Args:
        repo_root: Main repository root
        worktree_path: New worktree path
    """
    # Copy settings.local.json if exists
    source_settings = Path(repo_root) / ".claude" / "settings.local.json"
    dest_settings = Path(worktree_path) / ".claude" / "settings.local.json"

    if source_settings.exists():
        dest_settings.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_settings, dest_settings)


async def create_agent_worktree(slug: str) -> dict[str, Any]:
    """Create a lightweight worktree for a subagent.

    Does NOT touch global session state (currentWorktreeSession, cwd).

    Args:
        slug: Worktree slug

    Returns:
        Dict with worktree_path, worktree_branch, head_commit, etc.
    """
    validate_worktree_slug(slug)

    git_root = find_canonical_git_root()
    if not git_root:
        raise RuntimeError(
            "Cannot create agent worktree: not in a git repository. "
            "Configure WorktreeCreate hooks in settings.json for VCS-agnostic isolation."
        )

    result = await get_or_create_worktree(git_root, slug)

    if not result.existed:
        await perform_post_creation_setup(git_root, result.worktree_path)

    return {
        "worktree_path": result.worktree_path,
        "worktree_branch": result.worktree_branch,
        "head_commit": result.head_commit,
        "git_root": git_root,
    }


async def remove_agent_worktree(
    worktree_path: str,
    worktree_branch: Optional[str] = None,
    git_root: Optional[str] = None,
) -> bool:
    """Remove a worktree created by create_agent_worktree.

    Args:
        worktree_path: Worktree path
        worktree_branch: Branch name to delete
        git_root: Main repo git root

    Returns:
        True if removed successfully
    """
    if not git_root:
        git_root = find_canonical_git_root()

    if not git_root:
        return False

    # Remove worktree
    result = _run_git(
        ["worktree", "remove", "--force", worktree_path],
        cwd=git_root,
        check=False,
    )

    if result.returncode != 0:
        return False

    # Delete branch if provided
    if worktree_branch:
        await asyncio.sleep(0.1)  # Wait for git locks to release
        _run_git(["branch", "-D", worktree_branch], cwd=git_root, check=False)

    return True


async def has_worktree_changes(
    worktree_path: str,
    head_commit: str,
) -> bool:
    """Check if a worktree has uncommitted changes or new commits.

    Args:
        worktree_path: Worktree path
        head_commit: Original HEAD commit

    Returns:
        True if there are changes
    """
    # Check working tree status
    result = _run_git(["status", "--porcelain"], cwd=worktree_path, check=False)
    if result.returncode != 0:
        return True
    if result.stdout.strip():
        return True

    # Check for new commits
    result = _run_git(
        ["rev-list", "--count", f"{head_commit}..HEAD"],
        cwd=worktree_path,
        check=False,
    )
    if result.returncode != 0:
        return True

    try:
        count = int(result.stdout.strip())
        if count > 0:
            return True
    except ValueError:
        return True

    return False


# =============================================================================
# Session Worktree Management
# =============================================================================

# Global current worktree session (like TypeScript)
_current_worktree_session: Optional[WorktreeSession] = None


def get_current_worktree_session() -> Optional[WorktreeSession]:
    """Get the current worktree session.

    Returns:
        Current WorktreeSession or None
    """
    return _current_worktree_session


def restore_worktree_session(session: Optional[WorktreeSession]) -> None:
    """Restore worktree session on --resume.

    Args:
        session: WorktreeSession to restore
    """
    global _current_worktree_session
    _current_worktree_session = session


async def create_worktree_for_session(
    session_id: str,
    slug: str,
) -> WorktreeSession:
    """Create a worktree for a session and switch into it.

    Args:
        session_id: Session identifier
        slug: Worktree slug

    Returns:
        WorktreeSession
    """
    global _current_worktree_session

    validate_worktree_slug(slug)

    original_cwd = os.getcwd()

    git_root = find_git_root()
    if not git_root:
        raise RuntimeError(
            "Cannot create a worktree: not in a git repository. "
            "Configure WorktreeCreate hooks in settings.json for VCS-agnostic isolation."
        )

    original_branch = get_current_branch()
    result = await get_or_create_worktree(git_root, slug)

    if not result.existed:
        await perform_post_creation_setup(git_root, result.worktree_path)

    _current_worktree_session = WorktreeSession(
        original_cwd=original_cwd,
        worktree_path=result.worktree_path,
        worktree_name=slug,
        session_id=session_id,
        worktree_branch=result.worktree_branch,
        original_branch=original_branch,
        original_head_commit=result.head_commit,
        hook_based=False,
    )

    # Save to project config
    save_worktree_state(session_id, _current_worktree_session)

    return _current_worktree_session


async def keep_worktree() -> None:
    """Keep the worktree but exit the session."""
    global _current_worktree_session

    if not _current_worktree_session:
        return

    # Change back to original directory
    os.chdir(_current_worktree_session.original_cwd)

    # Clear session but keep worktree
    clear_worktree_state(_current_worktree_session.session_id)
    _current_worktree_session = None


async def cleanup_worktree() -> None:
    """Clean up the worktree completely."""
    global _current_worktree_session

    if not _current_worktree_session:
        return

    worktree_path = _current_worktree_session.worktree_path
    original_cwd = _current_worktree_session.original_cwd
    worktree_branch = _current_worktree_session.worktree_branch

    # Change back to original directory
    os.chdir(original_cwd)

    # Remove worktree
    git_root = find_canonical_git_root(original_cwd)
    if git_root:
        await remove_agent_worktree(worktree_path, worktree_branch, git_root)

    # Clear session
    clear_worktree_state(_current_worktree_session.session_id)
    _current_worktree_session = None


async def cleanup_stale_agent_worktrees(
    cutoff_date: datetime,
) -> int:
    """Remove stale agent/workflow worktrees older than cutoff date.

    Args:
        cutoff_date: Cutoff date for stale worktrees

    Returns:
        Number of worktrees removed
    """
    git_root = find_canonical_git_root()
    if not git_root:
        return 0

    dir_path = Path(worktrees_dir(git_root))
    if not dir_path.exists():
        return 0

    cutoff_ms = cutoff_date.timestamp() * 1000
    current_path = _current_worktree_session.worktree_path if _current_worktree_session else None
    removed = 0

    for slug_dir in dir_path.iterdir():
        if not slug_dir.is_dir():
            continue

        slug = slug_dir.name

        # Skip if not ephemeral pattern
        if not any(p.match(slug) for p in EPHEMERAL_WORKTREE_PATTERNS):
            continue

        worktree_path = str(slug_dir)

        # Skip current session
        if current_path == worktree_path:
            continue

        # Check mtime
        try:
            mtime_ms = slug_dir.stat().st_mtime * 1000
            if mtime_ms >= cutoff_ms:
                continue
        except Exception:
            continue

        # Check for changes (fail-closed)
        head_commit = get_head_commit(worktree_path)
        if head_commit and await has_worktree_changes(worktree_path, head_commit):
            continue

        # Remove
        if await remove_agent_worktree(worktree_path, worktree_branch_name(slug), git_root):
            removed += 1

    if removed > 0:
        _run_git(["worktree", "prune"], cwd=git_root, check=False)

    return removed