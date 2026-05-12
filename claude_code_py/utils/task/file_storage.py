"""File-based task storage with file locking.

This implements the V2 task system from TypeScript tasks.ts with:
- File-based storage with JSON files
- High water mark to prevent ID reuse after deletion/reset
- Cross-process file locking (using fcntl/portalocker)
- Task dependencies (blocks/blockedBy)
- Atomic task claiming with busy check
- Agent status tracking based on task ownership
- Teammate task unassignment on shutdown

Ported from: src/utils/tasks.ts
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Cross-Process File Locking
# =============================================================================

class FileLock:
    """Cross-process file lock using fcntl (Unix) or msvcrt (Windows).

    Provides advisory locking for concurrent access across multiple processes.
    Similar to proper-lockfile in TypeScript.
    """

    def __init__(self, lock_path: Path, timeout_ms: int = 5000, retries: int = 30):
        """Initialize lock.

        Args:
            lock_path: Path to lock file
            timeout_ms: Total timeout in milliseconds
            retries: Number of retries with backoff
        """
        self.lock_path = lock_path
        self.timeout_ms = timeout_ms
        self.retries = retries
        self._fd: Optional[int] = None
        self._locked = False

    def acquire(self) -> bool:
        """Acquire the lock with retries.

        Returns:
            True if lock acquired, False if timeout
        """
        import random

        # Ensure lock file exists
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lock_path.touch(exist_ok=True)
        except Exception:
            pass

        min_wait = 5
        max_wait = 100
        total_waited = 0

        for attempt in range(self.retries):
            try:
                self._fd = os.open(str(self.lock_path), os.O_RDWR)
                self._try_lock()
                self._locked = True
                return True
            except (IOError, OSError) as e:
                if self._fd is not None:
                    try:
                        os.close(self._fd)
                    except Exception:
                        pass
                    self._fd = None

                # Check if timeout exceeded
                wait_time = random.randint(min_wait, max_wait)
                total_waited += wait_time
                if total_waited >= self.timeout_ms:
                    logger.warning(f"File lock timeout after {total_waited}ms for {self.lock_path}")
                    return False

                time.sleep(wait_time / 1000.0)

        return False

    def _try_lock(self) -> None:
        """Try to acquire the lock using platform-specific method."""
        import sys

        if sys.platform == 'win32':
            # Windows: use msvcrt.locking
            import msvcrt
            msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
        else:
            # Unix: use fcntl.flock
            import fcntl
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self) -> None:
        """Release the lock."""
        if not self._locked or self._fd is None:
            return

        try:
            import sys
            if sys.platform == 'win32':
                import msvcrt
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:
            pass

        try:
            os.close(self._fd)
        except Exception:
            pass

        self._fd = None
        self._locked = False

    def __enter__(self) -> 'FileLock':
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock on {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class AsyncFileLock:
    """Async wrapper for FileLock."""

    def __init__(self, lock_path: Path, timeout_ms: int = 5000, retries: int = 30):
        self._lock = FileLock(lock_path, timeout_ms, retries)

    async def acquire(self) -> bool:
        """Acquire lock asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._lock.acquire)

    def release(self) -> None:
        """Release lock."""
        self._lock.release()

    async def __aenter__(self) -> 'AsyncFileLock':
        if not await self.acquire():
            raise TimeoutError(f"Could not acquire lock")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


# =============================================================================
# Task Types
# =============================================================================

class TaskStatus(str, Enum):
    """Task status for V2 tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


TASK_STATUSES = [TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]


@dataclass
class Task:
    """V2 Task definition matching TypeScript TaskSchema."""
    id: str
    subject: str
    description: str
    status: TaskStatus
    activeForm: Optional[str] = None  # Present continuous form for spinner
    owner: Optional[str] = None  # Agent ID owning this task
    blocks: List[str] = field(default_factory=list)  # Task IDs this task blocks
    blockedBy: List[str] = field(default_factory=list)  # Task IDs that block this task
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamMember:
    """Team member info for agent status."""
    agent_id: str
    name: str
    agent_type: Optional[str] = None


@dataclass
class AgentStatus:
    """Agent status based on task ownership."""
    agent_id: str
    name: str
    status: str  # 'idle' | 'busy'
    current_tasks: List[str] = field(default_factory=list)  # Task IDs the agent owns
    agent_type: Optional[str] = None


# =============================================================================
# Claim Task Result
# =============================================================================

@dataclass
class ClaimTaskResult:
    """Result of claiming a task."""
    success: bool
    reason: Optional[str] = None  # task_not_found | already_claimed | already_resolved | blocked | agent_busy
    task: Optional[Task] = None
    busy_with_tasks: Optional[List[str]] = None  # Task IDs agent is busy with
    blocked_by_tasks: Optional[List[str]] = None  # Task IDs blocking this task


@dataclass
class UnassignTasksResult:
    """Result of unassigning tasks from a teammate."""
    unassigned_tasks: List[Tuple[str, str]]  # List of (task_id, subject)
    notification_message: str


# =============================================================================
# Constants
# =============================================================================

HIGH_WATER_MARK_FILE = ".highwatermark"
LOCK_FILE = ".lock"
DEFAULT_TASKS_MODE_TASK_LIST_ID = "tasklist"

# Lock options matching TypeScript
LOCK_OPTIONS = {
    "retries": 30,
    "min_timeout_ms": 5,
    "max_timeout_ms": 100,
}


# =============================================================================
# Path Utilities
# =============================================================================

def sanitize_path_component(name: str) -> str:
    """Sanitize a string for safe use in file paths.

    Removes path traversal characters and other potentially dangerous characters.
    Only allows alphanumeric characters, hyphens, and underscores.
    """
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name)


def get_claude_config_home() -> Path:
    """Get the Claude config home directory."""
    config_home = os.environ.get("CLAUDE_CONFIG_HOME")
    if config_home:
        return Path(config_home)
    return Path.home() / ".claude"


def get_teams_dir() -> Path:
    """Get the teams directory."""
    return get_claude_config_home() / "teams"


def get_tasks_base_dir() -> Path:
    """Get the base directory for task storage."""
    return get_claude_config_home() / "tasks"


def get_tasks_dir(task_list_id: str) -> Path:
    """Get the directory for a specific task list."""
    return get_tasks_base_dir() / sanitize_path_component(task_list_id)


def get_task_path(task_list_id: str, task_id: str) -> Path:
    """Get the file path for a task."""
    return get_tasks_dir(task_list_id) / f"{sanitize_path_component(task_id)}.json"


def get_high_water_mark_path(task_list_id: str) -> Path:
    """Get the high water mark file path."""
    return get_tasks_dir(task_list_id) / HIGH_WATER_MARK_FILE


def get_task_list_lock_path(task_list_id: str) -> Path:
    """Get the lock file path for a task list."""
    return get_tasks_dir(task_list_id) / LOCK_FILE


def ensure_tasks_dir(task_list_id: str) -> Path:
    """Ensure the tasks directory exists, return the path."""
    dir_path = get_tasks_dir(task_list_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def ensure_task_list_lock_file(task_list_id: str) -> Path:
    """Ensure the lock file exists for a task list.

    proper-lockfile requires the target file to exist.
    """
    ensure_tasks_dir(task_list_id)
    lock_path = get_task_list_lock_path(task_list_id)

    # Create with exclusive flag so concurrent callers don't both create
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError:
        pass  # Already exists, which is fine

    return lock_path


# =============================================================================
# High Water Mark
# =============================================================================

def read_high_water_mark(task_list_id: str) -> int:
    """Read the high water mark for task IDs.

    The high water mark stores the maximum task ID ever assigned,
    preventing ID reuse after deletion/reset.
    """
    path = get_high_water_mark_path(task_list_id)
    try:
        content = path.read_text().strip()
        value = int(content)
        return max(value, 0)
    except (FileNotFoundError, ValueError, OSError):
        return 0


def write_high_water_mark(task_list_id: str, value: int) -> None:
    """Write the high water mark."""
    path = get_high_water_mark_path(task_list_id)
    ensure_tasks_dir(task_list_id)
    path.write_text(str(value))


def find_highest_task_id_from_files(task_list_id: str) -> int:
    """Find the highest task ID from existing task files (not including high water mark)."""
    dir_path = get_tasks_dir(task_list_id)
    try:
        files = list(dir_path.glob("*.json"))
    except FileNotFoundError:
        return 0

    highest = 0
    for f in files:
        # Skip hidden files like .lock and .highwatermark
        if f.stem.startswith("."):
            continue
        try:
            task_id = int(f.stem)
            if task_id > highest:
                highest = task_id
        except ValueError:
            continue

    return highest


def find_highest_task_id(task_list_id: str) -> int:
    """Find the highest task ID ever assigned, considering files and high water mark."""
    from_files = find_highest_task_id_from_files(task_list_id)
    from_mark = read_high_water_mark(task_list_id)
    return max(from_files, from_mark)


# =============================================================================
# Task List ID Resolution
# =============================================================================

# Leader team name set by TeamCreateTool
_leader_team_name: Optional[str] = None


def set_leader_team_name(team_name: str) -> None:
    """Set the leader's team name for task list resolution.

    Called by TeamCreateTool when a team is created.
    """
    global _leader_team_name
    if _leader_team_name == team_name:
        return
    _leader_team_name = team_name
    notify_tasks_updated()


def clear_leader_team_name() -> None:
    """Clear the leader's team name (when team is deleted)."""
    global _leader_team_name
    if _leader_team_name is None:
        return
    _leader_team_name = None
    notify_tasks_updated()


def get_task_list_id() -> str:
    """Get the task list ID based on current context.

    Priority:
    1. CLAUDE_CODE_TASK_LIST_ID - explicit task list ID
    2. In-process teammate: leader's team name (so teammates share the leader's task list)
    3. CLAUDE_CODE_TEAM_NAME - set when running as a process-based teammate
    4. Leader team name - set when the leader creates a team via TeamCreate
    5. Session ID fallback
    """
    # 1. Explicit env var
    task_list_id = os.environ.get("CLAUDE_CODE_TASK_LIST_ID")
    if task_list_id:
        return task_list_id

    # 2. In-process teammate context
    from claude_code_py.utils.teammate_context import get_teammate_context
    teammate_ctx = get_teammate_context()
    if teammate_ctx:
        return teammate_ctx.team_name

    # 3. Process-based teammate env var
    team_name = os.environ.get("CLAUDE_CODE_TEAM_NAME")
    if team_name:
        return team_name

    # 4. Leader team name
    if _leader_team_name:
        return _leader_team_name

    # 5. Session ID fallback (would need session state in full implementation)
    session_id = os.environ.get("CLAUDE_CODE_SESSION_ID", "default")
    return session_id


def is_task_v2_enabled() -> bool:
    """Check if V2 task system is enabled.

    Force-enable in non-interactive mode (SDK users who want Task tools over TodoWrite).
    """
    enable_env = os.environ.get("CLAUDE_CODE_ENABLE_TASKS", "")
    if enable_env.lower() == "true":
        return True

    # Default: enabled in interactive mode
    # In full implementation, this would check is_non_interactive_session
    return True


# =============================================================================
# Task CRUD Operations
# =============================================================================

def create_task(
    task_list_id: str,
    subject: str,
    description: str,
    activeForm: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    owner: Optional[str] = None,
    blocks: Optional[List[str]] = None,
    blockedBy: Optional[List[str]] = None,
) -> str:
    """Create a new task with a unique ID.

    Uses file locking to prevent race conditions when multiple processes
    create tasks concurrently.

    Args:
        task_list_id: Task list ID
        subject: Task subject/title
        description: Task description
        activeForm: Optional spinner text (e.g., "Running tests")
        metadata: Optional metadata dict
        owner: Optional owner agent ID
        blocks: Optional list of task IDs this task blocks
        blockedBy: Optional list of task IDs that block this task

    Returns:
        Task ID (string)
    """
    ensure_tasks_dir(task_list_id)
    lock_path = ensure_task_list_lock_file(task_list_id)

    lock = FileLock(lock_path, timeout_ms=5000, retries=30)

    with lock:
        # Read highest ID from disk while holding the lock
        highest_id = find_highest_task_id(task_list_id)
        task_id = str(highest_id + 1)

        task = Task(
            id=task_id,
            subject=subject,
            description=description,
            status=TaskStatus.PENDING,
            activeForm=activeForm,
            metadata=metadata or {},
            owner=owner,
            blocks=blocks or [],
            blockedBy=blockedBy or [],
        )

        path = get_task_path(task_list_id, task_id)
        path.write_text(json.dumps(task_to_dict(task), indent=2))

    notify_tasks_updated()
    return task_id


async def create_task_async(
    task_list_id: str,
    subject: str,
    description: str,
    activeForm: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    owner: Optional[str] = None,
    blocks: Optional[List[str]] = None,
    blockedBy: Optional[List[str]] = None,
) -> str:
    """Async version of create_task."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: create_task(
            task_list_id, subject, description, activeForm, metadata, owner, blocks, blockedBy
        )
    )


def get_task(task_list_id: str, task_id: str) -> Optional[Task]:
    """Get a task by ID.

    Args:
        task_list_id: Task list ID
        task_id: Task ID

    Returns:
        Task or None if not found
    """
    path = get_task_path(task_list_id, task_id)
    try:
        content = path.read_text()
        data = json.loads(content)

        # Migrate old status names (for backwards compatibility)
        if data.get("status") == "open":
            data["status"] = "pending"
        elif data.get("status") == "resolved":
            data["status"] = "completed"
        elif data.get("status") in ("planning", "implementing", "reviewing", "verifying"):
            data["status"] = "in_progress"

        return dict_to_task(data)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Failed to read task {task_id}: {e}")
        return None


def update_task_unsafe(
    task_list_id: str,
    task_id: str,
    updates: Dict[str, Any],
) -> Optional[Task]:
    """Update a task without locking (caller must hold lock).

    Args:
        task_list_id: Task list ID
        task_id: Task ID
        updates: Dict of fields to update

    Returns:
        Updated task or None if not found
    """
    existing = get_task(task_list_id, task_id)
    if not existing:
        return None

    # Apply updates
    if "subject" in updates and updates["subject"] is not None:
        existing.subject = updates["subject"]
    if "description" in updates and updates["description"] is not None:
        existing.description = updates["description"]
    if "activeForm" in updates:
        existing.activeForm = updates["activeForm"]
    if "status" in updates and updates["status"] is not None:
        existing.status = updates["status"]
    if "owner" in updates:
        existing.owner = updates["owner"]
    if "blocks" in updates:
        existing.blocks = updates["blocks"]
    if "blockedBy" in updates:
        existing.blockedBy = updates["blockedBy"]
    if "metadata" in updates:
        # Merge metadata, null values delete keys
        for key, value in updates["metadata"].items():
            if value is None:
                existing.metadata.pop(key, None)
            else:
                existing.metadata[key] = value

    path = get_task_path(task_list_id, task_id)
    path.write_text(json.dumps(task_to_dict(existing), indent=2))

    notify_tasks_updated()
    return existing


def update_task(
    task_list_id: str,
    task_id: str,
    subject: Optional[str] = None,
    description: Optional[str] = None,
    activeForm: Optional[str] = None,
    status: Optional[Union[TaskStatus, str]] = None,
    owner: Optional[str] = None,
    blocks: Optional[List[str]] = None,
    blockedBy: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Task]:
    """Update a task with file locking.

    Args:
        task_list_id: Task list ID
        task_id: Task ID
        subject: New subject
        description: New description
        activeForm: New activeForm
        status: New status
        owner: New owner
        blocks: New blocks list
        blockedBy: New blockedBy list
        metadata: Metadata to merge (set key to null to delete)

    Returns:
        Updated task or None if not found
    """
    task_path = get_task_path(task_list_id, task_id)

    # Check existence before locking
    task_before_lock = get_task(task_list_id, task_id)
    if not task_before_lock:
        return None

    lock = FileLock(task_path, timeout_ms=5000, retries=30)

    with lock:
        # Build updates dict
        updates: Dict[str, Any] = {}
        if subject is not None:
            updates["subject"] = subject
        if description is not None:
            updates["description"] = description
        if activeForm is not None:
            updates["activeForm"] = activeForm
        if status is not None:
            updates["status"] = TaskStatus(status) if isinstance(status, str) else status
        if owner is not None:
            updates["owner"] = owner
        if blocks is not None:
            updates["blocks"] = blocks
        if blockedBy is not None:
            updates["blockedBy"] = blockedBy
        if metadata is not None:
            updates["metadata"] = metadata

        return update_task_unsafe(task_list_id, task_id, updates)


async def update_task_async(
    task_list_id: str,
    task_id: str,
    **kwargs
) -> Optional[Task]:
    """Async version of update_task."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: update_task(task_list_id, task_id, **kwargs)
    )


def delete_task(task_list_id: str, task_id: str) -> bool:
    """Delete a task.

    Updates high water mark before deleting to prevent ID reuse.
    Also removes references to this task from other tasks' blocks/blockedBy.

    Args:
        task_list_id: Task list ID
        task_id: Task ID

    Returns:
        True if deleted, False if not found
    """
    task_path = get_task_path(task_list_id, task_id)
    lock_path = ensure_task_list_lock_file(task_list_id)

    lock = FileLock(lock_path, timeout_ms=5000, retries=30)

    with lock:
        # Update high water mark before deleting
        try:
            numeric_id = int(task_id)
            current_mark = read_high_water_mark(task_list_id)
            if numeric_id > current_mark:
                write_high_water_mark(task_list_id, numeric_id)
        except ValueError:
            pass

        # Delete the task file
        try:
            task_path.unlink()
        except FileNotFoundError:
            return False

        # Remove references from other tasks
        all_tasks = list_tasks(task_list_id)
        for task in all_tasks:
            new_blocks = [id for id in task.blocks if id != task_id]
            new_blocked_by = [id for id in task.blockedBy if id != task_id]

            if len(new_blocks) != len(task.blocks) or len(new_blocked_by) != len(task.blockedBy):
                update_task_unsafe(
                    task_list_id,
                    task.id,
                    {"blocks": new_blocks, "blockedBy": new_blocked_by}
                )

    notify_tasks_updated()
    return True


def list_tasks(task_list_id: str) -> List[Task]:
    """List all tasks in a task list.

    Args:
        task_list_id: Task list ID

    Returns:
        List of tasks
    """
    dir_path = get_tasks_dir(task_list_id)
    try:
        files = list(dir_path.glob("*.json"))
    except FileNotFoundError:
        return []

    tasks = []
    for f in files:
        # Skip hidden files
        if f.stem.startswith("."):
            continue
        task = get_task(task_list_id, f.stem)
        if task:
            tasks.append(task)

    return tasks


def reset_task_list(task_list_id: str) -> None:
    """Reset a task list for a new swarm - clears all existing tasks.

    Writes a high water mark file to prevent ID reuse after reset.
    Should be called when a new swarm is created.

    Args:
        task_list_id: Task list ID
    """
    ensure_tasks_dir(task_list_id)
    lock_path = ensure_task_list_lock_file(task_list_id)

    lock = FileLock(lock_path, timeout_ms=5000, retries=30)

    with lock:
        # Find current highest ID and save to high water mark
        current_highest = find_highest_task_id_from_files(task_list_id)
        if current_highest > 0:
            existing_mark = read_high_water_mark(task_list_id)
            if current_highest > existing_mark:
                write_high_water_mark(task_list_id, current_highest)

        # Delete all task files
        dir_path = get_tasks_dir(task_list_id)
        try:
            files = list(dir_path.glob("*.json"))
            for f in files:
                if not f.stem.startswith("."):
                    try:
                        f.unlink()
                    except FileNotFoundError:
                        pass
        except FileNotFoundError:
            pass

    notify_tasks_updated()


# =============================================================================
# Task Blocking Relationships
# =============================================================================

def block_task(
    task_list_id: str,
    from_task_id: str,
    to_task_id: str,
) -> bool:
    """Set up a blocking relationship between tasks.

    A blocks B means: B cannot start until A is completed.

    Args:
        task_list_id: Task list ID
        from_task_id: Task that blocks (A)
        to_task_id: Task that is blocked (B)

    Returns:
        True if successful, False if tasks not found
    """
    lock_path = ensure_task_list_lock_file(task_list_id)

    lock = FileLock(lock_path, timeout_ms=5000, retries=30)

    with lock:
        from_task = get_task(task_list_id, from_task_id)
        to_task = get_task(task_list_id, to_task_id)

        if not from_task or not to_task:
            return False

        # Update from_task: A blocks B
        if to_task_id not in from_task.blocks:
            from_task.blocks.append(to_task_id)
            update_task_unsafe(task_list_id, from_task_id, {"blocks": from_task.blocks})

        # Update to_task: B is blockedBy A
        if from_task_id not in to_task.blockedBy:
            to_task.blockedBy.append(from_task_id)
            update_task_unsafe(task_list_id, to_task_id, {"blockedBy": to_task.blockedBy})

    return True


# =============================================================================
# Task Claiming (Atomic)
# =============================================================================

def claim_task(
    task_list_id: str,
    task_id: str,
    claimant_agent_id: str,
    check_agent_busy: bool = False,
) -> ClaimTaskResult:
    """Attempt to claim a task for an agent atomically.

    With check_agent_busy=True, uses task-list-level lock to atomically check
    if the agent owns any other open tasks before claiming (prevents TOCTOU race).

    Args:
        task_list_id: Task list ID
        task_id: Task ID to claim
        claimant_agent_id: Agent ID claiming the task
        check_agent_busy: If True, check if agent is already busy with other tasks

    Returns:
        ClaimTaskResult with success/failure and details
    """
    if check_agent_busy:
        return _claim_task_with_busy_check(task_list_id, task_id, claimant_agent_id)

    task_path = get_task_path(task_list_id, task_id)

    # Check existence before locking
    task_before_lock = get_task(task_list_id, task_id)
    if not task_before_lock:
        return ClaimTaskResult(success=False, reason="task_not_found")

    lock = FileLock(task_path, timeout_ms=5000, retries=30)

    try:
        with lock:
            task = get_task(task_list_id, task_id)
            if not task:
                return ClaimTaskResult(success=False, reason="task_not_found")

            # Check if already claimed by another agent
            if task.owner and task.owner != claimant_agent_id:
                return ClaimTaskResult(success=False, reason="already_claimed", task=task)

            # Check if already completed
            if task.status == TaskStatus.COMPLETED:
                return ClaimTaskResult(success=False, reason="already_resolved", task=task)

            # Check for unresolved blockers
            all_tasks = list_tasks(task_list_id)
            unresolved_ids = set(t.id for t in all_tasks if t.status != TaskStatus.COMPLETED)
            blocked_by_tasks = [id for id in task.blockedBy if id in unresolved_ids]

            if blocked_by_tasks:
                return ClaimTaskResult(
                    success=False,
                    reason="blocked",
                    task=task,
                    blocked_by_tasks=blocked_by_tasks
                )

            # Claim the task
            updated = update_task_unsafe(task_list_id, task_id, {"owner": claimant_agent_id})
            return ClaimTaskResult(success=True, task=updated)

    except TimeoutError:
        logger.warning(f"Timeout claiming task {task_id}")
        return ClaimTaskResult(success=False, reason="task_not_found")


def _claim_task_with_busy_check(
    task_list_id: str,
    task_id: str,
    claimant_agent_id: str,
) -> ClaimTaskResult:
    """Claim a task with atomic check for agent busy status.

    Uses task-list-level lock to ensure busy check and claim are atomic.
    """
    lock_path = ensure_task_list_lock_file(task_list_id)

    lock = FileLock(lock_path, timeout_ms=5000, retries=30)

    try:
        with lock:
            # Read all tasks atomically
            all_tasks = list_tasks(task_list_id)

            # Find target task
            task = next((t for t in all_tasks if t.id == task_id), None)
            if not task:
                return ClaimTaskResult(success=False, reason="task_not_found")

            # Check if already claimed
            if task.owner and task.owner != claimant_agent_id:
                return ClaimTaskResult(success=False, reason="already_claimed", task=task)

            # Check if already completed
            if task.status == TaskStatus.COMPLETED:
                return ClaimTaskResult(success=False, reason="already_resolved", task=task)

            # Check for unresolved blockers
            unresolved_ids = set(t.id for t in all_tasks if t.status != TaskStatus.COMPLETED)
            blocked_by_tasks = [id for id in task.blockedBy if id in unresolved_ids]

            if blocked_by_tasks:
                return ClaimTaskResult(
                    success=False,
                    reason="blocked",
                    task=task,
                    blocked_by_tasks=blocked_by_tasks
                )

            # Check if agent is busy with other unresolved tasks
            agent_open_tasks = [
                t.id for t in all_tasks
                if t.status != TaskStatus.COMPLETED
                and t.owner == claimant_agent_id
                and t.id != task_id
            ]

            if agent_open_tasks:
                return ClaimTaskResult(
                    success=False,
                    reason="agent_busy",
                    task=task,
                    busy_with_tasks=agent_open_tasks
                )

            # Claim the task
            updated = update_task_unsafe(task_list_id, task_id, {"owner": claimant_agent_id})
            return ClaimTaskResult(success=True, task=updated)

    except TimeoutError:
        logger.warning(f"Timeout in claim_task_with_busy_check for {task_id}")
        return ClaimTaskResult(success=False, reason="task_not_found")


async def claim_task_async(
    task_list_id: str,
    task_id: str,
    claimant_agent_id: str,
    check_agent_busy: bool = False,
) -> ClaimTaskResult:
    """Async version of claim_task."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: claim_task(task_list_id, task_id, claimant_agent_id, check_agent_busy)
    )


# =============================================================================
# Agent Status Tracking
# =============================================================================

def _read_team_members(team_name: str) -> Optional[Tuple[str, List[TeamMember]]]:
    """Read team members from the team file.

    Returns:
        Tuple of (lead_agent_id, members) or None if team not found
    """
    teams_dir = get_teams_dir()
    team_file_path = teams_dir / sanitize_path_component(team_name) / "config.json"

    try:
        content = team_file_path.read_text()
        data = json.loads(content)

        lead_agent_id = data.get("leadAgentId", "")
        members = [
            TeamMember(
                agent_id=m.get("agentId", ""),
                name=m.get("name", ""),
                agent_type=m.get("agentType"),
            )
            for m in data.get("members", [])
        ]

        return lead_agent_id, members
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.debug(f"Failed to read team file for {team_name}: {e}")
        return None


def get_agent_statuses(team_name: str) -> Optional[List[AgentStatus]]:
    """Get the status of all agents in a team based on task ownership.

    An agent is considered "idle" if they don't own any open tasks.
    An agent is considered "busy" if they own at least one open task.

    Args:
        team_name: The team name (also used as task_list_id)

    Returns:
        Array of agent statuses, or None if team not found
    """
    team_data = _read_team_members(team_name)
    if not team_data:
        return None

    lead_agent_id, members = team_data
    task_list_id = sanitize_path_component(team_name)

    all_tasks = list_tasks(task_list_id)

    # Get unresolved tasks grouped by owner (pending or in_progress)
    unresolved_by_owner: Dict[str, List[str]] = {}
    for task in all_tasks:
        if task.status != TaskStatus.COMPLETED and task.owner:
            unresolved_by_owner.setdefault(task.owner, []).append(task.id)

    # Build status for each agent
    statuses = []
    for member in members:
        # Check both name and agentId for backwards compatibility
        tasks_by_name = unresolved_by_owner.get(member.name, [])
        tasks_by_id = unresolved_by_owner.get(member.agent_id, [])
        current_tasks = list(set(tasks_by_name + tasks_by_id))

        statuses.append(AgentStatus(
            agent_id=member.agent_id,
            name=member.name,
            agent_type=member.agent_type,
            status="idle" if not current_tasks else "busy",
            current_tasks=current_tasks,
        ))

    return statuses


# =============================================================================
# Teammate Task Unassignment
# =============================================================================

def unassign_teammate_tasks(
    team_name: str,
    teammate_id: str,
    teammate_name: str,
    reason: str,  # 'terminated' | 'shutdown'
) -> UnassignTasksResult:
    """Unassign all open tasks from a teammate when they shut down.

    Used when a teammate is killed or gracefully shuts down.
    Resets unassigned tasks to pending status with no owner.

    Args:
        team_name: The team/task list name
        teammate_id: The teammate's agent ID
        teammate_name: The teammate's display name
        reason: How the teammate exited ('terminated' | 'shutdown')

    Returns:
        UnassignTasksResult with unassigned tasks and notification message
    """
    task_list_id = sanitize_path_component(team_name)

    lock_path = ensure_task_list_lock_file(task_list_id)
    lock = FileLock(lock_path, timeout_ms=5000, retries=30)

    unassigned_tasks: List[Tuple[str, str]] = []

    with lock:
        tasks = list_tasks(task_list_id)

        # Find unresolved tasks owned by this teammate
        unresolved_owned = [
            t for t in tasks
            if t.status != TaskStatus.COMPLETED
            and (t.owner == teammate_id or t.owner == teammate_name)
        ]

        # Unassign each task
        for task in unresolved_owned:
            update_task_unsafe(
                task_list_id,
                task.id,
                {"owner": None, "status": TaskStatus.PENDING}
            )
            unassigned_tasks.append((task.id, task.subject))

    if unassigned_tasks:
        logger.debug(f"Unassigned {len(unassigned_tasks)} task(s) from {teammate_name}")

    notify_tasks_updated()

    # Build notification message
    action_verb = "was terminated" if reason == "terminated" else "has shut down"
    notification_message = f"{teammate_name} {action_verb}."

    if unassigned_tasks:
        task_list_str = ", ".join(f'#{id} "{subject}"' for id, subject in unassigned_tasks)
        notification_message += (
            f" {len(unassigned_tasks)} task(s) were unassigned: {task_list_str}. "
            f"Use TaskList to check availability and TaskUpdate with owner to reassign "
            f"them to idle teammates."
        )

    return UnassignTasksResult(
        unassigned_tasks=unassigned_tasks,
        notification_message=notification_message,
    )


async def unassign_teammate_tasks_async(
    team_name: str,
    teammate_id: str,
    teammate_name: str,
    reason: str,
) -> UnassignTasksResult:
    """Async version of unassign_teammate_tasks."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: unassign_teammate_tasks(team_name, teammate_id, teammate_name, reason)
    )


# =============================================================================
# Serialization
# =============================================================================

def task_to_dict(task: Task) -> Dict[str, Any]:
    """Convert a Task to a dict for JSON serialization."""
    return {
        "id": task.id,
        "subject": task.subject,
        "description": task.description,
        "status": task.status.value,
        "activeForm": task.activeForm,
        "owner": task.owner,
        "blocks": task.blocks,
        "blockedBy": task.blockedBy,
        "metadata": task.metadata,
    }


def dict_to_task(data: Dict[str, Any]) -> Task:
    """Convert a dict to a Task."""
    return Task(
        id=data["id"],
        subject=data["subject"],
        description=data["description"],
        status=TaskStatus(data["status"]),
        activeForm=data.get("activeForm"),
        owner=data.get("owner"),
        blocks=data.get("blocks", []),
        blockedBy=data.get("blockedBy", []),
        metadata=data.get("metadata", {}),
    )


# =============================================================================
# Update Notifications (Signal System)
# =============================================================================

# In-process update callbacks
_task_update_callbacks: List[Callable[[], None]] = []
_callback_lock = threading.Lock()


def on_tasks_updated(callback: Callable[[], None]) -> Callable[[], None]:
    """Register a callback for task updates.

    Returns an unsubscribe function.
    """
    with _callback_lock:
        _task_update_callbacks.append(callback)

    def unsubscribe():
        with _callback_lock:
            try:
                _task_update_callbacks.remove(callback)
            except ValueError:
                pass

    return unsubscribe


def notify_tasks_updated() -> None:
    """Notify all registered callbacks of task updates.

    Wraps emit in try/catch so listener failures never propagate.
    """
    with _callback_lock:
        callbacks = list(_task_update_callbacks)

    for cb in callbacks:
        try:
            cb()
        except Exception:
            pass  # Listener errors should not propagate


# =============================================================================
# Export All
# =============================================================================

__all__ = [
    # Types
    "TaskStatus",
    "Task",
    "TeamMember",
    "AgentStatus",
    "ClaimTaskResult",
    "UnassignTasksResult",
    # Locking
    "FileLock",
    "AsyncFileLock",
    # Path utilities
    "sanitize_path_component",
    "get_tasks_dir",
    "get_task_path",
    "get_task_list_lock_path",
    "ensure_tasks_dir",
    "ensure_task_list_lock_file",
    # High water mark
    "read_high_water_mark",
    "write_high_water_mark",
    "find_highest_task_id",
    "find_highest_task_id_from_files",
    # Task list ID
    "get_task_list_id",
    "is_task_v2_enabled",
    "set_leader_team_name",
    "clear_leader_team_name",
    # CRUD
    "create_task",
    "create_task_async",
    "get_task",
    "update_task",
    "update_task_async",
    "update_task_unsafe",
    "delete_task",
    "list_tasks",
    "reset_task_list",
    # Blocking
    "block_task",
    # Claiming
    "claim_task",
    "claim_task_async",
    # Agent status
    "get_agent_statuses",
    # Teammate unassignment
    "unassign_teammate_tasks",
    "unassign_teammate_tasks_async",
    # Serialization
    "task_to_dict",
    "dict_to_task",
    # Notifications
    "on_tasks_updated",
    "notify_tasks_updated",
    # Constants
    "TASK_STATUSES",
    "DEFAULT_TASKS_MODE_TASK_LIST_ID",
    "HIGH_WATER_MARK_FILE",
    "LOCK_FILE",
]