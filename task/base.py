"""Task base types and utilities.

This implements the Task system from Task.ts.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

from .types import TaskType, TaskStatus

if TYPE_CHECKING:
    from claude_code_py.state.app_state import AppState


# ID prefixes for different task types
TASK_ID_PREFIXES: dict[str, str] = {
    TaskType.LOCAL_BASH: "b",
    TaskType.LOCAL_AGENT: "a",
    TaskType.REMOTE_AGENT: "r",
    TaskType.IN_PROCESS_TEAMMATE: "t",
    TaskType.LOCAL_WORKFLOW: "w",
    TaskType.MONITOR_MCP: "m",
    TaskType.DREAM: "d",
}


def generate_task_id(task_type: TaskType) -> str:
    """Generate a unique task ID.

    Args:
        task_type: Type of task

    Returns:
        Unique task ID string
    """
    prefix = TASK_ID_PREFIXES.get(task_type, "x")
    # 8 random hex characters = 16^8 ≈ 4 billion combinations
    random_part = secrets.token_hex(8)
    return f"{prefix}{random_part}"


@dataclass
class TaskStateBase:
    """Base state for all tasks.

    This is the common fields shared by all task types.
    """

    id: str
    type: TaskType
    status: TaskStatus
    description: str
    tool_use_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_paused_ms: Optional[float] = None
    output_file: str = ""
    output_offset: int = 0
    notified: bool = False


@dataclass
class TaskHandle:
    """Handle for a running task.

    Provides a way to interact with and cleanup a task.
    """

    task_id: str
    cleanup: Optional[Callable[[], None]] = None


# Type alias for set_app_state (use string annotation to avoid circular import)
SetAppState = Callable[[Callable[[dict, Any], dict]], None]


@dataclass
class TaskContext:
    """Context passed to task operations."""

    abort_controller: Any  # AbortController
    get_app_state: Callable[[], AppState]
    set_app_state: SetAppState


@dataclass
class LocalShellSpawnInput:
    """Input for spawning a local shell task."""

    command: str
    description: str
    timeout: Optional[int] = None
    tool_use_id: Optional[str] = None
    agent_id: Optional[str] = None
    kind: str = "bash"  # "bash" or "monitor"


def create_task_state_base(
    id: str,
    task_type: TaskType,
    description: str,
    tool_use_id: Optional[str] = None,
) -> TaskStateBase:
    """Create a base task state.

    Args:
        id: Task ID
        task_type: Type of task
        description: Task description
        tool_use_id: Optional tool use ID

    Returns:
        TaskStateBase instance
    """
    from claude_code_py.utils.task.disk_output import get_task_output_path

    return TaskStateBase(
        id=id,
        type=task_type,
        status=TaskStatus.PENDING,
        description=description,
        tool_use_id=tool_use_id,
        start_time=time.time(),
        output_file=get_task_output_path(id),
        output_offset=0,
        notified=False,
    )