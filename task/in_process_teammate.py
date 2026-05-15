"""In-process Teammate Task implementation.

This implements the task state and execution for in-process teammates.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

from claude_code_py.task.base import TaskStateBase, TaskHandle, generate_task_id
from claude_code_py.task.types import TaskType, TaskStatus
from claude_code_py.utils.abort_controller import AbortController, AbortControllerPair

if TYPE_CHECKING:
    from claude_code_py.state.app_state import AppState
    from claude_code_py.core_types.message import Message


# =============================================================================
# Types
# =============================================================================


@dataclass
class TeammateIdentity:
    """Identity for an in-process teammate."""

    agent_id: str  # Full agent ID (e.g., "researcher@my-team")
    agent_name: str  # Display name (e.g., "researcher")
    team_name: str  # Team name
    parent_session_id: str  # Leader's session ID
    color: Optional[str] = None  # UI color
    plan_mode_required: bool = False  # Whether plan mode is required


@dataclass
class InProcessTeammateTaskState:
    """State for an in-process teammate task.

    Note: Does not inherit from TaskStateBase to avoid dataclass field ordering issues.
    Contains all base fields plus teammate-specific fields.
    """

    # Required fields (no defaults) - must come first
    id: str
    type: TaskType
    status: TaskStatus
    description: str
    identity: TeammateIdentity
    prompt: str
    abort_controller: AbortController

    # Optional fields with defaults
    tool_use_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    output_file: str = ""
    output_offset: int = 0
    notified: bool = False
    model: Optional[str] = None
    awaiting_plan_approval: bool = False
    permission_mode: str = "default"  # "default", "plan", "auto"
    is_idle: bool = False
    shutdown_requested: bool = False
    error: Optional[str] = None  # Error message if task failed

    # Progress tracking
    spinner_verb: str = "working"
    past_tense_verb: str = "completed"
    last_reported_tool_count: int = 0
    last_reported_token_count: int = 0
    token_count: int = 0  # Current token usage estimate
    color: Optional[str] = None  # Agent color for UI display

    # Messages
    pending_user_messages: list[dict] = field(default_factory=list)
    messages: list["Message"] = field(default_factory=list)

    # Two-level abort pattern (TypeScript: currentWorkAbortController)
    # - abort_controller (lifecycle): Kills the whole teammate
    # - current_work_abort_controller (work): Stops current turn only (Escape key)
    current_work_abort_controller: Optional[AbortController] = None

    # View lifecycle (TypeScript: LocalAgentTaskState)
    # retain=True blocks eviction, enables stream-append to messages, triggers disk bootstrap
    retain: bool = False
    # Bootstrap has read the sidechain JSONL and UUID-merged into messages. One-shot per retain cycle.
    disk_loaded: bool = False
    # Panel visibility deadline. None = no deadline (running or retained); timestamp = hide after.
    # Set on release for terminal tasks (now + PANEL_GRACE_MS); set to 0 for immediate dismiss.
    evict_after: Optional[float] = None


@dataclass
class InProcessSpawnConfig:
    """Configuration for spawning an in-process teammate."""

    name: str  # Display name
    team_name: str
    prompt: str
    color: Optional[str] = None
    plan_mode_required: bool = False
    model: Optional[str] = None


@dataclass
class InProcessSpawnOutput:
    """Result from spawning an in-process teammate."""

    success: bool
    agent_id: str
    task_id: Optional[str] = None
    abort_controller: Optional[AbortController] = None
    teammate_context: Optional[Any] = None  # TeammateContext
    error: Optional[str] = None


# =============================================================================
# ID Generation
# =============================================================================


def generate_agent_id(agent_name: str, team_name: str) -> str:
    """Generate a full agent ID.

    Args:
        agent_name: Agent display name
        team_name: Team name

    Returns:
        Full agent ID (e.g., "researcher@my-team")
    """
    return f"{agent_name}@{team_name}"


# =============================================================================
# Task Creation
# =============================================================================


def create_in_process_teammate_state(
    task_id: str,
    identity: TeammateIdentity,
    prompt: str,
    model: Optional[str] = None,
    abort_controller: Optional[AbortController] = None,
    tool_use_id: Optional[str] = None,
) -> InProcessTeammateTaskState:
    """Create a new in-process teammate task state.

    Args:
        task_id: Task ID
        identity: Teammate identity
        prompt: Initial prompt
        model: Optional model override
        abort_controller: Abort controller
        tool_use_id: Optional tool use ID

    Returns:
        InProcessTeammateTaskState
    """
    from claude_code_py.task.base import create_task_state_base

    base = create_task_state_base(
        id=task_id,
        task_type=TaskType.IN_PROCESS_TEAMMATE,
        description=f"{identity.agent_name}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}",
        tool_use_id=tool_use_id,
    )

    # Default abort controller if not provided
    if not abort_controller:
        abort_controller = AbortController()

    return InProcessTeammateTaskState(
        id=base.id,
        type=TaskType.IN_PROCESS_TEAMMATE,
        status=TaskStatus.RUNNING,
        description=base.description,
        tool_use_id=base.tool_use_id,
        start_time=base.start_time,
        output_file=base.output_file,
        output_offset=base.output_offset,
        notified=base.notified,
        identity=identity,
        prompt=prompt,
        model=model,
        abort_controller=abort_controller,
        permission_mode="plan" if identity.plan_mode_required else "default",
        is_idle=False,
        shutdown_requested=False,
        spinner_verb=_get_random_spinner_verb(),
        past_tense_verb=_get_random_completion_verb(),
        color=identity.color,  # Use color from identity
        token_count=0,
        pending_user_messages=[],
        messages=[],
    )


# =============================================================================
# Task Helpers
# =============================================================================


def find_teammate_task_by_agent_id(
    tasks: dict[str, Any],
    agent_id: str,
) -> Optional[InProcessTeammateTaskState]:
    """Find a teammate task by agent ID.

    Args:
        tasks: Dict of tasks from AppState
        agent_id: Agent ID to find

    Returns:
        InProcessTeammateTaskState or None
    """
    for task in tasks.values():
        if isinstance(task, InProcessTeammateTaskState):
            if task.identity.agent_id == agent_id:
                return task
    return None


def is_in_process_teammate_task(task: Any) -> bool:
    """Check if a task is an in-process teammate.

    Args:
        task: Task to check

    Returns:
        True if in-process teammate task
    """
    return isinstance(task, InProcessTeammateTaskState)


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