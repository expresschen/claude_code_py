"""Task manager for coordinating background tasks and in-process teammates.

This module provides:
- Global task registry for in-process teammates
- Task registration/unregistration functions
- Spawning and stopping functions for in-process teammates
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

# Debug flag - controlled by environment variable CLAUDE_CODE_DEBUG_TEAMMATE
DEBUG_MANAGER = os.environ.get("CLAUDE_CODE_DEBUG_TEAMMATE", "").lower() in ("1", "true", "yes")

# Import unified debug logger
from claude_code_py.utils.debug_log import debug_log

def _debug_print(msg: str) -> None:
    """Print debug message to console and log file."""
    debug_log("[TASK_MANAGER]", msg, DEBUG_MANAGER)

__all__ = [
    "register_task",
    "unregister_task",
    "get_task_by_id",
    "find_task_by_agent_id",
    "SpawnTeammateConfig",
    "SpawnTeammateResult",
    "spawn_in_process_teammate_v2",
    "stop_teammate",
    "_TASK_REGISTRY",
    "_THREAD_REGISTRY",
]

from .base import TaskStateBase, TaskHandle, generate_task_id, create_task_state_base
from .types import TaskType, TaskStatus, is_terminal_task_status
from .in_process_teammate import (
    InProcessTeammateTaskState,
    TeammateIdentity,
    InProcessSpawnConfig,
    InProcessSpawnOutput,
    generate_agent_id,
    create_in_process_teammate_state,
)

if TYPE_CHECKING:
    from claude_code_py.state.store import Store
    from claude_code_py.state.app_state import AppState
    from claude_code_py.utils.abort_controller import AbortController
    from claude_code_py.utils.teammate_context import TeammateContext
    from claude_code_py.utils.swarm.in_process_runner import InProcessRunnerConfig
    from claude_code_py.tool.context import ToolUseContext


# =============================================================================
# Global Task Registry
# =============================================================================

# In-process teammate task registry (fallback when AppState not available)
_TASK_REGISTRY: Dict[str, InProcessTeammateTaskState] = {}

# Thread handles for in-process teammates (real execution tracking)
_THREAD_REGISTRY: Dict[str, threading.Thread] = {}


# =============================================================================
# Task Registration Functions
# =============================================================================


def register_task(
    task_state: InProcessTeammateTaskState,
    set_app_state: Optional[Callable] = None,
) -> None:
    """Register a task in AppState or global registry.

    Args:
        task_state: Task state to register
        set_app_state: Optional AppState setter. If None, uses global registry.
    """
    _debug_print(f"→ register_task: task_id='{task_state.id}', agent_id='{task_state.identity.agent_id}'")
    _debug_print(f"   using_app_state={set_app_state is not None}")

    if set_app_state:
        set_app_state(
            lambda prev: replace(
                prev,
                tasks={**prev.tasks, task_state.id: task_state},
            )
        )
        _debug_print(f"✅ Task registered in AppState")
    else:
        _TASK_REGISTRY[task_state.id] = task_state
        _debug_print(f"✅ Task registered in global registry (fallback)")

    _debug_print(f"   Total tasks in registry: {len(_TASK_REGISTRY) if not set_app_state else 'N/A (AppState)'}")


def unregister_task(
    task_id: str,
    set_app_state: Optional[Callable] = None,
) -> None:
    """Unregister a task from AppState or global registry.

    Also cleans up the thread registry.

    Args:
        task_id: Task ID to unregister
        set_app_state: Optional AppState setter. If None, uses global registry.
    """
    _debug_print(f"→ unregister_task: task_id='{task_id}'")

    # Clean up thread registry
    thread = _THREAD_REGISTRY.pop(task_id, None)
    if thread:
        _debug_print(f"   Cleaned up thread for task_id (was_alive={thread.is_alive()})")
    else:
        _debug_print(f"   No thread found for task_id")

    if set_app_state:
        set_app_state(
            lambda prev: replace(
                prev,
                tasks={tid: t for tid, t in prev.tasks.items() if tid != task_id},
            )
        )
        _debug_print(f"✅ Task removed from AppState")
    else:
        _TASK_REGISTRY.pop(task_id, None)
        _debug_print(f"✅ Task removed from global registry")


def get_task_by_id(
    task_id: str,
    get_app_state: Optional[Callable] = None,
) -> Optional[InProcessTeammateTaskState]:
    """Get a task by ID from AppState or global registry.

    Args:
        task_id: Task ID to find
        get_app_state: Optional AppState getter. If None, uses global registry.

    Returns:
        InProcessTeammateTaskState or None
    """
    if get_app_state:
        state = get_app_state()
        task = state.tasks.get(task_id)
        if isinstance(task, InProcessTeammateTaskState):
            return task
        return None
    return _TASK_REGISTRY.get(task_id)


def find_task_by_agent_id(
    agent_id: str,
    get_app_state: Optional[Callable] = None,
) -> Optional[InProcessTeammateTaskState]:
    """Find a task by agent ID from AppState or global registry.

    Searches by identity.agent_id field.

    Args:
        agent_id: Agent ID to find (e.g., "researcher@my-team")
        get_app_state: Optional AppState getter. If None, uses global registry.

    Returns:
        InProcessTeammateTaskState or None
    """
    if get_app_state:
        state = get_app_state()
        tasks = state.tasks
        for task in tasks.values():
            if isinstance(task, InProcessTeammateTaskState):
                if task.identity.agent_id == agent_id:
                    return task
        return None
    else:
        for task in _TASK_REGISTRY.values():
            if task.identity.agent_id == agent_id:
                return task
        return None


class TaskManager:
    """Manager for background tasks.

    This coordinates all background task operations.
    """

    def __init__(self, store: "Store[AppState]"):
        """Initialize the task manager.

        Args:
            store: Application state store
        """
        self._store = store
        self._task_handles: dict[str, TaskHandle] = {}

    def spawn_task(
        self,
        task_type: TaskType,
        description: str,
        tool_use_id: Optional[str] = None,
    ) -> TaskStateBase:
        """Spawn a new background task.

        Args:
            task_type: Type of task
            description: Task description
            tool_use_id: Optional tool use ID

        Returns:
            Task state for the new task
        """
        task_id = generate_task_id(task_type)
        state = create_task_state_base(
            id=task_id,
            task_type=task_type,
            description=description,
            tool_use_id=tool_use_id,
        )

        # Add to app state
        self._store.set_state(
            lambda prev: replace(
                prev,
                tasks={**prev.tasks, task_id: state},
            )
        )

        return state

    async def kill_task(self, task_id: str) -> bool:
        """Kill a running task.

        Args:
            task_id: ID of task to kill

        Returns:
            True if task was killed, False if not found
        """
        # Get current task state
        state = self._store.get_state().tasks.get(task_id)
        if not state:
            return False

        if is_terminal_task_status(state.status):
            return False  # Already terminal

        # Update status
        self._update_task_status(task_id, TaskStatus.KILLED)

        # Call cleanup if available
        handle = self._task_handles.get(task_id)
        if handle and handle.cleanup:
            try:
                handle.cleanup()
            except Exception:
                pass

        return True

    def get_task(self, task_id: str) -> Optional[TaskStateBase]:
        """Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task state or None
        """
        return self._store.get_state().get("tasks", {}).get(task_id)

    def get_all_tasks(self) -> dict[str, TaskStateBase]:
        """Get all tasks.

        Returns:
            Dict of task ID to task state
        """
        return self._store.get_state().get("tasks", {})

    def get_tasks_by_status(self, status: TaskStatus) -> list[TaskStateBase]:
        """Get tasks by status.

        Args:
            status: Status to filter by

        Returns:
            List of matching tasks
        """
        tasks = self.get_all_tasks()
        return [t for t in tasks.values() if t.status == status]

    def get_running_tasks(self) -> list[TaskStateBase]:
        """Get all running tasks.

        Returns:
            List of running tasks
        """
        return self.get_tasks_by_status(TaskStatus.RUNNING)

    def _update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
    ) -> None:
        """Update a task's status.

        Args:
            task_id: Task ID
            status: New status
        """
        self._store.set_state(
            lambda prev: replace(
                prev,
                tasks={
                    **prev.tasks,
                    task_id: {
                        **prev.tasks.get(task_id, {}),
                        "status": status,
                        "end_time": time.time() if is_terminal_task_status(status) else None,
                    },
                },
            )
        )

    def _register_handle(
        self,
        task_id: str,
        handle: TaskHandle,
    ) -> None:
        """Register a task handle.

        Args:
            task_id: Task ID
            handle: Task handle
        """
        self._task_handles[task_id] = handle

    def _unregister_handle(self, task_id: str) -> None:
        """Unregister a task handle.

        Args:
            task_id: Task ID
        """
        self._task_handles.pop(task_id, None)


def get_task_output_path(task_id: str) -> str:
    """Get the output file path for a task.

    Args:
        task_id: Task ID

    Returns:
        Path to task output file
    """
    import os
    from pathlib import Path

    # Use temp directory for task outputs
    base_dir = Path(os.environ.get("CLAUDE_CODE_TASK_DIR", "/tmp/claude-code-tasks"))
    base_dir.mkdir(parents=True, exist_ok=True)

    return str(base_dir / f"{task_id}.log")


# =============================================================================
# In-process Teammate Spawning
# =============================================================================


async def spawn_in_process_teammate(
    config: InProcessSpawnConfig,
    context: dict[str, Any],
) -> InProcessSpawnOutput:
    """Spawn an in-process teammate.

    Creates the teammate's context, registers the task in AppState, and returns
    the spawn result. The actual agent execution is driven by the caller using
    run_with_teammate_context().

    Args:
        config: Spawn configuration
        context: Context with set_app_state, tool_use_id, session_id

    Returns:
        InProcessSpawnOutput with spawn result
    """
    from claude_code_py.utils.abort_controller import AbortController, create_abort_controller
    from claude_code_py.utils.teammate_context import create_teammate_context, TeammateContext
    from claude_code_py.task.base import generate_task_id

    set_app_state = context.get("set_app_state")
    tool_use_id = context.get("tool_use_id")
    parent_session_id = context.get("session_id", "default")

    # Generate agent ID
    agent_id = generate_agent_id(config.name, config.team_name)
    task_id = generate_task_id(TaskType.IN_PROCESS_TEAMMATE)

    # Create abort controller (independent for teammates)
    abort_controller = create_abort_controller()

    # Create teammate identity
    identity = TeammateIdentity(
        agent_id=agent_id,
        agent_name=config.name,
        team_name=config.team_name,
        color=config.color,
        plan_mode_required=config.plan_mode_required,
        parent_session_id=parent_session_id,
    )

    # Create teammate context for contextvars
    teammate_context = create_teammate_context(
        agent_id=agent_id,
        agent_name=config.name,
        team_name=config.team_name,
        color=config.color,
        plan_mode_required=config.plan_mode_required,
        parent_session_id=parent_session_id,
        abort_controller=abort_controller,
    )

    # Create task state
    task_state = create_in_process_teammate_state(
        task_id=task_id,
        identity=identity,
        prompt=config.prompt,
        model=config.model,
        abort_controller=abort_controller,
        tool_use_id=tool_use_id,
    )

    # Register task in AppState via set_app_state
    if set_app_state:
        set_app_state(
            lambda prev: replace(
                prev,
                tasks={**prev.tasks, task_id: task_state},
            )
        )

    return InProcessSpawnOutput(
        success=True,
        agent_id=agent_id,
        task_id=task_id,
        abort_controller=abort_controller,
        teammate_context=teammate_context,
    )


def update_teammate_task_status(
    set_app_state: Callable,
    task_id: str,
    status: TaskStatus,
    is_idle: Optional[bool] = None,
) -> None:
    """Update a teammate task's status.

    Args:
        set_app_state: State updater function
        task_id: Task ID
        status: New status
        is_idle: Optional idle flag
    """
    def updater(prev: AppState) -> AppState:
        tasks = prev.tasks
        task = tasks.get(task_id)
        if not task:
            return prev

        # Handle both dataclass and dict
        if hasattr(task, '__dataclass_fields__'):
            # dataclass - use replace
            updates = {"status": status}
            if is_idle is not None:
                updates["is_idle"] = is_idle
            if is_terminal_task_status(status):
                updates["end_time"] = time.time()
            new_task = replace(task, **updates)
        else:
            # dict - use dict update
            updates: dict[str, Any] = {"status": status}
            if is_idle is not None:
                updates["is_idle"] = is_idle
            if is_terminal_task_status(status):
                updates["end_time"] = time.time()
            new_task = {**task, **updates}

        return replace(
            prev,
            tasks={**tasks, task_id: new_task},
        )

    set_app_state(updater)


def add_pending_user_message(
    set_app_state: Callable,
    task_id: str,
    message: dict,
) -> None:
    """Add a pending user message to a teammate task.

    Args:
        set_app_state: State updater function
        task_id: Task ID
        message: Message to add
    """
    def updater(prev: AppState) -> AppState:
        tasks = prev.tasks
        task = tasks.get(task_id)
        if not task:
            return prev

        # Handle both dataclass and dict
        if hasattr(task, '__dataclass_fields__'):
            # dataclass - use attribute access and replace
            pending = list(task.pending_user_messages or [])
            pending.append(message)
            new_task = replace(task, pending_user_messages=pending)
        else:
            # dict - use .get() and dict update
            pending = list(task.get("pending_user_messages", []))
            pending.append(message)
            new_task = {**task, "pending_user_messages": pending}

        return replace(
            prev,
            tasks={**tasks, task_id: new_task},
        )

    set_app_state(updater)


# =============================================================================
# Spawn Configuration Types (Task V2)
# =============================================================================


@dataclass
@dataclass
class SpawnTeammateConfig:
    """Configuration for spawning an in-process teammate (Task V2).

    This is the config type used by spawn_in_process_teammate() with
    explicit set_app_state/get_app_state callbacks.
    """

    name: str  # Display name for the teammate
    team_name: str  # Team name
    prompt: str  # Initial prompt for the teammate
    description: Optional[str] = None  # Optional task description
    model: Optional[str] = None  # Optional model override
    color: Optional[str] = None  # Optional UI color
    plan_mode_required: bool = False  # Whether plan mode is required
    parent_session_id: Optional[str] = None  # Parent session ID
    tool_use_id: Optional[str] = None  # Tool use ID for correlation
    agent_type: Optional[str] = None  # Optional agent type specifier
    cwd: Optional[str] = None  # Working directory for teammate


@dataclass
class SpawnTeammateResult:
    """Result from spawning an in-process teammate."""

    success: bool
    agent_id: str  # Full agent ID (e.g., "researcher@my-team")
    task_id: Optional[str] = None  # Task ID if spawned successfully
    error: Optional[str] = None  # Error message if failed


# =============================================================================
# Spawn and Stop Functions (Task V2)
# =============================================================================


async def spawn_in_process_teammate_v2(
    config: SpawnTeammateConfig,
    set_app_state: Optional[Callable] = None,
    get_app_state: Optional[Callable] = None,
) -> SpawnTeammateResult:
    """Spawn an in-process teammate and start its runner.

    This is the main spawn function that:
    1. Generates agent_id using generate_agent_id()
    2. Generates task_id using generate_task_id()
    3. Creates abort_controller
    4. Creates TeammateIdentity
    5. Creates teammate_context using create_teammate_context()
    6. Creates task_state using create_in_process_teammate_state()
    7. Registers task in AppState or global registry
    8. Builds InProcessRunnerConfig
    9. Starts runner with start_in_process_teammate()
    10. Returns SpawnTeammateResult

    Args:
        config: Spawn configuration
        set_app_state: Optional AppState setter
        get_app_state: Optional AppState getter

    Returns:
        SpawnTeammateResult with spawn outcome
    """
    from claude_code_py.utils.abort_controller import create_abort_controller
    from claude_code_py.utils.teammate_context import create_teammate_context
    from claude_code_py.utils.swarm.spawn_in_process import generate_task_id as generate_swarm_task_id
    from claude_code_py.utils.swarm.in_process_runner import InProcessRunnerConfig, start_in_process_teammate

    _debug_print("=" * 60)
    _debug_print("spawn_in_process_teammate_v2: STARTING")
    _debug_print(f"  name: '{config.name}'")
    _debug_print(f"  team_name: '{config.team_name}'")
    _debug_print(f"  prompt: '{config.prompt[:80]}{'...' if len(config.prompt) > 80 else ''}'")
    _debug_print(f"  cwd: '{config.cwd}'")
    _debug_print(f"  model: '{config.model}'")
    _debug_print(f"  plan_mode_required: {config.plan_mode_required}")
    _debug_print(f"  set_app_state: {set_app_state is not None}")
    _debug_print(f"  get_app_state: {get_app_state is not None}")
    _debug_print("=" * 60)

    try:
        # Step a: Generate agent_id
        agent_id = generate_agent_id(config.name, config.team_name)
        _debug_print(f"[Step a] agent_id = '{agent_id}'")

        # Step b: Generate task_id
        task_id = generate_swarm_task_id("in_process_teammate")
        _debug_print(f"[Step b] task_id = '{task_id}'")

        # Step c: Create abort_controller
        abort_controller = create_abort_controller()
        _debug_print(f"[Step c] abort_controller created")

        # Step d: Create TeammateIdentity
        parent_session_id = config.parent_session_id or "default"
        identity = TeammateIdentity(
            agent_id=agent_id,
            agent_name=config.name,
            team_name=config.team_name,
            parent_session_id=parent_session_id,
            color=config.color,
            plan_mode_required=config.plan_mode_required,
        )
        _debug_print(f"[Step d] TeammateIdentity created")
        _debug_print(f"         parent_session_id='{parent_session_id}'")

        # Step e: Create teammate_context
        teammate_context = create_teammate_context(
            agent_id=agent_id,
            agent_name=config.name,
            team_name=config.team_name,
            parent_session_id=parent_session_id,
            abort_controller=abort_controller,
            color=config.color,
            plan_mode_required=config.plan_mode_required,
        )
        _debug_print(f"[Step e] teammate_context created")
        _debug_print(f"         is_in_process={teammate_context.is_in_process}")

        # Step f: Create task_state
        task_state = create_in_process_teammate_state(
            task_id=task_id,
            identity=identity,
            prompt=config.prompt,
            model=config.model,
            abort_controller=abort_controller,
            tool_use_id=config.tool_use_id,
        )
        _debug_print(f"[Step f] task_state created")
        _debug_print(f"         status='{task_state.status}'")
        _debug_print(f"         is_idle={task_state.is_idle}")

        # Step g: Register task
        _debug_print(f"[Step g] Registering task...")
        register_task(task_state, set_app_state)

        # Step h: Build InProcessRunnerConfig
        # Note: tool_use_context is required by InProcessRunnerConfig
        # Create minimal tool use context with the provided callbacks
        from claude_code_py.tool.context import create_default_tool_use_context

        _debug_print(f"[Step h] Creating tool_use_context...")
        # Create a minimal context - we'll set the required fields manually
        # First create a default context with empty tools
        tool_use_context = create_default_tool_use_context(
            tools=[],  # Empty tools list for teammate
            abort_controller=abort_controller,
            cwd=config.cwd or ".",  # Use config's cwd
        )

        # Set the required callbacks
        tool_use_context.get_app_state = get_app_state or (lambda: {})
        tool_use_context.set_app_state = set_app_state or (lambda _: None)
        tool_use_context.tool_use_id = config.tool_use_id
        _debug_print(f"         tool_use_context.cwd='{tool_use_context.get_cwd()}'")

        runner_config = InProcessRunnerConfig(
            identity=identity,
            task_id=task_id,
            prompt=config.prompt,
            tool_use_context=tool_use_context,
            abort_controller=abort_controller,
            description=config.description,
            teammate_context=teammate_context,
            model=config.model,
        )
        _debug_print(f"         InProcessRunnerConfig built")

        # Step i: Start runner and save thread handle
        _debug_print(f"[Step i] Starting in_process_teammate runner...")
        thread = start_in_process_teammate(runner_config)
        _THREAD_REGISTRY[task_id] = thread
        _debug_print(f"         Thread registered for task_id='{task_id}'")
        _debug_print(f"         thread.is_alive()={thread.is_alive()}")

        _debug_print("=" * 60)
        _debug_print("🎉 SPAWN SUCCESSFUL!")
        _debug_print(f"  agent_id: '{agent_id}'")
        _debug_print(f"  task_id: '{task_id}'")
        _debug_print("=" * 60)

        # Step j: Return SpawnTeammateResult
        return SpawnTeammateResult(
            success=True,
            agent_id=agent_id,
            task_id=task_id,
        )

    except Exception as e:
        _debug_print("=" * 60)
        _debug_print("❌ SPAWN FAILED!")
        _debug_print(f"  Exception: {type(e).__name__}: {e}")
        _debug_print("=" * 60)
        return SpawnTeammateResult(
            success=False,
            agent_id=generate_agent_id(config.name, config.team_name),
            error=str(e),
        )


def stop_teammate(
    agent_id: str,
    set_app_state: Optional[Callable] = None,
    get_app_state: Optional[Callable] = None,
) -> bool:
    """Stop an in-process teammate by agent ID.

    Aborts the teammate's controller, which signals the teammate loop to exit.
    The thread will naturally terminate when it detects abort_controller.aborted.

    Args:
        agent_id: Agent ID to stop (e.g., "researcher@my-team")
        set_app_state: Optional AppState setter
        get_app_state: Optional AppState getter

    Returns:
        True if teammate was stopped, False if not found
    """
    # Find the task by agent_id
    task_state = find_task_by_agent_id(agent_id, get_app_state)

    if not task_state:
        return False

    task_id = task_state.id

    # Abort the controller (signals teammate to exit its loop)
    task_state.abort_controller.abort()
    _debug_print(f"→ stop_teammate: aborted controller for '{agent_id}'")

    # Clean up thread registry (thread will exit on its own)
    thread = _THREAD_REGISTRY.pop(task_id, None)
    if thread:
        _debug_print(f"   Thread removed from registry (was_alive={thread.is_alive()})")

    # Update status to COMPLETED
    if set_app_state:
        update_teammate_task_status(
            set_app_state,
            task_id,
            TaskStatus.COMPLETED,
        )
    else:
        # Update in global registry directly
        task_state.status = TaskStatus.COMPLETED
        task_state.end_time = time.time()

    return True