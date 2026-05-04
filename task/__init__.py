"""Task system for background operations."""

from .base import TaskStateBase, TaskHandle, generate_task_id, create_task_state_base
from .types import TaskType, TaskStatus, is_terminal_task_status
from .manager import (
    TaskManager,
    spawn_in_process_teammate,
    spawn_in_process_teammate_v2,
    update_teammate_task_status,
    register_task,
    unregister_task,
    get_task_by_id,
    find_task_by_agent_id,
    stop_teammate,
    SpawnTeammateConfig,
    SpawnTeammateResult,
    _TASK_REGISTRY,
)
from .in_process_teammate import (
    TeammateIdentity,
    InProcessTeammateTaskState,
    InProcessSpawnConfig,
    InProcessSpawnOutput,
    generate_agent_id,
    create_in_process_teammate_state,
    find_teammate_task_by_agent_id,
    is_in_process_teammate_task,
)

__all__ = [
    "TaskStateBase",
    "TaskHandle",
    "generate_task_id",
    "create_task_state_base",
    "TaskType",
    "TaskStatus",
    "is_terminal_task_status",
    "TaskManager",
    "spawn_in_process_teammate",
    "spawn_in_process_teammate_v2",
    "update_teammate_task_status",
    "register_task",
    "unregister_task",
    "get_task_by_id",
    "find_task_by_agent_id",
    "stop_teammate",
    "SpawnTeammateConfig",
    "SpawnTeammateResult",
    "_TASK_REGISTRY",
    "TeammateIdentity",
    "InProcessTeammateTaskState",
    "InProcessSpawnConfig",
    "InProcessSpawnOutput",
    "generate_agent_id",
    "create_in_process_teammate_state",
    "find_teammate_task_by_agent_id",
    "is_in_process_teammate_task",
]