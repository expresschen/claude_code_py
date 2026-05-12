"""Task V2 system with file-based storage.

This module provides the complete Task V2 implementation:
- File-based storage with JSON files
- Cross-process file locking (fcntl/portalocker)
- High water mark to prevent ID reuse after deletion/reset
- Task CRUD operations (create, get, update, delete, list)
- Task dependencies (blocks/blockedBy)
- Atomic task claiming with busy check
- Agent status tracking based on task ownership
- Teammate task unassignment on shutdown

Ported from TypeScript: src/utils/tasks.ts
"""

from __future__ import annotations

from .file_storage import (
    # Types
    TaskStatus,
    Task,
    TeamMember,
    AgentStatus,
    ClaimTaskResult,
    UnassignTasksResult,

    # Locking
    FileLock,
    AsyncFileLock,

    # Path utilities
    sanitize_path_component,
    get_tasks_dir,
    get_task_path,
    get_task_list_lock_path,
    ensure_tasks_dir,
    ensure_task_list_lock_file,

    # High water mark
    read_high_water_mark,
    write_high_water_mark,
    find_highest_task_id,
    find_highest_task_id_from_files,

    # Task list ID
    get_task_list_id,
    is_task_v2_enabled,
    set_leader_team_name,
    clear_leader_team_name,

    # CRUD operations
    create_task,
    create_task_async,
    get_task,
    update_task,
    update_task_async,
    update_task_unsafe,
    delete_task,
    list_tasks,
    reset_task_list,

    # Blocking relationships
    block_task,

    # Atomic claiming
    claim_task,
    claim_task_async,

    # Agent status tracking
    get_agent_statuses,

    # Teammate unassignment
    unassign_teammate_tasks,
    unassign_teammate_tasks_async,

    # Serialization
    task_to_dict,
    dict_to_task,

    # Notifications
    on_tasks_updated,
    notify_tasks_updated,

    # Constants
    TASK_STATUSES,
    DEFAULT_TASKS_MODE_TASK_LIST_ID,
    HIGH_WATER_MARK_FILE,
    LOCK_FILE,
)

from .disk_output import (
    get_task_output_path,
    TaskOutputWriter,
)

# Re-export for convenience
task_to_json = task_to_dict
json_to_task = dict_to_task


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

    # CRUD operations
    "create_task",
    "create_task_async",
    "get_task",
    "update_task",
    "update_task_async",
    "update_task_unsafe",
    "delete_task",
    "list_tasks",
    "reset_task_list",

    # Blocking relationships
    "block_task",

    # Atomic claiming
    "claim_task",
    "claim_task_async",

    # Agent status tracking
    "get_agent_statuses",

    # Teammate unassignment
    "unassign_teammate_tasks",
    "unassign_teammate_tasks_async",

    # Serialization
    "task_to_dict",
    "dict_to_task",
    "task_to_json",
    "json_to_task",

    # Notifications
    "on_tasks_updated",
    "notify_tasks_updated",

    # Disk output (from V1 task system)
    "get_task_output_path",
    "TaskOutputWriter",

    # Constants
    "TASK_STATUSES",
    "DEFAULT_TASKS_MODE_TASK_LIST_ID",
    "HIGH_WATER_MARK_FILE",
    "LOCK_FILE",
]