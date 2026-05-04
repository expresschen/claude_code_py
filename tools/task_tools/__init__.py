"""Task tools package.

Provides TaskCreate, TaskUpdate, TaskList, TaskGet, TaskStop, TaskClaim tools
for managing a structured task list during coding sessions.

Features:
- File-based task storage with locking
- High water mark to prevent ID reuse
- Task dependencies (blocks/blockedBy)
- Atomic task claiming with agent busy check
"""

from __future__ import annotations

from .constants import (
    TASK_CREATE_TOOL_NAME,
    TASK_UPDATE_TOOL_NAME,
    TASK_LIST_TOOL_NAME,
    TASK_GET_TOOL_NAME,
    TASK_STOP_TOOL_NAME,
    TASK_STATUS_PENDING,
    TASK_STATUS_IN_PROGRESS,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_DELETED,
    VALID_TASK_STATUSES,
    TASK_CREATE_DESCRIPTION,
    TASK_UPDATE_DESCRIPTION,
    TASK_LIST_DESCRIPTION,
    TASK_GET_DESCRIPTION,
    TASK_STOP_DESCRIPTION,
)

from .prompt import (
    get_task_create_prompt,
    get_task_update_prompt,
    get_task_list_prompt,
    get_task_get_prompt,
    get_task_stop_prompt,
)

from .tool import (
    TaskCreateTool,
    TaskUpdateTool,
    TaskListTool,
    TaskGetTool,
    TaskStopTool,
    TaskClaimTool,
    TaskCreateInput,
    TaskUpdateInput,
    TaskListInput,
    TaskGetInput,
    TaskStopInput,
    TaskClaimInput,
    TaskCreateOutput,
    TaskUpdateOutput,
    TaskListOutput,
    TaskGetOutput,
    TaskStopOutput,
    TaskClaimOutput,
    task_create_tool,
    task_update_tool,
    task_list_tool,
    task_get_tool,
    task_stop_tool,
    task_claim_tool,
)


__all__ = [
    # Constants
    "TASK_CREATE_TOOL_NAME",
    "TASK_UPDATE_TOOL_NAME",
    "TASK_LIST_TOOL_NAME",
    "TASK_GET_TOOL_NAME",
    "TASK_STOP_TOOL_NAME",
    "TASK_STATUS_PENDING",
    "TASK_STATUS_IN_PROGRESS",
    "TASK_STATUS_COMPLETED",
    "TASK_STATUS_DELETED",
    "VALID_TASK_STATUSES",
    # Descriptions
    "TASK_CREATE_DESCRIPTION",
    "TASK_UPDATE_DESCRIPTION",
    "TASK_LIST_DESCRIPTION",
    "TASK_GET_DESCRIPTION",
    "TASK_STOP_DESCRIPTION",
    # Prompt functions
    "get_task_create_prompt",
    "get_task_update_prompt",
    "get_task_list_prompt",
    "get_task_get_prompt",
    "get_task_stop_prompt",
    # Tools
    "TaskCreateTool",
    "TaskUpdateTool",
    "TaskListTool",
    "TaskGetTool",
    "TaskStopTool",
    "TaskClaimTool",
    # Input/Output types
    "TaskCreateInput",
    "TaskUpdateInput",
    "TaskListInput",
    "TaskGetInput",
    "TaskStopInput",
    "TaskClaimInput",
    "TaskCreateOutput",
    "TaskUpdateOutput",
    "TaskListOutput",
    "TaskGetOutput",
    "TaskStopOutput",
    "TaskClaimOutput",
    # Tool instances
    "task_create_tool",
    "task_update_tool",
    "task_list_tool",
    "task_get_tool",
    "task_stop_tool",
    "task_claim_tool",
]