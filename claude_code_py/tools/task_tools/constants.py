"""Task tools constants."""

from __future__ import annotations

# Tool names
TASK_CREATE_TOOL_NAME = "TaskCreate"
TASK_UPDATE_TOOL_NAME = "TaskUpdate"
TASK_LIST_TOOL_NAME = "TaskList"
TASK_GET_TOOL_NAME = "TaskGet"
TASK_STOP_TOOL_NAME = "TaskStop"

# Task status values
TASK_STATUS_PENDING = "pending"
TASK_STATUS_IN_PROGRESS = "in_progress"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_DELETED = "deleted"

# Valid task statuses
VALID_TASK_STATUSES = [
    TASK_STATUS_PENDING,
    TASK_STATUS_IN_PROGRESS,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_DELETED,
]

# Description strings
TASK_CREATE_DESCRIPTION = "Create a new task in the task list"
TASK_UPDATE_DESCRIPTION = "Update a task in the task list"
TASK_LIST_DESCRIPTION = "List all tasks in the task list"
TASK_GET_DESCRIPTION = "Get a task by its ID from the task list"
TASK_STOP_DESCRIPTION = "Stop a running background task by its ID"