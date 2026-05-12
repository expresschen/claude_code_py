"""Task tools implementation.

Implements TaskCreateTool, TaskUpdateTool, TaskListTool, TaskGetTool, TaskStopTool.

Ported from TypeScript:
- src/tools/TaskCreateTool/TaskCreateTool.ts
- src/tools/TaskUpdateTool/TaskUpdateTool.ts
- src/tools/TaskListTool/TaskListTool.ts
- src/tools/TaskGetTool/TaskGetTool.ts
- src/tools/TaskStopTool/TaskStopTool.ts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List, Dict, TYPE_CHECKING

from pydantic import BaseModel, Field

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError

from .constants import (
    TASK_CREATE_TOOL_NAME,
    TASK_UPDATE_TOOL_NAME,
    TASK_LIST_TOOL_NAME,
    TASK_GET_TOOL_NAME,
    TASK_STOP_TOOL_NAME,
)
from .prompt import (
    get_task_create_prompt,
    get_task_update_prompt,
    get_task_list_prompt,
    get_task_get_prompt,
    get_task_stop_prompt,
)

if TYPE_CHECKING:
    from claude_code_py.utils.task.file_storage import Task, TaskStatus


# =============================================================================
# Input Schemas
# =============================================================================


class TaskCreateInput(BaseModel):
    """Input for TaskCreate tool."""

    subject: str = Field(description="A brief title for the task")
    description: str = Field(description="What needs to be done")
    activeForm: Optional[str] = Field(
        default=None,
        description="Present continuous form shown in spinner when in_progress (e.g., 'Running tests')",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary metadata to attach to the task",
    )


class TaskUpdateInput(BaseModel):
    """Input for TaskUpdate tool."""

    taskId: str = Field(description="The ID of the task to update")
    subject: Optional[str] = Field(default=None, description="New subject for the task")
    description: Optional[str] = Field(default=None, description="New description for the task")
    activeForm: Optional[str] = Field(
        default=None,
        description="Present continuous form shown in spinner when in_progress (e.g., 'Running tests')",
    )
    status: Optional[str] = Field(
        default=None,
        description="New status for the task (pending, in_progress, completed, or deleted)",
    )
    owner: Optional[str] = Field(default=None, description="New owner for the task")
    addBlocks: Optional[List[str]] = Field(
        default=None,
        description="Task IDs that this task blocks",
    )
    addBlockedBy: Optional[List[str]] = Field(
        default=None,
        description="Task IDs that block this task",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata keys to merge into the task. Set a key to null to delete it.",
    )


class TaskListInput(BaseModel):
    """Input for TaskList tool."""

    # No input fields - empty schema
    pass


class TaskGetInput(BaseModel):
    """Input for TaskGet tool."""

    taskId: str = Field(description="The ID of the task to retrieve")


class TaskStopInput(BaseModel):
    """Input for TaskStop tool."""

    task_id: str = Field(description="The ID of the background task to stop")


class TaskClaimInput(BaseModel):
    """Input for claiming a task."""

    taskId: str = Field(description="The ID of the task to claim")
    checkAgentBusy: Optional[bool] = Field(
        default=False,
        description="If true, check if agent is already busy with other tasks",
    )


# =============================================================================
# Output Types
# =============================================================================


@dataclass
class TaskCreateOutput:
    """Output from TaskCreate tool."""

    task_id: str
    subject: str


@dataclass
class TaskUpdateOutput:
    """Output from TaskUpdate tool."""

    success: bool
    task_id: str
    updated_fields: List[str]
    error: Optional[str] = None
    status_change: Optional[Dict[str, str]] = None


@dataclass
class TaskListOutput:
    """Output from TaskList tool."""

    tasks: List[Dict[str, Any]]


@dataclass
class TaskGetOutput:
    """Output from TaskGet tool."""

    task: Dict[str, Any]


@dataclass
class TaskStopOutput:
    """Output from TaskStop tool."""

    success: bool
    task_id: str


@dataclass
class TaskClaimOutput:
    """Output from TaskClaim tool."""

    success: bool
    task_id: str
    reason: Optional[str] = None
    task: Optional[Dict[str, Any]] = None
    blocked_by_tasks: Optional[List[str]] = None
    busy_with_tasks: Optional[List[str]] = None


# =============================================================================
# Tool Implementations
# =============================================================================


class TaskCreateTool(Tool[TaskCreateInput, TaskCreateOutput, Dict[str, Any]]):
    """Tool for creating tasks."""

    name = TASK_CREATE_TOOL_NAME
    input_schema = TaskCreateInput

    async def call(
        self,
        args: TaskCreateInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[TaskCreateOutput]:
        """Create a new task."""
        from claude_code_py.utils.task import (
            create_task,
            get_task_list_id,
            is_task_v2_enabled,
        )

        if not is_task_v2_enabled():
            raise ToolError("Task system is not enabled")

        task_list_id = get_task_list_id()

        # Create the task
        task_id = create_task(
            task_list_id,
            subject=args.subject,
            description=args.description,
            activeForm=args.activeForm,
            metadata=args.metadata,
        )

        output = TaskCreateOutput(
            task_id=task_id,
            subject=args.subject,
        )

        return ToolResult(data=output)

    async def description(self, input: TaskCreateInput, options: Dict[str, Any]) -> str:
        """Get tool description."""
        return f"Create task: {input.subject}"

    async def prompt(self, options: Dict[str, Any]) -> str:
        """Get tool prompt."""
        return get_task_create_prompt()

    def is_concurrency_safe(self, input: TaskCreateInput) -> bool:
        """Creating tasks is concurrency safe."""
        return True

    def to_auto_classifier_input(self, input: TaskCreateInput) -> str:
        """Get text for classifier."""
        return input.subject


class TaskUpdateTool(Tool[TaskUpdateInput, TaskUpdateOutput, Dict[str, Any]]):
    """Tool for updating tasks."""

    name = TASK_UPDATE_TOOL_NAME
    input_schema = TaskUpdateInput

    async def call(
        self,
        args: TaskUpdateInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[TaskUpdateOutput]:
        """Update a task."""
        from claude_code_py.utils.task import (
            get_task,
            get_task_list_id,
            update_task,
            delete_task,
            block_task,
            is_task_v2_enabled,
            TaskStatus,
            task_to_dict,
        )

        if not is_task_v2_enabled():
            raise ToolError("Task system is not enabled")

        task_list_id = get_task_list_id()

        # Check if task exists
        existing_task = get_task(task_list_id, args.taskId)
        if not existing_task:
            output = TaskUpdateOutput(
                success=False,
                task_id=args.taskId,
                updated_fields=[],
                error="Task not found",
            )
            return ToolResult(data=output)

        updated_fields: List[str] = []

        # Handle deletion
        if args.status == "deleted":
            deleted = delete_task(task_list_id, args.taskId)
            output = TaskUpdateOutput(
                success=deleted,
                task_id=args.taskId,
                updated_fields=["deleted"] if deleted else [],
                error=None if deleted else "Failed to delete task",
                status_change={"from": existing_task.status.value, "to": "deleted"} if deleted else None,
            )
            return ToolResult(data=output)

        # Build updates dict
        updates: Dict[str, Any] = {}

        if args.subject is not None and args.subject != existing_task.subject:
            updates["subject"] = args.subject
            updated_fields.append("subject")

        if args.description is not None and args.description != existing_task.description:
            updates["description"] = args.description
            updated_fields.append("description")

        if args.activeForm is not None and args.activeForm != existing_task.activeForm:
            updates["activeForm"] = args.activeForm
            updated_fields.append("activeForm")

        if args.owner is not None and args.owner != existing_task.owner:
            updates["owner"] = args.owner
            updated_fields.append("owner")

        if args.metadata is not None:
            updates["metadata"] = args.metadata
            updated_fields.append("metadata")

        if args.status is not None and args.status != existing_task.status.value:
            updates["status"] = TaskStatus(args.status)
            updated_fields.append("status")

        # Apply updates
        if updates:
            updated_task = update_task(
                task_list_id,
                args.taskId,
                **updates
            )

        # Handle blocks/blockedBy relationships
        if args.addBlocks:
            new_blocks = [id for id in args.addBlocks if id not in existing_task.blocks]
            for block_id in new_blocks:
                block_task(task_list_id, args.taskId, block_id)
            if new_blocks:
                updated_fields.append("blocks")

        if args.addBlockedBy:
            new_blocked_by = [id for id in args.addBlockedBy if id not in existing_task.blockedBy]
            for blocker_id in new_blocked_by:
                block_task(task_list_id, blocker_id, args.taskId)
            if new_blocked_by:
                updated_fields.append("blockedBy")

        status_change = None
        if "status" in updated_fields:
            status_change = {"from": existing_task.status.value, "to": args.status}

        output = TaskUpdateOutput(
            success=True,
            task_id=args.taskId,
            updated_fields=updated_fields,
            status_change=status_change,
        )

        return ToolResult(data=output)

    async def description(self, input: TaskUpdateInput, options: Dict[str, Any]) -> str:
        """Get tool description."""
        parts = [input.taskId]
        if input.status:
            parts.append(f"-> {input.status}")
        return f"Update task: {' '.join(parts)}"

    async def prompt(self, options: Dict[str, Any]) -> str:
        """Get tool prompt."""
        return get_task_update_prompt()

    def is_concurrency_safe(self, input: TaskUpdateInput) -> bool:
        """Updating tasks is concurrency safe."""
        return True

    def to_auto_classifier_input(self, input: TaskUpdateInput) -> str:
        """Get text for classifier."""
        parts = [input.taskId]
        if input.status:
            parts.append(input.status)
        if input.subject:
            parts.append(input.subject)
        return " ".join(parts)


class TaskListTool(Tool[TaskListInput, TaskListOutput, Dict[str, Any]]):
    """Tool for listing tasks."""

    name = TASK_LIST_TOOL_NAME
    input_schema = TaskListInput

    async def call(
        self,
        args: TaskListInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[TaskListOutput]:
        """List all tasks."""
        from claude_code_py.utils.task import (
            list_tasks,
            get_task_list_id,
            is_task_v2_enabled,
            task_to_dict,
            TaskStatus,
        )

        if not is_task_v2_enabled():
            raise ToolError("Task system is not enabled")

        task_list_id = get_task_list_id()

        all_tasks = list_tasks(task_list_id)

        # Filter out internal metadata tasks
        visible_tasks = [t for t in all_tasks if not t.metadata.get("_internal")]

        # Build output with resolved blockers
        resolved_ids = set(t.id for t in visible_tasks if t.status == TaskStatus.COMPLETED)

        task_summaries = []
        for task in visible_tasks:
            # Only show unresolved blockers
            blocked_by = [id for id in task.blockedBy if id not in resolved_ids]
            task_summaries.append({
                "id": task.id,
                "subject": task.subject,
                "status": task.status.value,
                "owner": task.owner,
                "blockedBy": blocked_by,
                "blocks": task.blocks,
            })

        output = TaskListOutput(tasks=task_summaries)
        return ToolResult(data=output)

    async def description(self, input: TaskListInput, options: Dict[str, Any]) -> str:
        """Get tool description."""
        return "List all tasks"

    async def prompt(self, options: Dict[str, Any]) -> str:
        """Get tool prompt."""
        return get_task_list_prompt()

    def is_concurrency_safe(self, input: TaskListInput) -> bool:
        """Listing is concurrency safe."""
        return True

    def is_read_only(self, input: TaskListInput) -> bool:
        """Listing is read-only."""
        return True


class TaskGetTool(Tool[TaskGetInput, TaskGetOutput, Dict[str, Any]]):
    """Tool for getting a single task."""

    name = TASK_GET_TOOL_NAME
    input_schema = TaskGetInput

    async def call(
        self,
        args: TaskGetInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[TaskGetOutput]:
        """Get a task by ID."""
        from claude_code_py.utils.task import (
            get_task,
            get_task_list_id,
            is_task_v2_enabled,
            task_to_dict,
        )

        if not is_task_v2_enabled():
            raise ToolError("Task system is not enabled")

        task_list_id = get_task_list_id()

        task = get_task(task_list_id, args.taskId)
        if not task:
            raise ToolError(f"Task {args.taskId} not found")

        output = TaskGetOutput(task=task_to_dict(task))
        return ToolResult(data=output)

    async def description(self, input: TaskGetInput, options: Dict[str, Any]) -> str:
        """Get tool description."""
        return f"Get task {input.taskId}"

    async def prompt(self, options: Dict[str, Any]) -> str:
        """Get tool prompt."""
        return get_task_get_prompt()

    def is_concurrency_safe(self, input: TaskGetInput) -> bool:
        """Getting is concurrency safe."""
        return True

    def is_read_only(self, input: TaskGetInput) -> bool:
        """Getting is read-only."""
        return True


class TaskStopTool(Tool[TaskStopInput, TaskStopOutput, Dict[str, Any]]):
    """Tool for stopping a background task."""

    name = TASK_STOP_TOOL_NAME
    input_schema = TaskStopInput

    async def call(
        self,
        args: TaskStopInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[TaskStopOutput]:
        """Stop a running background task.

        This integrates with TaskManager to kill background shell/agent tasks.
        """
        # In full implementation, this would:
        # 1. Look up the task in TaskManager
        # 2. Call task.kill() or abort_controller.abort()
        # 3. Update task status to KILLED

        # For now, return success for stub
        output = TaskStopOutput(
            success=True,
            task_id=args.task_id,
        )
        return ToolResult(data=output)

    async def description(self, input: TaskStopInput, options: Dict[str, Any]) -> str:
        """Get tool description."""
        return f"Stop task {input.task_id}"

    async def prompt(self, options: Dict[str, Any]) -> str:
        """Get tool prompt."""
        return get_task_stop_prompt()

    def is_concurrency_safe(self, input: TaskStopInput) -> bool:
        """Stopping is concurrency safe."""
        return True


class TaskClaimTool(Tool[TaskClaimInput, TaskClaimOutput, Dict[str, Any]]):
    """Tool for atomically claiming a task for an agent."""

    name = "TaskClaim"
    input_schema = TaskClaimInput

    async def call(
        self,
        args: TaskClaimInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[TaskClaimOutput]:
        """Claim a task atomically."""
        from claude_code_py.utils.task import (
            claim_task,
            get_task_list_id,
            is_task_v2_enabled,
            task_to_dict,
            get_current_agent_id,
        )

        if not is_task_v2_enabled():
            raise ToolError("Task system is not enabled")

        task_list_id = get_task_list_id()

        # Get agent ID from context or use a default
        agent_id = get_current_agent_id()
        if not agent_id:
            # Use tool_use_id as agent identifier for standalone sessions
            agent_id = context.tool_use_id or "main"

        result = claim_task(
            task_list_id,
            args.taskId,
            agent_id,
            check_agent_busy=args.checkAgentBusy,
        )

        output = TaskClaimOutput(
            success=result.success,
            task_id=args.taskId,
            reason=result.reason,
            task=task_to_dict(result.task) if result.task else None,
            blocked_by_tasks=result.blocked_by_tasks,
            busy_with_tasks=result.busy_with_tasks,
        )

        return ToolResult(data=output)

    async def description(self, input: TaskClaimInput, options: Dict[str, Any]) -> str:
        """Get tool description."""
        return f"Claim task {input.taskId}"

    async def prompt(self, options: Dict[str, Any]) -> str:
        """Get tool prompt."""
        return "Atomically claim a task for the current agent. Use checkAgentBusy=true to prevent claiming when already working on other tasks."

    def is_concurrency_safe(self, input: TaskClaimInput) -> bool:
        """Claiming needs coordination."""
        return False  # Uses internal locking


# =============================================================================
# Tool Instances
# =============================================================================


task_create_tool = TaskCreateTool()
task_update_tool = TaskUpdateTool()
task_list_tool = TaskListTool()
task_get_tool = TaskGetTool()
task_stop_tool = TaskStopTool()
task_claim_tool = TaskClaimTool()