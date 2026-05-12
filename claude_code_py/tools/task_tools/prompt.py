"""Task tools prompts.

Ported from TypeScript:
- src/tools/TaskCreateTool/prompt.ts
- src/tools/TaskUpdateTool/prompt.ts
- src/tools/TaskListTool/prompt.ts
"""

from __future__ import annotations

from .constants import (
    TASK_CREATE_TOOL_NAME,
    TASK_UPDATE_TOOL_NAME,
    TASK_LIST_TOOL_NAME,
    TASK_GET_TOOL_NAME,
)


def get_task_create_prompt() -> str:
    """Get TaskCreate tool prompt.

    Returns:
        Prompt string for task creation tool
    """
    from claude_code_py.utils.swarm.constants import is_agent_teams_enabled

    teams_enabled = is_agent_teams_enabled()
    teammate_context = " and potentially assigned to teammates" if teams_enabled else ""
    teammate_tips = (
        "- Include enough detail in the description for another agent to understand and complete the task\n"
        "- New tasks are created with status 'pending' and no owner - use TaskUpdate with the `owner` parameter to assign them\n"
    ) if teams_enabled else ""

    return f"""Use this tool to create a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool

Use this tool proactively in these scenarios:

- Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
- Non-trivial and complex tasks - Tasks that require careful planning or multiple operations{teammate_context}
- Plan mode - When using plan mode, create a task list to track the work
- User explicitly requests todo list - When the user directly asks you to use the todo list
- User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
- After receiving new instructions - Immediately capture user requirements as tasks
- When you start working on a task - Mark it as in_progress BEFORE beginning work
- After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
- There is only a single, straightforward task
- The task is trivial and tracking it provides no organizational benefit
- The task can be completed in less than 3 trivial steps
- The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

## Task Fields

- **subject**: A brief, actionable title in imperative form (e.g., "Fix authentication bug in login flow")
- **description**: What needs to be done
- **activeForm** (optional): Present continuous form shown in the spinner when the task is in_progress (e.g., "Fixing authentication bug"). If omitted, the spinner shows the subject instead.

All tasks are created with status `pending`.

## Tips

- Create tasks with clear, specific subjects that describe the outcome
- After creating tasks, use {TASK_UPDATE_TOOL_NAME} to set up dependencies (blocks/blockedBy) if needed
{teammate_tips}- Check {TASK_LIST_TOOL_NAME} first to avoid creating duplicate tasks
"""


def get_task_update_prompt() -> str:
    """Get TaskUpdate tool prompt.

    Returns:
        Prompt string for task update tool
    """
    return f"""Use this tool to update a task in the task list.

## When to Use This Tool

**Mark tasks as resolved:**
- When you have completed the work described in a task
- When a task is no longer needed or has been superseded
- IMPORTANT: Always mark your assigned tasks as resolved when you finish them
- After resolving, call {TASK_LIST_TOOL_NAME} to find your next task

- ONLY mark a task as completed when you have FULLY accomplished it
- If you encounter errors, blockers, or cannot finish, keep the task as in_progress
- When blocked, create a new task describing what needs to be resolved
- Never mark a task as completed if:
  - Tests are failing
  - Implementation is partial
  - You encountered unresolved errors
  - You couldn't find necessary files or dependencies

**Delete tasks:**
- When a task is no longer relevant or was created in error
- Setting status to `deleted` permanently removes the task

**Update task details:**
- When requirements change or become clearer
- When establishing dependencies between tasks

## Fields You Can Update

- **status**: The task status (see Status Workflow below)
- **subject**: Change the task title (imperative form, e.g., "Run tests")
- **description**: Change the task description
- **activeForm**: Present continuous form shown in spinner when in_progress (e.g., "Running tests")
- **owner**: Change the task owner (agent name)
- **metadata**: Merge metadata keys into the task (set a key to null to delete it)
- **addBlocks**: Mark tasks that cannot start until this one completes
- **addBlockedBy**: Mark tasks that must complete before this one can start

## Status Workflow

Status progresses: `pending` → `in_progress` → `completed`

Use `deleted` to permanently remove a task.

## Staleness

Make sure to read a task's latest state using `{TASK_GET_TOOL_NAME}` before updating it.

## Examples

Mark task as in progress when starting work:
```json
{{"taskId": "1", "status": "in_progress"}}
```

Mark task as completed after finishing work:
```json
{{"taskId": "1", "status": "completed"}}
```

Delete a task:
```json
{{"taskId": "1", "status": "deleted"}}
```

Claim a task by setting owner:
```json
{{"taskId": "1", "owner": "my-name"}}
```

Set up task dependencies:
```json
{{"taskId": "2", "addBlockedBy": ["1"]}}
```
"""


def get_task_list_prompt() -> str:
    """Get TaskList tool prompt.

    Returns:
        Prompt string for task list tool
    """
    from claude_code_py.utils.swarm.constants import is_agent_teams_enabled

    teams_enabled = is_agent_teams_enabled()
    teammate_use_case = "- To find tasks assigned to you as a teammate" if teams_enabled else ""
    teammate_workflow = (
        "\n## Teammate Workflow\n\n"
        "When working as a teammate:\n"
        "- Check TaskList periodically, especially after completing each task\n"
        "- Claim unassigned, unblocked tasks with TaskUpdate (set owner to your name)\n"
        "- Prefer tasks in ID order (lowest ID first) when multiple tasks are available\n"
        "- If blocked, focus on unblocking tasks or notify the team lead\n"
    ) if teams_enabled else ""

    return f"""Use this tool to list all tasks in the task list.

## When to Use This Tool

- To see what tasks are available to work on (status: 'pending', no owner, not blocked)
- To check overall progress on the project
- To find tasks that are blocked and need dependencies resolved
{teammate_use_case}- After completing a task, to check for newly unblocked work or claim the next available task
- **Prefer working on tasks in ID order** (lowest ID first) when multiple tasks are available, as earlier tasks often set up context for later ones

## Output

Returns a summary of each task:
- **id**: Task identifier (use with {TASK_GET_TOOL_NAME}, {TASK_UPDATE_TOOL_NAME})
- **subject**: Brief description of the task
- **status**: 'pending', 'in_progress', or 'completed'
- **owner**: Agent ID if assigned, empty if available
- **blockedBy**: List of open task IDs that must be resolved first (tasks with blockedBy cannot be claimed until dependencies resolve)

Use {TASK_GET_TOOL_NAME} with a specific task ID to view full details including description and comments.
{teammate_workflow}
"""


def get_task_get_prompt() -> str:
    """Get TaskGet tool prompt.

    Returns:
        Prompt string for task get tool
    """
    return f"""Use this tool to retrieve a task by its ID from the task list.

## When to Use This Tool

- When you need the full description and context before starting work on a task
- To understand task dependencies (what it blocks, what blocks it)
- After being assigned a task, to get complete requirements
- If the task is already completed, {TASK_GET_TOOL_NAME} is not needed - the task status and summary information from {TASK_LIST_TOOL_NAME} is sufficient

## Output

Returns full task details:
- **id**: Task identifier
- **subject**: Task title
- **description**: Detailed requirements and context
- **status**: 'pending', 'in_progress', or 'completed'
- **owner**: Agent ID if assigned, empty if available
- **blocks**: Tasks waiting on this one to complete
- **blockedBy**: Tasks that must complete before this one can start

## Staleness

A memory that names a specific function, file, or flag is a claim that it existed *when the memory was written*. It may have been renamed, removed, or never merged.

Before recommending from memory:
- If the memory names a file path: check the file exists.
- If the memory names a function or flag: grep for it.
- If the user is about to act on your recommendation (not just asking about history), verify first.

"The memory says X exists" is not the same as "X exists now."
"""


def get_task_stop_prompt() -> str:
    """Get TaskStop tool prompt.

    Returns:
        Prompt string for task stop tool
    """
    return f"""Use this tool to stop a running background task by its ID.

## When to Use This Tool

- When a background task is running too long and should be interrupted
- When you no longer need the results of a running background task
- To cancel a task that was started with `run_in_background` on Agent tool

## Behavior

- Stops the task immediately
- The task will not complete and no results will be available
- Use {TASK_LIST_TOOL_NAME} to find running tasks (status: 'in_progress', is_background: true)

## Examples

Stop a running background task:
```json
{{"taskId": "background-agent-1"}}
```
"""