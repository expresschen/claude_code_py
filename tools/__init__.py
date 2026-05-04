"""Built-in tools package."""

from .bash_tool import BashTool, BashInput, BashOutput, bash_tool
from .file_read_tool import FileReadTool, FileReadInput, FileReadOutput, file_read_tool
from .file_write_tool import FileWriteTool, FileWriteInput, file_write_tool
from .file_edit_tool import FileEditTool, FileEditInput, file_edit_tool
from .glob_tool import GlobTool, GlobInput, glob_tool
from .grep_tool import GrepTool, GrepInput, grep_tool
from .agent_tool import (
    AgentTool,
    AgentInput,
    AgentOutput,
    agent_tool,
    AGENT_TOOL_NAME,
    get_built_in_agents,
)
from .plan_mode import (
    EnterPlanModeTool,
    EnterPlanModeInput,
    EnterPlanModeOutput,
    enter_plan_mode_tool,
    ExitPlanModeTool,
    ExitPlanModeInput,
    ExitPlanModeOutput,
    exit_plan_mode_tool,
    ENTER_PLAN_MODE_TOOL_NAME,
    EXIT_PLAN_MODE_TOOL_NAME,
    is_in_plan_mode,
    get_plan_file_path,
)
from .ask_user_question import (
    AskUserQuestionTool,
    AskUserQuestionInput,
    AskUserQuestionOutput,
    ask_user_question_tool,
    ASK_USER_QUESTION_TOOL_NAME,
    Question,
    QuestionOption,
)
from .worktree_tool import (
    EnterWorktreeTool,
    ExitWorktreeTool,
    ENTER_WORKTREE_TOOL_NAME,
)
from .send_message_tool import (
    SendMessageTool,
    SEND_MESSAGE_TOOL_NAME,
)
from .task_tools import (
    TaskCreateTool,
    TaskUpdateTool,
    TaskListTool,
    TaskGetTool,
    TaskStopTool,
    TaskCreateInput,
    TaskUpdateInput,
    TaskListInput,
    TaskGetInput,
    TaskStopInput,
    task_create_tool,
    task_update_tool,
    task_list_tool,
    task_get_tool,
    task_stop_tool,
    TASK_CREATE_TOOL_NAME,
    TASK_UPDATE_TOOL_NAME,
    TASK_LIST_TOOL_NAME,
    TASK_GET_TOOL_NAME,
    TASK_STOP_TOOL_NAME,
    get_task_create_prompt,
    get_task_update_prompt,
    get_task_list_prompt,
    get_task_get_prompt,
    get_task_stop_prompt,
)

# Team tools - lazy import to avoid circular dependencies
def _get_team_create_tool():
    """Lazy import and instantiate TeamCreateTool."""
    from .team_tools import TeamCreateTool
    return TeamCreateTool()

def _get_team_delete_tool():
    """Lazy import and instantiate TeamDeleteTool."""
    from .team_tools import TeamDeleteTool
    return TeamDeleteTool()

def _is_agent_teams_enabled():
    """Check if agent teams feature is enabled."""
    import os
    return os.environ.get("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS", "").lower() in ("1", "true", "yes")

__all__ = [
    # Bash
    "BashTool",
    "BashInput",
    "BashOutput",
    "bash_tool",
    # File Read
    "FileReadTool",
    "FileReadInput",
    "FileReadOutput",
    "file_read_tool",
    # File Write
    "FileWriteTool",
    "FileWriteInput",
    "file_write_tool",
    # File Edit
    "FileEditTool",
    "FileEditInput",
    "file_edit_tool",
    # Glob
    "GlobTool",
    "GlobInput",
    "glob_tool",
    # Grep
    "GrepTool",
    "GrepInput",
    "grep_tool",
    # Agent
    "AgentTool",
    "AgentInput",
    "AgentOutput",
    "agent_tool",
    "AGENT_TOOL_NAME",
    "get_built_in_agents",
    # Plan Mode
    "EnterPlanModeTool",
    "EnterPlanModeInput",
    "EnterPlanModeOutput",
    "enter_plan_mode_tool",
    "ExitPlanModeTool",
    "ExitPlanModeInput",
    "ExitPlanModeOutput",
    "exit_plan_mode_tool",
    "ENTER_PLAN_MODE_TOOL_NAME",
    "EXIT_PLAN_MODE_TOOL_NAME",
    "is_in_plan_mode",
    "get_plan_file_path",
    # Ask User Question
    "AskUserQuestionTool",
    "AskUserQuestionInput",
    "AskUserQuestionOutput",
    "ask_user_question_tool",
    "ASK_USER_QUESTION_TOOL_NAME",
    "Question",
    "QuestionOption",
    # Worktree
    "EnterWorktreeTool",
    "ExitWorktreeTool",
    "ENTER_WORKTREE_TOOL_NAME",
    # SendMessage
    "SendMessageTool",
    "SEND_MESSAGE_TOOL_NAME",
    # Task Tools
    "TaskCreateTool",
    "TaskUpdateTool",
    "TaskListTool",
    "TaskGetTool",
    "TaskStopTool",
    "TaskCreateInput",
    "TaskUpdateInput",
    "TaskListInput",
    "TaskGetInput",
    "TaskStopInput",
    "task_create_tool",
    "task_update_tool",
    "task_list_tool",
    "task_get_tool",
    "task_stop_tool",
    "TASK_CREATE_TOOL_NAME",
    "TASK_UPDATE_TOOL_NAME",
    "TASK_LIST_TOOL_NAME",
    "TASK_GET_TOOL_NAME",
    "TASK_STOP_TOOL_NAME",
    "get_task_create_prompt",
    "get_task_update_prompt",
    "get_task_list_prompt",
    "get_task_get_prompt",
    "get_task_stop_prompt",
    # Tool registry
    "get_all_base_tools",
    "get_enabled_tool_names",
    "_is_agent_teams_enabled",
    "_get_team_create_tool",
    "_get_team_delete_tool",
]


def get_all_base_tools():
    """Get all base built-in tools.

    Returns list of tool instances.
    Similar to getAllBaseTools() in TypeScript tools.ts.
    """
    tools = [
        bash_tool,
        file_read_tool,
        file_edit_tool,
        file_write_tool,
        glob_tool,
        grep_tool,
        agent_tool,
        enter_plan_mode_tool,
        exit_plan_mode_tool,
        ask_user_question_tool,
        EnterWorktreeTool,
        ExitWorktreeTool,
        SendMessageTool,
        task_create_tool,
        task_update_tool,
        task_list_tool,
        task_get_tool,
        task_stop_tool,
    ]

    # Conditionally add Team tools when experimental flag is set
    # Matches TypeScript: ...(isAgentSwarmsEnabled() ? [getTeamCreateTool(), getTeamDeleteTool()] : [])
    if _is_agent_teams_enabled():
        tools.append(_get_team_create_tool())
        tools.append(_get_team_delete_tool())

    return tools


def get_enabled_tool_names():
    """Get names of all enabled tools.

    Filters tools by isEnabled() check, similar to getToolsForDefaultPreset() in TS.
    """
    tools = get_all_base_tools()
    return [tool.name for tool in tools if tool.is_enabled()]