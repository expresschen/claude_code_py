"""Tool execution context.

This defines the context passed to tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Union,
    Protocol,
)

# Import at runtime for default_factory usage
from claude_code_py.core_types.permissions import ToolPermissionContext

if TYPE_CHECKING:
    from claude_code_py.tool.base import Tool, ValidationResult
    from claude_code_py.core_types.message import AssistantMessage, Message, SystemMessage
    from claude_code_py.core_types.permissions import PermissionResult
    from claude_code_py.core_types.tools import ToolProgressData
    from claude_code_py.state.store import Store
    from claude_code_py.utils.abort_controller import AbortController


class CanUseToolFn(Protocol):
    """Protocol for the canUseTool function."""

    async def __call__(
        self,
        tool: "Tool",
        input: Any,
        context: "ToolUseContext",
        assistant_message: "AssistantMessage",
        tool_use_id: Optional[str],
        force_decision: Optional[str] = None,
    ) -> "PermissionResult":
        ...


class SetToolJSXFn(Protocol):
    """Protocol for the setToolJSX function."""

    def __call__(
        self,
        args: Optional[dict[str, Any]],
    ) -> None:
        ...


@dataclass
class AgentDefinition:
    """Definition of a custom agent."""

    name: str
    description: str
    model: Optional[str] = None
    tools: list[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    color: Optional[str] = None


@dataclass
class AgentDefinitionsResult:
    """Result of loading agent definitions."""

    active_agents: list[AgentDefinition]
    all_agents: list[AgentDefinition]


@dataclass
class Command:
    """Definition of a slash command."""

    name: str
    description: str
    handler: Optional[Callable] = None
    is_enabled: bool = True


@dataclass
class MCPServerConnection:
    """Connection to an MCP server."""

    name: str
    status: str  # "connected", "disconnected", "error"
    tools: list[Any] = field(default_factory=list)
    resources: list[Any] = field(default_factory=list)


@dataclass
class ServerResource:
    """MCP server resource."""

    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class ThinkingConfig:
    """Configuration for thinking mode."""

    type: str = "disabled"  # "disabled", "enabled", "adaptive"
    budget_tokens: Optional[int] = None


@dataclass
class ToolUseContext:
    """Context passed to tool calls.

    This is a comprehensive context object that provides tools with
    access to the application state, configuration, and utility functions.
    """

    # Core options
    options: "ToolUseContextOptions"

    # Control
    abort_controller: "AbortController"

    # State
    read_file_state: dict[str, Any] = field(default_factory=dict)
    get_app_state: Optional[Callable[[], Any]] = None
    set_app_state: Optional[Callable[[Callable], None]] = None

    # Messages
    messages: list["Message"] = field(default_factory=list)

    # UI callbacks
    set_tool_jsx: Optional["SetToolJSXFn"] = None
    add_notification: Optional[Callable] = None
    send_os_notification: Optional[Callable] = None

    # Memory and skills
    nested_memory_attachment_triggers: set[str] = field(default_factory=set)
    loaded_nested_memory_paths: set[str] = field(default_factory=set)
    dynamic_skill_dir_triggers: set[str] = field(default_factory=set)
    discovered_skill_names: set[str] = field(default_factory=set)

    # Tracking
    user_modified: bool = False
    set_in_progress_tool_use_ids: Optional[Callable] = None
    set_response_length: Optional[Callable] = None

    # Agent context
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    tool_use_id: Optional[str] = None

    # Teammate context 新增字段
    agent_name: Optional[str] = None
    team_name: Optional[str] = None
    parent_session_id: Optional[str] = None

    # Worktree context 新增字段
    worktree_path: Optional[str] = None
    is_worktree_session: bool = False

    # Additional context
    file_reading_limits: Optional[dict[str, int]] = None
    glob_limits: Optional[dict[str, int]] = None
    tool_decisions: Optional[dict[str, Any]] = None

    # For MCP tools
    handle_elicitation: Optional[Callable] = None

    def get_tool_permission_context(self) -> "ToolPermissionContext":
        """Get the current tool permission context."""
        from claude_code_py.core_types.permissions import ToolPermissionContext

        return self.options.tool_permission_context

    def get_cwd(self) -> str:
        """Get the current working directory."""
        return self.options.cwd

    def resolve_path(self, path: Optional[str] = None) -> str:
        """Resolve a path relative to the current working directory.

        Handles:
        - None: returns cwd
        - Absolute paths: returned directly
        - Relative paths: resolved relative to cwd
        - Home directory (~): expanded to user's home

        Args:
            path: Path to resolve (absolute, relative, or ~-based). If None, returns cwd.

        Returns:
            Absolute path
        """
        from pathlib import Path

        # Handle None - return cwd
        if path is None:
            return self.options.cwd

        # Expand home directory (~)
        if path.startswith("~"):
            path = str(Path(path).expanduser())
            p = Path(path)
        else:
            p = Path(path)

        if p.is_absolute():
            return str(p)
        return str(Path(self.options.cwd) / p)

    def get_tools(self) -> list["Tool"]:
        """Get available tools."""
        return self.options.tools


@dataclass
class ToolUseContextOptions:
    """Options within ToolUseContext."""

    # Working directory
    cwd: str = "."  # Current working directory for tool execution

    # Commands
    commands: list[Command] = field(default_factory=list)

    # Debug
    debug: bool = False
    verbose: bool = False

    # Model
    main_loop_model: str = "claude-sonnet-4-6"

    # Tools
    tools: list["Tool"] = field(default_factory=list)

    # Thinking
    thinking_config: ThinkingConfig = field(default_factory=ThinkingConfig)

    # MCP
    mcp_clients: list[MCPServerConnection] = field(default_factory=list)
    mcp_resources: dict[str, list[ServerResource]] = field(default_factory=dict)

    # Session
    is_non_interactive_session: bool = False

    # Agents
    agent_definitions: AgentDefinitionsResult = field(
        default_factory=lambda: AgentDefinitionsResult(active_agents=[], all_agents=[])
    )

    # Budget
    max_budget_usd: Optional[float] = None

    # Prompts
    custom_system_prompt: Optional[str] = None
    append_system_prompt: Optional[str] = None

    # Permissions
    tool_permission_context: "ToolPermissionContext" = field(
        default_factory=lambda: ToolPermissionContext()
    )

    # Query source
    query_source: Optional[str] = None

    # Tool refresh
    refresh_tools: Optional[Callable[[], list["Tool"]]] = None


def create_default_tool_use_context(
    tools: list["Tool"],
    abort_controller: "AbortController",
    cwd: str = ".",
) -> ToolUseContext:
    """Create a default tool use context.

    Args:
        tools: Available tools
        abort_controller: Abort controller
        cwd: Current working directory

    Returns:
        Tool use context with sensible defaults
    """
    from claude_code_py.core_types.permissions import ToolPermissionContext

    perm_context = ToolPermissionContext(cwd=cwd)

    options = ToolUseContextOptions(
        cwd=cwd,
        tools=tools,
        tool_permission_context=perm_context,
    )

    return ToolUseContext(
        options=options,
        abort_controller=abort_controller,
    )