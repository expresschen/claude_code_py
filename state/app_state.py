"""Application state definition.

This defines the central AppState type that holds all application state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from claude_code_py.core_types.permissions import ToolPermissionContext


@dataclass
class AppState:
    """Central application state.

    This is the single source of truth for the entire application.
    """

    # Messages
    messages: list[Any] = field(default_factory=list)

    # Permissions
    tool_permission_context: ToolPermissionContext = field(
        default_factory=ToolPermissionContext
    )

    # Model
    main_loop_model: str = "claude-sonnet-4-6"

    # MCP
    mcp_clients: list[Any] = field(default_factory=list)
    mcp_resources: dict[str, list[Any]] = field(default_factory=dict)

    # Plugins
    plugins: list[Any] = field(default_factory=list)

    # Agents
    agent_registry: list[Any] = field(default_factory=list)

    # Notifications
    notifications: list[Any] = field(default_factory=list)

    # Tasks
    tasks: dict[str, Any] = field(default_factory=dict)

    # Remote/Bridge state
    remote_bridge_state: Optional[Any] = None

    # UI state
    verbose: bool = False
    debug: bool = False

    # Working directory
    cwd: str = "."

    # Session
    session_id: Optional[str] = None

    # Features
    fast_mode: Optional[dict[str, Any]] = None

    # File history
    file_history: dict[str, Any] = field(default_factory=dict)

    # Attribution (for commits)
    attribution: dict[str, Any] = field(default_factory=dict)

    # Speculation
    speculation_state: Optional[dict[str, Any]] = None

    # Session isolation 新增字段
    session_project_dir: Optional[str] = None  # 跨项目 resume 时存储 transcript 目录
    worktree_path: Optional[str] = None         # 当前 worktree 路径
    worktree_branch: Optional[str] = None       # worktree 分支名
    worktree_session: Optional[Any] = None      # WorktreeSession 完整状态

    # Teammate 新增字段
    team_context: Optional[dict[str, Any]] = None  # Team context (teamName, leadAgentId, members, etc.)
    agent_color_map: dict[str, str] = field(default_factory=dict)  # agent_id -> color
    agent_color_index: int = 0
    teammate_mailbox: dict[str, list[Any]] = field(default_factory=dict)  # agent_name -> messages
    inbox: dict[str, Any] = field(default_factory=lambda: {"messages": []})  # Inbox for swarm messages
    pending_questions: list[Any] = field(default_factory=list)  # Questions from teammates awaiting user input
    pending_permissions: list[Any] = field(default_factory=list)  # Permission requests from teammates awaiting user approval

    # Session 环境新增字段
    session_env_dir: Optional[str] = None       # hook 环境脚本目录

    # Completion boundary
    completion_boundary: Optional[dict[str, Any]] = None

    # Loading state (for idle/busy tracking in InboxPoller)
    is_loading: bool = False                    # True when a query is running
    is_waiting_for_input: bool = False          # True when REPL is blocked on stdin


def get_default_app_state(cwd: str = ".") -> AppState:
    """Create a default AppState.

    Args:
        cwd: Initial working directory

    Returns:
        AppState with default values
    """
    return AppState(cwd=cwd)


# Type alias for the set_app_state callback
SetAppState = Callable[[Callable[[AppState], AppState]], None]