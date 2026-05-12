"""Core type definitions for Claude Code."""

from .ids import AgentId, SessionId, TaskId, as_agent_id, as_session_id
from .message import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ProgressMessage,
    AttachmentMessage,
    TombstoneMessage,
    ToolUseSummaryMessage,
    ContentBlock,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)
from .permissions import PermissionMode, PermissionResult, PermissionBehavior
from .tools import (
    ToolProgressData,
    BashProgress,
    FileReadProgress,
    WebFetchProgress,
)

__all__ = [
    # IDs
    "AgentId",
    "SessionId",
    "TaskId",
    "as_agent_id",
    "as_session_id",
    # Messages
    "Message",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ProgressMessage",
    "AttachmentMessage",
    "TombstoneMessage",
    "ToolUseSummaryMessage",
    "ContentBlock",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    # Permissions
    "PermissionMode",
    "PermissionResult",
    "PermissionBehavior",
    # Tool Progress
    "ToolProgressData",
    "BashProgress",
    "FileReadProgress",
    "WebFetchProgress",
]