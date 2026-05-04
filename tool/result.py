"""Tool result types.

This defines the result types returned by tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from claude_code_py.tool.context import ToolUseContext
    from claude_code_py.core_types.message import AssistantMessage, AttachmentMessage, SystemMessage, UserMessage
    from claude_code_py.core_types.tools import ToolProgressData


T = TypeVar("T")


@dataclass
class ToolResult(Generic[T]):
    """Result of a tool call.

    Attributes:
        data: The output data
        new_messages: Optional new messages to add to conversation
        context_modifier: Optional function to modify context after execution
        mcp_meta: Optional MCP protocol metadata
    """

    data: T
    new_messages: Optional[
        list[Union["UserMessage", "AssistantMessage", "AttachmentMessage", "SystemMessage"]]
    ] = None
    context_modifier: Optional[Callable[["ToolUseContext"], "ToolUseContext"]] = None
    mcp_meta: Optional[dict[str, Any]] = None

    @classmethod
    def success(cls, data: T) -> "ToolResult[T]":
        """Create a successful result with just data."""
        return cls(data=data)

    @classmethod
    def with_messages(
        cls,
        data: T,
        messages: list,
    ) -> "ToolResult[T]":
        """Create a result with additional messages."""
        return cls(data=data, new_messages=messages)


@dataclass
class ToolProgress(Generic[T]):
    """Progress update from a tool.

    Attributes:
        tool_use_id: ID of the tool use
        data: Progress data
    """

    tool_use_id: str
    data: T


# Type alias for progress callback
ToolCallProgress = Callable[[ToolProgress], None]


@dataclass
class ToolUseBlockParam:
    """Parameter for creating a tool use block."""

    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlockParam:
    """Parameter for creating a tool result block."""

    type: str = "tool_result"
    tool_use_id: str = ""
    content: Union[str, list[Any]] = ""
    is_error: bool = False


@dataclass
class MessageUpdate:
    """Update from tool execution to be yielded to the message stream."""

    message: Optional[Any] = None
    new_context: Optional["ToolUseContext"] = None
    context_modifier: Optional[dict[str, Any]] = None


# Error types
class ToolError(Exception):
    """Base error for tool execution."""

    def __init__(
        self,
        message: str,
        *,
        is_retryable: bool = False,
        error_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.is_retryable = is_retryable
        self.error_code = error_code


class PermissionDeniedError(ToolError):
    """Tool permission denied."""

    def __init__(self, tool_name: str, reason: Optional[str] = None):
        super().__init__(
            f"Permission denied for tool: {tool_name}"
            + (f" - {reason}" if reason else "")
        )
        self.tool_name = tool_name


class ValidationError(ToolError):
    """Tool input validation failed."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, error_code=400)
        self.field = field


class TimeoutError(ToolError):
    """Tool execution timed out."""

    def __init__(self, timeout_seconds: float):
        super().__init__(f"Tool execution timed out after {timeout_seconds}s", is_retryable=True)
        self.timeout_seconds = timeout_seconds


class SandboxError(ToolError):
    """Sandbox-related error."""

    def __init__(self, message: str):
        super().__init__(message, error_code=403)