"""Tool progress and message update types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from claude_code_py.tool.context import ToolUseContext
    from claude_code_py.core_types.message import Message
    from claude_code_py.core_types.tools import ToolProgressData


@dataclass
class ToolProgress:
    """Progress update from a tool."""

    tool_use_id: str
    data: "ToolProgressData"


@dataclass
class MessageUpdate:
    """Update from tool execution to be yielded to the message stream."""

    message: Optional["Message"] = None
    new_context: Optional["ToolUseContext"] = None


@dataclass
class MessageUpdateLazy:
    """Lazy message update with optional context modifier."""

    message: Optional["Message"] = None
    new_messages: Optional[list["Message"]] = None
    context_modifier: Optional[dict[str, Any]] = None


# Type alias for progress callback
ToolCallProgress = Callable[[ToolProgress], None]