"""Message type definitions using Pydantic for validation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# Content Block Types
# =============================================================================


class TextBlock(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """Tool use content block."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    """Tool result content block."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, list[ContentBlock]]
    is_error: bool = False


class ImageBlock(BaseModel):
    """Image content block."""

    type: Literal["image"] = "image"
    source: dict[str, Any]


# Union of all content block types
ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock]


# =============================================================================
# Message Types
# =============================================================================


class MessageType(str, Enum):
    """Message type enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    PROGRESS = "progress"
    ATTACHMENT = "attachment"
    TOMBSTONE = "tombstone"
    TOOL_USE_SUMMARY = "tool_use_summary"


class MessageRole(str, Enum):
    """Message role enumeration."""

    USER = "user"
    ASSISTANT = "assistant"


class BaseMessage(BaseModel):
    """Base message with common fields."""

    uuid: str = Field(default_factory=lambda: str(UUID(int=0)))
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


class UserMessage(BaseMessage):
    """User message."""

    type: Literal["user"] = "user"
    message: dict[str, Any]  # {role: "user", content: str | ContentBlock[]}
    tool_use_result: Optional[str] = None
    source_tool_assistant_uuid: Optional[str] = None
    is_meta: bool = False
    is_visible_in_transcript_only: bool = False
    is_compact_summary: bool = False


class AssistantMessage(BaseMessage):
    """Assistant message."""

    type: Literal["assistant"] = "assistant"
    message: dict[str, Any]  # {role: "assistant", content: ContentBlock[], stop_reason?: str}
    usage: Optional[dict[str, int]] = None
    stop_reason: Optional[str] = None
    is_api_error_message: bool = False
    api_error: Optional[str] = None


class SystemSubtype(str, Enum):
    """System message subtypes."""

    COMPACT_BOUNDARY = "compact_boundary"
    LOCAL_COMMAND = "local_command"
    API_ERROR = "api_error"


class SystemMessage(BaseMessage):
    """System message."""

    type: Literal["system"] = "system"
    subtype: Optional[SystemSubtype] = None
    content: Optional[str] = None
    compact_metadata: Optional[dict[str, Any]] = None
    retry_attempt: Optional[int] = None
    max_retries: Optional[int] = None
    retry_in_ms: Optional[int] = None
    error: Optional[dict[str, Any]] = None


class ProgressMessage(BaseMessage):
    """Progress message for tool execution updates."""

    type: Literal["progress"] = "progress"
    tool_use_id: str
    data: Optional[ToolProgressData] = None


class AttachmentMessage(BaseMessage):
    """Attachment message for binary data or special payloads."""

    type: Literal["attachment"] = "attachment"
    attachment: dict[str, Any]


class TombstoneMessage(BaseMessage):
    """Tombstone message for marking deleted messages."""

    type: Literal["tombstone"] = "tombstone"
    target_uuid: str


class ToolUseSummaryMessage(BaseMessage):
    """Tool use summary message."""

    type: Literal["tool_use_summary"] = "tool_use_summary"
    summary: str
    preceding_tool_use_ids: list[str]


# Union of all message types
Message = Union[
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ProgressMessage,
    AttachmentMessage,
    TombstoneMessage,
    ToolUseSummaryMessage,
]


# =============================================================================
# Forward reference resolution
# =============================================================================

# Update forward references for recursive models
ToolResultBlock.model_rebuild()


# =============================================================================
# Tool Progress Types (imported here for Message types)
# =============================================================================

from .tools import ToolProgressData  # noqa: E402, F401