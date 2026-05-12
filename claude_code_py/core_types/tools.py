"""Tool progress data types."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class ProgressEventType(str, Enum):
    """Types of progress events."""

    START = "start"
    UPDATE = "update"
    COMPLETE = "complete"
    ERROR = "error"


# =============================================================================
# Base Progress Data
# =============================================================================


class ToolProgressData(BaseModel):
    """Base class for tool progress data."""

    type: str = "base"


# =============================================================================
# Bash Tool Progress
# =============================================================================


class BashProgress(ToolProgressData):
    """Progress data for Bash tool."""

    type: Literal["bash"] = "bash"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    pid: Optional[int] = None
    duration_ms: Optional[int] = None


# =============================================================================
# File Tool Progress
# =============================================================================


class FileReadProgress(ToolProgressData):
    """Progress data for FileRead tool."""

    type: Literal["file_read"] = "file_read"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    path: str
    bytes_read: int = 0
    total_bytes: Optional[int] = None
    lines_read: int = 0


class FileEditProgress(ToolProgressData):
    """Progress data for FileEdit tool."""

    type: Literal["file_edit"] = "file_edit"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    path: str
    operation: str  # "replace", "insert", "delete"
    lines_modified: int = 0


class FileWriteProgress(ToolProgressData):
    """Progress data for FileWrite tool."""

    type: Literal["file_write"] = "file_write"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    path: str
    bytes_written: int = 0


# =============================================================================
# Search Tool Progress
# =============================================================================


class GlobProgress(ToolProgressData):
    """Progress data for Glob tool."""

    type: Literal["glob"] = "glob"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    pattern: str
    matches_found: int = 0
    directories_searched: int = 0


class GrepProgress(ToolProgressData):
    """Progress data for Grep tool."""

    type: Literal["grep"] = "grep"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    pattern: str
    matches_found: int = 0
    files_searched: int = 0


# =============================================================================
# Web Tool Progress
# =============================================================================


class WebFetchProgress(ToolProgressData):
    """Progress data for WebFetch tool."""

    type: Literal["web_fetch"] = "web_fetch"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    url: str
    bytes_received: int = 0
    status_code: Optional[int] = None


class WebSearchProgress(ToolProgressData):
    """Progress data for WebSearch tool."""

    type: Literal["web_search"] = "web_search"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    query: str
    results_found: int = 0


# =============================================================================
# Agent Tool Progress
# =============================================================================


class AgentToolProgress(ToolProgressData):
    """Progress data for Agent tool."""

    type: Literal["agent"] = "agent"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    agent_id: str
    agent_type: str
    status: str  # "starting", "running", "completed", "failed"
    turns_completed: int = 0


# =============================================================================
# Task Tool Progress
# =============================================================================


class TaskOutputProgress(ToolProgressData):
    """Progress data for TaskOutput tool."""

    type: Literal["task_output"] = "task_output"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    task_id: str
    bytes_available: int = 0
    is_complete: bool = False


# =============================================================================
# MCP Tool Progress
# =============================================================================


class MCPProgress(ToolProgressData):
    """Progress data for MCP tools."""

    type: Literal["mcp"] = "mcp"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    server_name: str
    tool_name: str
    progress_token: Optional[str] = None
    progress: Optional[float] = None
    message: Optional[str] = None


# =============================================================================
# Skill Tool Progress
# =============================================================================


class SkillToolProgress(ToolProgressData):
    """Progress data for Skill tool."""

    type: Literal["skill"] = "skill"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    skill_name: str
    step: str
    progress: float = 0.0


# =============================================================================
# REPL Tool Progress
# =============================================================================


class REPLToolProgress(ToolProgressData):
    """Progress data for REPL wrapper tool."""

    type: Literal["repl"] = "repl"
    event_type: ProgressEventType = ProgressEventType.UPDATE
    inner_tool_name: str
    inner_progress: Optional[ToolProgressData] = None


# =============================================================================
# Hook Progress
# =============================================================================


class HookProgress(ToolProgressData):
    """Progress data for hooks."""

    type: Literal["hook"] = "hook"
    hook_type: str  # "pre_tool_use", "post_tool_use", "stop", etc.
    hook_name: str
    status: str  # "running", "completed", "failed"


# =============================================================================
# Union of all progress types
# =============================================================================

AnyProgress = Union[
    BashProgress,
    FileReadProgress,
    FileEditProgress,
    FileWriteProgress,
    GlobProgress,
    GrepProgress,
    WebFetchProgress,
    WebSearchProgress,
    AgentToolProgress,
    TaskOutputProgress,
    MCPProgress,
    SkillToolProgress,
    REPLToolProgress,
    HookProgress,
]