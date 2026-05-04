"""Tool system for Claude Code.

This module defines the core Tool abstraction and related types.
"""

from .base import Tool, ToolDef, build_tool
from .context import ToolUseContext, CanUseToolFn, SetToolJSXFn
from .result import ToolResult, ToolCallProgress

__all__ = [
    "Tool",
    "ToolDef",
    "build_tool",
    "ToolUseContext",
    "CanUseToolFn",
    "SetToolJSXFn",
    "ToolResult",
    "ToolCallProgress",
]