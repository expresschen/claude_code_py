"""Tool call partitioning for concurrent execution.

This implements the partition_tool_calls logic from toolOrchestration.ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_code_py.tool.base import Tool
    from claude_code_py.tool.context import ToolUseContext
    from claude_code_py.core_types.message import ToolUseBlock


@dataclass
class Batch:
    """A batch of tool calls.

    Either all concurrency-safe (can run in parallel)
    or contains non-safe calls (must run serially).
    """

    is_concurrency_safe: bool
    blocks: list["ToolUseBlock"]


def partition_tool_calls(
    tool_use_messages: list["ToolUseBlock"],
    tool_use_context: "ToolUseContext",
) -> list[Batch]:
    """Partition tool calls into batches.

    Each batch is either:
    1. Multiple consecutive concurrency-safe tools (can run in parallel)
    2. A single non-safe tool (must run serially)

    Args:
        tool_use_messages: Tool use blocks to partition
        tool_use_context: Execution context with tools

    Returns:
        List of batches
    """
    from claude_code_py.tool.base import find_tool_by_name

    batches: list[Batch] = []

    for tool_use in tool_use_messages:
        # Find the tool
        tool = find_tool_by_name(tool_use_context.options.tools, tool_use.name)

        # Check if concurrency safe
        is_concurrency_safe = False
        if tool:
            try:
                # Parse input
                input_data = tool_use.input
                if hasattr(tool.input_schema, "model_validate"):
                    parsed = tool.input_schema.model_validate(input_data)
                else:
                    parsed = tool.input_schema(**input_data)

                is_concurrency_safe = tool.is_concurrency_safe(parsed)
            except Exception:
                # If parsing fails, treat as not safe
                is_concurrency_safe = False

        # Merge with previous batch if both are concurrency-safe
        if is_concurrency_safe and batches and batches[-1].is_concurrency_safe:
            batches[-1].blocks.append(tool_use)
        else:
            batches.append(Batch(
                is_concurrency_safe=is_concurrency_safe,
                blocks=[tool_use],
            ))

    return batches


def get_max_tool_use_concurrency() -> int:
    """Get the maximum concurrency for tool execution.

    Returns:
        Maximum number of concurrent tool calls
    """
    import os
    return int(os.environ.get("CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY", "10"))