"""Tool execution logic.

This implements the runTools, runToolsSerially, and runToolsConcurrently functions.
"""

from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Optional,
)

from .partition import partition_tool_calls, get_max_tool_use_concurrency
from .progress import MessageUpdate, MessageUpdateLazy
from claude_code_py.core_types.permissions import PermissionBehavior

if TYPE_CHECKING:
    from claude_code_py.tool.base import Tool, ToolCallProgress
    from claude_code_py.tool.context import ToolUseContext, CanUseToolFn
    from claude_code_py.tool.result import ToolResult
    from claude_code_py.core_types.message import AssistantMessage, Message, ToolUseBlock


async def run_tools(
    tool_use_messages: list["ToolUseBlock"],
    assistant_messages: list["AssistantMessage"],
    can_use_tool: "CanUseToolFn",
    tool_use_context: "ToolUseContext",
) -> AsyncGenerator[MessageUpdate, None]:
    """Execute tool calls.

    This partitions tool calls and runs them in appropriate batches.

    Args:
        tool_use_messages: Tool use blocks to execute
        assistant_messages: Parent assistant messages
        can_use_tool: Permission check function
        tool_use_context: Execution context

    Yields:
        Message updates from tool execution
    """
    current_context = tool_use_context

    for batch in partition_tool_calls(tool_use_messages, current_context):
        if batch.is_concurrency_safe:
            # Run concurrently
            queued_modifiers: dict[str, list[Callable]] = {}

            async for update in run_tools_concurrently(
                batch.blocks,
                assistant_messages,
                can_use_tool,
                current_context,
            ):
                if update.context_modifier:
                    tool_use_id = update.context_modifier.get("tool_use_id")
                    if tool_use_id:
                        if tool_use_id not in queued_modifiers:
                            queued_modifiers[tool_use_id] = []
                        queued_modifiers[tool_use_id].append(
                            update.context_modifier["modify_context"]
                        )

                yield MessageUpdate(
                    message=update.message,
                    new_context=current_context,
                )

            # Apply queued context modifiers
            for block in batch.blocks:
                modifiers = queued_modifiers.get(block.id, [])
                for modifier in modifiers:
                    current_context = modifier(current_context)

            yield MessageUpdate(new_context=current_context)

        else:
            # Run serially
            async for update in run_tools_serially(
                batch.blocks,
                assistant_messages,
                can_use_tool,
                current_context,
            ):
                if update.new_context:
                    current_context = update.new_context

                yield MessageUpdate(
                    message=update.message,
                    new_context=current_context,
                )


async def run_tools_serially(
    tool_use_messages: list["ToolUseBlock"],
    assistant_messages: list["AssistantMessage"],
    can_use_tool: "CanUseToolFn",
    tool_use_context: "ToolUseContext",
) -> AsyncGenerator[MessageUpdate, None]:
    """Execute tool calls serially, one at a time.

    Args:
        tool_use_messages: Tool use blocks to execute
        assistant_messages: Parent assistant messages
        can_use_tool: Permission check function
        tool_use_context: Execution context

    Yields:
        Message updates from tool execution
    """
    current_context = tool_use_context

    for tool_use in tool_use_messages:
        # Mark as in progress
        if tool_use_context.set_in_progress_tool_use_ids:
            tool_use_context.set_in_progress_tool_use_ids(
                lambda prev: prev | {tool_use.id}
            )

        # Find parent message
        parent_message = _find_parent_message(assistant_messages, tool_use.id)

        # Execute the tool
        async for update in run_tool_use(
            tool_use,
            parent_message,
            can_use_tool,
            current_context,
        ):
            if update.context_modifier:
                current_context = update.context_modifier["modify_context"](current_context)

            yield MessageUpdate(
                message=update.message,
                new_context=current_context,
            )

        # Mark as complete
        _mark_tool_use_complete(current_context, tool_use.id)


async def run_tools_concurrently(
    tool_use_messages: list["ToolUseBlock"],
    assistant_messages: list["AssistantMessage"],
    can_use_tool: "CanUseToolFn",
    tool_use_context: "ToolUseContext",
) -> AsyncGenerator[MessageUpdateLazy, None]:
    """Execute tool calls concurrently.

    Args:
        tool_use_messages: Tool use blocks to execute
        assistant_messages: Parent assistant messages
        can_use_tool: Permission check function
        tool_use_context: Execution context

    Yields:
        Message updates from tool execution
    """
    from claude_code_py.utils.generators import all

    async def run_one(
        tool_use: "ToolUseBlock",
    ) -> AsyncGenerator[MessageUpdateLazy, None]:
        # Mark as in progress
        if tool_use_context.set_in_progress_tool_use_ids:
            tool_use_context.set_in_progress_tool_use_ids(
                lambda prev: prev | {tool_use.id}
            )

        parent_message = _find_parent_message(assistant_messages, tool_use.id)

        async for update in run_tool_use(
            tool_use,
            parent_message,
            can_use_tool,
            tool_use_context,
        ):
            yield update

        _mark_tool_use_complete(tool_use_context, tool_use.id)

    # Run all with concurrency limit
    generators = [lambda t=tool_use: run_one(t) for tool_use in tool_use_messages]

    async for update in all(generators, get_max_tool_use_concurrency()):
        yield update


async def run_tool_use(
    tool_use: "ToolUseBlock",
    parent_message: "AssistantMessage",
    can_use_tool: "CanUseToolFn",
    tool_use_context: "ToolUseContext",
) -> AsyncGenerator[MessageUpdateLazy, None]:
    """Execute a single tool use.

    Args:
        tool_use: Tool use block
        parent_message: Parent assistant message
        can_use_tool: Permission check function
        tool_use_context: Execution context

    Yields:
        Message updates
    """
    from claude_code_py.tool.base import find_tool_by_name
    from claude_code_py.core_types.message import UserMessage

    tool = find_tool_by_name(tool_use_context.options.tools, tool_use.name)

    if not tool:
        # Tool not found - return error
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"Error: Tool '{tool_use.name}' not found",
                        "is_error": True,
                    }],
                }
            )
        )
        return

    try:
        # Parse input
        if hasattr(tool.input_schema, "model_validate"):
            parsed_input = tool.input_schema.model_validate(tool_use.input)
        else:
            parsed_input = tool.input_schema(**tool_use.input)

        # Check permissions
        perm_result = await can_use_tool(
            tool,
            parsed_input,
            tool_use_context,
            parent_message,
            tool_use.id,
        )

        if perm_result.behavior != PermissionBehavior.ALLOW:
            # Permission denied
            yield MessageUpdateLazy(
                message=UserMessage(
                    message={
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Permission denied: {perm_result.reason or 'No reason provided'}",
                            "is_error": True,
                        }],
                    }
                )
            )
            return

        # Execute tool
        result: ToolResult = await tool.call(
            parsed_input,
            tool_use_context,
            can_use_tool,
            parent_message,
        )

        # Create tool result message
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": _format_tool_result(result),
                        "is_error": False,
                    }],
                }
            ),
            new_messages=result.new_messages,
            context_modifier={
                "tool_use_id": tool_use.id,
                "modify_context": result.context_modifier,
            } if result.context_modifier else None,
        )

    except Exception as e:
        # Error executing tool
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"Error: {str(e)}",
                        "is_error": True,
                    }],
                }
            )
        )


def _find_parent_message(
    assistant_messages: list["AssistantMessage"],
    tool_use_id: str,
) -> Optional["AssistantMessage"]:
    """Find the assistant message containing a tool use.

    Args:
        assistant_messages: List of assistant messages
        tool_use_id: Tool use ID to find

    Returns:
        Parent message or None
    """
    for msg in assistant_messages:
        content = msg.message.get("content", [])
        for block in content:
            if isinstance(block, dict) and block.get("id") == tool_use_id:
                return msg
    return None


def _mark_tool_use_complete(
    context: "ToolUseContext",
    tool_use_id: str,
) -> None:
    """Mark a tool use as complete.

    Args:
        context: Tool use context
        tool_use_id: Tool use ID
    """
    if context.set_in_progress_tool_use_ids:
        context.set_in_progress_tool_use_ids(
            lambda prev: prev - {tool_use_id}
        )


def _format_tool_result(result: "ToolResult") -> str:
    """Format a tool result for the API.

    Args:
        result: Tool result

    Returns:
        Formatted string
    """
    if result.data is None:
        return "Success"

    if isinstance(result.data, str):
        return result.data

    if hasattr(result.data, "model_dump"):
        import json
        return json.dumps(result.data.model_dump(), indent=2)

    return str(result.data)