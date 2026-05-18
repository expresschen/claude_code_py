"""Tool execution logic.

This implements the runTools, runToolsSerially, and runToolsConcurrently functions.
"""

from __future__ import annotations

import asyncio
import logging
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
from claude_code_py.tool.result import (
    ToolError,
    ShellError,
    TimeoutError as ToolTimeoutError,
    ValidationError as ToolValidationError,
    PermissionDeniedError,
)
from claude_code_py.utils.abort_controller import AbortError
from claude_code_py.utils.tool_errors import (
    format_error,
    format_validation_error,
    classify_tool_error,
)

if TYPE_CHECKING:
    from claude_code_py.tool.base import Tool, ToolCallProgress
    from claude_code_py.tool.context import ToolUseContext, CanUseToolFn
    from claude_code_py.tool.result import ToolResult
    from claude_code_py.core_types.message import AssistantMessage, Message, ToolUseBlock

logger = logging.getLogger(__name__)


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
    """Execute tool calls concurrently with Bash sibling error cascade.

    When a Bash tool produces an error result, all other concurrent tools
    are cancelled with a synthetic error message. Only Bash errors cascade;
    read-only tool failures (Read, Glob, etc.) do not affect siblings.

    This mirrors StreamingToolExecutor.ts in the TypeScript implementation.

    Args:
        tool_use_messages: Tool use blocks to execute
        assistant_messages: Parent assistant messages
        can_use_tool: Permission check function
        tool_use_context: Execution context

    Yields:
        Message updates from tool execution
    """
    from claude_code_py.utils.abort_controller import (
        AbortController,
        AbortSignal,
    )
    from claude_code_py.core_types.message import UserMessage

    # Sibling abort controller: child of the parent tool_use_context abort.
    # Fires when a Bash tool errors so sibling subprocesses die immediately.
    # Aborting this does NOT abort the parent — the query loop won't end the turn.
    sibling_controller = AbortController(
        parent_signal=tool_use_context.abort_controller.signal,
    )

    # Track whether a Bash tool has errored (triggers sibling cascade)
    has_errored = False
    errored_tool_description = ""

    # Track which tools have already produced their own error
    # (to avoid giving them a duplicate "sibling error" message)
    tools_that_errored: set[str] = set()

    # Track which tools are still pending (not yet started or in progress)
    pending_tool_ids: set[str] = {t.id for t in tool_use_messages}

    BASH_TOOL_NAME = "Bash"

    def _is_error_result(update: MessageUpdateLazy) -> bool:
        """Check if an update contains an error tool_result."""
        if not update.message or not hasattr(update.message, "message"):
            return False
        msg = update.message.message
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("is_error") is True:
                    return True
        return False

    def _get_tool_name_for_id(tool_use_id: str) -> str:
        """Get tool name from tool use ID."""
        for t in tool_use_messages:
            if t.id == tool_use_id:
                return t.name
        return ""

    def _get_tool_description(tool_use: "ToolUseBlock") -> str:
        """Get a human-readable description for a tool use."""
        name = tool_use.name
        inp = tool_use.input
        if name == BASH_TOOL_NAME and isinstance(inp, dict):
            desc = inp.get("description", "")
            cmd = inp.get("command", "")
            if desc:
                return f"Bash({desc})"
            if cmd:
                return f"Bash({cmd[:50]})"
        return name

    def _create_sibling_error_message(
        tool_use_id: str,
    ) -> UserMessage:
        """Create a synthetic error message for a tool cancelled by sibling error."""
        desc = errored_tool_description
        msg = (
            f"Cancelled: parallel tool call {desc} errored"
            if desc
            else "Cancelled: parallel tool call errored"
        )
        return UserMessage(
            message={
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": f"<tool_use_error>{msg}</tool_use_error>",
                    "is_error": True,
                }],
            }
        )

    async def run_one(
        tool_use: "ToolUseBlock",
    ) -> AsyncGenerator[MessageUpdateLazy, None]:
        nonlocal has_errored, errored_tool_description

        # Check if already cancelled by a sibling error before starting
        if has_errored and tool_use.id not in tools_that_errored:
            yield MessageUpdateLazy(
                message=_create_sibling_error_message(tool_use.id),
            )
            _mark_tool_use_complete(tool_use_context, tool_use.id)
            pending_tool_ids.discard(tool_use.id)
            return

        # Mark as in progress
        if tool_use_context.set_in_progress_tool_use_ids:
            tool_use_context.set_in_progress_tool_use_ids(
                lambda prev: prev | {tool_use.id}
            )

        parent_message = _find_parent_message(assistant_messages, tool_use.id)

        # Execute the tool, checking for sibling abort between yields
        this_tool_errored = False
        async for update in run_tool_use(
            tool_use,
            parent_message,
            can_use_tool,
            tool_use_context,
        ):
            # Check if a sibling Bash error occurred while this tool was running
            if has_errored and not this_tool_errored and tool_use.id not in tools_that_errored:
                yield MessageUpdateLazy(
                    message=_create_sibling_error_message(tool_use.id),
                )
                break

            # Check if this update is an error result
            if _is_error_result(update):
                this_tool_errored = True
                tools_that_errored.add(tool_use.id)

                # Only Bash errors cancel siblings
                if tool_use.name == BASH_TOOL_NAME:
                    has_errored = True
                    errored_tool_description = _get_tool_description(tool_use)
                    sibling_controller.abort("sibling_error")

            yield update

        _mark_tool_use_complete(tool_use_context, tool_use.id)
        pending_tool_ids.discard(tool_use.id)

    # Run all with concurrency limit
    from claude_code_py.utils.generators import all

    generators = [lambda t=tool_use: run_one(t) for tool_use in tool_use_messages]

    async for update in all(generators, get_max_tool_use_concurrency()):
        yield update

    # After all concurrent tools complete, emit synthetic errors for any
    # pending tools that were never started due to sibling error
    if has_errored:
        for tool_use_id in list(pending_tool_ids):
            if tool_use_id not in tools_that_errored:
                yield MessageUpdateLazy(
                    message=_create_sibling_error_message(tool_use_id),
                )
                _mark_tool_use_complete(tool_use_context, tool_use_id)


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
    from pydantic import ValidationError as PydanticValidationError

    tool = find_tool_by_name(tool_use_context.options.tools, tool_use.name)

    if not tool:
        # Tool not found - return error with <tool_use_error> wrapper
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"<tool_use_error>Error: No such tool available: {tool_use.name}</tool_use_error>",
                        "is_error": True,
                    }],
                }
            )
        )
        return

    try:
        # Parse input with validation error handling
        try:
            if hasattr(tool.input_schema, "model_validate"):
                parsed_input = tool.input_schema.model_validate(tool_use.input)
            else:
                parsed_input = tool.input_schema(**tool_use.input)
        except PydanticValidationError as e:
            # Format Pydantic validation errors into human-readable messages
            content = format_validation_error(tool.name, e)
            yield MessageUpdateLazy(
                message=UserMessage(
                    message={
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"<tool_use_error>{content}</tool_use_error>",
                            "is_error": True,
                        }],
                    }
                )
            )
            return

        # Run tool-level validate_input
        validation_result = await tool.validate_input(parsed_input, tool_use_context)
        if not validation_result.result:
            content = validation_result.message or "Input validation failed"
            yield MessageUpdateLazy(
                message=UserMessage(
                    message={
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"<tool_use_error>{content}</tool_use_error>",
                            "is_error": True,
                        }],
                    }
                )
            )
            return

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
            reason = perm_result.reason or "No reason provided"
            yield MessageUpdateLazy(
                message=UserMessage(
                    message={
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Permission denied: {reason}",
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

    except AbortError as e:
        # User-initiated abort - not logged as a tool failure
        content = format_error(e)
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": content,
                        "is_error": True,
                    }],
                }
            )
        )

    except ShellError as e:
        # Shell errors are expected operational errors, not bugs
        # Don't log via logError, just format and return to model
        content = format_error(e)
        logger.debug(f"{tool.name} tool error: {content[:200]}")
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": content,
                        "is_error": True,
                    }],
                }
            )
        )

    except ToolTimeoutError as e:
        # Timeout errors are retryable
        content = format_error(e)
        logger.debug(f"{tool.name} timed out: {content[:200]}")
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": content,
                        "is_error": True,
                    }],
                }
            )
        )

    except ToolValidationError as e:
        # Tool-level validation error
        content = format_error(e)
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"<tool_use_error>{content}</tool_use_error>",
                        "is_error": True,
                    }],
                }
            )
        )

    except PermissionDeniedError as e:
        # Permission denied from within tool execution
        content = format_error(e)
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": content,
                        "is_error": True,
                    }],
                }
            )
        )

    except ToolError as e:
        # Generic tool error - logged as error
        content = format_error(e)
        logger.error(f"{tool.name} tool error: {classify_tool_error(e)}")
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"<tool_use_error>Error calling tool ({tool.name}): {content}</tool_use_error>",
                        "is_error": True,
                    }],
                }
            )
        )

    except Exception as e:
        # Catch-all for unexpected errors - always logged
        content = format_error(e)
        tool_info = f" ({tool.name})" if tool else ""
        logger.error(f"Error calling tool{tool_info}: {classify_tool_error(e)}")
        yield MessageUpdateLazy(
            message=UserMessage(
                message={
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"<tool_use_error>Error calling tool{tool_info}: {content}</tool_use_error>",
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