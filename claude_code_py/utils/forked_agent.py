"""Forked Agent implementation.

This provides the ability to run background agents that share
the parent's prompt cache for efficient execution.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional, Union

from claude_code_py.core_types.message import (
    Message,
    UserMessage,
    AssistantMessage,
)
from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext, CanUseToolFn
from claude_code_py.utils.abort_controller import AbortController, create_abort_controller
from claude_code_py.utils.async_helpers import create_task_with_yield


# =============================================================================
# Global Cache Params Storage
# =============================================================================


# Module-level singleton for last cache safe params
# Written after each turn so post-turn forks (compact, prompt suggestion, etc.)
# can share the main loop's prompt cache without each caller threading params.
_last_cache_safe_params: Optional[CacheSafeParams] = None


def save_cache_safe_params(params: Optional[CacheSafeParams]) -> None:
    """Save cache safe params for later use by forked agents.

    Called at the end of each turn to enable post-turn forks
    (compact summaries, prompt suggestions, etc.) to share
    the main loop's prompt cache.

    Args:
        params: Cache safe params to save, or None to clear
    """
    global _last_cache_safe_params
    _last_cache_safe_params = params


def get_last_cache_safe_params() -> Optional[CacheSafeParams]:
    """Get the last saved cache safe params.

    Used by forked agents to share the parent's prompt cache.

    Returns:
        Last saved cache params, or None if not saved
    """
    return _last_cache_safe_params


# =============================================================================
# Types
# =============================================================================


@dataclass
class CacheSafeParams:
    """Parameters that must be identical between fork and parent for cache sharing."""

    system_prompt: str
    user_context: dict[str, str]
    system_context: dict[str, str]
    tool_use_context: ToolUseContext
    fork_context_messages: list[Message]


@dataclass
class ForkedAgentParams:
    """Parameters for running a forked agent."""

    prompt_messages: list[Message]
    cache_safe_params: CacheSafeParams
    can_use_tool: CanUseToolFn
    query_source: str
    fork_label: str

    # Optional overrides
    max_output_tokens: Optional[int] = None
    max_turns: Optional[int] = None
    skip_transcript: bool = False
    skip_cache_write: bool = False

    # Callbacks
    on_message: Optional[Callable[[Message], None]] = None


@dataclass
class ForkedAgentResult:
    """Result from running a forked agent."""

    messages: list[Message]
    total_usage: dict[str, int]


@dataclass
class SubagentContextOverrides:
    """Overrides for creating subagent context."""

    options: Optional[Any] = None
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    messages: Optional[list[Message]] = None
    read_file_state: Optional[dict[str, Any]] = None
    abort_controller: Optional[AbortController] = None
    get_app_state: Optional[Callable] = None

    # Sharing options
    share_set_app_state: bool = False
    share_set_response_length: bool = False
    share_abort_controller: bool = False


# =============================================================================
# Context Creation
# =============================================================================


def create_cache_safe_params(
    system_prompt: str,
    user_context: dict[str, str],
    system_context: dict[str, str],
    tool_use_context: ToolUseContext,
    messages: list[Message],
) -> CacheSafeParams:
    """Create cache-safe parameters from context.

    Args:
        system_prompt: System prompt string
        user_context: User context dict
        system_context: System context dict
        tool_use_context: Tool use context
        messages: Current messages

    Returns:
        Cache safe parameters
    """
    return CacheSafeParams(
        system_prompt=system_prompt,
        user_context=user_context,
        system_context=system_context,
        tool_use_context=tool_use_context,
        fork_context_messages=messages.copy(),
    )


def create_subagent_context(
    parent_context: ToolUseContext,
    overrides: Optional[SubagentContextOverrides] = None,
) -> ToolUseContext:
    """Create an isolated context for subagents.

    By default, all mutable state is isolated to prevent interference.

    Args:
        parent_context: Parent's ToolUseContext
        overrides: Optional overrides

    Returns:
        New isolated context
    """
    overrides = overrides or SubagentContextOverrides()

    # Determine abort controller
    if overrides.abort_controller:
        abort_controller = overrides.abort_controller
    elif overrides.share_abort_controller:
        abort_controller = parent_context.abort_controller
    else:
        # Create child controller linked to parent's signal
        parent_signal = parent_context.abort_controller.signal if parent_context.abort_controller else None
        abort_controller = create_abort_controller(parent_signal)

    # Clone file state
    if overrides.read_file_state:
        read_file_state = overrides.read_file_state
    else:
        read_file_state = copy.deepcopy(parent_context.read_file_state)

    # Get app state function
    if overrides.get_app_state:
        get_app_state = overrides.get_app_state
    elif overrides.share_abort_controller:
        get_app_state = parent_context.get_app_state
    else:
        # Wrap to set should_avoid_permission_prompts
        def get_app_state() -> Any:
            state = parent_context.get_app_state()
            if hasattr(state, "tool_permission_context") and \
               getattr(state.tool_permission_context, "should_avoid_permission_prompts", False):
                return state
            # Set flag to avoid UI prompts in background agents
            return state  # For now, just return the state

    # Set app state callback
    if overrides.share_set_app_state:
        set_app_state = parent_context.set_app_state
    else:
        # No-op for background agents
        def set_app_state(state: Any) -> None:
            pass

    # Create new context
    return ToolUseContext(
        options=overrides.options or parent_context.options,
        abort_controller=abort_controller,
        messages=overrides.messages or [],
        read_file_state=read_file_state,
        get_app_state=get_app_state,
        set_app_state=set_app_state,
    )


# =============================================================================
# Forked Agent Execution
# =============================================================================


async def run_forked_agent(
    params: ForkedAgentParams,
) -> ForkedAgentResult:
    """Run a forked agent with cache sharing.

    This runs a separate query loop in the background with:
    - Shared prompt cache (identical system prompt, tools, model)
    - Isolated state (separate messages, file cache)
    - Custom permission handler (can_use_tool)

    Args:
        params: Forked agent parameters

    Returns:
        Result with messages and usage
    """
    from claude_code_py.engine.query import query, QueryParams

    # Create isolated context
    context = create_subagent_context(
        params.cache_safe_params.tool_use_context,
        SubagentContextOverrides(
            read_file_state=params.cache_safe_params.tool_use_context.read_file_state.copy(),
        ),
    )

    # Build initial messages: parent conversation history + new prompt
    # This enables the forked agent to understand context from parent conversation
    initial_messages = params.cache_safe_params.fork_context_messages + params.prompt_messages

    # Build query params
    query_params = QueryParams(
        messages=initial_messages,
        system_prompt=params.cache_safe_params.system_prompt,
        user_context=params.cache_safe_params.user_context,
        system_context=params.cache_safe_params.system_context,
        can_use_tool=params.can_use_tool,
        tool_use_context=context,
        query_source=params.query_source,
        max_turns=params.max_turns,
        skip_cache_write=params.skip_cache_write,
    )

    # Run query loop
    all_messages: list[Message] = []
    total_usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }

    try:
        async for event in query(query_params):
            if isinstance(event, Message):
                all_messages.append(event)

                # Call on_message callback if provided
                if params.on_message:
                    params.on_message(event)

                # Accumulate usage from assistant messages
                if isinstance(event, AssistantMessage) and hasattr(event, "usage"):
                    usage = event.usage or {}
                    total_usage["input_tokens"] += usage.get("input_tokens", 0)
                    total_usage["output_tokens"] += usage.get("output_tokens", 0)
                    total_usage["cache_read_input_tokens"] += usage.get("cache_read_input_tokens", 0)
                    total_usage["cache_creation_input_tokens"] += usage.get("cache_creation_input_tokens", 0)

    except asyncio.CancelledError:
        # Agent was cancelled
        pass
    except Exception as e:
        # Log error but don't propagate - forked agents run in background
        import traceback
        traceback.print_exc()

    return ForkedAgentResult(
        messages=all_messages,
        total_usage=total_usage,
    )


async def run_forked_agent_background(
    params: ForkedAgentParams,
) -> asyncio.Task[ForkedAgentResult]:
    """Run a forked agent in the background.

    This returns immediately with a Task that can be awaited later.
    Uses create_task_with_yield to ensure the agent starts immediately.

    Args:
        params: Forked agent parameters

    Returns:
        Asyncio Task for the result
    """
    # Create task with immediate execution guarantee
    task = await create_task_with_yield(run_forked_agent(params))

    return task


# =============================================================================
# Helper Functions
# =============================================================================


def create_user_message(content: Union[str, list[dict[str, Any]]]) -> UserMessage:
    """Create a user message from content.

    Args:
        content: String or content blocks

    Returns:
        UserMessage
    """
    if isinstance(content, str):
        message_content = content
    else:
        message_content = content

    return UserMessage(
        message={
            "role": "user",
            "content": message_content,
        }
    )


def extract_text_from_messages(messages: list[Message]) -> str:
    """Extract text content from messages.

    Args:
        messages: Message list

    Returns:
        Combined text content
    """
    texts = []

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            content = msg.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))

    return "\n".join(texts)


def get_last_assistant_message(messages: list[Message]) -> Optional[AssistantMessage]:
    """Get the last assistant message from a list.

    Args:
        messages: Message list

    Returns:
        Last assistant message or None
    """
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            return msg
    return None


# =============================================================================
# Permission Helpers for Forked Agents
# =============================================================================


def create_memory_file_can_use_tool(memory_path: str) -> CanUseToolFn:
    """Create a can_use_tool function that only allows Edit on memory file.

    Args:
        memory_path: Path to the memory file

    Returns:
        Permission check function
    """
    async def can_use_tool(
        tool: Tool,
        input: Any,
        context: ToolUseContext,
        assistant_message: AssistantMessage,
        tool_use_id: Optional[str] = None,
        force_decision: Optional[str] = None,
    ) -> Any:
        """Permission check for memory file editing."""
        from claude_code_py.core_types.permissions import PermissionResult, PermissionBehavior

        tool_name = getattr(tool, "name", "")

        # Allow Edit tool on exact memory path
        if tool_name == "Edit":
            if hasattr(input, "file_path"):
                input_path = str(input.file_path)
                if input_path == memory_path:
                    return PermissionResult.allow(updated_input=input)
            elif isinstance(input, dict):
                input_path = input.get("file_path", "")
                if input_path == memory_path:
                    return PermissionResult.allow(updated_input=input)

        # Allow Write tool for creating the file
        if tool_name == "Write":
            if hasattr(input, "file_path"):
                input_path = str(input.file_path)
                if input_path == memory_path:
                    return PermissionResult.allow(updated_input=input)
            elif isinstance(input, dict):
                input_path = input.get("file_path", "")
                if input_path == memory_path:
                    return PermissionResult.allow(updated_input=input)

        # Allow Read tool for reading the file
        if tool_name == "Read":
            return PermissionResult.allow(updated_input=input)

        # Deny everything else
        return PermissionResult.deny(
            reason=f"Only Read/Write/Edit on {memory_path} is allowed for session memory extraction"
        )

    return can_use_tool


def create_read_only_can_use_tool() -> CanUseToolFn:
    """Create a can_use_tool function that only allows read-only tools.

    Returns:
        Permission check function
    """
    read_only_tools = {"Read", "Glob", "Grep", "LSP", "ToolSearch"}

    async def can_use_tool(
        tool: Tool,
        input: Any,
        context: ToolUseContext,
        assistant_message: AssistantMessage,
        tool_use_id: Optional[str] = None,
        force_decision: Optional[str] = None,
    ) -> Any:
        """Permission check for read-only tools."""
        from claude_code_py.core_types.permissions import PermissionResult

        tool_name = getattr(tool, "name", "")

        if tool_name in read_only_tools:
            return PermissionResult.allow(updated_input=input)

        return PermissionResult.deny(
            reason="Only read-only tools are allowed for this forked agent"
        )

    return can_use_tool