"""Query loop implementation.

This implements the main query loop that processes messages and executes tools,
with integrated auto-compact handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any, AsyncGenerator, Callable, Optional, Union

from claude_code_py.tool.base import Tool, find_tool_by_name
from claude_code_py.tool.context import ToolUseContext, CanUseToolFn
from claude_code_py.tool.result import ToolResult, ToolError
from claude_code_py.core_types.message import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ProgressMessage,
    ToolUseBlock,
)
from claude_code_py.utils.abort_controller import AbortError
from claude_code_py.services.compact import (
    auto_compact_if_needed,
    execute_compact_flow,
    build_post_compact_messages,
    CompactResult,
)
from claude_code_py.services.micro_compact import (
    maybe_microcompact,
    MicrocompactResult,
    consume_pending_cache_edits,
    get_pinned_cache_edits,
    mark_tools_sent_to_api_state,
    reset_microcompact_state,
    is_model_supported_for_cache_editing,
    is_cached_microcompact_enabled,
)
from claude_code_py.utils.context import CAPPED_DEFAULT_MAX_TOKENS
from claude_code_py.utils.forked_agent import (
    save_cache_safe_params,
    create_cache_safe_params,
)
from claude_code_py.constants import QuerySource, is_main_thread_source
from claude_code_py.utils.debug_log import debug_log

logger = logging.getLogger(__name__)


# =============================================================================
# Extraction frequency threshold (like TypeScript's tengu_bramble_lintel)
# =============================================================================

EXTRACTION_TURN_THRESHOLD = 1  # Default: every eligible turn
MIN_MESSAGES_FOR_EXTRACTION = 5  # Minimum messages to trigger extraction


# =============================================================================
# Task Notification Handling
# =============================================================================


def is_task_notification(content: str) -> Optional[dict]:
    """Parse task-notification from message content.

    Args:
        content: Message content string to parse

    Returns:
        Parsed notification dict with task_id, status, summary, result
        or None if not a task notification
    """
    match = re.search(
        r'<task-notification>\s*<task-id>([^<]+)</task-id>\s*<status>([^<]+)</status>\s*<summary>([^<]+)</summary>\s*<result>([^<]+)</result>\s*</task-notification>',
        content,
        re.DOTALL
    )
    if match:
        return {
            "task_id": match.group(1),
            "status": match.group(2),
            "summary": match.group(3),
            "result": match.group(4),
        }
    return None


def handle_task_notification(notification: dict, context: Any) -> None:
    """Handle a task notification from a worker.

    Args:
        notification: Parsed notification dict with task_id, status, summary, result
        context: Execution context (ToolUseContext or similar)
    """
    task_id = notification["task_id"]
    status = notification["status"]
    summary = notification["summary"]

    logger.info(f"Task notification: {task_id} -> {status}")

    # Update task state if in AppState
    if hasattr(context, "set_app_state"):
        context.set_app_state(lambda prev: replace(
            prev,
            tasks={
                **prev.tasks,
                task_id: {
                    **prev.tasks.get(task_id, {}),
                    "status": status,
                    "summary": summary,
                },
            },
        ))


@dataclass
class QueryParams:
    """Parameters for the query function."""

    messages: list[Message]
    system_prompt: str
    user_context: dict[str, str]
    system_context: dict[str, str]
    can_use_tool: CanUseToolFn
    tool_use_context: ToolUseContext
    fallback_model: Optional[str] = None
    query_source: str = QuerySource.REPL_MAIN_THREAD  # Default to main thread
    max_output_tokens_override: Optional[int] = None
    max_turns: Optional[int] = None
    skip_cache_write: bool = False
    task_budget: Optional[dict[str, int]] = None


@dataclass
class State:
    """Mutable state carried between query loop iterations."""

    messages: list[Message]
    tool_use_context: ToolUseContext
    auto_compact_tracking: Optional[dict[str, Any]] = None
    max_output_tokens_recovery_count: int = 0
    has_attempted_reactive_compact: bool = False
    max_output_tokens_override: Optional[int] = None
    pending_tool_use_summary: Optional[Any] = None
    stop_hook_active: Optional[bool] = None
    turn_count: int = 1
    transition: Optional[str] = None
    last_compact_result: Optional[CompactResult] = None
    consecutive_compact_failures: int = 0
    done: bool = False  # Flag to signal conversation completion
    # Microcompact tracking (moved from auto_compact_if_needed)
    last_microcompact_result: Optional[MicrocompactResult] = None
    pending_cache_edits: Optional[list[dict[str, Any]]] = None


@dataclass
class Continue:
    """Continue state marker."""

    reason: str
    state: State


async def query(
    params: QueryParams,
) -> AsyncGenerator[Union[Message, Any], None]:
    """Execute the query loop.

    This is the main loop that:
    1. Checks for auto-compact need
    2. Sends messages to the API
    3. Processes the streaming response
    4. Executes tool calls
    5. Returns tool results
    6. Repeats until done

    Args:
        params: Query parameters

    Yields:
        Messages and events from the conversation
    """
    # Initialize state
    state = State(
        messages=params.messages,
        tool_use_context=params.tool_use_context,
    )

    # Main loop
    while True:
        try:
            # Check for abort
            params.tool_use_context.abort_controller.signal.throw_if_aborted()

            # Check for task notifications in user messages
            for message in state.messages:
                if hasattr(message, "content") and getattr(message, "type", None) == "user":
                    content = str(message.content) if hasattr(message, "content") else str(message.message.get("content", ""))
                    notification = is_task_notification(content)
                    if notification:
                        handle_task_notification(notification, params.tool_use_context)
                        # Remove the notification message from state to prevent re-processing
                        state.messages.remove(message)
                        break  # Process one notification per iteration

            # NEW: Apply microcompact BEFORE autocompact (moved from auto_compact_if_needed)
            async for event in _check_microcompact(params, state):
                yield event

            # Check for auto-compact before each iteration (now only session_memory + standard)
            async for event in _check_auto_compact(params, state):
                yield event

            # Process one iteration
            async for event in _process_iteration(params, state):
                yield event

            # Check for session memory extraction after iteration
            await _check_session_memory(params, state)

            # Check if conversation is done (no more tool calls)
            if state.done:
                # Save cache params for post-turn forks
                _save_cache_params_at_turn_end(params, state)
                # Turn ended - trigger auto memory extraction (fire-and-forget)
                _trigger_auto_memory_extraction(params, state)
                break  # Exit generator - conversation complete

            # Check if we should continue
            if state.turn_count > (params.max_turns or float("inf")):
                yield create_max_turns_message(state.turn_count)
                break  # Exit generator

            state.turn_count += 1

        except AbortError:
            yield create_error_message("Aborted by user")
            break
        except ContextLengthError as e:
            # Handle 413 context_length_exceeded
            async for event in _handle_context_overflow(params, state, str(e)):
                yield event
            continue
        except Exception as e:
            yield create_error_message(str(e))
            break


# =============================================================================
# Cache Params Saving
# =============================================================================


def _save_cache_params_at_turn_end(
    params: QueryParams,
    state: State,
) -> None:
    """Save cache params at turn end for post-turn forks.

    This enables forked agents (compact, prompt suggestions, etc.)
    to share the main loop's prompt cache by using identical
    cache-key params.

    Only saves for main session queries (repl_main_thread*, sdk).

    Args:
        params: Query parameters
        state: Mutable state (contains messages)
    """
    # Only save for main session queries (repl_main_thread and its variants, sdk)
    if not is_main_thread_source(params.query_source) and params.query_source != QuerySource.SDK:
        return

    # Create and save cache params
    cache_params = create_cache_safe_params(
        system_prompt=params.system_prompt,
        user_context=params.user_context,
        system_context=params.system_context,
        tool_use_context=params.tool_use_context,
        messages=state.messages,
    )
    save_cache_safe_params(cache_params)


# =============================================================================
# Auto Memory Extraction
# =============================================================================


def _trigger_auto_memory_extraction(
    params: QueryParams,
    state: State,
) -> None:
    """Trigger auto memory extraction at turn end.

    This implements the stop hook trigger from TypeScript's handleStopHooks.
    Runs as fire-and-forget background task, does not block main conversation.

    Args:
        params: Query parameters
        state: Mutable state (contains messages)
    """
    # Skip if not main thread (no agentId in TypeScript)
    agent_id = getattr(params.tool_use_context, "agent_id", None)
    if agent_id:
        return

    # Skip if not in extract mode (similar to isExtractModeActive)
    from claude_code_py.memory import is_auto_memory_enabled
    if not is_auto_memory_enabled():
        return

    # Skip if bare/simple mode
    import os
    if os.environ.get("CLAUDE_CODE_SIMPLE"):
        return

    # Use extract.py's state for turn threshold check
    from claude_code_py.memory.extract import get_extraction_state

    extraction_state = get_extraction_state()

    # Check turn threshold (tengu_bramble_lintel equivalent)
    extraction_state.turns_since_last_extraction += 1
    if extraction_state.turns_since_last_extraction < EXTRACTION_TURN_THRESHOLD:
        return

    # Reset counter
    extraction_state.turns_since_last_extraction = 0

    # Check minimum messages
    visible_count = sum(1 for m in state.messages if m.type in ("user", "assistant"))
    if visible_count < MIN_MESSAGES_FOR_EXTRACTION:
        return

    # Fire-and-forget: create background task
    # execute_extract_memories handles in_progress check and pending_context stashing
    asyncio.create_task(_run_auto_memory_extraction(params, state))


async def _run_auto_memory_extraction(
    params: QueryParams,
    state: State,
) -> None:
    """Run auto memory extraction in background.

    Delegates to extract.py which has its own state management and
    trailing run handling.

    Args:
        params: Query parameters (for tool_use_context)
        state: Mutable state (contains messages)
    """
    # Import and delegate to extract.py
    # extract.py has its own _extraction_state with in_progress,
    # pending_context, and trailing run handling
    from claude_code_py.memory.extract import execute_extract_memories

    # Build context for forked agent
    context = {
        "tool_use_context": params.tool_use_context,
        "system_prompt": params.system_prompt,
        "user_context": params.user_context,
        "system_context": params.system_context,
    }

    # Call the extraction function with context
    await execute_extract_memories(
        messages=state.messages,
        context=context,
    )


async def drain_pending_extraction(timeout_ms: int = 60000) -> None:
    """Wait for pending auto memory extraction to complete.

    Called during shutdown to ensure extractions finish.

    Args:
        timeout_ms: Maximum time to wait in milliseconds
    """
    from claude_code_py.memory.extract import is_extraction_in_progress

    if not is_extraction_in_progress():
        return

    start = asyncio.get_event_loop().time()
    while is_extraction_in_progress():
        if (asyncio.get_event_loop().time() - start) * 1000 > timeout_ms:
            break
        await asyncio.sleep(0.1)


# =============================================================================
# Microcompact Check (moved from auto_compact_if_needed)
# =============================================================================


async def _check_microcompact(
    params: QueryParams,
    state: State,
) -> AsyncGenerator[Union[Message, Any], None]:
    """Check and execute microcompact if appropriate.

    Microcompact is now called separately at the beginning of each query loop
    iteration, before auto-compact. This matches the TypeScript architecture
    where microcompact is called independently before autocompact.

    Priority:
    1. Time-based microcompact (if cache expired)
    2. Cached microcompact (if cache_edits supported)

    IMPORTANT: For cached microcompact, messages are NOT modified locally.
    The cache_edits are queued for the API layer to handle.

    Args:
        params: Query parameters
        state: Mutable state

    Yields:
        Progress message if microcompact was applied
    """
    # Guard: only for main thread queries (repl_main_thread, sdk)
    # Time-based: needs main thread/sdk to track cache expiration
    # Cached: needs main REPL thread (not sdk) because cachedMCState is global
    if not is_main_thread_source(params.query_source):
        return

    # Determine model
    model = params.fallback_model or "claude-sonnet-4-"

    # Execute microcompact
    result = await maybe_microcompact(
        messages=state.messages,
        model=model,
        query_source=params.query_source,
    )

    if result and result.success:
        state.last_microcompact_result = result

        # KEY CHANGE: Only apply modified_messages for time_based
        # Cached microcompact keeps messages unchanged (handled at API layer)
        if result.type == "time_based" and result.modified_messages:
            state.messages = result.modified_messages
            # Reset cached MC state since we modified content directly
            reset_microcompact_state()

        # Store pending cache_edits for later API call (cached microcompact)
        if result.pending_cache_edits:
            state.pending_cache_edits = result.pending_cache_edits

        # Yield progress message
        yield ProgressMessage(
            content=f"Microcompact ({result.type}): {result.messages_removed} tool results removed"
        )


# =============================================================================
# Auto Compact Check
# =============================================================================


async def _check_auto_compact(
    params: QueryParams,
    state: State,
) -> AsyncGenerator[Union[Message, Any], None]:
    """Check and execute auto-compact if needed.

    Args:
        params: Query parameters
        state: Mutable state

    Yields:
        Post-compact messages (boundary + summary + messagesToKeep + attachments)
    """
    # Guard: only for main thread queries (repl_main_thread, sdk)
    # Forked agents (session_memory, compact) would deadlock if blocked here
    if not is_main_thread_source(params.query_source) and params.query_source != QuerySource.SDK:
        return

    # Recursion guard: forked agents inherit full conversation
    if params.query_source in ('session_memory', 'compact'):
        return

    # Skip if too many failures
    if state.consecutive_compact_failures >= 3:
        return

    # Determine model
    model = params.fallback_model or "claude-sonnet-4-"

    # Get cache params for prompt cache sharing
    from claude_code_py.utils.forked_agent import get_last_cache_safe_params
    cache_params = get_last_cache_safe_params()

    # Progress callback - yields messages
    progress_msgs: list[ProgressMessage] = []

    def on_progress(msg: str) -> None:
        progress_msgs.append(ProgressMessage(content=msg))

    # Check for auto-compact (with cache sharing support)
    compact_result = await auto_compact_if_needed(
        messages=state.messages,
        model=model,
        on_progress=on_progress,
        cache_safe_params=cache_params,
        query_source=params.query_source,
    )

    # Yield progress messages
    for msg in progress_msgs:
        yield msg

    if compact_result:
        state.last_compact_result = compact_result

        if compact_result.success:
            state.consecutive_compact_failures = 0

            # Build post-compact messages (matching TypeScript)
            # This includes: boundary + summary + messagesToKeep + attachments + hookResults
            post_compact_messages = build_post_compact_messages(compact_result, state.messages)

            # Yield all post-compact messages (matching TypeScript query.ts)
            for message in post_compact_messages:
                yield message

            # Update state with new messages
            state.messages = post_compact_messages
        else:
            state.consecutive_compact_failures += 1


async def _check_session_memory(
    params: QueryParams,
    state: State,
) -> None:
    """Check and extract session memory if needed.

    This runs in the background without blocking the main conversation.

    Args:
        params: Query parameters
        state: Mutable state
    """
    # Only run on main REPL thread (repl_main_thread and variants, sdk)
    if not is_main_thread_source(params.query_source) and params.query_source != QuerySource.SDK:
        return

    # Check if we should extract
    from claude_code_py.memory.session_memory import (
        should_extract_memory,
        extract_session_memory,
        is_session_memory_gate_enabled,
    )

    # Skip if gate disabled
    if not is_session_memory_gate_enabled():
        return

    # Check thresholds
    if not should_extract_memory(state.messages):
        return

    # Run extraction in background (fire and forget)
    # Don't await - let it run independently
    try:
        import asyncio

        async def run_extraction():
            await extract_session_memory(
                messages=state.messages,
                tool_use_context=params.tool_use_context,
                system_prompt=params.system_prompt,
                user_context=params.user_context,
                system_context=params.system_context,
            )

        # Create background task
        asyncio.create_task(run_extraction())

    except Exception:
        # Don't fail the main conversation if session memory extraction fails
        pass


async def _handle_context_overflow(
    params: QueryParams,
    state: State,
    error_message: str,
) -> AsyncGenerator[Union[Message, Any], None]:
    """Handle 413 context_length_exceeded error.

    Args:
        params: Query parameters
        state: Mutable state
        error_message: Error message

    Yields:
        Messages from recovery attempt
    """
    if state.has_attempted_reactive_compact:
        yield create_error_message("Context overflow recovery failed")
        return

    state.has_attempted_reactive_compact = True

    # Yield progress
    yield ProgressMessage(content="Handling context overflow...")

    # Try reactive compact
    model = params.fallback_model or "claude-sonnet-4-"

    compact_result = await execute_compact_flow(
        messages=state.messages,
        model=model,
        query_source=params.query_source,
        is_reactive=True,
        error_message=error_message,
    )

    if compact_result and compact_result.success:
        post_compact_messages = build_post_compact_messages(compact_result, state.messages)
        state.messages = post_compact_messages
        # Yield all post-compact messages (matching TypeScript)
        for message in post_compact_messages:
            yield message
    else:
        yield create_error_message("Unable to recover from context overflow")


async def _process_iteration(
    params: QueryParams,
    state: State,
) -> AsyncGenerator[Union[Message, Any], None]:
    """Process one iteration of the query loop.

    Args:
        params: Query parameters
        state: Mutable state

    Yields:
        Messages and events
    """
    # 1. Normalize messages for API
    api_messages = normalize_messages_for_api(state.messages)

    # Debug: Log messages sent to API (only for team-lead)
    from claude_code_py.utils.teammate_context import is_team_lead, get_current_team_name
    team_name = get_current_team_name()
    if team_name and is_team_lead():
        debug_log("[TEAM_LEAD]", f"=== SENDING TO API ===")
        debug_log("[TEAM_LEAD]", f"query_source: {params.query_source}")
        debug_log("[TEAM_LEAD]", f"team_name: {team_name}")
        debug_log("[TEAM_LEAD]", f"message_count: {len(api_messages)}")
        for i, msg in enumerate(api_messages):
            role = msg.get("role", "unknown")
            # Truncate content for readability
            content = msg.get("content", "")
            if isinstance(content, str):
                preview = content[:500] + "..." if len(content) > 500 else content
            elif isinstance(content, list):
                # Summarize content blocks
                block_types = [b.get("type", "unknown") for b in content]
                preview = f"[{len(content)} blocks: {', '.join(block_types)}]"
            else:
                preview = str(content)[:200]
            debug_log("[TEAM_LEAD]", f"  [{i}] role={role}, content={preview}")
        debug_log("[TEAM_LEAD]", f"======================")

    # 2. Call API with cache breakpoints
    assistant_message = await _call_api(
        messages=api_messages,
        system_prompt=params.system_prompt,
        tools=params.tool_use_context.options.tools,
        query_source=params.query_source,
        pending_cache_edits=state.pending_cache_edits,
        thinking_config=params.tool_use_context.options.thinking_config,
        max_output_tokens_override=params.max_output_tokens_override,
        abort_controller=params.tool_use_context.abort_controller,
    )

    # Debug: Log assistant response (only for team-lead)
    if team_name and is_team_lead():
        debug_log("[TEAM_LEAD]", f"=== ASSISTANT RESPONSE ===")
        debug_log("[TEAM_LEAD]", f"uuid: {assistant_message.uuid}")
        content = assistant_message.message.get("content", [])
        if isinstance(content, list):
            for i, block in enumerate(content):
                block_type = block.get("type", "unknown")
                if block_type == "text":
                    text = block.get("text", "")
                    preview = text[:300] + "..." if len(text) > 300 else text
                    debug_log("[TEAM_LEAD]", f"  [{i}] type=text, text={preview}")
                elif block_type == "thinking":
                    thinking = block.get("thinking", "")
                    preview = thinking[:200] + "..." if len(thinking) > 200 else thinking
                    debug_log("[TEAM_LEAD]", f"  [{i}] type=thinking, thinking={preview}")
                elif block_type == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_id = block.get("id", "unknown")
                    debug_log("[TEAM_LEAD]", f"  [{i}] type=tool_use, name={tool_name}, id={tool_id}")
                else:
                    debug_log("[TEAM_LEAD]", f"  [{i}] type={block_type}")
        debug_log("[TEAM_LEAD]", f"==========================")

    # Yield assistant message
    yield assistant_message
    state.messages.append(assistant_message)

    # 3. Extract tool use blocks
    tool_use_blocks = extract_tool_use_blocks(assistant_message)

    if not tool_use_blocks:
        # No tools to execute - conversation is complete
        state.done = True
        return

    # 4. Execute tools
    async for update in _execute_tools(
        tool_use_blocks=tool_use_blocks,
        assistant_message=assistant_message,
        params=params,
        state=state,
    ):
        yield update


async def _call_api(
    messages: list[dict[str, Any]],
    system_prompt: str,
    tools: list[Tool],
    query_source: Optional[str] = None,
    pending_cache_edits: Optional[list[dict[str, Any]]] = None,
    thinking_config: Optional[Any] = None,
    max_output_tokens_override: Optional[int] = None,
    abort_controller: Optional[Any] = None,
) -> AssistantMessage:
    """Call the API using the configured endpoint.

    Args:
        messages: Normalized messages
        system_prompt: System prompt
        tools: Available tools
        query_source: Source of the query
        pending_cache_edits: Pending cache_edits from cached microcompact
        thinking_config: Thinking configuration (ThinkingConfig from tool/context.py)
        max_output_tokens_override: Override for max output tokens
        abort_controller: Optional abort controller to check during API call

    Returns:
        Assistant message from the API
    """
    # Check API configuration first
    from claude_code_py.utils.api_config import get_api_config
    from claude_code_py.utils.cache import add_cache_breakpoints

    config = get_api_config()
    model = config.model

    # Check if cached MC should be used
    use_cached_mc = (
        is_model_supported_for_cache_editing(model) and
        is_cached_microcompact_enabled() and
        is_main_thread_source(query_source) and
        query_source != "sdk"
    )

    # Get pending and pinned cache edits
    new_cache_edits = None
    pinned_edits = []

    if use_cached_mc:
        # Consume pending cache_edits from microcompact
        new_cache_edits = consume_pending_cache_edits()
        pinned_edits = [
            {"userMessageIndex": p.userMessageIndex, "block": p.block}
            for p in get_pinned_cache_edits()
        ]

    # Add cache breakpoints before API call
    # This adds: cache_control, cache_edits, and cache_reference
    cached_messages = add_cache_breakpoints(
        messages=messages,
        enable_prompt_caching=True,
        query_source=query_source,
        use_cached_mc=use_cached_mc,
        new_cache_edits=new_cache_edits,
        pinned_edits=pinned_edits,
    )

    # Check if anthropic is available
    try:
        import anthropic
    except ImportError:
        # Build a helpful message based on available tools
        tool_names = [t.name for t in tools] if tools else []
        tool_list = ", ".join(tool_names[:5]) if tool_names else "none"
        if len(tool_names) > 5:
            tool_list += f" and {len(tool_names) - 5} more"

        # Get the last user message for context
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user_msg = content[:100]
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            last_user_msg = block.get("text", "")[:100]
                            break
                break

        return AssistantMessage(
            message={
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"""⚠️ **API Not Configured**

To use Claude Code, you need to install the Anthropic SDK and configure your API key:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your-key
```

Or use a custom endpoint:
```bash
export ANTHROPIC_BASE_URL=https://your-endpoint
export ANTHROPIC_AUTH_TOKEN=your-token
```

**Available Tools:** {tool_list}

**Your message:** "{last_user_msg}"

I cannot process your request without a valid API connection."""
                }],
            }
        )

    # Check if API is configured
    if not config.is_valid():
        return AssistantMessage(
            message={
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "⚠️ No API key configured. Please set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN environment variable."
                }],
            }
        )

    client = anthropic.AsyncAnthropic(**config.to_anthropic_kwargs())

    # Build tool schemas
    # Call tool.prompt() to get the description for API
    tool_schemas = []
    for tool in tools:
        # prompt() is async and returns the tool description for API
        tool_desc = await tool.prompt({})

        # Get input schema - use inputJSONSchema if available (MCP tools), else convert from Pydantic
        if hasattr(tool, 'input_json_schema') and tool.input_json_schema:
            input_schema = tool.input_json_schema
        elif hasattr(tool.input_schema, "model_json_schema"):
            input_schema = tool.input_schema.model_json_schema()
        else:
            input_schema = {"type": "object"}

        tool_schemas.append({
            "name": tool.name,
            "description": tool_desc,
            "input_schema": input_schema,
        })

    # Make API call
    model = config.model

    # Determine max_tokens
    max_tokens = max_output_tokens_override or CAPPED_DEFAULT_MAX_TOKENS

    # Build thinking parameter
    from claude_code_py.utils.thinking import (
        ThinkingConfig as UtilsThinkingConfig,
        build_thinking_param,
    )

    # Convert thinking_config if provided
    thinking_param = None
    if thinking_config is not None:
        # Handle both ThinkingConfig from tool/context.py and utils/thinking.py
        if hasattr(thinking_config, 'type'):
            config_obj = UtilsThinkingConfig(
                type=thinking_config.type,
                budget_tokens=getattr(thinking_config, 'budget_tokens', None),
            )
            thinking_param = build_thinking_param(
                thinking_config=config_obj,
                model=model,
                max_output_tokens=max_tokens,
            )

    # Debug: Log full API request (only for team-lead)
    from claude_code_py.utils.teammate_context import is_team_lead, get_current_team_name
    team_name = get_current_team_name()
    if team_name and is_team_lead():
        debug_log("[TEAM_LEAD]", f"=== FULL API REQUEST ===")
        debug_log("[TEAM_LEAD]", f"model: {model}")
        debug_log("[TEAM_LEAD]", f"max_tokens: {max_tokens}")
        debug_log("[TEAM_LEAD]", f"thinking: {thinking_param}")
        debug_log("[TEAM_LEAD]", f"system_prompt length: {len(system_prompt)}")
        debug_log("[TEAM_LEAD]", f"tools count: {len(tool_schemas)}")
        debug_log("[TEAM_LEAD]", f"messages count: {len(cached_messages)}")
        debug_log("[TEAM_LEAD]", f"--- SYSTEM PROMPT ---")
        debug_log("[TEAM_LEAD]", system_prompt[:2000] + "..." if len(system_prompt) > 2000 else system_prompt)
        debug_log("[TEAM_LEAD]", f"--- MESSAGES (JSON) ---")
        try:
            messages_json = json.dumps(cached_messages, indent=2, ensure_ascii=False)
            # Limit output size
            if len(messages_json) > 10000:
                debug_log("[TEAM_LEAD]", messages_json[:10000] + "\n... (truncated)")
            else:
                debug_log("[TEAM_LEAD]", messages_json)
        except Exception as e:
            debug_log("[TEAM_LEAD]", f"Failed to serialize messages: {e}")
        debug_log("[TEAM_LEAD]", f"========================")

    # Build API request params
    api_params = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": cached_messages,
    }

    # Add tools if present
    if tool_schemas:
        api_params["tools"] = tool_schemas

    # Add thinking parameter if configured
    if thinking_param is not None:
        api_params["thinking"] = thinking_param

    # Make API call with abort support
    import asyncio

    async def make_api_call():
        return await client.messages.create(**api_params)

    # Create task for API call
    api_task = asyncio.create_task(make_api_call())

    # Poll for completion or abort
    while not api_task.done():
        # Check abort controller
        if abort_controller and abort_controller.signal.aborted:
            api_task.cancel()
            try:
                await api_task
            except asyncio.CancelledError:
                pass
            raise AbortError("API call aborted by user")

        await asyncio.sleep(0.05)  # Check every 50ms for faster response

    # Get result
    try:
        response = api_task.result()
    except asyncio.CancelledError:
        raise AbortError("API call cancelled")

    # Debug: Log full API response (only for team-lead)
    if team_name and is_team_lead():
        debug_log("[TEAM_LEAD]", f"=== FULL API RESPONSE ===")
        debug_log("[TEAM_LEAD]", f"model: {response.model}")
        debug_log("[TEAM_LEAD]", f"stop_reason: {response.stop_reason}")
        debug_log("[TEAM_LEAD]", f"usage: {response.usage}")
        debug_log("[TEAM_LEAD]", f"content blocks: {len(response.content)}")
        debug_log("[TEAM_LEAD]", f"--- CONTENT BLOCKS (JSON) ---")
        response_blocks = []
        for block in response.content:
            if hasattr(block, "text"):
                response_blocks.append({"type": "text", "text": block.text})
            elif hasattr(block, "thinking"):
                response_blocks.append({"type": "thinking", "thinking": block.thinking})
            elif hasattr(block, "name"):
                response_blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        try:
            response_json = json.dumps(response_blocks, indent=2, ensure_ascii=False)
            if len(response_json) > 10000:
                debug_log("[TEAM_LEAD]", response_json[:10000] + "\n... (truncated)")
            else:
                debug_log("[TEAM_LEAD]", response_json)
        except Exception as e:
            debug_log("[TEAM_LEAD]", f"Failed to serialize response: {e}")
        debug_log("[TEAM_LEAD]", f"=========================")

    # Mark tools sent to API after successful response (cached MC)
    if use_cached_mc:
        mark_tools_sent_to_api_state()

    # Build assistant message
    content_blocks = []
    for block in response.content:
        if hasattr(block, "text"):
            content_blocks.append({"type": "text", "text": block.text})
        elif hasattr(block, "thinking"):
            # Handle thinking blocks from some API providers
            content_blocks.append({"type": "text", "text": block.thinking})
        elif hasattr(block, "name"):  # tool_use
            content_blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })

    return AssistantMessage(
        message={
            "role": "assistant",
            "content": content_blocks,
        }
    )


async def _execute_tools(
    tool_use_blocks: list[ToolUseBlock],
    assistant_message: AssistantMessage,
    params: QueryParams,
    state: State,
) -> AsyncGenerator[Union[Message, Any], None]:
    """Execute tool calls.

    Args:
        tool_use_blocks: Tool use blocks to execute
        assistant_message: Parent assistant message
        params: Query parameters
        state: Mutable state

    Yields:
        Progress messages and tool results
    """
    from claude_code_py.orchestration.executor import run_tools

    # Track tool results to add to messages
    tool_result_messages = []

    async for update in run_tools(
        tool_use_messages=tool_use_blocks,
        assistant_messages=[assistant_message],
        can_use_tool=params.can_use_tool,
        tool_use_context=state.tool_use_context,
    ):
        if update.message:
            yield update.message
            # Add tool result to state messages for next API call
            state.messages.append(update.message)
            tool_result_messages.append(update.message)

        if update.new_context:
            state.tool_use_context = update.new_context

    # After all tools executed, need to call API again for next turn
    # The tool results are now in state.messages


def normalize_messages_for_api(messages: list[Message]) -> list[dict[str, Any]]:
    """Normalize messages for the API.

    Args:
        messages: Internal message list

    Returns:
        API-formatted messages
    """
    result = []

    for msg in messages:
        if msg.type == "user":
            content = msg.message.get("content", "")
            # Handle tool_result format (list of content blocks)
            if isinstance(content, list):
                # Already in correct format (tool_result blocks)
                result.append({
                    "role": "user",
                    "content": content,
                })
            else:
                # Plain text user message
                result.append({
                    "role": "user",
                    "content": content,
                })
        elif msg.type == "assistant":
            result.append({
                "role": "assistant",
                "content": msg.message.get("content", []),
            })

    return result


def extract_tool_use_blocks(message: AssistantMessage) -> list[ToolUseBlock]:
    """Extract tool use blocks from an assistant message.

    Args:
        message: Assistant message

    Returns:
        List of tool use blocks
    """
    blocks = []
    content = message.message.get("content", [])

    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            blocks.append(ToolUseBlock(
                id=block.get("id", ""),
                name=block.get("name", ""),
                input=block.get("input", {}),
            ))

    return blocks


def create_max_turns_message(turn_count: int) -> Message:
    """Create a max turns reached message.

    Args:
        turn_count: Number of turns

    Returns:
        System message
    """
    return SystemMessage(
        type="system",
        content=f"Reached maximum number of turns ({turn_count})",
    )


def create_error_message(error: str) -> Message:
    """Create an error message.

    Args:
        error: Error message

    Returns:
        System message
    """
    return SystemMessage(
        type="system",
        subtype="api_error",
        content=error,
    )


class ContextLengthError(Exception):
    """Exception for 413 context_length_exceeded errors."""

    def __init__(self, message: str = "context_length_exceeded"):
        super().__init__(message)
        self.error_type = "context_length_exceeded"