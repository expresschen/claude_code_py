"""Microcompact Implementation.

This implements two types of microcompact:
1. Time-based: Clears stale tool results when cache has expired (>60 min gap)
2. Cached: Uses cache_edits API to delete tool_results without breaking cache prefix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from claude_code_py.core_types.message import (
    Message,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ToolUseBlock,
)
from claude_code_py.constants import QuerySource, is_main_thread_source
from .cached_mc_state import (
    COMPACTABLE_TOOLS,
    CachedMCState,
    CachedMCConfig,
    PinnedCacheEdits,
    create_cached_mc_state,
    get_cached_mc_config,
    collect_compactable_tool_ids,
    register_tool_result,
    register_tool_message,
    get_tool_results_to_delete,
    create_cache_edits_block,
    mark_tools_sent_to_api,
    reset_cached_mc_state,
    pin_cache_edits as _pin_cache_edits,
)
from .compact import (
    CompactResult,
    CompactMetadata,
)


# =============================================================================
# Constants
# =============================================================================

# Time threshold for cache expiration (minutes)
CACHE_EXPIRATION_THRESHOLD_MINUTES = 60

# Maximum tool results to delete in cached microcompact
MAX_TOOL_RESULTS_TO_DELETE = 50

# Models supported for cache editing
CACHE_EDIT_SUPPORTED_MODELS = [
    "claude-sonnet-4-",
    "claude-opus-4-",
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
]


# =============================================================================
# Module-Level State (matches TypeScript module-level cachedMCState)
# =============================================================================

# Lazy-initialized cached MC state to avoid importing in external builds.
# State lives inside feature() checks in TypeScript for dead code elimination.
_cached_mc_state: Optional[CachedMCState] = None
_pending_cache_edits: Optional[dict[str, Any]] = None


def _get_cached_mc_state() -> CachedMCState:
    """Get or create cached microcompact state singleton."""
    global _cached_mc_state
    if _cached_mc_state is None:
        _cached_mc_state = create_cached_mc_state()
    return _cached_mc_state


def consume_pending_cache_edits() -> Optional[dict[str, Any]]:
    """Get new pending cache edits to be included in the next API request.

    Returns null if there are no new pending edits.
    Clears the pending state (caller must pin them after insertion).

    Returns:
        cache_edits block or None
    """
    global _pending_cache_edits
    edits = _pending_cache_edits
    _pending_cache_edits = None
    return edits


def get_pinned_cache_edits() -> list[PinnedCacheEdits]:
    """Get all previously-pinned cache edits that must be re-sent at their
    original positions for cache hits.

    Returns:
        List of pinned cache edits
    """
    state = _cached_mc_state
    if state is None:
        return []
    return state.pinnedEdits.copy()


def pin_cache_edits(user_message_index: int, block: dict[str, Any]) -> None:
    """Pin a new cache_edits block to a specific user message position.

    Called after inserting new edits so they are re-sent in subsequent calls.

    Args:
        user_message_index: Position to re-insert the block
        block: cache_edits block to pin
    """
    state = _get_cached_mc_state()
    _pin_cache_edits(state, user_message_index, block)


def mark_tools_sent_to_api_state() -> None:
    """Marks all registered tools as sent to the API.
    Called after a successful API response.
    """
    state = _cached_mc_state
    if state is not None:
        mark_tools_sent_to_api(state)


def reset_microcompact_state() -> None:
    """Reset all cached microcompact state.

    Called after /clear or time-based microcompact.
    """
    global _cached_mc_state, _pending_cache_edits
    if _cached_mc_state is not None:
        reset_cached_mc_state(_cached_mc_state)
    _pending_cache_edits = None


# =============================================================================
# Types
# =============================================================================


@dataclass
class MicrocompactResult:
    """Result of microcompact operation."""

    success: bool
    type: str  # "time_based" or "cached"
    messages_removed: int = 0
    messages_kept: int = 0
    # Modified messages to apply after microcompact
    modified_messages: Optional[list[Message]] = None
    # Pending cache_edits for API call (cached microcompact only)
    pending_cache_edits: Optional[list[dict[str, Any]]] = None
    cache_references_deleted: list[str] = field(default_factory=list)
    error: Optional[str] = None


# =============================================================================
# Time-based Microcompact
# =============================================================================


def should_do_time_based_microcompact(
    messages: list[Message],
    query_source: str,
) -> bool:
    """Check if time-based microcompact should trigger.

    Time-based microcompact triggers when there's a large time gap
    between messages, indicating the cache has expired (>60 min).
    In this case, clearing tool results doesn't break the cache
    since it's already expired.

    Only runs for main thread queries (repl_main_thread and variants, sdk).

    Args:
        messages: Message list
        query_source: Source of the query

    Returns:
        True if time-based microcompact should run
    """
    # Only for main thread queries (prefix match for output style variants)
    if not is_main_thread_source(query_source) and query_source != QuerySource.SDK:
        return False

    # Check for large time gap
    if len(messages) < 2:
        return False

    # Find the gap
    last_assistant_time = None
    current_user_time = None

    for msg in reversed(messages):
        if msg.type == "assistant":
            # Get timestamp from message
            last_assistant_time = _get_message_timestamp(msg)
            break

    # Get current time from latest user message
    for msg in reversed(messages):
        if msg.type == "user":
            current_user_time = _get_message_timestamp(msg)
            break

    if last_assistant_time and current_user_time:
        gap_minutes = (current_user_time - last_assistant_time) / 60
        return gap_minutes >= CACHE_EXPIRATION_THRESHOLD_MINUTES

    return False


def execute_time_based_microcompact(
    messages: list[Message],
) -> MicrocompactResult:
    """Execute time-based microcompact.

    Clears all tool results since the cache is already expired.
    This is safe because expired cache can't be preserved anyway.

    Args:
        messages: Message list

    Returns:
        MicrocompactResult
    """
    # Find all tool result blocks
    messages_to_keep = []
    tool_results_removed = 0

    for msg in messages:
        if msg.type == "user":
            content = msg.message.get("content", [])
            if isinstance(content, list):
                # Filter out tool_result blocks
                new_content = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            tool_results_removed += 1
                        else:
                            new_content.append(block)

                if new_content != content:
                    # Create modified message
                    messages_to_keep.append(UserMessage(
                        uuid=msg.uuid,
                        message={"role": "user", "content": new_content},
                    ))
                else:
                    messages_to_keep.append(msg)
            else:
                messages_to_keep.append(msg)
        else:
            messages_to_keep.append(msg)

    return MicrocompactResult(
        success=True,
        type="time_based",
        messages_removed=tool_results_removed,
        messages_kept=len(messages_to_keep),
        modified_messages=messages_to_keep,
    )


# =============================================================================
# Cached Microcompact (cache_edits)
# =============================================================================


def is_model_supported_for_cache_editing(model: str) -> bool:
    """Check if model supports cache_edits API.

    Args:
        model: Model identifier

    Returns:
        True if supported
    """
    model_lower = model.lower()
    for supported in CACHE_EDIT_SUPPORTED_MODELS:
        if supported.lower() in model_lower:
            return True
    return False


def is_cached_microcompact_enabled() -> bool:
    """Check if cached microcompact feature is enabled.

    Returns:
        True if enabled
    """
    import os

    # Check environment override
    if os.environ.get("DISABLE_CACHED_MICROCOMPACT", "").lower() in ("1", "true", "yes"):
        return False

    # Check feature flag (would normally check feature('CACHED_MICROCOMPACT'))
    # For now, default to enabled for supported models
    return True


def should_do_cached_microcompact(
    messages: list[Message],
    model: str,
    query_source: str,
) -> bool:
    """Check if cached microcompact should trigger.

    Cached microcompact uses the cache_edits API to delete tool_results
    while preserving the cached prefix. Only works for supported models
    and first-party API provider.

    Only runs for main REPL thread (repl_main_thread and variants).
    Does NOT run for SDK queries (different conversation context).

    Args:
        messages: Message list
        model: Model being used
        query_source: Source of the query

    Returns:
        True if cached microcompact should run
    """
    # Only for main REPL thread queries (prefix match for output style variants)
    # SDK is excluded because cachedMCState is global and SDK has different context
    if not is_main_thread_source(query_source) or query_source == QuerySource.SDK:
        return False

    # Check feature enabled
    if not is_cached_microcompact_enabled():
        return False

    # Check model supported
    if not is_model_supported_for_cache_editing(model):
        return False

    # Check for tool results that can be deleted
    tool_result_count = _count_tool_results(messages)

    return tool_result_count > 0


def execute_cached_microcompact(
    messages: list[Message],
    model: str,
) -> MicrocompactResult:
    """Execute cached microcompact using state-based logic.

    KEY CHANGES from original:
    1. Only consider COMPACTABLE_TOOLS (not all tool_results)
    2. Use state tracking (registerToolResult, getToolResultsToDelete)
    3. Do NOT modify messages directly
    4. Return pending_cache_edits for API layer

    Matches TypeScript's cachedMicrocompactPath().

    Args:
        messages: Message list
        model: Model identifier

    Returns:
        MicrocompactResult with pending_cache_edits (messages unchanged)
    """
    global _pending_cache_edits

    state = _get_cached_mc_state()
    config = get_cached_mc_config()

    # 1. Collect compactable tool IDs (only COMPACTABLE_TOOLS)
    compactable_tool_ids = set(collect_compactable_tool_ids(messages))

    # 2. Register tool results with state
    for message in messages:
        if message.type == "user":
            content = message.message.get("content", [])
            if isinstance(content, list):
                group_ids: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        # Only register if compactable and not already registered
                        if tool_use_id in compactable_tool_ids and tool_use_id not in state.registeredTools:
                            register_tool_result(state, tool_use_id)
                            group_ids.append(tool_use_id)
                register_tool_message(state, group_ids)

    # 3. Get tools to delete based on thresholds
    tools_to_delete = get_tool_results_to_delete(state)

    if not tools_to_delete:
        # No compaction needed
        return MicrocompactResult(
            success=True,
            type="cached",
            messages_removed=0,
            messages_kept=len(messages),
            modified_messages=None,  # Return messages unchanged
            pending_cache_edits=None,
            cache_references_deleted=[],
        )

    # 4. Create cache_edits block
    cache_edits = create_cache_edits_block(state, tools_to_delete)

    # 5. Queue for API layer (do NOT modify messages)
    _pending_cache_edits = cache_edits

    # Log event
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Cached MC deleting {len(tools_to_delete)} tools: {', '.join(tools_to_delete[:5])}...")

    # Return messages unchanged - cache_edits handled at API layer
    return MicrocompactResult(
        success=True,
        type="cached",
        messages_removed=len(tools_to_delete),
        messages_kept=len(messages),
        modified_messages=None,  # KEY: messages unchanged
        pending_cache_edits=[cache_edits],
        cache_references_deleted=tools_to_delete,
    )


# =============================================================================
# Main Microcompact Function
# =============================================================================


async def maybe_microcompact(
    messages: list[Message],
    model: str,
    query_source: str,
) -> Optional[MicrocompactResult]:
    """Check and execute microcompact if appropriate.

    Priority:
    1. Time-based microcompact (if cache expired)
    2. Cached microcompact (if cache_edits supported)

    IMPORTANT: For cached microcompact, messages are NOT modified.
    The cache_edits are queued for the API layer to handle.

    Args:
        messages: Message list
        model: Model identifier
        query_source: Source of the query

    Returns:
        MicrocompactResult if triggered, None otherwise
    """
    # Check time-based first (takes precedence)
    if should_do_time_based_microcompact(messages, query_source):
        # Reset cached MC state since we're modifying content directly
        reset_microcompact_state()
        return execute_time_based_microcompact(messages)

    # Check cached microcompact
    if should_do_cached_microcompact(messages, model, query_source):
        return execute_cached_microcompact(messages, model)

    return None


# =============================================================================
# Helper Functions
# =============================================================================


def _get_message_timestamp(message: Message) -> Optional[float]:
    """Get timestamp from a message.

    Args:
        message: Message to extract timestamp from

    Returns:
        Timestamp or None
    """
    # Check for timestamp attribute
    if hasattr(message, "timestamp"):
        return message.timestamp

    # Check in message dict
    if hasattr(message, "message"):
        return message.message.get("timestamp")

    # Default to None
    return None


def _count_tool_results(messages: list[Message]) -> int:
    """Count tool result blocks in messages.

    Args:
        messages: Message list

    Returns:
        Count of tool result blocks
    """
    count = 0

    for msg in messages:
        if msg.type == "user":
            content = msg.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        count += 1

    return count


def build_messages_with_cache_edits(
    messages: list[Message],
    cache_edits_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build API messages with cache_edits blocks.

    Args:
        messages: Internal message list
        cache_edits_blocks: Cache edits to include

    Returns:
        API-formatted messages with cache_edits
    """
    result = []

    for msg in messages:
        if msg.type == "user":
            content = msg.message.get("content", [])
            api_msg = {"role": "user", "content": content}
            result.append(api_msg)
        elif msg.type == "assistant":
            content = msg.message.get("content", [])
            api_msg = {"role": "assistant", "content": content}
            result.append(api_msg)

    # Add cache_edits as special block
    if cache_edits_blocks:
        # Cache edits go at the end of the messages array
        for edit_block in cache_edits_blocks:
            result.append(edit_block)

    return result


# =============================================================================
# Statistics
# =============================================================================


def get_microcompact_stats(messages: list[Message], model: str) -> dict[str, Any]:
    """Get microcompact statistics.

    Args:
        messages: Message list
        model: Model identifier

    Returns:
        Stats dict
    """
    return {
        "time_based_eligible": should_do_time_based_microcompact(messages, QuerySource.REPL_MAIN_THREAD),
        "cached_eligible": should_do_cached_microcompact(messages, model, QuerySource.REPL_MAIN_THREAD),
        "model_supported_for_cache_editing": is_model_supported_for_cache_editing(model),
        "tool_result_count": _count_tool_results(messages),
        "cache_edit_enabled": is_cached_microcompact_enabled(),
    }