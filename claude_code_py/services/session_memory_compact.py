"""Session Memory-based Compaction.

This implements compact execution using pre-extracted session memory,
allowing zero-latency compaction without API calls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from claude_code_py.memory import (
    SessionMemory,
    get_session_memory_path,
    is_auto_memory_enabled,
)
from claude_code_py.core_types.message import (
    Message,
    SystemMessage,
    AssistantMessage,
    UserMessage,
)
from .compact_types import (
    CompactResult,
    CompactMetadata,
    MAX_COMPACT_TURNS,
)

# Session memory gate feature flag
SESSION_MEMORY_COMPACT_ENABLED = True

# Maximum wait time for session memory extraction (ms)
MAX_WAIT_FOR_EXTRACTION_MS = 30_000


# =============================================================================
# Configuration (matching TypeScript defaults)
# =============================================================================


@dataclass
class SessionMemoryCompactConfig:
    """Configuration for session memory compaction thresholds."""

    min_tokens: int = 10_000  # Minimum tokens to preserve after compaction
    min_text_block_messages: int = 5  # Minimum messages with text blocks
    max_tokens: int = 40_000  # Maximum tokens to preserve (hard cap)


# Global config (can be overridden)
_sm_compact_config = SessionMemoryCompactConfig()


def set_session_memory_compact_config(config: dict[str, int]) -> None:
    """Set session memory compact configuration.

    Args:
        config: Partial config dict with keys: min_tokens, min_text_block_messages, max_tokens
    """
    global _sm_compact_config
    if "min_tokens" in config and config["min_tokens"] > 0:
        _sm_compact_config.min_tokens = config["min_tokens"]
    if "min_text_block_messages" in config and config["min_text_block_messages"] > 0:
        _sm_compact_config.min_text_block_messages = config["min_text_block_messages"]
    if "max_tokens" in config and config["max_tokens"] > 0:
        _sm_compact_config.max_tokens = config["max_tokens"]


def get_session_memory_compact_config() -> SessionMemoryCompactConfig:
    """Get current session memory compact configuration."""
    return _sm_compact_config


# =============================================================================
# Session Memory Compact Result
# =============================================================================


@dataclass
class SessionMemoryCompactState:
    """State for session memory compact tracking."""

    last_summarized_message_uuid: Optional[str] = None  # Use UUID instead of index (more stable)
    extraction_in_progress: bool = False
    last_extraction_time: Optional[float] = None


# Global state
_session_memory_compact_state = SessionMemoryCompactState()


# =============================================================================
# Main Functions
# =============================================================================


def is_session_memory_compact_enabled() -> bool:
    """Check if session memory compact is enabled.

    Returns:
        True if enabled
    """
    # Check environment override
    import os
    if os.environ.get("DISABLE_SESSION_MEMORY_COMPACT", "").lower() in ("1", "true", "yes"):
        return False

    return SESSION_MEMORY_COMPACT_ENABLED


async def try_session_memory_compaction(
    messages: list[Message],
    auto_compact_threshold: int,
) -> Optional[CompactResult]:
    """Attempt session memory-based compaction.

    This uses pre-extracted session memory to create a compact result
    without making an API call. If session memory is not ready, returns None.

    Handles two scenarios:
    1. Normal case: last_summarized_message_uuid is set, find it and keep messages after
    2. Resumed session: last_summarized_message_uuid is not set but session memory exists,
       treat as resumed session - set lastSummarizedIndex to last message

    Args:
        messages: Current message list
        auto_compact_threshold: Token threshold for compact trigger

    Returns:
        CompactResult if session memory compact succeeded, None otherwise
    """
    if not is_session_memory_compact_enabled():
        return None

    # Check if session memory extraction has run
    if not _has_session_memory_been_extracted():
        return None

    # Wait for any in-progress extraction
    await _wait_for_extraction_complete(MAX_WAIT_FOR_EXTRACTION_MS)

    # Read session memory file
    session_memory = SessionMemory()

    if not session_memory.exists():
        return None

    memory_content = session_memory.read()
    if not memory_content:
        return None

    # Check for empty template (matches TypeScript isSessionMemoryEmpty)
    if _is_session_memory_empty(memory_content):
        return None

    try:
        # Get last summarized message UUID (matching TypeScript getLastSummarizedMessageId)
        last_summarized_uuid = _session_memory_compact_state.last_summarized_message_uuid
        last_summarized_index: int

        if last_summarized_uuid:
            # Normal case: we know exactly which message was last summarized
            # Find the index by UUID (more stable than storing index directly)
            last_summarized_index = -1
            for i, msg in enumerate(messages):
                if hasattr(msg, "uuid") and msg.uuid == last_summarized_uuid:
                    last_summarized_index = i
                    break

            if last_summarized_index == -1:
                # The summarized message UUID doesn't exist in current messages
                # This can happen if messages were modified - fall back to legacy compact
                return None
        else:
            # Resumed session case: session memory has content but we don't know the boundary
            # Set lastSummarizedIndex to last message so startIndex becomes len(messages) initially
            last_summarized_index = len(messages) - 1 if len(messages) > 0 else -1

        # Calculate the starting index for messages to keep
        # This starts from lastSummarizedIndex, expands to meet minimums,
        # and adjusts to not split tool_use/tool_result pairs
        start_index = _calculate_messages_to_keep_index(
            messages,
            last_summarized_index,
            auto_compact_threshold,
        )

        # Filter out old compact boundary messages from messagesToKeep.
        # After REPL pruning, old boundaries re-yielded would trigger unwanted second prune
        messages_to_keep = [
            m for m in messages[start_index:]
            if not _is_compact_boundary_message(m)
        ]

        # Create compact result from session memory
        result = _create_compaction_result_from_session_memory(
            messages=messages,
            memory_content=memory_content,
            messages_to_keep=messages_to_keep,
            start_index=start_index,
        )

        # Build post-compact messages and check threshold (matching TypeScript)
        # Use lazy import to avoid circular dependency with compact.py
        from .compact import build_post_compact_messages
        post_compact_messages = build_post_compact_messages(result, messages)
        post_compact_tokens = sum(_estimate_message_tokens(m) for m in post_compact_messages)

        # Only check threshold if one was provided (for autocompact)
        if auto_compact_threshold > 0 and post_compact_tokens >= auto_compact_threshold:
            return None

        # Update state with the last kept message UUID (not index)
        if messages_to_keep:
            last_kept_msg = messages_to_keep[-1]
            if hasattr(last_kept_msg, "uuid"):
                _session_memory_compact_state.last_summarized_message_uuid = last_kept_msg.uuid

        return result

    except Exception:
        # Errors are expected (e.g., file issues) - return null to fall back to legacy compact
        return None


def _is_session_memory_empty(content: str) -> bool:
    """Check if session memory content matches empty template.

    Args:
        content: Session memory file content

    Returns:
        True if content is empty or matches template
    """
    # Simple check: if content is very short or contains only template markers
    if not content or len(content.strip()) < 100:
        return True

    # Check for common empty template markers
    empty_markers = [
        "# Memory Index",
        "## Entries",
        "*No entries yet*",
    ]
    for marker in empty_markers:
        if marker in content and len(content) < 500:
            return True

    return False


def _estimate_session_memory_tokens(content: str) -> int:
    """Estimate token count for session memory content.

    Args:
        content: Session memory text

    Returns:
        Estimated token count
    """
    from claude_code_py.utils.context import rough_token_count_estimation
    return rough_token_count_estimation(content)


# =============================================================================
# Helper Functions
# =============================================================================


def _has_session_memory_been_extracted() -> bool:
    """Check if session memory has been extracted at least once.

    Returns:
        True if extraction has occurred
    """
    return (
        _session_memory_compact_state.last_extraction_time is not None or
        SessionMemory().exists()
    )


async def _wait_for_extraction_complete(timeout_ms: int) -> None:
    """Wait for session memory extraction to complete.

    Args:
        timeout_ms: Maximum wait time in milliseconds
    """
    if not _session_memory_compact_state.extraction_in_progress:
        return

    start = asyncio.get_event_loop().time()
    timeout_sec = timeout_ms / 1000

    while _session_memory_compact_state.extraction_in_progress:
        elapsed = asyncio.get_event_loop().time() - start
        if elapsed > timeout_sec:
            break
        await asyncio.sleep(0.1)


def _calculate_messages_to_keep_index(
    messages: list[Message],
    last_summarized_index: int,
    auto_compact_threshold: int,
) -> int:
    """Calculate the starting index for messages to keep.

    This follows the TypeScript logic:
    1. Start from lastSummarizedIndex + 1 (or messages.length if -1)
    2. Calculate tokens and text-block count from startIndex to end
    3. Check if we already meet min/max thresholds
    4. Expand backwards to meet minimums (floor at last compact boundary)
    5. Adjust to preserve tool_use/tool_result pairs

    Args:
        messages: Message list
        last_summarized_index: Index where last summary ended (-1 if not found)
        auto_compact_threshold: Token threshold (used for fallback calculation)

    Returns:
        Index to start keeping messages from
    """
    if len(messages) == 0:
        return 0

    config = get_session_memory_compact_config()

    # Correct initial value calculation (matching TypeScript)
    # If lastSummarizedIndex is -1, start from messages.length (no messages kept initially)
    # Otherwise start from lastSummarizedIndex + 1
    start_index = (
        last_summarized_index + 1 if last_summarized_index >= 0 else len(messages)
    )

    # Calculate current tokens and text-block message count from startIndex to end
    total_tokens = 0
    text_block_message_count = 0
    for i in range(start_index, len(messages)):
        msg = messages[i]
        total_tokens += _estimate_message_tokens(msg)
        if _has_text_blocks(msg):
            text_block_message_count += 1

    # Check if we already hit the max cap
    if total_tokens >= config.max_tokens:
        return _adjust_index_to_preserve_tool_pairs(messages, start_index)

    # Check if we already meet both minimums
    if (
        total_tokens >= config.min_tokens
        and text_block_message_count >= config.min_text_block_messages
    ):
        return _adjust_index_to_preserve_tool_pairs(messages, start_index)

    # Find floor: last compact boundary (can't expand past this)
    floor = _find_last_compact_boundary_index(messages) + 1

    # Expand backwards until we meet both minimums or hit max cap
    for i in range(start_index - 1, floor, -1):
        msg = messages[i]
        msg_tokens = _estimate_message_tokens(msg)
        total_tokens += msg_tokens
        if _has_text_blocks(msg):
            text_block_message_count += 1
        start_index = i

        # Stop if we hit the max cap
        if total_tokens >= config.max_tokens:
            break

        # Stop if we meet both minimums
        if (
            total_tokens >= config.min_tokens
            and text_block_message_count >= config.min_text_block_messages
        ):
            break

    # Adjust for tool_use/tool_result pairs
    return _adjust_index_to_preserve_tool_pairs(messages, start_index)


def _estimate_message_tokens(message: Message) -> int:
    """Estimate token count for a single message.

    Args:
        message: Message to estimate

    Returns:
        Estimated token count
    """
    from claude_code_py.utils.context import rough_token_count_estimation

    content = ""
    if hasattr(message, "message"):
        msg_dict = message.message
        content = msg_dict.get("content", "")
        if isinstance(content, str):
            return rough_token_count_estimation(content)
        elif isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total += rough_token_count_estimation(block.get("text", ""))
                    elif block.get("type") == "image":
                        total += 1000  # Images ~1000 tokens
                    elif block.get("type") == "tool_use":
                        total += 100  # Tool overhead
                        total += rough_token_count_estimation(str(block.get("input", {})))
            return total

    return rough_token_count_estimation(content)


def _has_text_blocks(message: Message) -> bool:
    """Check if a message contains text blocks (text content for user/assistant interaction).

    Args:
        message: Message to check

    Returns:
        True if message has text blocks
    """
    if message.type == "assistant":
        content = message.message.get("content", [])
        if isinstance(content, list):
            return any(block.get("type") == "text" for block in content if isinstance(block, dict))
        return False

    if message.type == "user":
        content = message.message.get("content", "")
        if isinstance(content, str):
            return len(content) > 0
        if isinstance(content, list):
            return any(
                block.get("type") == "text"
                for block in content
                if isinstance(block, dict)
            )

    return False


def _find_last_compact_boundary_index(messages: list[Message]) -> int:
    """Find the index of the last compact boundary message.

    Args:
        messages: Message list

    Returns:
        Index of last compact boundary, or -1 if none found
    """
    for i in range(len(messages) - 1, -1, -1):
        if _is_compact_boundary_message(messages[i]):
            return i
    return -1


def _get_tool_result_ids(message: Message) -> list[str]:
    """Get tool_result IDs from a message.

    Args:
        message: Message to check

    Returns:
        List of tool_use_ids from tool_result blocks
    """
    if message.type != "user":
        return []

    content = message.message.get("content", [])
    if not isinstance(content, list):
        return []

    ids: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tool_use_id = block.get("tool_use_id", "")
            if tool_use_id:
                ids.append(tool_use_id)

    return ids


def _has_tool_use_with_ids(message: Message, tool_use_ids: set[str]) -> bool:
    """Check if a message contains tool_use blocks with any of the given IDs.

    Args:
        message: Message to check
        tool_use_ids: Set of tool_use IDs to match

    Returns:
        True if message has matching tool_use blocks
    """
    if message.type != "assistant":
        return False

    content = message.message.get("content", [])
    if not isinstance(content, list):
        return False

    return any(
        block.get("type") == "tool_use" and block.get("id") in tool_use_ids
        for block in content
        if isinstance(block, dict)
    )


def _adjust_index_to_preserve_tool_pairs(
    messages: list[Message],
    start_index: int,
) -> int:
    """Adjust start index to ensure tool_use/tool_result pairs are not split.

    If ANY message in the kept range contains tool_result blocks, we need to
    include the preceding assistant message(s) that contain matching tool_use blocks.

    This prevents API errors from orphan tool_results.

    Args:
        messages: Full message list
        start_index: Proposed start index

    Returns:
        Adjusted start index
    """
    if start_index <= 0 or start_index >= len(messages):
        return start_index

    adjusted_index = start_index

    # Step 1: Handle tool_use/tool_result pairs
    # Collect tool_result IDs from ALL messages in the kept range
    all_tool_result_ids: list[str] = []
    for i in range(start_index, len(messages)):
        all_tool_result_ids.extend(_get_tool_result_ids(messages[i]))

    if all_tool_result_ids:
        # Collect tool_use IDs already in the kept range
        tool_use_ids_in_kept_range: set[str] = set()
        for i in range(adjusted_index, len(messages)):
            msg = messages[i]
            if msg.type == "assistant":
                content = msg.message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_id = block.get("id", "")
                            if tool_id:
                                tool_use_ids_in_kept_range.add(tool_id)

        # Only look for tool_uses that are NOT already in the kept range
        needed_tool_use_ids = set(
            id_ for id_ in all_tool_result_ids if id_ not in tool_use_ids_in_kept_range
        )

        # Find the assistant message(s) with matching tool_use blocks
        for i in range(adjusted_index - 1, -1, -1):
            if not needed_tool_use_ids:
                break
            message = messages[i]
            if _has_tool_use_with_ids(message, needed_tool_use_ids):
                adjusted_index = i
                # Remove found tool_use_ids from the set
                if message.type == "assistant":
                    content = message.message.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                tool_id = block.get("id", "")
                                if tool_id in needed_tool_use_ids:
                                    needed_tool_use_ids.remove(tool_id)

    return adjusted_index


def _is_compact_boundary_message(message: Message) -> bool:
    """Check if a message is a compact boundary marker.

    Args:
        message: Message to check

    Returns:
        True if it's a compact boundary
    """
    if message.type == "system":
        if hasattr(message, "subtype"):
            return message.subtype == "compact_boundary"
        if hasattr(message, "message"):
            return message.message.get("subtype") == "compact_boundary"
    return False


def _create_compaction_result_from_session_memory(
    messages: list[Message],
    memory_content: str,
    messages_to_keep: list[Message],
    start_index: int,
) -> CompactResult:
    """Create a compact result from session memory content.

    Args:
        messages: Original message list
        memory_content: Session memory file content
        messages_to_keep: Messages to preserve after compact
        start_index: Index where messages were cut

    Returns:
        CompactResult with pre-built messages
    """
    # Truncate oversized sections (matching TypeScript truncateSessionMemoryForCompact)
    max_length = 10000
    truncated_content = memory_content
    was_truncated = False
    if len(memory_content) > max_length:
        truncated_content = memory_content[:max_length] + "\n\n[Session memory truncated for length]"
        was_truncated = True

    # 1. Create boundary system message
    boundary_msg = SystemMessage(
        type="system",
        subtype="compact_boundary",
        compact_metadata={
            "summary": truncated_content,
            "timestamp": datetime.now().isoformat(),
            "source": "session_memory",
        },
    )

    # 2. Create summary user message
    summary_content = _format_session_memory_summary(truncated_content)
    if was_truncated:
        summary_content += "\n\nSome session memory sections were truncated for length."

    summary_msg = UserMessage(
        uuid=_generate_uuid(),
        message={
            "role": "user",
            "content": summary_content,
        },
    )

    # Calculate tokens
    tokens_before = sum(_estimate_message_tokens(m) for m in messages)
    tokens_after = sum(_estimate_message_tokens(m) for m in messages_to_keep) + _estimate_message_tokens(summary_msg)

    metadata = CompactMetadata(
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_removed=tokens_before - tokens_after,
        messages_before=len(messages),
        messages_after=len(messages_to_keep) + 1,  # +1 for summary
        messages_removed=start_index,
        compact_type="session_memory",
    )

    return CompactResult(
        success=True,
        summary=memory_content,
        metadata=metadata,
        # Pre-built messages for build_post_compact_messages
        boundary_marker=boundary_msg,
        summary_messages=[summary_msg],
        messages_to_keep=messages_to_keep,
        attachments=[],  # Would be populated from file cache in full implementation
        hook_results=[],  # Would be populated from hooks in full implementation
        pre_compact_token_count=tokens_before,
        post_compact_token_count=tokens_after,
    )


def _format_session_memory_summary(content: str) -> str:
    """Format session memory content as a summary message.

    Args:
        content: Raw session memory content

    Returns:
        Formatted summary string
    """
    # Truncate if too long (matching TypeScript truncateSessionMemoryForCompact)
    max_length = 10000  # Reasonable limit
    if len(content) > max_length:
        content = content[:max_length] + "\n\n[Session memory truncated for length]"

    return f"[Session Memory Summary]\n\n{content}"


def _generate_uuid() -> str:
    """Generate a UUID for new messages.

    Returns:
        UUID string
    """
    import uuid
    return str(uuid.uuid4())




# =============================================================================
# Integration with Session Memory Extraction
# =============================================================================


def mark_extraction_started() -> None:
    """Mark that session memory extraction has started."""
    _session_memory_compact_state.extraction_in_progress = True


def mark_extraction_complete() -> None:
    """Mark that session memory extraction has completed."""
    _session_memory_compact_state.extraction_in_progress = False
    _session_memory_compact_state.last_extraction_time = datetime.now().timestamp()


def get_session_memory_compact_stats() -> dict[str, Any]:
    """Get session memory compact statistics.

    Returns:
        Stats dict
    """
    return {
        "enabled": is_session_memory_compact_enabled(),
        "last_summarized_message_uuid": _session_memory_compact_state.last_summarized_message_uuid,
        "extraction_in_progress": _session_memory_compact_state.extraction_in_progress,
        "last_extraction_time": _session_memory_compact_state.last_extraction_time,
    }


def reset_session_memory_compact_state() -> None:
    """Reset session memory compact state."""
    global _session_memory_compact_state
    _session_memory_compact_state = SessionMemoryCompactState()


def set_last_summarized_message_uuid(uuid: Optional[str]) -> None:
    """Set the last summarized message UUID.

    Args:
        uuid: UUID of the last message that was summarized, or None to reset
    """
    _session_memory_compact_state.last_summarized_message_uuid = uuid


def get_last_summarized_message_uuid() -> Optional[str]:
    """Get the last summarized message UUID.

    Returns:
        UUID string or None if not set
    """
    return _session_memory_compact_state.last_summarized_message_uuid