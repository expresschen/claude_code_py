"""Conversation Compaction Service.

This implements the automatic conversation compaction system that summarizes
conversation history when context limits are approached.

Compact Types:
1. Standard Compact: Full API-based summarization
2. Session Memory Compact: Uses pre-extracted memory (zero-latency)
3. Microcompact: Light-weight deletion of tool results
4. Reactive Compact: Triggered by 413 context_length_exceeded errors

Cache Optimization:
- build_messages_with_cache_edits: Preserve cache_control on kept messages
- apply_cache_edits_after_compact: Restore caching after compact boundary
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from claude_code_py.core_types.message import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    AttachmentMessage,
)
from claude_code_py.utils.context import (
    get_auto_compact_threshold,
    get_effective_context_window,
    POST_COMPACT_TOKEN_BUDGET,
    POST_COMPACT_MAX_FILES_TO_RESTORE,
    POST_COMPACT_MAX_TOKENS_PER_FILE,
)
from claude_code_py.utils.side_query import get_default_sonnet_model

# Import types from separate module to avoid circular imports
from .compact_types import (
    CompactResult,
    CompactMetadata,
    CompactOptions,
    CompactState,
    MAX_COMPACT_TURNS,
)

# Import microcompact helpers (maybe_microcompact now called from query loop)
from .micro_compact import (
    build_messages_with_cache_edits,
)

# Import new compact prompts
from .compact_prompt import (
    get_base_compact_prompt,
    get_partial_compact_prompt,
    format_compact_summary,
    get_compact_user_summary_message,
)


# =============================================================================
# Constants
# =============================================================================

MAX_COMPACT_TURNS = 1  # Compaction should complete in one turn
MAX_CONSECUTIVE_FAILURES = 3

# Reactive compact (413 error recovery)
MAX_REACTIVE_COMPACT_ATTEMPTS = 2
REACTIVE_COMPACT_TOKEN_REDUCTION_RATIO = 0.5  # Aim to reduce to 50% of threshold

# Legacy prompt (kept for backwards compatibility)
# Use get_base_compact_prompt() for new code with <analysis> block
COMPACT_SYSTEM_PROMPT_LEGACY = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created.
4. Errors and fixes: List all errors that you ran into, and how you fixed them.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary.
9. Optional Next Step: List the next step that you will take, if applicable.

Format your response as:
<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]

3. Files and Code Sections:
   - [File Name]
     - [Why important]
     - [Code snippet or changes]

4. Errors and fixes:
   - [Error]: [How fixed]

5. Problem Solving:
   [Description]

6. All user messages:
   - [Message 1]
   - [Message 2]

7. Pending Tasks:
   - [Task 1]

8. Current Work:
   [Description]

9. Optional Next Step:
   [Next step if applicable]
</summary>"""


# =============================================================================
# Types (imported from compact_types.py)
# =============================================================================

# Note: CompactResult, CompactMetadata are imported from compact_types

@dataclass
class AutoCompactState:
    """State for auto-compact tracking."""

    enabled: bool = True
    consecutive_failures: int = 0
    last_compact_time: Optional[float] = None
    in_progress: bool = False


# Global state
_auto_compact_state = AutoCompactState()


# =============================================================================
# Lazy imports for circular dependencies
# =============================================================================


def _get_session_memory_compact_functions():
    """Lazy import to avoid circular dependency."""
    from .session_memory_compact import (
        try_session_memory_compaction,
        is_session_memory_compact_enabled,
        mark_extraction_complete,
    )
    return (
        try_session_memory_compaction,
        is_session_memory_compact_enabled,
        mark_extraction_complete,
    )


# =============================================================================
# Main Compaction Functions
# =============================================================================


async def compact_conversation(
    messages: list[Message],
    model: str,
    on_progress: Optional[Callable[[str], None]] = None,
    cache_safe_params: Optional[Any] = None,
) -> CompactResult:
    """Compact a conversation by summarizing old messages.

    When cache_safe_params is provided, the summary generation
    uses run_forked_agent to share the main thread's prompt cache,
    significantly reducing cache_creation_input_tokens.

    Args:
        messages: Current conversation messages
        model: Model being used
        on_progress: Optional progress callback
        cache_safe_params: Optional cache params for prompt cache sharing

    Returns:
        CompactResult with summary and metadata
    """
    from claude_code_py.utils.context import rough_token_count_estimation_for_messages

    tokens_before = rough_token_count_estimation_for_messages(messages)

    try:
        if on_progress:
            on_progress("Analyzing conversation...")

        # Strip images from messages (they use a lot of tokens)
        stripped_messages = strip_images_from_messages(messages)

        # Build compact prompt
        conversation_text = messages_to_compact_text(stripped_messages)

        if on_progress:
            on_progress("Generating summary...")

        # Use side query to generate summary (with cache sharing if params provided)
        summary = await generate_compact_summary(
            conversation_text,
            model,
            cache_safe_params=cache_safe_params,
        )

        if on_progress:
            on_progress("Summary complete")

        # Calculate token savings
        tokens_after = rough_token_count_estimation_for_messages(
            create_compact_boundary_messages(summary, messages)
        )

        metadata = CompactMetadata(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            messages_removed=len(messages),
            messages_kept=1,  # Just the summary
        )

        return CompactResult(
            success=True,
            summary=summary,
            metadata=metadata,
        )

    except Exception as e:
        return CompactResult(
            success=False,
            error=str(e),
        )


async def auto_compact_if_needed(
    messages: list[Message],
    model: str,
    on_progress: Optional[Callable[[str], None]] = None,
    cache_safe_params: Optional[Any] = None,
    query_source: Optional[str] = None,
) -> Optional[CompactResult]:
    """Auto-compact if token threshold is exceeded.

    Priority:
    1. Session Memory Compact (zero-latency, no API call)
    2. Standard Compact (full API summarization)

    Note: Microcompact is now handled separately in the query loop,
    executed before this function is called.

    Note: query_source guards are handled in _check_auto_compact() in query.py.
    This function assumes the caller has already validated the source.

    Args:
        messages: Current messages
        model: Model being used
        on_progress: Optional progress callback
        cache_safe_params: Optional cache params for prompt cache sharing
        query_source: Source of the query (for logging/stats)

    Returns:
        CompactResult if compaction occurred, None otherwise
    """
    global _auto_compact_state

    if not _auto_compact_state.enabled:
        return None

    # Check if already in progress
    if _auto_compact_state.in_progress:
        return None

    # Check consecutive failures
    if _auto_compact_state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        return None

    # Check if compaction needed
    from claude_code_py.utils.context import should_auto_compact

    should_compact, token_usage = should_auto_compact(messages, model)
    if not should_compact:
        return None

    _auto_compact_state.in_progress = True

    try:
        # Get threshold for session memory compact
        threshold = get_auto_compact_threshold(model)

        # Try session memory compact first (fastest, no API call)
        # Use lazy import to avoid circular dependency
        try_session_memory_compaction, is_session_memory_compact_enabled, _ = _get_session_memory_compact_functions()

        if is_session_memory_compact_enabled():
            if on_progress:
                on_progress("Checking session memory...")

            session_result = await try_session_memory_compaction(messages, threshold)
            if session_result:
                _auto_compact_state.consecutive_failures = 0
                _auto_compact_state.last_compact_time = datetime.now().timestamp()
                if on_progress:
                    on_progress("Session memory compact complete")
                return session_result

        # Fall back to standard compact (microcompact now handled in query loop)
        if on_progress:
            on_progress("Running full compact...")

        result = await compact_conversation(
            messages,
            model,
            on_progress,
            cache_safe_params=cache_safe_params,
        )

        if result.success:
            _auto_compact_state.consecutive_failures = 0
            _auto_compact_state.last_compact_time = datetime.now().timestamp()
        else:
            _auto_compact_state.consecutive_failures += 1

        return result

    finally:
        _auto_compact_state.in_progress = False


# =============================================================================
# Reactive Compact (413 Error Recovery)
# =============================================================================


async def reactive_compact(
    messages: list[Message],
    model: str,
    error_message: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Optional[CompactResult]:
    """Handle 413 context_length_exceeded error with emergency compact.

    This is triggered when the API rejects a request due to context overflow.
    It aggressively removes messages to fit within the limit.

    Args:
        messages: Current messages
        model: Model being used
        error_message: The error message from API
        on_progress: Optional progress callback

    Returns:
        CompactResult if successful, None if no more compaction possible
    """
    global _auto_compact_state

    # Track reactive attempts
    if not hasattr(_auto_compact_state, "reactive_attempts"):
        _auto_compact_state.reactive_attempts = 0

    if _auto_compact_state.reactive_attempts >= MAX_REACTIVE_COMPACT_ATTEMPTS:
        return None

    _auto_compact_state.reactive_attempts += 1

    if on_progress:
        on_progress(f"Reactive compact (attempt {_auto_compact_state.reactive_attempts})...")

    # Get current token usage
    from claude_code_py.utils.context import (
        rough_token_count_estimation_for_messages,
        get_effective_context_window,
    )

    current_tokens = rough_token_count_estimation_for_messages(messages)
    max_tokens = get_effective_context_window(model)

    # Target: reduce to 50% of max
    target_tokens = int(max_tokens * REACTIVE_COMPACT_TOKEN_REDUCTION_RATIO)

    # Strategy: remove oldest messages aggressively
    messages_to_keep = _calculate_reactive_compact_messages(
        messages,
        target_tokens,
    )

    if len(messages_to_keep) >= len(messages):
        # Can't remove anything
        return None

    # Create compact result
    removed_count = len(messages) - len(messages_to_keep)

    # Add boundary marker
    boundary_msg = SystemMessage(
        type="system",
        subtype="compact_boundary",
        compact_metadata={
            "summary": f"Reactive compact: removed {removed_count} messages due to context overflow",
            "timestamp": datetime.now().isoformat(),
            "source": "reactive",
            "error": error_message,
        },
    )

    result = CompactResult(
        success=True,
        summary=f"Reactive compact: {removed_count} messages removed",
        metadata=CompactMetadata(
            tokens_before=current_tokens,
            tokens_after=target_tokens,
            messages_removed=removed_count,
            messages_kept=len(messages_to_keep) + 1,
        ),
    )

    return result


def _calculate_reactive_compact_messages(
    messages: list[Message],
    target_tokens: int,
) -> list[Message]:
    """Calculate messages to keep after reactive compact.

    Args:
        messages: Full message list
        target_tokens: Target token count

    Returns:
        Messages to keep
    """
    from claude_code_py.utils.context import rough_token_count_estimation_for_messages

    # Walk backwards to find cutoff point
    current_tokens = 0
    cutoff_idx = len(messages)

    # Always keep last few messages
    min_keep = 5

    for i in range(len(messages) - 1, min_keep, -1):
        # Calculate tokens for this range
        test_messages = messages[i:]
        test_tokens = rough_token_count_estimation_for_messages(test_messages)

        if test_tokens <= target_tokens:
            cutoff_idx = i
            break

    return messages[cutoff_idx:]


# =============================================================================
# Compact Execution Orchestrator
# =============================================================================


async def execute_compact_flow(
    messages: list[Message],
    model: str,
    query_source: str,
    on_progress: Optional[Callable[[str], None]] = None,
    is_reactive: bool = False,
    error_message: Optional[str] = None,
) -> Optional[CompactResult]:
    """Execute the complete compact flow.

    This orchestrates compact types for reactive and manual scenarios.

    Note: Microcompact is now handled separately in the query loop,
    before this function is called. This function handles:
    - Reactive compact (413 error recovery)
    - Session memory compact
    - Standard compact

    Args:
        messages: Current messages
        model: Model identifier
        query_source: Query source type
        on_progress: Progress callback
        is_reactive: Whether this is a reactive compact
        error_message: Error message if reactive

    Returns:
        CompactResult if compact occurred
    """
    # Reactive compact takes priority
    if is_reactive and error_message:
        return await reactive_compact(messages, model, error_message, on_progress)

    # Try session memory compact (microcompact now in query loop)
    # Use lazy import to avoid circular dependency
    try_session_memory_compaction, _, _ = _get_session_memory_compact_functions()
    threshold = get_auto_compact_threshold(model)
    session_result = await try_session_memory_compaction(messages, threshold)
    if session_result:
        return session_result

    # Check if standard compact needed
    should_compact, _ = should_auto_compact(messages, model)
    if should_compact:
        return await compact_conversation(messages, model, on_progress)

    return None


# =============================================================================
# Summary Generation
# =============================================================================


async def generate_compact_summary(
    conversation_text: str,
    model: str,
    cache_safe_params: Optional[Any] = None,
) -> str:
    """Generate a compact summary using the API.

    When cache_safe_params is provided, uses run_forked_agent to
    share the main thread's prompt cache for efficient summarization.

    CRITICAL: Do not set max_output_tokens when using cache sharing,
    as it changes budget_tokens and invalidates the cache key.

    Args:
        conversation_text: Text representation of conversation
        model: Model to use for summarization
        cache_safe_params: Optional cache params for prompt cache sharing

    Returns:
        Summary string
    """
    from claude_code_py.utils.side_query import side_query, SideQueryOptions, QuerySource
    from .compact_prompt import get_base_compact_prompt

    # Use Sonnet for fast summarization
    compact_model = get_default_sonnet_model()

    # Get the proper compact prompt
    system_prompt = get_base_compact_prompt()

    # Try cache sharing path first if params provided
    if cache_safe_params:
        try:
            from claude_code_py.utils.forked_agent import (
                run_forked_agent,
                ForkedAgentParams,
                create_read_only_can_use_tool,
                extract_text_from_messages,
            )
            from claude_code_py.core_types.message import UserMessage

            # Create summary request message
            summary_request = UserMessage(
                message={
                    "role": "user",
                    "content": f"Summarize the following conversation:\n\n{conversation_text}",
                }
            )

            # Run forked agent with cache sharing
            # DO NOT set max_output_tokens - would break cache sharing!
            result = await run_forked_agent(
                ForkedAgentParams(
                    prompt_messages=[summary_request],
                    cache_safe_params=cache_safe_params,
                    can_use_tool=create_read_only_can_use_tool(),
                    query_source="compact",
                    fork_label="compact_summary",
                    max_turns=1,
                    skip_cache_write=True,
                    # max_output_tokens intentionally NOT set
                )
            )

            # Extract summary text from result messages
            text = extract_text_from_messages(result.messages)

            # Extract content between <summary> tags if present
            if "<summary>" in text:
                start = text.find("<summary>") + len("<summary>")
                end = text.find("</summary>")
                if end > start:
                    return text[start:end].strip()

            return text.strip() if text else "Unable to generate summary."

        except Exception as e:
            # Log error and fall back to side_query
            import logging
            logging.debug(f"[compact] cache sharing failed, falling back: {e}")

    # Fallback: side_query without cache sharing
    opts = SideQueryOptions(
        model=compact_model,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"Summarize the following conversation:\n\n{conversation_text}",
        }],
        max_tokens=8192,
        query_source=QuerySource.MEMORY_EXTRACTION,
        skip_system_prompt_prefix=True,
    )

    result = await side_query(opts)

    # Extract summary from response
    for block in result.content:
        if block.get("type") == "text":
            text = block.get("text", "")
            # Extract content between <summary> tags
            if "<summary>" in text:
                start = text.find("<summary>") + len("<summary>")
                end = text.find("</summary>")
                if end > start:
                    return text[start:end].strip()
            return text.strip()

    return "Unable to generate summary."


# =============================================================================
# Message Utilities
# =============================================================================


def strip_images_from_messages(messages: list[Message]) -> list[Message]:
    """Remove image blocks from messages to save tokens.

    Args:
        messages: Original messages

    Returns:
        Messages with images replaced by placeholders
    """
    result = []

    for msg in messages:
        if msg.type == "user":
            content = msg.message.get("content", [])
            if isinstance(content, list):
                new_content = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "image":
                            # Replace with placeholder
                            new_content.append({
                                "type": "text",
                                "text": "[Image attached]",
                            })
                        else:
                            new_content.append(block)
                # Create modified message
                result.append(UserMessage(
                    uuid=msg.uuid,
                    message={"role": "user", "content": new_content},
                ))
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


def messages_to_compact_text(messages: list[Message]) -> str:
    """Convert messages to text for compaction.

    Args:
        messages: Message list

    Returns:
        Text representation
    """
    lines = []

    for msg in messages:
        if msg.type == "user":
            content = msg.message.get("content", "")
            if isinstance(content, str):
                lines.append(f"USER: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        lines.append(f"USER: {block.get('text', '')}")

        elif msg.type == "assistant":
            content = msg.message.get("content", [])
            if isinstance(content, str):
                lines.append(f"ASSISTANT: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        lines.append(f"ASSISTANT: {block.get('text', '')}")
                    elif isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        lines.append(f"TOOL_USE: {tool_name}({tool_input})")

    return "\n".join(lines)


def create_compact_boundary_messages(
    summary: str,
    original_messages: list[Message],
) -> list[Message]:
    """Create messages after compaction.

    Args:
        summary: Generated summary
        original_messages: Original messages (for extracting preserved items)

    Returns:
        New message list with summary
    """
    # Create compact boundary system message
    boundary_msg = SystemMessage(
        type="system",
        subtype="compact_boundary",
        compact_metadata={
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        },
    )

    return [boundary_msg]


# =============================================================================
# Post-Compact Restoration
# =============================================================================


async def restore_files_after_compact(
    original_messages: list[Message],
    file_cache: dict[str, Any],
) -> list[AttachmentMessage]:
    """Restore important files after compaction.

    Args:
        original_messages: Original messages before compaction
        file_cache: Cache of file contents

    Returns:
        List of file attachment messages to restore
    """
    # Find recently read files
    read_files: dict[str, dict[str, Any]] = {}

    for msg in reversed(original_messages):
        if msg.type == "assistant":
            content = msg.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if block.get("name") == "Read":
                            file_path = block.get("input", {}).get("file_path")
                            if file_path and file_path not in read_files:
                                read_files[file_path] = {
                                    "path": file_path,
                                    "tokens": 0,  # Would calculate from cache
                                }

        if len(read_files) >= POST_COMPACT_MAX_FILES_TO_RESTORE:
            break

    # Create attachment messages for restored files
    restored: list[AttachmentMessage] = []

    for file_path in list(read_files.keys())[:POST_COMPACT_MAX_FILES_TO_RESTORE]:
        if file_path in file_cache:
            content = file_cache[file_path]
            if isinstance(content, dict):
                content = content.get("content", "")

            # Truncate if needed
            if len(str(content)) > POST_COMPACT_MAX_TOKENS_PER_FILE * 4:
                content = str(content)[:POST_COMPACT_MAX_TOKENS_PER_FILE * 4]

            restored.append(AttachmentMessage(
                type="attachment",
                attachment={
                    "type": "file",
                    "path": file_path,
                    "content": content,
                },
            ))

    return restored


# =============================================================================
# Statistics
# =============================================================================


def get_compact_stats() -> dict[str, Any]:
    """Get compaction statistics.

    Returns:
        Stats dict
    """
    return {
        "enabled": _auto_compact_state.enabled,
        "consecutive_failures": _auto_compact_state.consecutive_failures,
        "last_compact_time": _auto_compact_state.last_compact_time,
        "in_progress": _auto_compact_state.in_progress,
    }


def reset_compact_state() -> None:
    """Reset auto-compact state."""
    global _auto_compact_state
    _auto_compact_state = AutoCompactState()


# =============================================================================
# Post-Compact Message Building
# =============================================================================


def build_post_compact_messages(
    compact_result: CompactResult,
    original_messages: list[Message],
) -> list[Message]:
    """Build the message list after compaction.

    This creates the messages to use after a compact:
    - Boundary message with summary
    - Summary message(s)
    - Messages to keep (from result or calculated)
    - Attachments and hook results

    If compact_result has pre-built messages (session memory compact),
    use those directly. Otherwise, build from metadata.

    Args:
        compact_result: Result from compaction
        original_messages: Original messages before compact (used as fallback)

    Returns:
        New message list
    """
    # Session memory compact: use pre-built messages directly
    if compact_result.boundary_marker is not None:
        return [
            compact_result.boundary_marker,
            *compact_result.summary_messages,
            *compact_result.messages_to_keep,
            *compact_result.attachments,
            *compact_result.hook_results,
        ]

    # Standard compact: build messages from metadata
    result = []

    # 1. Create boundary message
    boundary_msg = SystemMessage(
        type="system",
        subtype="compact_boundary",
        compact_metadata={
            "summary": compact_result.summary or "",
            "timestamp": datetime.now().isoformat(),
            "tokens_removed": (
                compact_result.metadata.tokens_before - compact_result.metadata.tokens_after
                if compact_result.metadata else 0
            ),
        },
    )
    result.append(boundary_msg)

    # 2. Add messages to keep
    if compact_result.messages_to_keep:
        # Use pre-calculated messages to keep if available
        result.extend(compact_result.messages_to_keep)
    else:
        # Calculate which messages to preserve from metadata
        messages_to_keep = _get_messages_to_keep(
            original_messages,
            compact_result.metadata.messages_removed if compact_result.metadata else 0,
        )
        result.extend(messages_to_keep)

    # 3. Add attachments
    result.extend(compact_result.attachments)

    # 4. Add hook results
    result.extend(compact_result.hook_results)

    return result


def _get_messages_to_keep(
    original_messages: list[Message],
    messages_removed: int,
) -> list[Message]:
    """Get messages to keep after compaction.

    Args:
        original_messages: Original message list
        messages_removed: Number of messages removed

    Returns:
        Messages to preserve
    """
    if messages_removed <= 0:
        return original_messages

    # Keep messages after the removed portion
    # Skip any existing compact boundary messages
    start_idx = messages_removed
    result = []

    for msg in original_messages[start_idx:]:
        if not _is_compact_boundary_message(msg):
            result.append(msg)

    return result


def _is_compact_boundary_message(message: Message) -> bool:
    """Check if message is a compact boundary.

    Args:
        message: Message to check

    Returns:
        True if compact boundary
    """
    if message.type == "system":
        if hasattr(message, "subtype"):
            return message.subtype == "compact_boundary"
        if hasattr(message, "message"):
            return message.message.get("subtype") == "compact_boundary"
    return False


# =============================================================================
# Cache Optimization Integration
# =============================================================================


def apply_cache_edits_after_compact(
    messages: list[Message],
    compact_boundary_uuid: str,
    last_message_uuid: str,
) -> list[Message]:
    """Apply cache_control edits after compact boundary.

    This preserves prompt caching on kept messages by:
    1. Adding ephemeral cache_control to the compact boundary message
    2. Ensuring subsequent messages maintain proper cache positions

    Args:
        messages: Post-compact message list
        compact_boundary_uuid: UUID of the compact boundary message
        last_message_uuid: UUID of the last message before compact

    Returns:
        Messages with cache_control properly applied
    """
    result: list[Message] = []

    for i, msg in enumerate(messages):
        # Clone message for modification
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump()
        else:
            msg_dict = dict(msg)

        # Apply cache_control to boundary message (first position)
        if i == 0 and msg_dict.get("type") == "system":
            msg_dict["cache_control"] = {"type": "ephemeral"}

        # Apply cache_control to last user message
        if msg_dict.get("type") == "user":
            # Check if this is the last user message before API call
            is_last_user = i == len(messages) - 1 or \
                all(m.get("type") != "user" for m in messages[i+1:])
            if is_last_user:
                msg_dict["cache_control"] = {"type": "ephemeral"}

        result.append(msg_dict)

    return result


def preserve_cache_on_kept_messages(
    original_messages: list[Message],
    kept_start_index: int,
) -> list[Message]:
    """Preserve cache_control on kept messages after truncation.

    Args:
        original_messages: Original message list
        kept_start_index: Index where kept messages start

    Returns:
        Kept messages with preserved cache_control
    """
    kept_messages = original_messages[kept_start_index:]

    # Preserve cache_control on first kept message
    result: list[Message] = []

    for i, msg in enumerate(kept_messages):
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump()
        else:
            msg_dict = dict(msg)

        # First kept message gets cache_control
        if i == 0:
            msg_dict["cache_control"] = {"type": "ephemeral"}

        result.append(msg_dict)

    return result


# =============================================================================
# Session Storage Integration
# =============================================================================


async def record_compact_to_session(
    session_storage: Any,
    compact_result: CompactResult,
    original_messages: list[Message],
    kept_messages: list[Message],
) -> None:
    """Record compact boundary and kept messages to session storage.

    Args:
        session_storage: SessionStorage instance
        compact_result: Compact result with metadata
        original_messages: Original messages before compact
        kept_messages: Messages kept after compact
    """
    from claude_code_py.core_types.message import SystemMessage

    # Get last parent UUID from original messages
    last_parent_uuid = None
    if original_messages:
        last_msg = original_messages[-1]
        last_parent_uuid = getattr(last_msg, "uuid", None)

    # Create compact boundary message
    boundary_msg = SystemMessage(
        type="system",
        subtype="compact_boundary",
        compact_metadata={
            "summary": compact_result.summary or "",
            "timestamp": datetime.now().isoformat(),
            "tokens_removed": compact_result.metadata.tokens_before - compact_result.metadata.tokens_after if compact_result.metadata else 0,
            "messages_removed": compact_result.metadata.messages_removed if compact_result.metadata else 0,
        },
    )

    # Record boundary to session
    session_storage.append_message(boundary_msg, parent_uuid=last_parent_uuid)

    # Record kept messages
    prev_uuid = getattr(boundary_msg, "uuid", None)
    for msg in kept_messages:
        session_storage.append_message(msg, parent_uuid=prev_uuid)
        prev_uuid = getattr(msg, "uuid", None)


# =============================================================================
# Compact Statistics and Monitoring
# =============================================================================


@dataclass
class CompactStats:
    """Statistics for compact monitoring."""

    total_compacts: int = 0
    session_memory_compacts: int = 0
    microcompacts: int = 0
    standard_compacts: int = 0
    reactive_compacts: int = 0
    failed_compacts: int = 0
    total_tokens_saved: int = 0
    total_messages_removed: int = 0
    last_compact_time: Optional[float] = None


# Global stats
_compact_stats = CompactStats()


def record_compact_stats(
    compact_type: str,
    tokens_saved: int,
    messages_removed: int,
    success: bool,
) -> None:
    """Record compact statistics.

    Args:
        compact_type: Type of compact (session_memory, microcompact, standard, reactive)
        tokens_saved: Tokens saved by compact
        messages_removed: Messages removed
        success: Whether compact succeeded
    """
    global _compact_stats

    if success:
        _compact_stats.total_compacts += 1
        _compact_stats.total_tokens_saved += tokens_saved
        _compact_stats.total_messages_removed += messages_removed
        _compact_stats.last_compact_time = datetime.now().timestamp()

        if compact_type == "session_memory":
            _compact_stats.session_memory_compacts += 1
        elif compact_type == "microcompact":
            _compact_stats.microcompacts += 1
        elif compact_type == "standard":
            _compact_stats.standard_compacts += 1
        elif compact_type == "reactive":
            _compact_stats.reactive_compacts += 1
    else:
        _compact_stats.failed_compacts += 1


def get_compact_stats() -> CompactStats:
    """Get compact statistics.

    Returns:
        CompactStats
    """
    return _compact_stats


def reset_compact_stats() -> None:
    """Reset compact statistics."""
    global _compact_stats
    _compact_stats = CompactStats()