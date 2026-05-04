"""Context Management - Token estimation and compaction.

This implements the context management system for handling token limits
and automatic conversation compaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from claude_code_py.core_types.message import Message, AssistantMessage


# =============================================================================
# Constants
# =============================================================================

# Context window sizes (tokens)
CONTEXT_WINDOW_CLAUDE_4 = 200_000
CONTEXT_WINDOW_CLAUDE_3_5 = 200_000

# Reserved tokens for output
MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000

# Max output tokens for API calls (slot-reservation optimization)
# Business data shows p99 output ~4,911 tokens, so 8k default is efficient
# Larger defaults (32k/64k) over-reserve slot capacity by 8-16x
CAPPED_DEFAULT_MAX_TOKENS = 8_000
ESCALATED_MAX_TOKENS = 64_000
MAX_OUTPUT_TOKENS_DEFAULT = 32_000
MAX_OUTPUT_TOKENS_UPPER_LIMIT = 64_000

# Threshold buffers
AUTOCOMPACT_BUFFER_TOKENS = 13_000
WARNING_THRESHOLD_BUFFER_TOKENS = 20_000
ERROR_THRESHOLD_BUFFER_TOKENS = 20_000
MANUAL_COMPACT_BUFFER_TOKENS = 3_000

# Post-compact restoration limits
POST_COMPACT_TOKEN_BUDGET = 50_000
POST_COMPACT_MAX_FILES_TO_RESTORE = 5
POST_COMPACT_MAX_TOKENS_PER_FILE = 5_000
POST_COMPACT_SKILLS_TOKEN_BUDGET = 25_000
POST_COMPACT_MAX_TOKENS_PER_SKILL = 5_000

# Token estimation
TOKENS_PER_CHAR = 0.25  # ~4 chars per token


# =============================================================================
# Token State
# =============================================================================


class TokenWarningLevel(str, Enum):
    """Token warning level."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    AUTO_COMPACT = "auto_compact"
    BLOCKING = "blocking"


@dataclass
class TokenWarningState:
    """State of token usage warnings."""

    token_usage: int
    percent_left: int
    is_above_warning_threshold: bool
    is_above_error_threshold: bool
    is_above_auto_compact_threshold: bool
    is_at_blocking_limit: bool
    level: TokenWarningLevel = TokenWarningLevel.OK


@dataclass
class TokenStats:
    """Token usage statistics by category."""

    tool_requests: dict[str, int] = field(default_factory=dict)
    tool_results: dict[str, int] = field(default_factory=dict)
    human_messages: int = 0
    assistant_messages: int = 0
    local_command_outputs: int = 0
    other: int = 0
    attachments: dict[str, int] = field(default_factory=dict)
    duplicate_file_reads: dict[str, dict[str, Any]] = field(default_factory=dict)
    total: int = 0


# =============================================================================
# Token Estimation
# =============================================================================


def get_context_window_for_model(model: str) -> int:
    """Get the context window size for a model.

    Args:
        model: Model identifier

    Returns:
        Context window size in tokens
    """
    model_lower = model.lower()

    if "claude-4" in model_lower or "claude-sonnet-4" in model_lower:
        return CONTEXT_WINDOW_CLAUDE_4
    elif "claude-3-5" in model_lower or "claude-3.5" in model_lower:
        return CONTEXT_WINDOW_CLAUDE_3_5
    elif "claude-opus" in model_lower or "claude-sonnet" in model_lower:
        return 200_000
    elif "claude-haiku" in model_lower:
        return 200_000

    # Default
    return 200_000


def get_effective_context_window(model: str) -> int:
    """Get effective context window after reserving output tokens.

    Args:
        model: Model identifier

    Returns:
        Effective context window in tokens
    """
    import os

    context_window = get_context_window_for_model(model)

    # Check for environment override
    auto_compact_window = os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW")
    if auto_compact_window:
        try:
            parsed = int(auto_compact_window)
            if parsed > 0:
                context_window = min(context_window, parsed)
        except ValueError:
            pass

    return context_window - MAX_OUTPUT_TOKENS_FOR_SUMMARY


def rough_token_count_estimation(text: str) -> int:
    """Estimate token count from text.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return int(len(text) * TOKENS_PER_CHAR)


def rough_token_count_estimation_for_messages(
    messages: list[Message],
) -> int:
    """Estimate token count from messages.

    Args:
        messages: Message list

    Returns:
        Estimated token count
    """
    total = 0

    for msg in messages:
        if msg.type == "user":
            content = msg.message.get("content", "")
            if isinstance(content, str):
                total += rough_token_count_estimation(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total += rough_token_count_estimation(
                                block.get("text", "")
                            )
                        elif block.get("type") == "image":
                            # Images are roughly 1000 tokens each
                            total += 1000

        elif msg.type == "assistant":
            content = msg.message.get("content", [])
            if isinstance(content, str):
                total += rough_token_count_estimation(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total += rough_token_count_estimation(
                                block.get("text", "")
                            )
                        elif block.get("type") == "tool_use":
                            # Tool use has overhead
                            total += 100
                            input_dict = block.get("input", {})
                            total += rough_token_count_estimation(
                                str(input_dict)
                            )

        elif msg.type == "attachment":
            # Attachments vary widely
            attachment = msg.attachment
            content = attachment.get("content", "")
            if isinstance(content, str):
                total += rough_token_count_estimation(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        total += rough_token_count_estimation(
                            item.get("content", "")
                        )

    return total


def get_token_count_from_usage(usage: dict[str, int]) -> int:
    """Get total token count from API usage data.

    Args:
        usage: Usage dict from API response

    Returns:
        Total token count
    """
    return (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("output_tokens", 0)
    )


def token_count_from_last_api_response(messages: list[Message]) -> int:
    """Get token count from the last API response.

    Args:
        messages: Message list

    Returns:
        Token count from last response, or 0 if not found
    """
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            usage = msg.usage
            if usage:
                return get_token_count_from_usage(usage)

    return 0


# =============================================================================
# Threshold Calculation
# =============================================================================


def get_auto_compact_threshold(model: str) -> int:
    """Get the auto-compact threshold for a model.

    Args:
        model: Model identifier

    Returns:
        Auto-compact threshold in tokens
    """
    import os

    effective_window = get_effective_context_window(model)
    threshold = effective_window - AUTOCOMPACT_BUFFER_TOKENS

    # Check for environment override
    env_percent = os.environ.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE")
    if env_percent:
        try:
            parsed = float(env_percent)
            if 0 < parsed <= 100:
                percentage_threshold = int(effective_window * (parsed / 100))
                return min(percentage_threshold, threshold)
        except ValueError:
            pass

    return threshold


def calculate_token_warning_state(
    token_usage: int,
    model: str,
    auto_compact_enabled: bool = True,
) -> TokenWarningState:
    """Calculate token warning state.

    Args:
        token_usage: Current token usage
        model: Model identifier
        auto_compact_enabled: Whether auto-compact is enabled

    Returns:
        TokenWarningState
    """
    auto_compact_threshold = get_auto_compact_threshold(model)
    effective_window = get_effective_context_window(model)

    threshold = auto_compact_threshold if auto_compact_enabled else effective_window

    percent_left = max(0, int(((threshold - token_usage) / threshold) * 100))

    warning_threshold = threshold - WARNING_THRESHOLD_BUFFER_TOKENS
    error_threshold = threshold - ERROR_THRESHOLD_BUFFER_TOKENS

    is_above_warning = token_usage >= warning_threshold
    is_above_error = token_usage >= error_threshold
    is_above_auto_compact = auto_compact_enabled and token_usage >= auto_compact_threshold

    blocking_limit = effective_window - MANUAL_COMPACT_BUFFER_TOKENS
    is_blocking = token_usage >= blocking_limit

    # Determine level
    if is_blocking:
        level = TokenWarningLevel.BLOCKING
    elif is_above_auto_compact:
        level = TokenWarningLevel.AUTO_COMPACT
    elif is_above_error:
        level = TokenWarningLevel.ERROR
    elif is_above_warning:
        level = TokenWarningLevel.WARNING
    else:
        level = TokenWarningLevel.OK

    return TokenWarningState(
        token_usage=token_usage,
        percent_left=percent_left,
        is_above_warning_threshold=is_above_warning,
        is_above_error_threshold=is_above_error,
        is_above_auto_compact_threshold=is_above_auto_compact,
        is_at_blocking_limit=is_blocking,
        level=level,
    )


# =============================================================================
# Context Analysis
# =============================================================================


def analyze_context(messages: list[Message]) -> TokenStats:
    """Analyze token usage by category.

    Args:
        messages: Message list

    Returns:
        TokenStats with breakdown
    """
    stats = TokenStats()
    tool_ids_to_names: dict[str, str] = {}
    file_read_stats: dict[str, dict[str, Any]] = {}

    for msg in messages:
        if msg.type == "attachment":
            att_type = msg.attachment.get("type", "unknown")
            stats.attachments[att_type] = stats.attachments.get(att_type, 0) + 1

        if msg.type == "user":
            content = msg.message.get("content", "")
            if isinstance(content, str):
                tokens = rough_token_count_estimation(content)
                stats.total += tokens
                stats.human_messages += tokens
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        _process_block(
                            block, stats, tool_ids_to_names, file_read_stats
                        )

        elif msg.type == "assistant":
            content = msg.message.get("content", [])
            if isinstance(content, str):
                tokens = rough_token_count_estimation(content)
                stats.total += tokens
                stats.assistant_messages += tokens
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        _process_block(
                            block, stats, tool_ids_to_names, file_read_stats
                        )

    # Calculate duplicate file reads
    for path, data in file_read_stats.items():
        if data["count"] > 1:
            avg_tokens = data["total_tokens"] // data["count"]
            duplicate_tokens = avg_tokens * (data["count"] - 1)
            stats.duplicate_file_reads[path] = {
                "count": data["count"],
                "tokens": duplicate_tokens,
            }

    return stats


def _process_block(
    block: dict[str, Any],
    stats: TokenStats,
    tool_ids_to_names: dict[str, str],
    file_read_stats: dict[str, dict[str, Any]],
) -> None:
    """Process a content block for token analysis."""
    block_type = block.get("type", "")

    if block_type == "text":
        tokens = rough_token_count_estimation(block.get("text", ""))
        stats.total += tokens

    elif block_type == "tool_use":
        tool_name = block.get("name", "unknown")
        tool_id = block.get("id", "")
        tool_ids_to_names[tool_id] = tool_name

        # Estimate input tokens
        input_dict = block.get("input", {})
        tokens = rough_token_count_estimation(str(input_dict)) + 100  # overhead
        stats.tool_requests[tool_name] = (
            stats.tool_requests.get(tool_name, 0) + tokens
        )
        stats.total += tokens

        # Track file reads
        if tool_name == "Read" and "file_path" in input_dict:
            path = input_dict["file_path"]
            if path not in file_read_stats:
                file_read_stats[path] = {"count": 0, "total_tokens": 0}
            file_read_stats[path]["count"] += 1
            file_read_stats[path]["total_tokens"] += tokens

    elif block_type == "tool_result":
        tool_use_id = block.get("tool_use_id", "")
        tool_name = tool_ids_to_names.get(tool_use_id, "unknown")

        content = block.get("content", "")
        if isinstance(content, str):
            tokens = rough_token_count_estimation(content)
        else:
            tokens = rough_token_count_estimation(str(content))

        stats.tool_results[tool_name] = (
            stats.tool_results.get(tool_name, 0) + tokens
        )
        stats.total += tokens


# =============================================================================
# Auto Compact Check
# =============================================================================


def is_auto_compact_enabled() -> bool:
    """Check if auto-compact is enabled.

    Returns:
        True if auto-compact is enabled
    """
    import os

    if os.environ.get("DISABLE_COMPACT", "").lower() in ("1", "true", "yes"):
        return False

    return True


def should_auto_compact(
    messages: list[Message],
    model: str,
) -> tuple[bool, int]:
    """Check if auto-compact should trigger.

    Args:
        messages: Current messages
        model: Model identifier

    Returns:
        Tuple of (should_compact, current_tokens)
    """
    if not is_auto_compact_enabled():
        return False, 0

    # Get token count
    token_usage = token_count_from_last_api_response(messages)
    if token_usage == 0:
        token_usage = rough_token_count_estimation_for_messages(messages)

    threshold = get_auto_compact_threshold(model)

    return token_usage >= threshold, token_usage