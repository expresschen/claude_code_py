"""Session Memory implementation.

This implements the session memory system for current conversation summaries.
Session memory runs in the background using forked agents to extract key information.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .paths import get_session_memory_path, get_session_memory_dir
from .session_memory_prompts import (
    load_session_memory_template,
    build_session_memory_update_prompt,
    build_session_memory_init_prompt,
)


# =============================================================================
# Constants
# =============================================================================

# Thresholds for extraction
MINIMUM_MESSAGE_TOKENS_TO_INIT = 10000
MINIMUM_TOKENS_BETWEEN_UPDATE = 5000
TOOL_CALLS_BETWEEN_UPDATES = 3

# Extraction timeouts
EXTRACTION_WAIT_TIMEOUT_MS = 15000
EXTRACTION_STALE_THRESHOLD_MS = 60000  # 1 minute


# =============================================================================
# State Management
# =============================================================================


@dataclass
class SessionMemoryState:
    """State for session memory tracking."""

    initialized: bool = False
    last_memory_message_uuid: Optional[str] = None
    last_token_count: int = 0
    extraction_started_at: Optional[float] = None
    gate_checked: bool = False
    gate_enabled: bool = False


# Global state (module-level singleton)
_session_memory_state = SessionMemoryState()


def get_session_memory_state() -> SessionMemoryState:
    """Get the global session memory state."""
    return _session_memory_state


def reset_session_memory_state() -> None:
    """Reset the session memory state."""
    global _session_memory_state
    _session_memory_state = SessionMemoryState()


def mark_extraction_started() -> None:
    """Mark that extraction has started."""
    _session_memory_state.extraction_started_at = datetime.now().timestamp()


def mark_extraction_completed() -> None:
    """Mark that extraction has completed."""
    _session_memory_state.extraction_started_at = None


async def wait_for_session_memory_extraction() -> None:
    """Wait for any in-progress extraction to complete.

    Returns immediately if no extraction is in progress or if extraction is stale.
    """
    start_time = datetime.now().timestamp()

    while _session_memory_state.extraction_started_at:
        extraction_age = datetime.now().timestamp() - _session_memory_state.extraction_started_at

        # Extraction is stale (>1 min), don't wait
        if extraction_age > EXTRACTION_STALE_THRESHOLD_MS / 1000:
            return

        # Timeout reached
        elapsed = datetime.now().timestamp() - start_time
        if elapsed > EXTRACTION_WAIT_TIMEOUT_MS / 1000:
            return

        await asyncio.sleep(1)


# =============================================================================
# Threshold Checks
# =============================================================================


def is_session_memory_gate_enabled() -> bool:
    """Check if session memory feature gate is enabled.

    Returns:
        True if enabled
    """
    # Check environment override
    env_value = os.environ.get("CLAUDE_CODE_SESSION_MEMORY", "")
    if env_value.lower() in ("1", "true", "yes"):
        return True
    if env_value.lower() in ("0", "false", "no"):
        return False

    # Default: enabled if auto-compact is enabled
    auto_compact_disabled = os.environ.get("DISABLE_COMPACT", "").lower() in ("1", "true", "yes")
    return not auto_compact_disabled


def should_extract_memory(messages: list[Any]) -> bool:
    """Check if session memory should be extracted.

    Args:
        messages: Current message list

    Returns:
        True if memory should be extracted
    """
    global _session_memory_state

    # Check gate
    if not is_session_memory_gate_enabled():
        return False

    # Estimate token count
    current_token_count = sum(_estimate_tokens(msg) for msg in messages)

    # Check initialization threshold
    if not _session_memory_state.initialized:
        if current_token_count < MINIMUM_MESSAGE_TOKENS_TO_INIT:
            return False
        _session_memory_state.initialized = True

    # Check token threshold for updates
    has_met_token_threshold = (
        current_token_count - _session_memory_state.last_token_count
        >= MINIMUM_TOKENS_BETWEEN_UPDATE
    )

    # Check tool call threshold
    tool_calls_since_last = _count_tool_calls_since(
        messages, _session_memory_state.last_memory_message_uuid
    )
    has_met_tool_call_threshold = tool_calls_since_last >= TOOL_CALLS_BETWEEN_UPDATES

    # Check if last turn had tool calls
    has_tool_calls_in_last_turn = _has_tool_calls_in_last_assistant_turn(messages)

    # Trigger extraction when:
    # 1. Both thresholds met (tokens AND tool calls), OR
    # 2. Token threshold met AND no tool calls in last turn (natural breakpoint)
    should_extract = (
        (has_met_token_threshold and has_met_tool_call_threshold) or
        (has_met_token_threshold and not has_tool_calls_in_last_turn)
    )

    if should_extract:
        # Update state
        if messages:
            last_msg = messages[-1]
            _session_memory_state.last_memory_message_uuid = getattr(last_msg, "uuid", None)
        _session_memory_state.last_token_count = current_token_count
        return True

    return False


def _estimate_tokens(message: Any) -> int:
    """Estimate token count for a message.

    Args:
        message: Message object

    Returns:
        Estimated token count
    """
    # Rough approximation: ~4 chars per token
    try:
        if hasattr(message, "message"):
            content = message.message.get("content", "")
            if isinstance(content, str):
                return len(content) // 4
            elif isinstance(content, list):
                total = 0
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "") or block.get("content", "")
                        total += len(str(text)) // 4
                return total
    except Exception:
        pass
    return 0


def _count_tool_calls_since(messages: list[Any], since_uuid: Optional[str]) -> int:
    """Count tool calls since a specific message.

    Args:
        messages: Message list
        since_uuid: UUID to count after

    Returns:
        Tool call count
    """
    count = 0
    counting = since_uuid is None

    for msg in messages:
        if not counting:
            if hasattr(msg, "uuid") and msg.uuid == since_uuid:
                counting = True
            continue

        try:
            if hasattr(msg, "message"):
                content = msg.message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            count += 1
        except Exception:
            pass

    return count


def _has_tool_calls_in_last_assistant_turn(messages: list[Any]) -> bool:
    """Check if the last assistant turn had tool calls.

    Args:
        messages: Message list

    Returns:
        True if last assistant had tool calls
    """
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "assistant":
            try:
                content = msg.message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            return True
            except Exception:
                pass
            break

    return False


# =============================================================================
# Session Memory Manager
# =============================================================================


class SessionMemory:
    """Manager for session memory."""

    def __init__(self, session_id: Optional[str] = None, cwd: Optional[str] = None):
        """Initialize session memory.

        Args:
            session_id: Optional session ID
            cwd: Working directory (defaults to current)
        """
        self.session_id = session_id
        self._cwd = cwd
        self._memory_path = get_session_memory_path(cwd)
        self._state = _session_memory_state

    @property
    def path(self) -> Path:
        """Get the session memory file path."""
        return self._memory_path

    def exists(self) -> bool:
        """Check if session memory exists.

        Returns:
            True if memory file exists
        """
        return self._memory_path.exists()

    async def initialize(self) -> None:
        """Initialize the session memory file with secure permissions."""
        memory_dir = get_session_memory_dir(self._cwd)

        # Create directory with secure permissions
        memory_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(memory_dir, 0o700)
        except Exception:
            pass

        # Create file if it doesn't exist
        if not self._memory_path.exists():
            template = load_session_memory_template()
            self._memory_path.write_text(template, encoding="utf-8")
            try:
                os.chmod(self._memory_path, 0o600)
            except Exception:
                pass

    def read(self) -> str:
        """Read the session memory content.

        Returns:
            Memory content or empty string
        """
        try:
            return self._memory_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def write(self, content: str) -> None:
        """Write to the session memory.

        Args:
            content: Content to write
        """
        self._memory_path.write_text(content, encoding="utf-8")
        try:
            os.chmod(self._memory_path, 0o600)
        except Exception:
            pass

    def append(self, content: str) -> None:
        """Append to the session memory.

        Args:
            content: Content to append
        """
        with open(self._memory_path, "a", encoding="utf-8") as f:
            f.write(content)

    def reset(self) -> None:
        """Reset the session memory state."""
        reset_session_memory_state()


# =============================================================================
# Extraction Implementation
# =============================================================================


async def setup_session_memory_file(
    context: Any,
) -> tuple[str, str]:
    """Set up the session memory file.

    Args:
        context: Tool use context

    Returns:
        Tuple of (memory_path, current_memory)
    """
    cwd = context.get_cwd() if hasattr(context, "get_cwd") else None

    memory_dir = get_session_memory_dir(cwd)
    memory_path = get_session_memory_path(cwd)

    # Create directory
    try:
        memory_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        os.chmod(memory_dir, 0o700)
    except Exception:
        pass

    # Create file if it doesn't exist
    if not memory_path.exists():
        template = load_session_memory_template()
        try:
            memory_path.write_text(template, encoding="utf-8")
        except Exception:
            pass
        try:
            os.chmod(memory_path, 0o600)
        except Exception:
            pass

    # Read current content
    current_memory = ""
    try:
        current_memory = memory_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass

    return str(memory_path), current_memory


async def extract_session_memory(
    messages: list[Any],
    tool_use_context: Any,
    system_prompt: str,
    user_context: dict[str, str],
    system_context: dict[str, str],
) -> bool:
    """Extract and update session memory.

    This runs a forked agent to extract key information from the conversation
    and update the session memory file.

    Args:
        messages: Current conversation messages
        tool_use_context: Tool use context
        system_prompt: System prompt for cache sharing
        user_context: User context
        system_context: System context

    Returns:
        True if extraction succeeded
    """
    from claude_code_py.utils.forked_agent import (
        run_forked_agent_background,
        ForkedAgentParams,
        CacheSafeParams,
        create_cache_safe_params,
        create_user_message,
        create_memory_file_can_use_tool,
    )

    # Mark extraction started
    mark_extraction_started()

    try:
        # Set up file
        memory_path, current_memory = await setup_session_memory_file(tool_use_context)

        # Build extraction prompt
        user_prompt = build_session_memory_update_prompt(current_memory, memory_path)

        # Create cache-safe params
        cache_params = create_cache_safe_params(
            system_prompt=system_prompt,
            user_context=user_context,
            system_context=system_context,
            tool_use_context=tool_use_context,
            messages=messages,
        )

        # Create forked agent params
        fork_params = ForkedAgentParams(
            prompt_messages=[create_user_message(user_prompt)],
            cache_safe_params=cache_params,
            can_use_tool=create_memory_file_can_use_tool(memory_path),
            query_source="session_memory",
            fork_label="session_memory",
            max_turns=5,
        )

        # Run forked agent in background
        task = await run_forked_agent_background(fork_params)

        # Wait for completion (with timeout)
        try:
            result = await asyncio.wait_for(task, timeout=60.0)
        except asyncio.TimeoutError:
            # Extraction timed out
            pass

        mark_extraction_completed()
        return True

    except Exception:
        mark_extraction_completed()
        return False


async def manually_extract_session_memory(
    messages: list[Any],
    tool_use_context: Any,
    system_prompt: str,
    user_context: dict[str, str],
    system_context: dict[str, str],
) -> dict[str, Any]:
    """Manually trigger session memory extraction.

    This bypasses threshold checks and extracts immediately.
    Used by /summary command.

    Args:
        messages: Current conversation messages
        tool_use_context: Tool use context
        system_prompt: System prompt
        user_context: User context
        system_context: System context

    Returns:
        Result dict with success/error
    """
    from claude_code_py.utils.forked_agent import (
        run_forked_agent,
        ForkedAgentParams,
        CacheSafeParams,
        create_cache_safe_params,
        create_user_message,
        create_memory_file_can_use_tool,
    )

    if not messages:
        return {"success": False, "error": "No messages to summarize"}

    mark_extraction_started()

    try:
        # Set up file
        memory_path, current_memory = await setup_session_memory_file(tool_use_context)

        # Build extraction prompt
        user_prompt = build_session_memory_update_prompt(current_memory, memory_path)

        # Create cache-safe params
        cache_params = create_cache_safe_params(
            system_prompt=system_prompt,
            user_context=user_context,
            system_context=system_context,
            tool_use_context=tool_use_context,
            messages=messages,
        )

        # Create forked agent params
        fork_params = ForkedAgentParams(
            prompt_messages=[create_user_message(user_prompt)],
            cache_safe_params=cache_params,
            can_use_tool=create_memory_file_can_use_tool(memory_path),
            query_source="session_memory",
            fork_label="session_memory_manual",
            max_turns=5,
        )

        # Run forked agent (foreground for manual extraction)
        result = await run_forked_agent(fork_params)

        mark_extraction_completed()

        return {
            "success": True,
            "memory_path": memory_path,
            "messages_extracted": len(result.messages),
        }

    except Exception as e:
        mark_extraction_completed()
        return {"success": False, "error": str(e)}


# =============================================================================
# Post-Sampling Hook Registration
# =============================================================================


# Track registered hook
_hook_registered = False


def init_session_memory() -> None:
    """Initialize session memory by registering the post-turn hook.

    This is called during startup to enable session memory extraction.
    """
    global _hook_registered

    if _hook_registered:
        return

    # Check if session memory should be enabled
    if not is_session_memory_gate_enabled():
        return

    _hook_registered = True


def get_session_memory_content() -> Optional[str]:
    """Get the current session memory content.

    Returns:
        Content or None if file doesn't exist
    """
    memory_path = get_session_memory_path()

    if not memory_path.exists():
        return None

    try:
        return memory_path.read_text(encoding="utf-8")
    except Exception:
        return None