"""Cache Optimization Utilities.

This implements cache_control placement for optimal prompt caching:
- Global scope: System prompt static prefix (cross-user caching)
- Org scope: Memory/project context (cross-session caching)
- Ephemeral scope: Last message (per-session caching)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


# =============================================================================
# Constants
# =============================================================================

# Boundary marker for static vs dynamic system prompt
SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "<system_prompt_dynamic_boundary>"

# Cache scopes
class CacheScope(str, Enum):
    """Cache scope levels."""

    GLOBAL = "global"      # Cross-user caching (system prompt)
    ORG = "org"            # Cross-session caching (memory)
    EPHEMERAL = "ephemeral"  # Per-session caching (last message)


# =============================================================================
# Types
# =============================================================================


@dataclass
class CacheControlBlock:
    """Cache control block for API."""

    type: str = "ephemeral"
    scope: Optional[str] = None


@dataclass
class SystemPromptBlock:
    """Block of system prompt with cache info."""

    content: str
    cache_scope: Optional[CacheScope] = None
    cache_control: Optional[CacheControlBlock] = None


# =============================================================================
# System Prompt Splitting
# =============================================================================


def split_system_prompt_for_cache(
    system_prompt: str,
    memory_content: Optional[str] = None,
    user_context: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Split system prompt into cacheable blocks.

    Creates blocks with appropriate cache_control markers:
    1. Static prefix -> global scope (cross-user)
    2. Memory/context -> org scope (cross-session)
    3. Dynamic parts -> no cache
    4. Tools -> ephemeral scope
    5. Last message -> ephemeral scope

    Args:
        system_prompt: Full system prompt
        memory_content: Optional memory content
        user_context: Optional user context

    Returns:
        List of blocks with cache_control
    """
    blocks = []

    # Split by boundary marker
    if SYSTEM_PROMPT_DYNAMIC_BOUNDARY in system_prompt:
        static_part, dynamic_part = system_prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY, 1)
    else:
        # No boundary, treat first half as static
        split_idx = len(system_prompt) // 2
        static_part = system_prompt[:split_idx]
        dynamic_part = system_prompt[split_idx:]

    # Static prefix with global cache
    if static_part.strip():
        blocks.append({
            "type": "text",
            "text": static_part.strip(),
            "cache_control": {"type": "ephemeral", "scope": "global"},
        })

    # Memory content with org cache
    if memory_content and memory_content.strip():
        blocks.append({
            "type": "text",
            "text": "\n\n" + memory_content.strip(),
            "cache_control": {"type": "ephemeral", "scope": "org"},
        })

    # User context with org cache
    if user_context:
        context_text = format_user_context(user_context)
        if context_text.strip():
            blocks.append({
                "type": "text",
                "text": context_text,
                "cache_control": {"type": "ephemeral", "scope": "org"},
            })

    # Dynamic parts (no cache)
    if dynamic_part.strip():
        blocks.append({
            "type": "text",
            "text": dynamic_part.strip(),
        })

    return blocks


def format_user_context(user_context: dict[str, str]) -> str:
    """Format user context for system prompt.

    Args:
        user_context: Dict of context key -> value

    Returns:
        Formatted context string
    """
    lines = ["\n<user_context>"]
    for key, value in user_context.items():
        lines.append(f"{key}: {value}")
    lines.append("</user_context>")
    return "\n".join(lines)


# =============================================================================
# Message Cache Control
# =============================================================================


def add_cache_control_to_last_message(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add ephemeral cache_control to the last message.

    This enables caching of the conversation prefix.

    Args:
        messages: API-formatted messages

    Returns:
        Messages with cache_control on last message
    """
    if not messages:
        return messages

    result = messages.copy()

    # Add cache_control to last message's content
    last_msg = result[-1]
    content = last_msg.get("content", [])

    if isinstance(content, str):
        # Convert to block format
        result[-1] = {
            "role": last_msg["role"],
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    elif isinstance(content, list):
        # Add cache_control to last block
        if content:
            new_blocks = content.copy()
            last_block = new_blocks[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}
            result[-1] = {
                "role": last_msg["role"],
                "content": new_blocks,
            }

    return result


def add_cache_control_to_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add ephemeral cache_control to tools definition.

    Args:
        tools: Tool definitions

    Returns:
        Tools with cache_control
    """
    result = []

    for tool in tools:
        tool_copy = tool.copy()
        # Add cache_control to tool schema
        if "input_schema" in tool_copy:
            tool_copy["cache_control"] = {"type": "ephemeral"}
        result.append(tool_copy)

    return result


# =============================================================================
# Cache Reference Tracking
# =============================================================================


@dataclass
class CacheReference:
    """Reference to a cached block."""

    id: str
    type: str  # "tool_use" or "content_block"
    cache_scope: CacheScope
    created_at: float


class CacheReferenceTracker:
    """Track cache references for cache_edits."""

    def __init__(self):
        """Initialize tracker."""
        self._references: dict[str, CacheReference] = {}
        self._next_id = 1

    def register_tool_use(self, tool_use_id: str, cache_scope: CacheScope) -> str:
        """Register a tool_use block for potential deletion.

        Args:
            tool_use_id: The tool_use.id from API
            cache_scope: Cache scope of the block

        Returns:
            Internal reference ID
        """
        import time

        ref_id = f"ref_{self._next_id}"
        self._next_id += 1

        self._references[ref_id] = CacheReference(
            id=ref_id,
            type="tool_use",
            cache_scope=cache_scope,
            created_at=time.time(),
        )

        # Also map tool_use_id -> ref_id
        self._references[tool_use_id] = self._references[ref_id]

        return ref_id

    def get_reference_for_tool_use(self, tool_use_id: str) -> Optional[CacheReference]:
        """Get cache reference for a tool_use ID.

        Args:
            tool_use_id: Tool use ID from API

        Returns:
            CacheReference if found
        """
        return self._references.get(tool_use_id)

    def mark_for_deletion(self, tool_use_id: str) -> Optional[dict[str, Any]]:
        """Create delete edit for a tool_use.

        Args:
            tool_use_id: Tool use ID to delete

        Returns:
            Delete edit dict
        """
        ref = self.get_reference_for_tool_use(tool_use_id)
        if ref:
            return {
                "type": "delete",
                "cache_reference": tool_use_id,
            }
        return None

    def clear(self) -> None:
        """Clear all references."""
        self._references.clear()
        self._next_id = 1


# =============================================================================
# Cache Stats
# =============================================================================


def estimate_cache_savings(
    messages: list[dict[str, Any]],
    system_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate potential cache savings.

    Args:
        messages: API messages
        system_blocks: System prompt blocks

    Returns:
        Cache savings estimate
    """
    from .context import rough_token_count_estimation

    global_tokens = 0
    org_tokens = 0
    ephemeral_tokens = 0

    # Count system prompt cached tokens
    for block in system_blocks:
        cc = block.get("cache_control", {})
        scope = cc.get("scope", "ephemeral")
        text = block.get("text", "")
        tokens = rough_token_count_estimation(text)

        if scope == "global":
            global_tokens += tokens
        elif scope == "org":
            org_tokens += tokens
        else:
            ephemeral_tokens += tokens

    # Count message tokens
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            ephemeral_tokens += rough_token_count_estimation(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    ephemeral_tokens += rough_token_count_estimation(block.get("text", ""))

    return {
        "global_cached_tokens": global_tokens,
        "org_cached_tokens": org_tokens,
        "ephemeral_cached_tokens": ephemeral_tokens,
        "total_cached_tokens": global_tokens + org_tokens + ephemeral_tokens,
        "cache_hit_potential": global_tokens * 0.9 + org_tokens * 0.7 + ephemeral_tokens * 0.5,
    }


# =============================================================================
# Build Full API Request
# =============================================================================


def build_cached_api_request(
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    memory_content: Optional[str] = None,
    user_context: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Build API request with optimal cache placement.

    Args:
        system_prompt: System prompt text
        messages: Conversation messages
        tools: Tool definitions
        memory_content: Optional memory content
        user_context: Optional user context

    Returns:
        API request dict with cache_control markers
    """
    # Split system prompt
    system_blocks = split_system_prompt_for_cache(
        system_prompt,
        memory_content=memory_content,
        user_context=user_context,
    )

    # Add cache to last message
    cached_messages = add_cache_control_to_last_message(messages)

    # Add cache to tools
    cached_tools = add_cache_control_to_tools(tools)

    return {
        "system": system_blocks,
        "messages": cached_messages,
        "tools": cached_tools,
    }


# =============================================================================
# Cache Breakpoints (for cached microcompact)
# =============================================================================


def add_cache_breakpoints(
    messages: list[dict[str, Any]],
    enable_prompt_caching: bool,
    query_source: Optional[str] = None,
    use_cached_mc: bool = False,
    new_cache_edits: Optional[dict[str, Any]] = None,
    pinned_edits: Optional[list[dict[str, Any]]] = None,
    skip_cache_write: bool = False,
) -> list[dict[str, Any]]:
    """Add cache_control and cache_edits to messages for API.

    Matches TypeScript's addCacheBreakpoints() in claude.ts:3063-3211.

    Key responsibilities:
    1. Add cache_control to last message (ephemeral)
    2. Insert pinned cache_edits at their original positions
    3. Insert new cache_edits in last user message
    4. Add cache_reference to tool_result blocks before cache_control marker

    Args:
        messages: API-formatted messages
        enable_prompt_caching: Whether caching is enabled
        query_source: Source of the query
        use_cached_mc: Whether to use cached microcompact
        new_cache_edits: New cache_edits block to insert
        pinned_edits: Previously pinned cache_edits to re-send
        skip_cache_write: Skip cache write (for fire-and-forget forks)

    Returns:
        Messages with cache_control, cache_edits, and cache_reference
    """
    # Create a deep copy to avoid mutating original
    result = []
    for msg in messages:
        msg_copy = msg.copy()
        if isinstance(msg_copy.get("content"), list):
            msg_copy["content"] = [b.copy() if isinstance(b, dict) else b for b in msg_copy["content"]]
        result.append(msg_copy)

    # 1. Add cache_control to appropriate message
    # For fire-and-forget forks (skipCacheWrite), shift marker to second-to-last
    # message: that's the last shared-prefix point
    marker_index = len(result) - 2 if skip_cache_write else len(result) - 1
    if marker_index >= 0 and marker_index < len(result):
        marker_msg = result[marker_index]
        content = marker_msg.get("content", [])
        if isinstance(content, list) and content:
            # Add cache_control to last block of marker message
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}

    if not use_cached_mc:
        return result

    # Track seen cache_references to prevent duplicates
    seen_delete_refs: set[str] = set()

    def deduplicate_edits(block: dict[str, Any]) -> dict[str, Any]:
        """Filter out duplicate deletions."""
        edits = block.get("edits", [])
        unique_edits = []
        for edit in edits:
            ref = edit.get("cache_reference", "")
            if ref and ref not in seen_delete_refs:
                seen_delete_refs.add(ref)
                unique_edits.append(edit)
        return {"type": "cache_edits", "edits": unique_edits}

    def insert_block_after_tool_results(content: list[Any], block: dict[str, Any]) -> None:
        """Insert cache_edits block after all tool_result blocks."""
        # Find last tool_result index
        last_tool_result_idx = -1
        for i, c in enumerate(content):
            if isinstance(c, dict) and c.get("type") == "tool_result":
                last_tool_result_idx = i

        # Insert after tool_results or at end
        if last_tool_result_idx >= 0:
            content.insert(last_tool_result_idx + 1, block)
        else:
            content.append(block)

    # 2. Re-insert pinned cache_edits at original positions
    for pinned in pinned_edits or []:
        msg_index = pinned.get("userMessageIndex", 0)
        block = pinned.get("block", {})
        if msg_index < len(result):
            msg = result[msg_index]
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    deduped_block = deduplicate_edits(block)
                    if deduped_block.get("edits"):
                        insert_block_after_tool_results(content, deduped_block)

    # 3. Insert new cache_edits into last user message and pin them
    if new_cache_edits:
        deduped_new = deduplicate_edits(new_cache_edits)
        if deduped_new.get("edits"):
            for i in range(len(result) - 1, -1, -1):
                msg = result[i]
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        insert_block_after_tool_results(content, deduped_new)
                        # Pin for subsequent calls
                        try:
                            from claude_code_py.services.micro_compact import pin_cache_edits
                            pin_cache_edits(i, new_cache_edits)
                        except ImportError:
                            pass
                    break

    # 4. Add cache_reference to tool_result blocks before cache_control
    if enable_prompt_caching:
        # Find the last message containing a cache_control marker
        last_cc_msg = -1
        for i, msg in enumerate(result):
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        last_cc_msg = i

        # Add cache_reference to tool_results strictly before last_cc_msg
        # The API requires cache_reference to appear "before or on" the last
        # cache_control — we use strict "before" to avoid edge cases
        if last_cc_msg >= 0:
            for i in range(last_cc_msg):
                msg = result[i]
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for j, block in enumerate(content):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        # Create new object to avoid mutation
                        tool_use_id = block.get("tool_use_id", "")
                        if tool_use_id:
                            content[j] = {**block, "cache_reference": tool_use_id}

    return result