"""Cached Microcompact State Management.

This implements the state tracking for cached microcompact, matching
the TypeScript cachedMicrocompact.js module functionality:
- Track registered tool results
- Track deleted references
- Maintain tool order for LRU eviction
- Pin cache_edits for re-sending in subsequent calls
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from claude_code_py.core_types.message import Message


# =============================================================================
# Constants
# =============================================================================

# Tool names eligible for cached microcompact (matching TypeScript COMPACTABLE_TOOLS)
COMPACTABLE_TOOLS = frozenset([
    "Read",       # FILE_READ_TOOL_NAME
    "Bash",       # SHELL_TOOL_NAMES
    "Grep",       # GREP_TOOL_NAME
    "Glob",       # GLOB_TOOL_NAME
    "WebSearch",  # WEB_SEARCH_TOOL_NAME
    "WebFetch",   # WEB_FETCH_TOOL_NAME
    "Edit",       # FILE_EDIT_TOOL_NAME
    "Write",      # FILE_WRITE_TOOL_NAME
])


# =============================================================================
# Types
# =============================================================================


@dataclass
class PinnedCacheEdits:
    """Pinned cache_edits block with its position.

    Used to re-send cache_edits at their original positions in subsequent API calls.
    """

    userMessageIndex: int
    block: dict[str, Any]


@dataclass
class CachedMCConfig:
    """Configuration for cached microcompact.

    Matches TypeScript's config from GrowthBook:
    - triggerThreshold: Number of tools to trigger deletion
    - keepRecent: Number of most recent tools to keep
    """

    enabled: bool = True
    triggerThreshold: int = 20  # Delete when this many tools registered
    keepRecent: int = 5  # Keep this many most recent


@dataclass
class CachedMCState:
    """State for cached microcompact tracking.

    Matches TypeScript's CachedMCState from cachedMicrocompact.js:
    - registeredTools: Set of tool_use_ids that have been registered
    - deletedRefs: Set of tool_use_ids that have been deleted
    - toolOrder: List maintaining insertion order of tool_use_ids
    - pinnedEdits: List of pinned cache_edits for re-sending
    """

    registeredTools: set[str] = field(default_factory=set)
    deletedRefs: set[str] = field(default_factory=set)
    toolOrder: list[str] = field(default_factory=list)
    pinnedEdits: list[PinnedCacheEdits] = field(default_factory=list)
    toolsSentToAPI: bool = False


# =============================================================================
# Core Functions
# =============================================================================


def create_cached_mc_state() -> CachedMCState:
    """Factory function to create a new CachedMCState.

    Returns:
        Fresh CachedMCState instance
    """
    return CachedMCState()


def get_cached_mc_config() -> CachedMCConfig:
    """Get cached microcompact config.

    In production, would read from GrowthBook/config system.
    For now, returns defaults matching TypeScript.

    Returns:
        CachedMCConfig with current settings
    """
    # Could integrate with feature flag system in future
    # Environment overrides for testing
    import os

    config = CachedMCConfig()

    # Check environment overrides
    if os.environ.get("DISABLE_CACHED_MICROCOMPACT", "").lower() in ("1", "true", "yes"):
        config.enabled = False

    threshold_override = os.environ.get("CACHED_MC_TRIGGER_THRESHOLD")
    if threshold_override:
        try:
            config.triggerThreshold = int(threshold_override)
        except ValueError:
            pass

    keep_override = os.environ.get("CACHED_MC_KEEP_RECENT")
    if keep_override:
        try:
            config.keepRecent = int(keep_override)
        except ValueError:
            pass

    return config


def collect_compactable_tool_ids(messages: list["Message"]) -> list[str]:
    """Collect tool_use IDs from compactable tools in encounter order.

    Only includes tools whose name is in COMPACTABLE_TOOLS.
    Matches TypeScript's collectCompactableToolIds().

    Args:
        messages: Message list

    Returns:
        List of tool_use IDs in order of appearance
    """
    ids: list[str] = []
    for message in messages:
        if message.type == "assistant":
            content = message.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_name = block.get("name", "")
                            if tool_name in COMPACTABLE_TOOLS:
                                tool_id = block.get("id", "")
                                if tool_id:
                                    ids.append(tool_id)
    return ids


def register_tool_result(state: CachedMCState, tool_use_id: str) -> None:
    """Register a tool result for potential deletion.

    Adds to registeredTools and toolOrder.
    Matches TypeScript's registerToolResult().

    Args:
        state: CachedMCState instance
        tool_use_id: Tool use ID to register
    """
    if tool_use_id and tool_use_id not in state.registeredTools:
        state.registeredTools.add(tool_use_id)
        state.toolOrder.append(tool_use_id)


def register_tool_message(state: CachedMCState, group_ids: list[str]) -> None:
    """Register a group of tool results from same user message.

    Matches TypeScript's registerToolMessage().
    Currently just tracks - no special handling needed in Python implementation.

    Args:
        state: CachedMCState instance
        group_ids: List of tool_use IDs from the same message
    """
    # In TypeScript, this tracks message boundaries
    # For Python implementation, we simplify since we track by tool_use_id
    pass


def get_tool_results_to_delete(state: CachedMCState) -> list[str]:
    """Get tool_use_ids to delete based on config thresholds.

    Uses triggerThreshold and keepRecent from config.

    Logic:
    - If registered count < triggerThreshold, return empty (no deletion)
    - Delete all except keepRecent most recent
    - Exclude already-deleted IDs

    Matches TypeScript's getToolResultsToDelete().

    Args:
        state: CachedMCState instance

    Returns:
        List of tool_use IDs to delete
    """
    config = get_cached_mc_config()

    if not config.enabled:
        return []

    # Count active (registered but not deleted)
    active_tools = [
        id for id in state.toolOrder
        if id not in state.deletedRefs
    ]

    # Don't delete if under threshold
    if len(active_tools) < config.triggerThreshold:
        return []

    # Keep keepRecent most recent
    # Floor at 1: slice(-0) returns full array (paradoxically keeps everything)
    # Always keep at least the last result
    keep_count = max(1, config.keepRecent)
    tools_to_keep = set(active_tools[-keep_count:])

    # Delete older tools (before keepRecent window)
    tools_to_delete = [
        id for id in active_tools[:-keep_count]
        if id not in state.deletedRefs
    ]

    return tools_to_delete


def create_cache_edits_block(
    state: CachedMCState,
    tools_to_delete: list[str],
) -> dict[str, Any]:
    """Create cache_edits block for API.

    Also marks the tools as deleted in state.
    Matches TypeScript's createCacheEditsBlock().

    Args:
        state: CachedMCState instance
        tools_to_delete: List of tool_use IDs to delete

    Returns:
        cache_edits block dict for API
    """
    edits = [
        {"type": "delete", "cache_reference": tool_id}
        for tool_id in tools_to_delete
    ]

    # Mark as deleted in state
    for tool_id in tools_to_delete:
        state.deletedRefs.add(tool_id)

    return {
        "type": "cache_edits",
        "edits": edits,
    }


def mark_tools_sent_to_api(state: CachedMCState) -> None:
    """Mark all registered tools as sent to API.

    Called after successful API response.

    Args:
        state: CachedMCState instance
    """
    state.toolsSentToAPI = True


def reset_cached_mc_state(state: CachedMCState) -> None:
    """Reset cached microcompact state.

    Called after time-based microcompact or /clear.
    Matches TypeScript's resetCachedMCState().

    Args:
        state: CachedMCState instance
    """
    state.registeredTools.clear()
    state.deletedRefs.clear()
    state.toolOrder.clear()
    state.pinnedEdits.clear()
    state.toolsSentToAPI = False


def pin_cache_edits(
    state: CachedMCState,
    user_message_index: int,
    block: dict[str, Any],
) -> None:
    """Pin cache_edits to re-send in subsequent calls.

    Args:
        state: CachedMCState instance
        user_message_index: Position to re-insert the block
        block: cache_edits block to pin
    """
    state.pinnedEdits.append(PinnedCacheEdits(
        userMessageIndex=user_message_index,
        block=block,
    ))