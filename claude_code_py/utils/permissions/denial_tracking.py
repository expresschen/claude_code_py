"""Denial tracking for permissions.

This tracks permission denials to prevent repetitive prompts
for the same operation and provide better UX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from collections import defaultdict


# =============================================================================
# Constants
# =============================================================================


# Maximum denials before fallback to prompting
MAX_DENIALS_BEFORE_PROMPT = 3

# Cooldown period (seconds) before resetting denial count
DENIAL_COOLDOWN_SECONDS = 60


# =============================================================================
# Denial Tracking State
# =============================================================================


@dataclass
class DenialRecord:
    """Record of a single denial."""

    tool_name: str
    tool_input_hash: str
    reason: str
    timestamp: float
    user_message: Optional[str] = None


@dataclass
class DenialTrackingState:
    """State for tracking denials."""

    # Denials per tool + input hash
    denials: dict[str, list[DenialRecord]] = field(default_factory=lambda: defaultdict(list))

    # Last success timestamp per tool (to reset denial count)
    last_success: dict[str, float] = field(default_factory=dict)

    # Session start time
    session_start: float = field(default_factory=lambda: datetime.now().timestamp())


def create_denial_tracking_state() -> DenialTrackingState:
    """Create a new denial tracking state.

    Returns:
        Fresh tracking state
    """
    return DenialTrackingState()


# =============================================================================
# Input Hashing
# =============================================================================


def hash_tool_input(tool_name: str, input: Any) -> str:
    """Create a hash key for tool + input combination.

    This is used to identify repeated attempts of the same operation.

    Args:
        tool_name: Tool name
        input: Tool input

    Returns:
        Hash key string
    """
    import hashlib
    import json

    # Convert input to stable string
    if hasattr(input, "model_dump"):
        input_dict = input.model_dump()
    elif isinstance(input, dict):
        input_dict = input
    else:
        input_dict = {"value": str(input)}

    # Sort keys for stability
    input_str = json.dumps(input_dict, sort_keys=True, default=str)

    # Hash
    hash_obj = hashlib.sha256(f"{tool_name}:{input_str}".encode())
    return hash_obj.hexdigest()[:16]  # Use first 16 chars


# =============================================================================
# Recording Functions
# =============================================================================


def record_denial(
    state: DenialTrackingState,
    tool_name: str,
    input: Any,
    reason: str,
    user_message: Optional[str] = None,
) -> None:
    """Record a denial in tracking state.

    Args:
        state: Tracking state
        tool_name: Tool name
        input: Tool input
        reason: Denial reason
        user_message: Optional user feedback message
    """
    input_hash = hash_tool_input(tool_name, input)
    record = DenialRecord(
        tool_name=tool_name,
        tool_input_hash=input_hash,
        reason=reason,
        timestamp=datetime.now().timestamp(),
        user_message=user_message,
    )

    state.denials[input_hash].append(record)


def record_success(
    state: DenialTrackingState,
    tool_name: str,
    input: Any,
) -> None:
    """Record a successful permission grant.

    This resets the denial count for that operation.

    Args:
        state: Tracking state
        tool_name: Tool name
        input: Tool input
    """
    input_hash = hash_tool_input(tool_name, input)

    # Clear denials for this input
    if input_hash in state.denials:
        state.denials[input_hash] = []

    # Record success time
    state.last_success[tool_name] = datetime.now().timestamp()


# =============================================================================
# Count Functions
# =============================================================================


def get_denial_count(
    state: DenialTrackingState,
    tool_name: str,
    input: Any,
) -> int:
    """Get number of denials for a tool + input.

    Only counts recent denials (within cooldown period).

    Args:
        state: Tracking state
        tool_name: Tool name
        input: Tool input

    Returns:
        Number of recent denials
    """
    input_hash = hash_tool_input(tool_name, input)
    records = state.denials.get(input_hash, [])

    # Filter by cooldown
    now = datetime.now().timestamp()
    recent_records = [
        r for r in records
        if now - r.timestamp < DENIAL_COOLDOWN_SECONDS
    ]

    return len(recent_records)


def get_total_tool_denials(
    state: DenialTrackingState,
    tool_name: str,
) -> int:
    """Get total denials for a tool (all inputs).

    Args:
        state: Tracking state
        tool_name: Tool name

    Returns:
        Total denials for tool
    """
    total = 0
    for records in state.denials.values():
        for record in records:
            if record.tool_name == tool_name:
                total += 1
    return total


# =============================================================================
# Fallback Logic
# =============================================================================


def should_fallback_to_prompting(
    state: DenialTrackingState,
    tool_name: str,
    input: Any,
) -> bool:
    """Check if we should fallback to prompting user.

    After too many denials, we stop auto-blocking and let the user
    decide explicitly.

    Args:
        state: Tracking state
        tool_name: Tool name
        input: Tool input

    Returns:
        True if should prompt user instead of auto-block
    """
    denial_count = get_denial_count(state, tool_name, input)

    # If denied too many times, let user decide
    return denial_count >= MAX_DENIALS_BEFORE_PROMPT


def get_last_denial_reason(
    state: DenialTrackingState,
    tool_name: str,
    input: Any,
) -> Optional[str]:
    """Get the reason for the most recent denial.

    Args:
        state: Tracking state
        tool_name: Tool name
        input: Tool input

    Returns:
        Last denial reason or None
    """
    input_hash = hash_tool_input(tool_name, input)
    records = state.denials.get(input_hash, [])

    if not records:
        return None

    # Return most recent
    return records[-1].reason


def get_denial_history_message(
    state: DenialTrackingState,
    tool_name: str,
    input: Any,
) -> Optional[str]:
    """Get a message describing denial history for this operation.

    Args:
        state: Tracking state
        tool_name: Tool name
        input: Tool input

    Returns:
        History message or None
    """
    denial_count = get_denial_count(state, tool_name, input)

    if denial_count == 0:
        return None

    last_reason = get_last_denial_reason(state, tool_name, input)

    if denial_count == 1:
        return f"This operation was previously denied: {last_reason}"
    else:
        return f"This operation has been denied {denial_count} times. Last reason: {last_reason}"


# =============================================================================
# Session Stats
# =============================================================================


def get_session_stats(state: DenialTrackingState) -> dict[str, Any]:
    """Get statistics for the current session.

    Args:
        state: Tracking state

    Returns:
        Stats dictionary
    """
    total_denials = sum(len(records) for records in state.denials.values())

    # Denials by tool
    denials_by_tool: dict[str, int] = defaultdict(int)
    for records in state.denials.values():
        for record in records:
            denials_by_tool[record.tool_name] += 1

    return {
        "session_duration_seconds": datetime.now().timestamp() - state.session_start,
        "total_denials": total_denials,
        "denials_by_tool": dict(denials_by_tool),
        "unique_denied_operations": len(state.denials),
        "tools_with_success": len(state.last_success),
    }


# =============================================================================
# Reset Functions
# =============================================================================


def reset_denial_tracking(state: DenialTrackingState) -> None:
    """Reset all denial tracking state.

    Called on /clear or /compact to start fresh.

    Args:
        state: Tracking state
    """
    state.denials.clear()
    state.last_success.clear()
    state.session_start = datetime.now().timestamp()


def reset_tool_denials(state: DenialTrackingState, tool_name: str) -> None:
    """Reset denials for a specific tool.

    Args:
        state: Tracking state
        tool_name: Tool name
    """
    # Remove all records for this tool
    for input_hash, records in list(state.denials.items()):
        state.denials[input_hash] = [
            r for r in records if r.tool_name != tool_name
        ]
        if not state.denials[input_hash]:
            del state.denials[input_hash]

    # Remove success record
    if tool_name in state.last_success:
        del state.last_success[tool_name]