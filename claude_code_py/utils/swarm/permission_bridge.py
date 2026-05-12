"""Permission bridge for in-process teammates.

This module provides module-level setters that allow in-process teammates
to access the leader's UI for permission dialogs.

Ported from: src/utils/swarm/leaderPermissionBridge.ts
"""

from __future__ import annotations

from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Module-Level State
# =============================================================================

_permission_queue_setter: Optional[Callable[[Callable[[List], List]], None]] = None
_permission_context_setter: Optional[Callable[[Any, Optional[Dict]], None]] = None


# =============================================================================
# Registration Functions
# =============================================================================

def register_leader_permission_queue(
    setter: Callable[[Callable[[List], List]], None]
) -> None:
    """Register the leader's permission queue setter."""
    global _permission_queue_setter
    _permission_queue_setter = setter
    logger.debug("Registered leader permission queue setter")


def unregister_leader_permission_queue() -> None:
    """Unregister the leader's permission queue setter."""
    global _permission_queue_setter
    _permission_queue_setter = None
    logger.debug("Unregistered leader permission queue setter")


def register_leader_permission_context_setter(
    setter: Callable[[Any, Optional[Dict]], None]
) -> None:
    """Register the leader's permission context setter."""
    global _permission_context_setter
    _permission_context_setter = setter
    logger.debug("Registered leader permission context setter")


def unregister_leader_permission_context_setter() -> None:
    """Unregister the leader's permission context setter."""
    global _permission_context_setter
    _permission_context_setter = None
    logger.debug("Unregistered leader permission context setter")


# =============================================================================
# Getter Functions
# =============================================================================

def get_leader_permission_queue() -> Optional[Callable[[Callable[[List], List]], None]]:
    """Get the registered leader permission queue setter."""
    return _permission_queue_setter


def get_leader_permission_context_setter() -> Optional[Callable[[Any, Optional[Dict]], None]]:
    """Get the registered leader permission context setter."""
    return _permission_context_setter


# =============================================================================
# Types
# =============================================================================

@dataclass
class PermissionQueueItem:
    """An item in the ToolUseConfirm queue from a worker teammate."""

    assistant_message: Any
    tool: Any
    description: str
    input: Dict[str, Any]
    tool_use_context: Any
    tool_use_id: str
    permission_result: Any
    worker_badge: Optional[Dict[str, str]] = None
    permission_prompt_start_time_ms: int = 0
    on_user_interaction: Callable[[], None] = field(default_factory=lambda: lambda: None)
    on_abort: Callable[[], None] = field(default_factory=lambda: lambda: None)
    on_allow: Callable = field(default_factory=lambda: lambda: None)
    on_reject: Callable = field(default_factory=lambda: lambda: None)
    recheck_permission: Callable = field(default_factory=lambda: lambda: None)


@dataclass
class WorkerBadge:
    """Badge identifying a worker teammate in permission dialogs."""
    name: str
    color: str

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "color": self.color}


def enqueue_permission_request(item: PermissionQueueItem) -> bool:
    """Add a permission request to the leader's queue."""
    setter = get_leader_permission_queue()
    if not setter:
        logger.warning("Cannot enqueue permission: bridge not registered")
        return False
    setter(lambda prev: prev + [item])
    logger.debug(f"Enqueued permission request for tool {item.tool_use_id}")
    return True


__all__ = [
    "register_leader_permission_queue",
    "unregister_leader_permission_queue",
    "register_leader_permission_context_setter",
    "unregister_leader_permission_context_setter",
    "get_leader_permission_queue",
    "get_leader_permission_context_setter",
    "PermissionQueueItem",
    "WorkerBadge",
    "enqueue_permission_request",
]