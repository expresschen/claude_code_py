"""AbortController implementation for async cancellation.

This mirrors the JavaScript AbortController API for Python async code.
Enhanced with two-level abort pattern (lifecycle vs per-turn) and timeout support.

Ported from: src/utils/abortController.ts (implicit patterns in swarm code)
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class AbortError(Exception):
    """Raised when an operation is aborted."""

    def __init__(self, message: str = "Operation aborted", reason: Optional[Exception] = None):
        super().__init__(message)
        self.name = "AbortError"
        self.reason = reason


@dataclass
class AbortSignal:
    """Signal object that can be used to abort async operations.

    Enhanced with:
    - Parent linking for hierarchical abort
    - Timeout support
    - Listener management
    """

    aborted: bool = False
    reason: Optional[Exception] = None
    _parent: Optional[AbortSignal] = field(default=None, repr=False)
    _listeners: List[Callable[[], None]] = field(default_factory=list, repr=False)
    _timeout_task: Optional[asyncio.Task] = field(default=None, repr=False)

    def throw_if_aborted(self) -> None:
        """Throw AbortError if the signal is aborted."""
        if self.aborted:
            msg = str(self.reason) if self.reason else "Operation aborted"
            raise AbortError(msg, self.reason)

    def add_event_listener(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Add a listener for abort events.

        Returns a function to remove the listener.
        """
        self._listeners.append(callback)
        return lambda: self._remove_listener(callback)

    def _remove_listener(self, callback: Callable[[], None]) -> None:
        """Remove a listener."""
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def _abort(self, reason: Optional[Exception] = None) -> None:
        """Internal method to abort the signal."""
        if self.aborted:
            return

        self.aborted = True
        self.reason = reason

        # Cancel timeout task if exists
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None

        # Notify all listeners
        for listener in self._listeners:
            try:
                listener()
            except Exception as e:
                logger.debug(f"Abort listener error: {e}")

        # Clear listeners after firing
        self._listeners.clear()

    def link_parent(self, parent: AbortSignal) -> None:
        """Link this signal to a parent signal.

        When parent aborts, this signal also aborts.
        """
        if self._parent:
            return

        self._parent = parent

        # If parent is already aborted, abort this one too
        if parent.aborted:
            self._abort(parent.reason)
            return

        # Register listener for parent abort
        parent.add_event_listener(lambda: self._on_parent_abort())

    def _on_parent_abort(self) -> None:
        """Called when parent signal aborts."""
        if self._parent and self._parent.aborted and not self.aborted:
            self._abort(self._parent.reason)


class AbortController:
    """Controller for aborting async operations.

    Enhanced with:
    - Parent signal linking for hierarchical abort
    - Timeout support (auto-abort after duration)
    """

    def __init__(
        self,
        parent_signal: Optional[AbortSignal] = None,
        timeout_ms: Optional[int] = None,
    ) -> None:
        self._signal = AbortSignal()

        # Link parent if provided
        if parent_signal:
            self._signal.link_parent(parent_signal)

        # Set up timeout if provided
        if timeout_ms:
            self._schedule_timeout(timeout_ms)

    def _schedule_timeout(self, timeout_ms: int) -> None:
        """Schedule auto-abort after timeout."""
        async def timeout_abort():
            await asyncio.sleep(timeout_ms / 1000.0)
            if not self._signal.aborted:
                self.abort(AbortError(f"Operation timed out after {timeout_ms}ms"))

        try:
            loop = asyncio.get_event_loop()
            self._signal._timeout_task = loop.create_task(timeout_abort())
        except RuntimeError:
            # No event loop running, schedule later
            pass

    @property
    def signal(self) -> AbortSignal:
        """Get the abort signal."""
        return self._signal

    def abort(self, reason: Optional[Exception] = None) -> None:
        """Abort the operation."""
        self._signal._abort(reason)


def create_abort_controller(
    parent_signal: Optional[AbortSignal] = None,
    timeout_ms: Optional[int] = None,
) -> AbortController:
    """Factory function to create an AbortController.

    Args:
        parent_signal: Optional parent signal to link to.
            When parent aborts, child also aborts.
        timeout_ms: Optional timeout in milliseconds.
            After this duration, auto-abort with timeout error.

    Returns:
        New AbortController with optional linking and timeout.
    """
    return AbortController(parent_signal, timeout_ms)


async def check_abort(signal: AbortSignal) -> None:
    """Check if the signal is aborted and raise if so.

    This is a checkpoint that can be called in async code.
    """
    if signal.aborted:
        msg = str(signal.reason) if signal.reason else "Operation aborted"
        raise AbortError(msg, signal.reason)
    await asyncio.sleep(0)  # Yield to event loop


# =============================================================================
# Two-Level Abort Pattern (Lifecycle vs Work)
# =============================================================================

@dataclass
class AbortControllerPair:
    """A pair of abort controllers for two-level abort pattern.

    Used by in-process teammates:
    - lifecycle_controller: Kills the whole teammate
    - work_controller: Stops current turn only (Escape pressed)

    The work controller is NOT linked to lifecycle - pressing Escape
    stops current work but keeps the teammate alive.
    """

    lifecycle: AbortController
    work: AbortController

    @property
    def lifecycle_signal(self) -> AbortSignal:
        return self.lifecycle.signal

    @property
    def work_signal(self) -> AbortSignal:
        return self.work.signal

    def abort_work(self, reason: Optional[Exception] = None) -> None:
        """Abort current work turn (not whole teammate)."""
        self.work.abort(reason)

    def abort_lifecycle(self, reason: Optional[Exception] = None) -> None:
        """Abort the whole teammate."""
        self.lifecycle.abort(reason)
        # Also abort current work
        self.work.abort(reason)


def create_abort_controller_pair(
    parent_signal: Optional[AbortSignal] = None,
) -> AbortControllerPair:
    """Create a pair of abort controllers for two-level abort.

    The lifecycle controller may be linked to a parent.
    The work controller is independent (not linked).

    Args:
        parent_signal: Optional parent for lifecycle controller

    Returns:
        AbortControllerPair with independent work controller
    """
    lifecycle = create_abort_controller(parent_signal)
    work = create_abort_controller()  # NOT linked to lifecycle

    return AbortControllerPair(lifecycle, work)


__all__ = [
    "AbortError",
    "AbortSignal",
    "AbortController",
    "AbortControllerPair",
    "create_abort_controller",
    "create_abort_controller_pair",
    "check_abort",
]