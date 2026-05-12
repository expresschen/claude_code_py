"""Generic Store implementation.

This mirrors the Zustand store pattern from JavaScript/TypeScript.

IMPORTANT: Thread-safe for cross-thread state access (e.g., background loop teammates).
"""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class Store(Generic[T]):
    """A simple state store with subscription support.

    This is similar to Zustand's createStore pattern.

    Thread-safe: Uses a lock to protect state access across threads.
    This is essential when background event loop threads access AppState
    via set_app_state / get_app_state callbacks.

    Example:
        ```python
        store = Store(initial_state={"count": 0})

        # Subscribe to changes
        def on_change(state):
            print(f"Count is now {state['count']}")

        unsubscribe = store.subscribe(on_change)

        # Update state (thread-safe)
        store.set_state(lambda prev: {**prev, "count": prev["count"] + 1})
        ```
    """

    def __init__(
        self,
        initial_state: T,
        on_change: Optional[Callable[[T, T], None]] = None,
    ):
        """Initialize the store.

        Args:
            initial_state: Initial state value
            on_change: Optional callback called with (new_state, old_state) on changes
        """
        self._state = initial_state
        self._listeners: list[Callable[[T], None]] = []
        self._on_change = on_change
        self._lock = threading.Lock()  # Thread-safe lock

    def get_state(self) -> T:
        """Get the current state (thread-safe).

        Returns:
            Current state (deep copy to prevent cross-thread mutation)
        """
        with self._lock:
            # Return deep copy for thread safety
            if isinstance(self._state, dict):
                return deepcopy(self._state)
            elif hasattr(self._state, "__dataclass_fields__"):
                return deepcopy(self._state)
            else:
                return self._state

    def set_state(self, updater: Callable[[T], T]) -> None:
        """Update the state (thread-safe).

        Args:
            updater: Function that receives current state and returns new state.
                    The updater should create a new object, not mutate the input.
        """
        with self._lock:
            old_state = self._state

            # Deep copy to ensure immutability
            if isinstance(old_state, dict):
                old_state_copy = deepcopy(old_state)
            elif hasattr(old_state, "__dataclass_fields__"):
                old_state_copy = deepcopy(old_state)
            else:
                old_state_copy = old_state

            new_state = updater(old_state_copy)
            self._state = new_state

        # Notify listeners (outside lock to avoid blocking)
        if self._on_change:
            self._on_change(new_state, old_state)

        for listener in self._listeners:
            try:
                listener(new_state)
            except Exception:
                pass  # Ignore listener errors

    def subscribe(self, listener: Callable[[T], None]) -> Callable[[], None]:
        """Subscribe to state changes.

        Args:
            listener: Callback called with new state on changes

        Returns:
            Unsubscribe function
        """
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def __repr__(self) -> str:
        return f"Store({self._state!r})"


def create_store(
    initial_state: T,
    on_change: Optional[Callable[[T, T], None]] = None,
) -> Store[T]:
    """Factory function to create a store.

    Args:
        initial_state: Initial state value
        on_change: Optional callback for changes

    Returns:
        New Store instance
    """
    return Store(initial_state, on_change)