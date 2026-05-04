"""Global AppState store.

This provides a singleton Store instance for AppState,
matching TypeScript's AppStateContext pattern.
"""

from __future__ import annotations

from typing import Callable, Optional

from .store import Store
from .app_state import AppState


# Global AppState store instance
_app_store: Optional[Store[AppState]] = None


def get_app_store() -> Store[AppState]:
    """Get the global AppState store.

    Creates the store on first access if it doesn't exist.

    Returns:
        Store[AppState] singleton instance
    """
    global _app_store
    if _app_store is None:
        _app_store = Store[AppState](
            initial_state=AppState(),
        )
    return _app_store


def set_app_state(updater: Callable[[AppState], AppState]) -> None:
    """Update the global AppState.

    This is equivalent to TypeScript's context.setAppState(prev => ...).

    Args:
        updater: Function that receives current state and returns new state
    """
    store = get_app_store()
    store.set_state(updater)


def get_current_app_state() -> AppState:
    """Get the current AppState snapshot.

    Returns:
        Current AppState
    """
    return get_app_store().get_state()


def initialize_app_store(initial_state: Optional[AppState] = None) -> Store[AppState]:
    """Initialize the global AppState store with custom initial state.

    This should be called once at application startup.

    Args:
        initial_state: Optional custom initial state

    Returns:
        The initialized store
    """
    global _app_store
    _app_store = Store[AppState](
        initial_state=initial_state or AppState(),
    )
    return _app_store


def reset_app_store() -> None:
    """Reset the global AppState store.

    Used for testing or when switching sessions.
    """
    global _app_store
    _app_store = None