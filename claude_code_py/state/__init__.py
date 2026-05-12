"""State management system.

This implements a Zustand-like store pattern for Python.
"""

from .store import Store
from .app_state import AppState, get_default_app_state
from .context import ToolPermissionContext

__all__ = [
    "Store",
    "AppState",
    "get_default_app_state",
    "ToolPermissionContext",
]