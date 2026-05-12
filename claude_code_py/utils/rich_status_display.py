"""Rich Live display for teammate status.

This provides real-time visualization of running agents in the REPL,
similar to TypeScript's TeammateSpinnerTree component.

Usage:
    from claude_code_py.utils.rich_status_display import TeammateStatusDisplay

    display = TeammateStatusDisplay(get_app_state, set_app_state)
    display.start()

    # ... REPL loop runs ...

    display.stop()
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel


# Color mapping (matching TypeScript spinner colors)
COLOR_MAP = {
    "green": "green",
    "blue": "blue",
    "yellow": "yellow",
    "red": "red",
    "magenta": "magenta",
    "cyan": "cyan",
    "white": "white",
    "orange": "orange3",
    "purple": "purple",
    "pink": "pink",
}

# Spinner characters (cycling animation)
SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Default verbs for different states
DEFAULT_VERBS = {
    "thinking": "thinking",
    "reading": "reading files",
    "writing": "writing code",
    "tool_call": "using tools",
    "waiting": "waiting",
    "idle": "idle",
    "default": "working",
}


class TeammateStatusDisplay:
    """Real-time display of running teammate agents.

    Uses Rich Live to show a status table that updates periodically.
    Runs in a background thread to avoid blocking the main REPL loop.
    """

    def __init__(
        self,
        get_app_state: Callable[[], Any],
        set_app_state: Callable[[Callable[[Any], Any]], None],
        refresh_per_second: int = 4,
        console: Optional[Console] = None,
    ):
        """Initialize the status display.

        Args:
            get_app_state: Function to get current AppState
            set_app_state: Function to update AppState
            refresh_per_second: How often to refresh display
            console: Optional Rich Console (creates new if None)
        """
        self._get_app_state = get_app_state
        self._set_app_state = set_app_state
        self._refresh_rate = refresh_per_second
        self._console = console or Console()

        self._live: Optional[Live] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False
        self._spinner_index = 0
        self._lock = threading.Lock()  # Protect Live operations

    def start(self) -> None:
        """Start the status display in background thread."""
        if self._running:
            return

        self._running = True

        def run_display():
            """Run Rich Live display in dedicated thread."""
            try:
                # Create Live display
                with self._lock:
                    self._live = Live(
                        self._generate_table(),
                        console=self._console,
                        refresh_per_second=self._refresh_rate,
                        transient=True,  # Clear when done
                        vertical_overflow="visible",
                        auto_refresh=True,
                    )
                    self._live.start()

                # Keep updating while running
                while self._running:
                    if not self._paused:
                        self._spinner_index = (self._spinner_index + 1) % len(SPINNER_CHARS)
                        with self._lock:
                            if self._live and not self._paused:
                                try:
                                    self._live.update(self._generate_table())
                                except Exception:
                                    pass  # Live may have been stopped
                    time.sleep(1.0 / self._refresh_rate)

            except Exception:
                pass  # Silently exit on errors

            finally:
                with self._lock:
                    if self._live:
                        try:
                            self._live.stop()
                        except Exception:
                            pass

        self._thread = threading.Thread(
            target=run_display,
            name="teammate-status-display",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the status display."""
        self._running = False

        with self._lock:
            if self._live:
                try:
                    self._live.stop()
                except Exception:
                    pass

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def pause(self) -> None:
        """Pause the display temporarily (e.g., while waiting for input).

        This stops the Live display without stopping the background thread,
        allowing normal Console output to appear without interference.
        """
        self._paused = True
        with self._lock:
            if self._live:
                try:
                    self._live.stop()
                    self._live = None  # Clear reference, will recreate on resume
                except Exception:
                    pass

    def resume(self) -> None:
        """Resume the display after pausing.

        Recreate the Live display to show teammate status again.
        """
        self._paused = False
        with self._lock:
            if not self._live and self._running:
                try:
                    self._live = Live(
                        self._generate_table(),
                        console=self._console,
                        refresh_per_second=self._refresh_rate,
                        transient=True,
                        vertical_overflow="visible",
                        auto_refresh=True,
                    )
                    self._live.start()
                except Exception:
                    pass

    def _generate_table(self) -> Table:
        """Generate the status display table.

        Returns:
            Rich Table containing the status (no Panel wrapper to avoid duplication)
        """
        try:
            app_state = self._get_app_state()
            tasks = app_state.tasks if hasattr(app_state, "tasks") else {}
            team_context = app_state.team_context if hasattr(app_state, "team_context") else None
        except Exception:
            tasks = {}
            team_context = None

        # Collect teammates from both sources:
        # 1. tasks (InProcessTeammateTaskState) - running agents
        # 2. team_context.teammates - agent metadata including idle state

        teammates_info = {}  # agent_name -> {is_idle, color, verb, task_id}

        # Source 1: tasks (InProcessTeammateTaskState)
        for task_id, task in tasks.items():
            if hasattr(task, "identity"):
                # InProcessTeammateTaskState dataclass
                name = task.identity.agent_name
                is_idle = task.is_idle if hasattr(task, "is_idle") else False
                color = task.color or task.identity.color or "white"
                verb = task.spinner_verb or "working"
                teammates_info[name] = {
                    "is_idle": is_idle,
                    "color": color,
                    "verb": verb,
                    "task_id": task_id,
                }
            elif isinstance(task, dict) and (task.get("type") == "in_process_teammate" or task.get("agent_name")):
                # Dict-style task
                name = task.get("agent_name") or task.get("name") or "unknown"
                is_idle = task.get("is_idle", False)
                color = task.get("color") or "white"
                verb = task.get("spinner_verb") or "working"
                teammates_info[name] = {
                    "is_idle": is_idle,
                    "color": color,
                    "verb": verb,
                    "task_id": task_id,
                }

        # Source 2: team_context.teammates (metadata with isIdle)
        if team_context and "teammates" in team_context:
            for agent_id, teammate_info in team_context["teammates"].items():
                name = teammate_info.get("name")
                if name and name not in teammates_info:
                    # Not in tasks, but exists in team_context
                    is_idle = teammate_info.get("isIdle", True)
                    color = teammate_info.get("color") or "white"
                    teammates_info[name] = {
                        "is_idle": is_idle,
                        "color": color,
                        "verb": "idle" if is_idle else "available",
                        "task_id": None,
                    }
                elif name in teammates_info:
                    # Update idle state from team_context if available
                    if "isIdle" in teammate_info:
                        teammates_info[name]["is_idle"] = teammate_info["isIdle"]

        # Filter running teammates (not idle)
        running_teammates = [name for name, info in teammates_info.items() if not info["is_idle"]]

        # Build a simple table without Panel wrapper
        table = Table(
            show_header=False,
            show_edge=False,
            padding=(0, 1),
            expand=False,
        )
        table.add_column("status")

        # If no teammates at all, return empty table
        if not teammates_info:
            return table

        # If no running teammates, show idle indicator with all teammate names
        if not running_teammates:
            spinner = SPINNER_CHARS[self._spinner_index]
            idle_names = list(teammates_info.keys())
            # Show all idle teammates in one row
            idle_text = Text()
            idle_text.append(f"{spinner} ", style="dim")
            for name in idle_names:
                info = teammates_info[name]
                rich_color = COLOR_MAP.get(info["color"], "white")
                idle_text.append(f"● {name} ", style=rich_color)
            idle_text.append("(idle)", style="dim")
            table.add_row(idle_text)
            return table

        # Build status rows for running teammates
        for name in running_teammates:
            info = teammates_info[name]
            rich_color = COLOR_MAP.get(info["color"], "white")
            verb = info["verb"]

            # Get spinner char
            spinner = SPINNER_CHARS[self._spinner_index]

            # Build status line
            # Format: ● name: verb
            status_text = Text()
            status_text.append("● ", style=rich_color)
            status_text.append(f"{name}: ", style="bold")
            status_text.append(f"{spinner} {verb}", style=rich_color)

            table.add_row(status_text)

        return table

    def is_running(self) -> bool:
        """Check if display is running."""
        return self._running


def create_status_display(
    get_app_state: Callable[[], Any],
    set_app_state: Callable[[Callable[[Any], Any]], None],
    console: Optional[Console] = None,
) -> TeammateStatusDisplay:
    """Factory function to create a status display.

    Args:
        get_app_state: Function to get current AppState
        set_app_state: Function to update AppState
        console: Optional Rich Console (shared from REPL to avoid conflicts)

    Returns:
        TeammateStatusDisplay instance
    """
    return TeammateStatusDisplay(get_app_state, set_app_state, console=console)


__all__ = [
    "TeammateStatusDisplay",
    "create_status_display",
    "COLOR_MAP",
    "SPINNER_CHARS",
]