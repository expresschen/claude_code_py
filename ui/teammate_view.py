"""Teammate View - Rich-based UI for viewing agent status and messages.

This provides a terminal UI for viewing individual agent status, messages,
and navigating between agents in a team.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Callable, Any
from enum import Enum

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from claude_code_py.state.app_state import AppState, SetAppState
from claude_code_py.task.in_process_teammate import (
    InProcessTeammateTaskState,
    is_in_process_teammate_task,
    TeammateIdentity,
)


# =============================================================================
# View Selection Mode
# =============================================================================


class ViewSelectionMode(str, Enum):
    """View selection mode for agent navigation."""

    NONE = "none"
    SELECTING = "selecting-agent"
    VIEWING = "viewing-agent"


class ExpandedView(str, Enum):
    """Expanded view state."""

    NONE = "none"
    TEAMMATES = "teammates"


# =============================================================================
# AppState Extensions
# =============================================================================


def extend_app_state_for_viewing(app_state: AppState) -> dict[str, Any]:
    """Get viewing-related fields from AppState with defaults.

    AppState 需要添加以下字段:
    - viewing_agent_task_id: Optional[str]
    - view_selection_mode: str (default "none")
    - selected_agent_index: int (default -1)
    - expanded_view: str (default "none")
    """
    return {
        "viewing_agent_task_id": getattr(app_state, "viewing_agent_task_id", None),
        "view_selection_mode": getattr(app_state, "view_selection_mode", ViewSelectionMode.NONE.value),
        "selected_agent_index": getattr(app_state, "selected_agent_index", -1),
        "expanded_view": getattr(app_state, "expanded_view", ExpandedView.NONE.value),
    }


# =============================================================================
# Selectors
# =============================================================================


def get_viewed_teammate_task(app_state: AppState) -> Optional[InProcessTeammateTaskState]:
    """Get the currently viewed teammate task."""
    viewing_id = app_state.viewing_agent_task_id
    if not viewing_id:
        return None

    task = app_state.tasks.get(viewing_id)
    if not task or not is_in_process_teammate_task(task):
        return None

    return task


def get_running_teammates_sorted(app_state: AppState) -> list[InProcessTeammateTaskState]:
    """Get all running teammates sorted by agent name."""
    teammates = [
        t for t in app_state.tasks.values()
        if is_in_process_teammate_task(t) and t.status == "running"
    ]
    return sorted(teammates, key=lambda t: t.identity.agent_name)


def get_teammate_count(app_state: AppState) -> int:
    """Get count of running teammates."""
    return len(get_running_teammates_sorted(app_state))


# =============================================================================
# View Helpers
# =============================================================================


PANEL_GRACE_MS = 30_000  # 30 seconds before evicting completed tasks


def enter_teammate_view(
    task_id: str,
    set_app_state: SetAppState,
) -> None:
    """Enter teammate view to see their status and messages.

    Sets viewing_agent_task_id and retain=True to prevent eviction.
    """
    from dataclasses import replace

    def updater(prev: AppState) -> AppState:
        task = prev.tasks.get(task_id)
        if not task or not is_in_process_teammate_task(task):
            return prev

        new_tasks = dict(prev.tasks)

        # Release previous viewed task if any
        prev_id = prev.viewing_agent_task_id
        if prev_id and prev_id != task_id:
            prev_task = new_tasks.get(prev_id)
            if prev_task and is_in_process_teammate_task(prev_task):
                new_tasks[prev_id] = _release_task(prev_task)

        # Set retain on new task
        new_tasks[task_id] = _set_retain(task, True)

        return replace(
            prev,
            viewing_agent_task_id=task_id,
            view_selection_mode=ViewSelectionMode.VIEWING.value,
            tasks=new_tasks,
        )

    set_app_state(updater)


def exit_teammate_view(set_app_state: SetAppState) -> None:
    """Exit teammate view and return to leader view."""
    from dataclasses import replace

    def updater(prev: AppState) -> AppState:
        viewing_id = prev.viewing_agent_task_id
        new_tasks = dict(prev.tasks)

        if viewing_id:
            task = new_tasks.get(viewing_id)
            if task and is_in_process_teammate_task(task):
                new_tasks[viewing_id] = _release_task(task)

        return replace(
            prev,
            viewing_agent_task_id=None,
            view_selection_mode=ViewSelectionMode.NONE.value,
            tasks=new_tasks,
        )

    set_app_state(updater)


def step_teammate_selection(
    delta: int,  # +1 or -1
    set_app_state: SetAppState,
    app_state: AppState,
) -> None:
    """Step teammate selection by delta with wrapping."""
    from dataclasses import replace

    count = get_teammate_count(app_state)
    if count == 0:
        return

    current = app_state.selected_agent_index
    expanded = app_state.expanded_view

    # Expand if collapsed
    if expanded != ExpandedView.TEAMMATES.value:
        set_app_state(lambda prev: replace(
            prev,
            expanded_view=ExpandedView.TEAMMATES.value,
            view_selection_mode=ViewSelectionMode.SELECTING.value,
            selected_agent_index=-1,
        ))
        return

    # Step with wrapping
    max_idx = count  # hide row
    if delta == 1:
        next_idx = -1 if current >= max_idx else current + 1
    else:
        next_idx = max_idx if current <= -1 else current - 1

    set_app_state(lambda prev: replace(
        prev,
        selected_agent_index=next_idx,
        view_selection_mode=ViewSelectionMode.SELECTING.value,
    ))


def _release_task(task: InProcessTeammateTaskState) -> InProcessTeammateTaskState:
    """Release task back to stub form."""
    # Create new state with retain=False and messages cleared
    return InProcessTeammateTaskState(
        id=task.id,
        type=task.type,
        status=task.status,
        description=task.description,
        identity=task.identity,
        prompt=task.prompt,
        abort_controller=task.abort_controller,
        tool_use_id=task.tool_use_id,
        start_time=task.start_time,
        end_time=task.end_time,
        output_file=task.output_file,
        output_offset=task.output_offset,
        notified=task.notified,
        model=task.model,
        awaiting_plan_approval=task.awaiting_plan_approval,
        permission_mode=task.permission_mode,
        is_idle=task.is_idle,
        shutdown_requested=task.shutdown_requested,
        error=task.error,
        spinner_verb=task.spinner_verb,
        past_tense_verb=task.past_tense_verb,
        last_reported_tool_count=task.last_reported_tool_count,
        last_reported_token_count=task.last_reported_token_count,
        token_count=task.token_count,
        color=task.color,
        pending_user_messages=[],
        messages=[],  # Clear messages
    )


def _set_retain(task: InProcessTeammateTaskState, retain: bool) -> InProcessTeammateTaskState:
    """Set retain flag on task (conceptually - we just return the task as-is for now)."""
    # In Python, we don't have a retain field yet, but we can add it
    # For now, just return the task
    return task


# =============================================================================
# Rich UI Components
# =============================================================================


class TeammateViewUI:
    """Rich-based UI for viewing teammate status and messages."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def render_teammate_header(self, task: InProcessTeammateTaskState) -> Panel:
        """Render header showing which agent is being viewed."""
        color = task.color or "cyan"
        name = task.identity.agent_name

        header_text = Text()
        header_text.append("Viewing ", style="dim")
        header_text.append(f"@{name}", style=f"bold {color}")
        header_text.append("  ·  ", style="dim")
        header_text.append("esc", style="bold")
        header_text.append(" return", style="dim")

        prompt_text = Text(task.prompt[:100], style="dim")
        if len(task.prompt) > 100:
            prompt_text.append("...", style="dim")

        return Panel(
            Group(header_text, prompt_text),
            border_style=color,
            padding=(0, 1),
        )

    def render_status_line(self, task: InProcessTeammateTaskState) -> Text:
        """Render status line with running/idle indicator."""
        status_text = Text()

        # Status indicator
        if task.status == "running":
            if task.is_idle:
                status_text.append("● ", style="yellow")
                status_text.append("idle", style="yellow")
            else:
                status_text.append("● ", style="green")
                status_text.append("running", style="green")
        elif task.status == "completed":
            status_text.append("● ", style="blue")
            status_text.append("completed", style="blue")
        elif task.status == "killed":
            status_text.append("● ", style="red")
            status_text.append("killed", style="red")
        elif task.status == "failed":
            status_text.append("● ", style="red")
            status_text.append("failed", style="red")
        else:
            status_text.append(f"● {task.status}", style="dim")

        # Token count
        if task.token_count > 0:
            status_text.append(f"  ·  {task.token_count:,} tokens", style="dim")

        # Elapsed time
        import time
        elapsed = time.time() - task.start_time
        if elapsed < 60:
            status_text.append(f"  ·  {int(elapsed)}s", style="dim")
        else:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            status_text.append(f"  ·  {mins}m {secs}s", style="dim")

        return status_text

    def render_messages(self, task: InProcessTeammateTaskState, limit: int = 20) -> Panel:
        """Render message history."""
        messages = task.messages[-limit:] if task.messages else []

        if not messages:
            return Panel(
                Text("No messages yet", style="dim italic"),
                title="Messages",
                border_style="dim",
            )

        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle content blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown")
                            text_parts.append(f"[tool: {tool_name}]")
                        elif block.get("type") == "tool_result":
                            text_parts.append("[tool result]")
                    else:
                        text_parts.append(str(block))
                content = " ".join(text_parts)
            elif not isinstance(content, str):
                content = str(content)

            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."

            # Style by role
            line = Text()
            if role == "user":
                line.append("User: ", style="bold blue")
                line.append(content)
            elif role == "assistant":
                line.append("Assistant: ", style="bold green")
                line.append(content)
            else:
                line.append(f"{role}: ", style="bold dim")
                line.append(content, style="dim")

            lines.append(line)

        return Panel(
            Group(*lines) if lines else Text("No messages", style="dim"),
            title=f"Messages ({len(task.messages)} total)",
            border_style="dim",
        )

    def render_full_view(self, task: InProcessTeammateTaskState) -> Group:
        """Render full teammate view."""
        return Group(
            self.render_teammate_header(task),
            Text(),  # Spacer
            self.render_status_line(task),
            Text(),  # Spacer
            self.render_messages(task),
        )

    def render_agent_list(
        self,
        app_state: AppState,
        teammates: list[InProcessTeammateTaskState],
    ) -> Panel:
        """Render list of agents for selection."""
        selected_idx = app_state.selected_agent_index
        viewing_id = app_state.viewing_agent_task_id

        lines = []

        # Leader row
        leader_style = "bold cyan" if selected_idx == -1 else None
        leader_marker = "●" if viewing_id is None else "○"
        leader_line = Text()
        leader_line.append(f"  {leader_marker} ", style=leader_style)
        leader_line.append("main", style=leader_style or "dim")
        lines.append(leader_line)

        # Teammate rows
        for i, task in enumerate(teammates):
            is_selected = selected_idx == i
            is_viewed = viewing_id == task.id

            color = task.color or "white"
            marker = "●" if is_viewed else "○"

            line = Text()
            if is_selected:
                line.append("▶ ", style="bold")

            # Status indicator
            if task.status == "running":
                if task.is_idle:
                    line.append("⏸ ", style="yellow")
                else:
                    line.append("▶ ", style="green")
            else:
                line.append("⏹ ", style="dim")

            line.append(f"{marker} ", style=color)
            line.append(task.identity.agent_name, style=f"bold {color}" if is_selected else color)

            # Token count
            if task.token_count > 0:
                line.append(f" · {task.token_count:,} tokens", style="dim")

            lines.append(line)

        # Hide row (if expanded)
        expanded = app_state.expanded_view
        if expanded == ExpandedView.TEAMMATES.value:
            hide_selected = selected_idx == len(teammates)
            hide_line = Text()
            if hide_selected:
                hide_line.append("▶ ", style="bold")
            hide_line.append("  ○ hide", style="bold" if hide_selected else "dim")
            lines.append(hide_line)

        return Panel(
            Group(*lines),
            title="Agents",
            border_style="dim",
        )


# =============================================================================
# Keyboard Navigation Handler
# =============================================================================


class KeyboardHandler:
    """Handle keyboard input for agent navigation."""

    def __init__(
        self,
        app_state: AppState,
        set_app_state: SetAppState,
        console: Optional[Console] = None,
    ):
        self.app_state = app_state
        self.set_app_state = set_app_state
        self.ui = TeammateViewUI(console)

    def handle_key(self, key: str, shift: bool = False) -> bool:
        """Handle a key press. Returns True if handled."""
        view_mode = getattr(self.app_state, "view_selection_mode", ViewSelectionMode.NONE.value)
        teammates = get_running_teammates_sorted(self.app_state)
        count = len(teammates)

        # Escape handling
        if key == "escape":
            if view_mode == ViewSelectionMode.VIEWING.value:
                # Exit view
                exit_teammate_view(self.set_app_state)
                return True
            elif view_mode == ViewSelectionMode.SELECTING.value:
                # Exit selection
                updates = {
                    "view_selection_mode": ViewSelectionMode.NONE.value,
                    "selected_agent_index": -1,
                }
                self.set_app_state(lambda prev: _apply_updates(prev, updates))
                return True

        # Shift+Up/Down: Navigate between agents
        if shift and key in ("up", "down"):
            step_teammate_selection(
                1 if key == "down" else -1,
                self.set_app_state,
                self.app_state,
            )
            return True

        # Enter: Confirm selection
        if key == "return" and view_mode == ViewSelectionMode.SELECTING.value:
            idx = getattr(self.app_state, "selected_agent_index", -1)
            if idx == -1:
                # Leader selected
                exit_teammate_view(self.set_app_state)
            elif idx >= count:
                # Hide row selected
                updates = {
                    "expanded_view": ExpandedView.NONE.value,
                    "view_selection_mode": ViewSelectionMode.NONE.value,
                    "selected_agent_index": -1,
                }
                self.set_app_state(lambda prev: _apply_updates(prev, updates))
            else:
                # Teammate selected
                task = teammates[idx]
                enter_teammate_view(task.id, self.set_app_state)
            return True

        # 'f': View transcript
        if key == "f" and view_mode == ViewSelectionMode.SELECTING.value:
            idx = getattr(self.app_state, "selected_agent_index", -1)
            if 0 <= idx < count:
                enter_teammate_view(teammates[idx].id, self.set_app_state)
                return True

        # 'k': Kill selected teammate
        if key == "k" and view_mode == ViewSelectionMode.SELECTING.value:
            idx = getattr(self.app_state, "selected_agent_index", -1)
            if 0 <= idx < count:
                task = teammates[idx]
                if task.status == "running":
                    task.abort_controller.abort("user-cancel")
                return True

        return False


# =============================================================================
# Demo / Testing
# =============================================================================


def demo():
    """Demo the teammate view UI."""
    console = Console()
    ui = TeammateViewUI(console)

    # Create a fake task for demo
    from claude_code_py.task.in_process_teammate import TeammateIdentity
    from claude_code_py.utils.abort_controller import AbortController

    identity = TeammateIdentity(
        agent_id="researcher@demo-team",
        agent_name="researcher",
        team_name="demo-team",
        parent_session_id="session-123",
        color="cyan",
    )

    task = InProcessTeammateTaskState(
        id="task-1",
        type="in_process_teammate",
        status="running",
        description="researcher: Analyzing codebase structure",
        identity=identity,
        prompt="Analyze the codebase structure and identify key components",
        abort_controller=AbortController(),
        messages=[
            {"role": "user", "content": "Analyze the codebase"},
            {"role": "assistant", "content": "I'll start by exploring the directory structure..."},
            {"role": "user", "content": "Focus on the core modules"},
            {"role": "assistant", "content": "Found 3 core modules: engine, state, tools"},
        ],
        token_count=1234,
        is_idle=False,
    )

    console.clear()
    console.print(ui.render_full_view(task))
    console.print()
    console.print("[dim]Press Shift+Down to navigate, Enter to select, Escape to exit[/dim]")


if __name__ == "__main__":
    demo()
