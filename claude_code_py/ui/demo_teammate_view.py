"""Interactive demo for teammate view UI with keyboard navigation."""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.text import Text

from claude_code_py.ui.teammate_view import (
    TeammateViewUI,
    ViewSelectionMode,
    get_running_teammates_sorted,
    get_viewed_teammate_task,
    enter_teammate_view,
    exit_teammate_view,
    step_teammate_selection,
)
from claude_code_py.task.in_process_teammate import (
    InProcessTeammateTaskState,
    TeammateIdentity,
)
from claude_code_py.utils.abort_controller import AbortController


def create_demo_tasks() -> list[InProcessTeammateTaskState]:
    """Create demo tasks for testing."""
    tasks = []

    # Researcher agent
    identity1 = TeammateIdentity(
        agent_id="researcher@demo-team",
        agent_name="researcher",
        team_name="demo-team",
        parent_session_id="session-123",
        color="cyan",
    )
    task1 = InProcessTeammateTaskState(
        id="task-1",
        type="in_process_teammate",
        status="running",
        description="researcher: Analyzing codebase",
        identity=identity1,
        prompt="Analyze the codebase structure and identify key components",
        abort_controller=AbortController(),
        messages=[
            {"role": "user", "content": "Analyze the codebase"},
            {"role": "assistant", "content": "I'll explore the directory structure..."},
            {"role": "assistant", "content": "Found main modules: engine, state, tools"},
        ],
        token_count=1500,
        is_idle=False,
        start_time=time.time() - 45,  # 45 seconds ago
    )
    tasks.append(task1)

    # Tester agent
    identity2 = TeammateIdentity(
        agent_id="tester@demo-team",
        agent_name="tester",
        team_name="demo-team",
        parent_session_id="session-123",
        color="green",
    )
    task2 = InProcessTeammateTaskState(
        id="task-2",
        type="in_process_teammate",
        status="running",
        description="tester: Running tests",
        identity=identity2,
        prompt="Run all tests and report failures",
        abort_controller=AbortController(),
        messages=[
            {"role": "user", "content": "Run the tests"},
            {"role": "assistant", "content": "Running pytest..."},
            {"role": "assistant", "content": "12 tests passed, 2 failed"},
        ],
        token_count=800,
        is_idle=True,  # Idle, waiting for next task
        start_time=time.time() - 120,  # 2 minutes ago
    )
    tasks.append(task2)

    # Builder agent
    identity3 = TeammateIdentity(
        agent_id="builder@demo-team",
        agent_name="builder",
        team_name="demo-team",
        parent_session_id="session-123",
        color="yellow",
    )
    task3 = InProcessTeammateTaskState(
        id="task-3",
        type="in_process_teammate",
        status="running",
        description="builder: Building feature",
        identity=identity3,
        prompt="Implement the new authentication feature",
        abort_controller=AbortController(),
        messages=[
            {"role": "user", "content": "Implement auth feature"},
            {"role": "assistant", "content": "Creating auth module..."},
        ],
        token_count=2000,
        is_idle=False,
        start_time=time.time() - 30,
    )
    tasks.append(task3)

    return tasks


class DemoAppState:
    """Simple demo state container."""

    def __init__(self, tasks: list[InProcessTeammateTaskState]):
        self.tasks = {t.id: t for t in tasks}
        self.viewing_agent_task_id: Optional[str] = None
        self.view_selection_mode: str = ViewSelectionMode.NONE.value
        self.selected_agent_index: int = -1
        self.expanded_view: str = "none"


def demo_interactive():
    """Interactive demo with keyboard input simulation."""
    console = Console()
    ui = TeammateViewUI(console)

    tasks = create_demo_tasks()
    state = DemoAppState(tasks)

    def set_app_state(updater):
        """Update state."""
        new_state = updater(state)
        # Copy attributes
        for attr in ["viewing_agent_task_id", "view_selection_mode", "selected_agent_index", "expanded_view", "tasks"]:
            if hasattr(new_state, attr):
                setattr(state, attr, getattr(new_state, attr))

    def render():
        """Render current state."""
        viewed_task = get_viewed_teammate_task(state)
        teammates = get_running_teammates_sorted(state)

        if viewed_task:
            # Viewing mode - show full transcript
            return ui.render_full_view(viewed_task)
        else:
            # Show agent list
            return ui.render_agent_list(state, teammates)

    console.clear()
    console.print("[bold]Teammate View Demo[/bold]")
    console.print("[dim]Commands: n=next, p=prev, v=view, e=exit, q=quit[/dim]")
    console.print()

    # Initial render
    console.print(render())

    # Simple command loop
    while True:
        try:
            cmd = console.input("[bold cyan]Command:[/bold cyan] ").strip().lower()

            if cmd == "q":
                console.print("[dim]Goodbye![/dim]")
                break

            elif cmd == "n":
                # Next agent (simulate Shift+Down)
                step_teammate_selection(1, set_app_state, state)

            elif cmd == "p":
                # Previous agent (simulate Shift+Up)
                step_teammate_selection(-1, set_app_state, state)

            elif cmd == "v":
                # View selected agent (simulate Enter)
                if state.view_selection_mode == ViewSelectionMode.SELECTING.value:
                    idx = state.selected_agent_index
                    teammates = get_running_teammates_sorted(state)
                    if 0 <= idx < len(teammates):
                        enter_teammate_view(teammates[idx].id, set_app_state)
                    elif idx == -1:
                        console.print("[dim]Leader selected - nothing to view[/dim]")

            elif cmd == "e":
                # Exit view (simulate Escape)
                if state.view_selection_mode == ViewSelectionMode.VIEWING.value:
                    exit_teammate_view(set_app_state)
                elif state.view_selection_mode == ViewSelectionMode.SELECTING.value:
                    state.view_selection_mode = ViewSelectionMode.NONE.value
                    state.selected_agent_index = -1

            elif cmd == "l":
                # List all agents
                state.expanded_view = "teammates"
                state.view_selection_mode = ViewSelectionMode.SELECTING.value
                state.selected_agent_index = -1

            elif cmd == "s":
                # Show status
                console.print(f"[dim]State: mode={state.view_selection_mode}, idx={state.selected_agent_index}, viewing={state.viewing_agent_task_id}[/dim]")

            # Re-render
            console.clear()
            console.print("[bold]Teammate View Demo[/bold]")
            console.print("[dim]Commands: n=next, p=prev, v=view, e=exit, l=list, s=status, q=quit[/dim]")
            console.print()
            console.print(render())

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def demo_static():
    """Static demo showing all UI components."""
    console = Console()
    ui = TeammateViewUI(console)

    tasks = create_demo_tasks()

    console.print("\n[bold]=== Teammate View Demo ===[/bold]\n")

    # Show each task's full view
    for task in tasks:
        console.print(ui.render_full_view(task))
        console.print()

    # Show agent list
    state = DemoAppState(tasks)
    state.expanded_view = "teammates"
    state.view_selection_mode = ViewSelectionMode.SELECTING.value
    state.selected_agent_index = 0  # Select first agent

    console.print("\n[bold]=== Agent Selection List ===[/bold]\n")
    console.print(ui.render_agent_list(state, tasks))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        demo_interactive()
    else:
        demo_static()
        print("\nRun with 'interactive' argument for interactive demo:")
        print("  python -m claude_code_py.ui.demo_teammate_view interactive")