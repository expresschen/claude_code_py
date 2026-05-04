"""Agent Resume implementation.

Provides functionality to resume stopped agents with new prompts.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any, Callable, Optional, TYPE_CHECKING

from claude_code_py.task.types import TaskStatus, is_terminal_task_status
from claude_code_py.task.manager import (
    find_task_by_agent_id,
    get_task_by_id,
    update_teammate_task_status,
)

if TYPE_CHECKING:
    from claude_code_py.task.in_process_teammate import InProcessTeammateTaskState
    from claude_code_py.tool.context import ToolUseContext
    from claude_code_py.state.app_state import AppState


# =============================================================================
# Agent ID Validation
# =============================================================================


def is_agent_id_format(value: str) -> bool:
    """Check if a string matches agent ID format.

    Agent IDs look like:
    - "agent-abc123" (TypeScript format)
    - "researcher@my-team" (swarm teammate format)

    Args:
        value: String to check

    Returns:
        True if matches agent ID format
    """
    if not value:
        return False

    # Swarm teammate format: name@team
    if "@" in value:
        parts = value.split("@", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return True

    # TypeScript agent format: agent-<hex>
    if value.startswith("agent-"):
        remainder = value[6:]
        if remainder and all(c in "0123456789abcdefABCDEF" for c in remainder):
            return True

    return False


def to_agent_id(value: str) -> Optional[str]:
    """Convert a value to agent ID if valid format.

    Args:
        value: String to convert

    Returns:
        Agent ID string or None if invalid
    """
    if is_agent_id_format(value):
        return value
    return None


# =============================================================================
# Pending Message Queue
# =============================================================================


def queue_pending_message(
    agent_id: str,
    message: str,
    set_app_state: Callable,
) -> bool:
    """Queue a message for delivery to a running agent.

    The message will be delivered at the agent's next tool round.

    Args:
        agent_id: Agent ID to queue message for
        message: Message content to queue
        set_app_state: AppState setter function

    Returns:
        True if message was queued, False if agent not found
    """
    # Find task by agent_id
    task = find_task_by_agent_id(agent_id, lambda: {})

    if not task:
        return False

    # Create user message dict
    user_message = {
        "role": "user",
        "content": message,
    }

    # Add to pending_user_messages
    def updater(prev: AppState) -> AppState:
        tasks = prev.tasks
        task_data = tasks.get(task.id)
        if not task_data:
            return prev

        pending = list(task_data.get("pending_user_messages", []))
        pending.append(user_message)

        new_task = {**task_data, "pending_user_messages": pending}

        return replace(
            prev,
            tasks={**tasks, task.id: new_task},
        )

    set_app_state(updater)
    return True


# =============================================================================
# Resume Agent Background
# =============================================================================


async def resume_agent_background(
    agent_id: str,
    prompt: str,
    tool_use_context: "ToolUseContext",
    invoking_request_id: Optional[str] = None,
) -> dict:
    """Resume a stopped agent with a new prompt.

    Args:
        agent_id: Agent ID to resume
        prompt: New prompt message
        tool_use_context: Tool use context
        invoking_request_id: Optional request ID for correlation

    Returns:
        Dict with agent_id, output_file, and status
    """
    from claude_code_py.task.manager import get_task_output_path
    from claude_code_py.utils.teammate_context import get_teammate_context

    get_app_state = tool_use_context.get_app_state
    set_app_state = tool_use_context.set_app_state_for_tasks or tool_use_context.set_app_state

    # Find existing task
    task = find_task_by_agent_id(agent_id, get_app_state)

    if not task:
        # Try by task_id directly
        task = get_task_by_id(agent_id, get_app_state)

    if not task:
        raise ValueError(f"No task found for agent ID: {agent_id}")

    # Check if running - if so, queue message instead
    if task.status == TaskStatus.RUNNING:
        queue_pending_message(agent_id, prompt, set_app_state)
        return {
            "agent_id": agent_id,
            "output_file": task.output_file or get_task_output_path(task.id),
            "status": "running",
            "message": "Agent was running; message queued for delivery.",
        }

    # Check if terminal status
    if is_terminal_task_status(task.status):
        # Need to restart - update status and reset state
        update_teammate_task_status(
            set_app_state,
            task.id,
            TaskStatus.RUNNING,
        )

        # Add prompt as pending message
        user_message = {
            "role": "user",
            "content": prompt,
        }

        def add_message(prev: AppState) -> AppState:
            tasks = prev.tasks
            task_data = tasks.get(task.id)
            if not task_data:
                return prev

            pending = list(task_data.get("pending_user_messages", []))
            pending.append(user_message)

            return replace(
                prev,
                tasks={**tasks, task.id: {**task_data, "pending_user_messages": pending}},
            )

        set_app_state(add_message)

        # Start the agent runner
        output_file = get_task_output_path(task.id)

        # Create background task to run agent
        async def run_resumed_agent():
            """Run the resumed agent in background."""
            from claude_code_py.utils.swarm.in_process_runner import (
                InProcessRunnerConfig,
                start_in_process_teammate,
            )
            from claude_code_py.utils.teammate_context import create_teammate_context

            # Get teammate context from existing task
            identity = task.identity

            # Create teammate context
            teammate_context = create_teammate_context(
                agent_id=identity.agent_id,
                agent_name=identity.agent_name,
                team_name=identity.team_name,
                parent_session_id=identity.parent_session_id,
                abort_controller=task.abort_controller,
                color=identity.color,
            )

            # Build runner config
            runner_config = InProcessRunnerConfig(
                identity=identity,
                task_id=task.id,
                prompt=prompt,
                tool_use_context=tool_use_context,
                abort_controller=task.abort_controller,
                teammate_context=teammate_context,
            )

            # Start runner
            start_in_process_teammate(runner_config)

        # Create background task
        asyncio.create_task(run_resumed_agent())

        return {
            "agent_id": agent_id,
            "output_file": output_file,
            "status": "resumed",
            "message": f"Agent '{agent_id}' was stopped ({task.status}); resumed in background.",
        }

    raise ValueError(f"Agent '{agent_id}' has unexpected status: {task.status}")


# =============================================================================
# Find Agent by ID or Name
# =============================================================================


def find_agent_for_send(
    to: str,
    get_app_state: Callable,
) -> Optional["InProcessTeammateTaskState"]:
    """Find an agent for sending messages.

    Searches by:
    1. Agent name registry (if available)
    2. Agent ID format (direct match)
    3. Task ID

    Args:
        to: Recipient identifier (name, agent ID, or task ID)
        get_app_state: AppState getter

    Returns:
        InProcessTeammateTaskState or None
    """
    # Try by agent ID format first
    agent_id = to_agent_id(to)
    if agent_id:
        task = find_task_by_agent_id(agent_id, get_app_state)
        if task:
            return task

    # Try by task ID
    task = get_task_by_id(to, get_app_state)
    if task:
        return task

    # Try by agent name (name@team format or bare name)
    if "@" in to:
        task = find_task_by_agent_id(to, get_app_state)
        if task:
            return task

    # Try bare name - search all tasks
    state = get_app_state()
    tasks = state.tasks
    for t in tasks.values():
        if hasattr(t, "identity") and t.identity.agent_name == to:
            return t

    return None