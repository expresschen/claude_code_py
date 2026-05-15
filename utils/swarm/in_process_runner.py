"""In-process teammate runner.

Wraps run_agent() for in-process teammates, providing context isolation,
progress tracking, idle notifications, and cleanup.

Ported from: src/utils/swarm/inProcessRunner.ts
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Optional, List, Dict, TYPE_CHECKING
import logging

# Debug flag - controlled by environment variable CLAUDE_CODE_DEBUG_TEAMMATE
DEBUG_RUNNER = os.environ.get("CLAUDE_CODE_DEBUG_TEAMMATE", "").lower() in ("1", "true", "yes")

# Import unified debug logger
from claude_code_py.utils.debug_log import debug_log

def _debug_print(msg: str) -> None:
    """Print debug message to console and log file."""
    debug_log("[IN_PROCESS_RUNNER]", msg, DEBUG_RUNNER)

from claude_code_py.utils.abort_controller import (
    AbortController,
    create_abort_controller,
    check_abort,
)
from claude_code_py.utils.teammate_mailbox import (
    read_mailbox,
    read_unread_messages,
    write_to_mailbox,
    mark_messages_as_read,
    mark_message_as_read_by_index,
    create_idle_notification,
    create_shutdown_approved_message,
    is_shutdown_request,
    is_permission_response,
    is_plan_approval_response,
    is_structured_protocol_message,
    TeammateMessage,
    TEAMMATE_MESSAGE_TAG,
)
from claude_code_py.utils.team import TEAM_LEAD_NAME
from claude_code_py.storage.session import SessionStorage
from claude_code_py.utils.task.file_storage import (
    list_tasks,
    claim_task,
    update_task,
    ClaimTaskResult,
)
from claude_code_py.utils.teammate_context import (
    TeammateContext,
    run_with_teammate_context_async,
    get_current_agent_id,
)
from claude_code_py.task.types import TaskStatus
from claude_code_py.utils.swarm.spawn_in_process import (
    generate_task_id,
)
# Note: Uses get_background_loop() for non-blocking execution
# See start_in_process_teammate() for implementation

if TYPE_CHECKING:
    from claude_code_py.task.in_process_teammate import TeammateIdentity
    from claude_code_py.tool.context import ToolUseContext
    from claude_code_py.core_types.message import Message

logger = logging.getLogger(__name__)

POLL_INTERVAL_MS = 500

# Compaction threshold (in estimated tokens)
COMPACT_THRESHOLD_TOKENS = 100000


# =============================================================================
# Progress Tracking
# =============================================================================


@dataclass
class ProgressTracker:
    """Tracks progress for an in-process teammate."""

    last_update_time: float = 0.0
    total_tool_calls: int = 0
    total_messages: int = 0
    current_activity: Optional[str] = None

    def update(self, activity: Optional[str] = None) -> None:
        """Update progress state."""
        self.last_update_time = time.time()
        if activity:
            self.current_activity = activity


def create_progress_tracker() -> ProgressTracker:
    """Create a new progress tracker."""
    return ProgressTracker()


def update_progress_from_message(
    tracker: ProgressTracker,
    message: Any,
) -> None:
    """Update progress from a message."""
    tracker.total_messages += 1
    tracker.last_update_time = time.time()

    # Extract activity from message content
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str) and len(content) > 0:
            # Use first 50 chars as activity hint
            tracker.current_activity = content[:50]


def get_progress_update(tracker: ProgressTracker) -> Dict[str, Any]:
    """Get progress update for AppState."""
    return {
        "last_update_time": tracker.last_update_time,
        "total_tool_calls": tracker.total_tool_calls,
        "total_messages": tracker.total_messages,
        "current_activity": tracker.current_activity,
    }


# =============================================================================
# Configuration and Result Types
# =============================================================================


@dataclass
class InProcessRunnerConfig:
    """Configuration for running an in-process teammate."""

    # Required fields (no defaults) - must come first
    identity: "TeammateIdentity"
    task_id: str
    prompt: str
    tool_use_context: "ToolUseContext"
    abort_controller: AbortController

    # Optional fields (with defaults) - must come after required
    description: Optional[str] = None
    teammate_context: Optional[TeammateContext] = None
    model: Optional[str] = None
    agent_type: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    disallowed_tools: Optional[List[str]] = None
    allow_permission_prompts: bool = False
    system_prompt: Optional[str] = None
    system_prompt_mode: str = "default"  # 'default', 'replace', 'append'
    max_turns: Optional[int] = None


@dataclass
class InProcessRunnerResult:
    """Result from running an in-process teammate."""

    success: bool
    messages: List["Message"]
    error: Optional[str] = None


@dataclass
class WaitResult:
    """Result of waiting for messages."""

    type: str  # 'shutdown_request', 'new_message', 'aborted', 'task_claimed'
    request: Optional[Dict] = None
    message: Optional[str] = None
    from_agent: Optional[str] = None
    color: Optional[str] = None
    summary: Optional[str] = None
    original_message: Optional[str] = None
    task_id: Optional[str] = None  # For task_claimed type


# =============================================================================
# Task State Updates
# =============================================================================


def update_task_state(
    task_id: str,
    updater: Callable[[Any], Any],
    set_app_state: Callable,
) -> None:
    """Update task state in AppState.

    Handles both dict and dataclass task states.
    """
    def state_updater(prev: Any) -> Any:
        task = prev.tasks.get(task_id)
        if task is None:
            return prev

        updated_task = updater(task)

        # Use replace for dataclass, dict update for dict
        if hasattr(updated_task, '__dataclass_fields__'):
            # dataclass - use replace
            return replace(prev, tasks={**prev.tasks, task_id: updated_task})
        else:
            # dict
            return replace(prev, tasks={**prev.tasks, task_id: updated_task})

    set_app_state(state_updater)


def append_teammate_message(
    task_id: str,
    message: Any,
    set_app_state: Callable,
) -> None:
    """Append a message to the task's message list."""
    update_task_state(
        task_id,
        lambda task: replace(task, messages=(task.messages or []) + [message]) if hasattr(task, '__dataclass_fields__') else {**task, "messages": (task.get("messages") or []) + [message]},
        set_app_state,
    )


# =============================================================================
# Idle Notification
# =============================================================================


async def send_idle_notification(
    agent_name: str,
    agent_color: Optional[str],
    team_name: str,
    options: Optional[Dict[str, Any]] = None,
) -> None:
    """Send idle notification to leader mailbox."""
    opts = options or {}
    notification = create_idle_notification(
        agent_id=agent_name,
        options={
            "idle_reason": opts.get("idle_reason", "available"),
            "summary": opts.get("summary"),
            "completed_task_id": opts.get("completed_task_id"),
            "completed_status": opts.get("completed_status"),
            "failure_reason": opts.get("failure_reason"),
        },
    )

    _debug_print(f"→ send_idle_notification: agent='{agent_name}', team='{team_name}'")
    _debug_print(f"   idle_reason: '{opts.get('idle_reason', 'available')}'")
    _debug_print(f"   sending to: TEAM_LEAD_NAME='{TEAM_LEAD_NAME}'")

    await write_to_mailbox(
        TEAM_LEAD_NAME,
        TeammateMessage(
            from_agent=agent_name,
            text=json.dumps({
                "type": notification.type,
                "from": notification.from_agent,
                "timestamp": notification.timestamp,
                "idle_reason": notification.idle_reason,
                "summary": notification.summary,
                "completed_task_id": notification.completed_task_id,
                "completed_status": notification.completed_status,
                "failure_reason": notification.failure_reason,
            }),
            timestamp=datetime.now().isoformat(),
            color=agent_color,
        ),
        team_name,
    )
    _debug_print(f"✅ Idle notification sent to leader mailbox")


# =============================================================================
# Task Claiming
# =============================================================================


def find_available_task(tasks: List[Any], agent_name: Optional[str] = None) -> Optional[Any]:
    """Find an available task to claim.

    Args:
        tasks: List of tasks to search
        agent_name: Current agent name - if provided, allows claiming tasks
                    that are already owned by this agent (to continue interrupted work)

    Returns:
        Available task or None
    """
    unresolved_ids = set(t.id for t in tasks if t.status != "completed")

    for task in tasks:
        if task.status != "pending":
            continue
        # Skip if owned by different agent
        if task.owner and task.owner != agent_name:
            continue
        # Check if blocked by unresolved tasks
        blocked = any(bid in unresolved_ids for bid in task.blocked_by or [])
        if blocked:
            continue
        return task
    return None


def format_task_as_prompt(task: Any) -> str:
    """Format a task as a prompt for the teammate."""
    prompt = f"Complete all open tasks. Start with task #{task.id}: \n\n {task.subject}"
    if task.description:
        prompt += f"\n\n{task.description}"
    return prompt


@dataclass
class ClaimedTask:
    """Result from claiming a task."""

    task_id: str
    prompt: str
    subject: str


async def try_claim_next_task(task_list_id: str, agent_name: str) -> Optional[ClaimedTask]:
    """Try to claim an available task.

    Args:
        task_list_id: Task list ID
        agent_name: Current agent name - allows claiming tasks already owned by this agent

    Returns:
        ClaimedTask with task_id and prompt if task found and claimed, None otherwise
    """
    try:
        loop = asyncio.get_event_loop()
        tasks = await loop.run_in_executor(None, list_tasks, task_list_id)
        available_task = find_available_task(tasks, agent_name)

        if not available_task:
            return None

        # If already owned by this agent, just update status
        if available_task.owner == agent_name:
            await loop.run_in_executor(
                None, update_task, task_list_id, available_task.id, {"status": "in_progress"}
            )
            logger.debug(f"Resuming owned task #{available_task.id}: {available_task.subject}")
        else:
            # Claim the task
            result = await loop.run_in_executor(
                None, claim_task, task_list_id, available_task.id, agent_name, False
            )

            if not result.success:
                logger.debug(f"Failed to claim task #{available_task.id}: {result.reason}")
                return None

            # Set status to in_progress
            await loop.run_in_executor(
                None, update_task, task_list_id, available_task.id, {"status": "in_progress"}
            )
            logger.debug(f"Claimed task #{available_task.id}: {available_task.subject}")

        return ClaimedTask(
            task_id=available_task.id,
            prompt=format_task_as_prompt(available_task),
            subject=available_task.subject,
        )

    except Exception as e:
        logger.debug(f"try_claim_next_task error: {e}")
        return None


# =============================================================================
# Message Formatting
# =============================================================================


def format_teammate_xml(
    from_agent: str,
    content: str,
    color: Optional[str] = None,
    summary: Optional[str] = None,
) -> str:
    """Format message as teammate XML."""
    color_attr = f' color="{color}"' if color else ""
    summary_attr = f' summary="{summary}"' if summary else ""
    return f'<{TEAMMATE_MESSAGE_TAG} teammate_id="{from_agent}"{color_attr}{summary_attr}>\n{content}\n</{TEAMMATE_MESSAGE_TAG}>'


# =============================================================================
# Wait for Next Prompt
# =============================================================================


async def wait_for_next_prompt(
    identity: "TeammateIdentity",
    abort_controller: AbortController,
    task_id: str,
    get_app_state: Callable,
    set_app_state: Callable,
    task_list_id: str,
) -> WaitResult:
    """Poll mailbox for new messages or shutdown.

    Priority order:
    1. Pending user messages (in-memory queue)
    2. Shutdown requests (highest priority - prevents starvation)
    3. Team-lead messages (represents user intent and coordination)
    4. Other messages (FIFO)
    5. Unclaimed tasks

    Ported from: src/utils/swarm/inProcessRunner.ts waitForNextPromptOrShutdown
    """
    poll_count = 0

    _debug_print(f"→ wait_for_next_prompt: Starting poll loop")
    _debug_print(f"   agent_id: '{identity.agent_id}'")
    _debug_print(f"   task_list_id: '{task_list_id}'")
    _debug_print(f"   abort_controller.signal.aborted: {abort_controller.signal.aborted}")

    while not abort_controller.signal.aborted:
        # Check for in-memory pending messages (from transcript viewing)
        app_state = get_app_state()
        tasks = app_state.tasks
        task = tasks.get(task_id)

        # Get pending messages - handle both dataclass and dict
        if task is None:
            pending_messages = []
        elif hasattr(task, '__dataclass_fields__'):
            pending_messages = task.pending_user_messages or []
        else:
            pending_messages = task.get("pendingUserMessages", [])

        _debug_print(f"   Poll #{poll_count}: Checking pending messages...")
        if pending_messages and len(pending_messages) > 0:
            _debug_print(f"   ✅ Found {len(pending_messages)} pending user message(s)")
            message = pending_messages[0]
            # Pop the message from the queue
            if hasattr(task, '__dataclass_fields__'):
                # dataclass - use replace
                set_app_state(lambda prev: replace(
                    prev,
                    tasks={
                        **prev.tasks,
                        task_id: replace(task, pending_user_messages=pending_messages[1:]),
                    },
                ))
            else:
                # dict
                set_app_state(lambda prev: replace(
                    prev,
                    tasks={
                        **prev.tasks,
                        task_id: {
                            **prev.tasks.get(task_id, {}),
                            "pendingUserMessages": pending_messages[1:],
                        },
                    },
                ))
            _debug_print(f"   → Returning new_message from pending queue")
            return WaitResult(
                type="new_message",
                message=message,
                from_agent="user",
            )
        else:
            _debug_print(f"   ℹ️ No pending user messages")

        # Wait before next poll (skip on first iteration to check immediately)
        if poll_count > 0:
            await asyncio.sleep(POLL_INTERVAL_MS / 1000)

        poll_count += 1  # Increment AFTER sleep check

        # Check for abort
        if abort_controller.signal.aborted:
            _debug_print(f"   ⚠️ Abort detected at poll #{poll_count}")
            return WaitResult(type="aborted")

        # Check for messages in mailbox
        _debug_print(f"   Poll #{poll_count}: Checking mailbox...")
        try:
            all_messages = await read_mailbox(identity.agent_name, identity.team_name)
            _debug_print(f"   ℹ️ Mailbox has {len(all_messages)} total messages")

            unread_count = sum(1 for m in all_messages if m and not m.read)
            _debug_print(f"   ℹ️ Unread messages: {unread_count}")

            # Scan all unread messages for shutdown requests (highest priority).
            # Shutdown requests are prioritized over regular messages to prevent
            # starvation when peer-to-peer messages flood the queue.
            shutdown_index = -1
            shutdown_parsed = None
            for i, m in enumerate(all_messages):
                if m and not m.read:
                    try:
                        parsed = json.loads(m.text)
                        if parsed.get("type") == "shutdown_request":
                            shutdown_index = i
                            shutdown_parsed = parsed
                            break
                    except json.JSONDecodeError:
                        pass

            if shutdown_index != -1:
                msg = all_messages[shutdown_index]
                skipped_unread = sum(1 for m in all_messages[:shutdown_index] if m and not m.read)
                _debug_print(f"   ⚠️ SHUTDOWN REQUEST FOUND at index {shutdown_index}")
                _debug_print(f"      from_agent: '{msg.from_agent}'")
                _debug_print(f"      skipped {skipped_unread} other unread messages")
                await mark_message_as_read_by_index(
                    identity.agent_name,
                    shutdown_index,
                    identity.team_name,
                )
                return WaitResult(
                    type="shutdown_request",
                    request=shutdown_parsed,
                    from_agent=msg.from_agent,
                    original_message=msg.text,
                )

            # No shutdown request found. Prioritize team-lead messages over peer
            # messages — the leader represents user intent and coordination, so
            # their messages should not be starved behind peer-to-peer chatter.
            selected_index = -1

            # Check for unread team-lead messages first
            for i, m in enumerate(all_messages):
                if m and not m.read and m.from_agent == TEAM_LEAD_NAME:
                    selected_index = i
                    _debug_print(f"   ✅ Found TEAM_LEAD message at index {i}")
                    break

            # Fall back to first unread message (any sender), excluding protocol messages
            if selected_index == -1:
                for i, m in enumerate(all_messages):
                    if m and not m.read:
                        # Skip structured protocol messages (handled elsewhere)
                        try:
                            parsed = json.loads(m.text)
                            msg_type = parsed.get("type")
                            if msg_type in ("permission_response", "plan_approval_response"):
                                continue
                            if is_structured_protocol_message(m.text):
                                continue
                        except json.JSONDecodeError:
                            pass
                        selected_index = i
                        _debug_print(f"   ✅ Found regular message at index {i}")
                        _debug_print(f"      from_agent: '{m.from_agent}'")
                        break

            if selected_index != -1:
                msg = all_messages[selected_index]
                if msg:
                    _debug_print(f"   → Processing message from '{msg.from_agent}'")
                    await mark_message_as_read_by_index(
                        identity.agent_name,
                        selected_index,
                        identity.team_name,
                    )
                    return WaitResult(
                        type="new_message",
                        message=msg.text,
                        from_agent=msg.from_agent,
                        color=msg.color,
                        summary=msg.summary,
                    )

        except Exception as e:
            _debug_print(f"   ❌ Mailbox poll error: {type(e).__name__}: {e}")
            # Continue polling even if one read fails

        # Check the team's task list for unclaimed tasks
        _debug_print(f"   Poll #{poll_count}: Checking for unclaimed tasks...")
        claimed_task = await try_claim_next_task(task_list_id, identity.agent_name)
        if claimed_task:
            _debug_print(f"   ✅ Claimed a task! task_id=#{claimed_task.task_id}")
            return WaitResult(
                type="task_claimed",
                message=claimed_task.prompt,
                task_id=claimed_task.task_id,
            )
        else:
            _debug_print(f"   ℹ️ No available tasks to claim")

    _debug_print(f"⚠️ Exiting poll loop (abort={abort_controller.signal.aborted})")
    return WaitResult(type="aborted")


# =============================================================================
# Compaction
# =============================================================================


def estimate_token_count(messages: List[Any]) -> int:
    """Estimate token count for messages."""
    # Simple estimation: ~4 chars per token
    total_chars = 0
    for msg in messages:
        if hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total_chars += len(block.get("text", ""))
    return total_chars // 4


async def check_and_compact(
    messages: List[Any],
    abort_controller: AbortController,
    system_prompt: Optional[str] = None,
) -> List[Any]:
    """Check if messages need compaction and compact if needed."""
    token_count = estimate_token_count(messages)

    if token_count >= COMPACT_THRESHOLD_TOKENS:
        logger.debug(f"Compacting messages: {token_count} tokens")

        max_messages = 50  # Keep last 50 messages
        if len(messages) > max_messages:
            return messages[-max_messages:]

    return messages


# =============================================================================
# Main Runner
# =============================================================================


async def run_in_process_teammate(config: InProcessRunnerConfig) -> InProcessRunnerResult:
    """Run in-process teammate with continuous prompt loop."""
    identity = config.identity
    abort_controller = config.abort_controller
    tool_use_context = config.tool_use_context
    set_app_state = tool_use_context.set_app_state
    task_list_id = identity.parent_session_id
    all_messages: List["Message"] = []

    # Create session storage for sidechain writes (agent transcript persistence).
    # Uses the leader's session ID so agent transcripts live under the same session.
    session_storage = SessionStorage(identity.parent_session_id, config.tool_use_context.get_cwd())

    # Track the last recorded message UUID for parent chain continuity.
    # Matches TypeScript runAgent.ts: lastRecordedUuid pattern.
    last_recorded_uuid: Optional[str] = None

    _debug_print("=" * 70)
    _debug_print("run_in_process_teammate: STARTING")
    _debug_print(f"  agent_id: '{identity.agent_id}'")
    _debug_print(f"  agent_name: '{identity.agent_name}'")
    _debug_print(f"  team_name: '{identity.team_name}'")
    _debug_print(f"  task_id: '{config.task_id}'")
    _debug_print(f"  parent_session_id: '{identity.parent_session_id}'")
    _debug_print(f"  task_list_id: '{task_list_id}'")
    _debug_print(f"  abort_controller.signal.aborted: {abort_controller.signal.aborted}")
    _debug_print(f"  model: '{config.model}'")
    _debug_print(f"  max_turns: {config.max_turns}")
    _debug_print("=" * 70)

    # Create progress tracker
    progress_tracker = create_progress_tracker()
    _debug_print("✅ Progress tracker created")

    # Create teammate context for context isolation
    teammate_context = config.teammate_context
    if not teammate_context:
        _debug_print("⚠️ No teammate_context provided, creating one...")
        teammate_context = TeammateContext(
            agent_id=identity.agent_id,
            agent_name=identity.agent_name,
            team_name=identity.team_name,
            parent_session_id=identity.parent_session_id,
            abort_controller=abort_controller,
            color=identity.color,
            plan_mode_required=False,
            is_in_process=True,
        )
    _debug_print(f"✅ Teammate context ready: is_in_process={teammate_context.is_in_process}")

    # Wrap initial prompt with XML for proper styling
    current_prompt = format_teammate_xml(TEAM_LEAD_NAME, config.prompt)
    _debug_print(f"✅ Initial prompt wrapped (len={len(current_prompt)})")
    should_exit = False
    iteration_count = 0
    current_task_id: Optional[str] = None  # Track current task for completion reporting
    last_error: Optional[str] = None  # Track last error for idle notification summary

    # Try to claim an available task immediately so the UI can show activity
    # from the very start. The idle loop handles claiming for subsequent tasks.
    # Use parent_session_id as the task list ID since the leader creates tasks
    # under its session ID (which equals the team name).
    _debug_print("→ Attempting to claim initial task...")
    initial_claimed_task = await try_claim_next_task(
        task_list_id,
        identity.agent_name,
    )
    if initial_claimed_task:
        _debug_print(f"✅ Claimed initial task!")
        _debug_print(f"   Task ID: #{initial_claimed_task.task_id}")
        _debug_print(f"   Task prompt preview: '{initial_claimed_task.prompt[:100]}{'...' if len(initial_claimed_task.prompt) > 100 else ''}'")
        current_prompt = initial_claimed_task.prompt
        current_task_id = initial_claimed_task.task_id
    else:
        _debug_print("ℹ️ No initial task available, using spawn prompt")

    # Run within teammate context for isolation
    async def run_loop() -> InProcessRunnerResult:
        nonlocal should_exit, iteration_count, current_prompt, all_messages, current_task_id
        _debug_print("→ Entering main run loop...")
        while not abort_controller.signal.aborted and not should_exit:
            iteration_count += 1
            _debug_print("")
            _debug_print("-" * 50)
            _debug_print(f"ITERATION #{iteration_count}")
            _debug_print(f"  abort_controller.signal.aborted: {abort_controller.signal.aborted}")
            _debug_print(f"  should_exit: {should_exit}")
            _debug_print("-" * 50)

            # Create work abort controller (for current iteration only)
            # This allows Escape to stop current work without killing the whole teammate
            work_abort = create_abort_controller()

            # Store work controller in task state so UI can abort it
            _debug_print("→ Storing current_work_abort_controller in task state...")
            update_task_state(
                config.task_id,
                lambda task: {
                    **task,
                    "current_work_abort_controller": work_abort,
                } if isinstance(task, dict) else replace(
                    task,
                    current_work_abort_controller=work_abort,
                ),
                set_app_state,
            )

            # Mark as not idle
            _debug_print("→ Marking task as NOT idle...")
            update_task_state(
                config.task_id,
                lambda task: {
                    **task,
                    "is_idle": False,
                    "spinner_verb": "thinking",
                    "color": identity.color or "green",
                } if isinstance(task, dict) else replace(
                    task,
                    is_idle=False,
                    spinner_verb="thinking",
                    color=identity.color or "green",
                ),
                set_app_state,
            )

            iteration_messages = []

            try:
                await check_abort(work_abort.signal)
                _debug_print("✅ Abort check passed")

                # Execute agent
                _debug_print("→ Executing agent...")
                _debug_print(f"   prompt preview: '{current_prompt[:100]}{'...' if len(current_prompt) > 100 else ''}'")

                from claude_code_py.tools.agent_tool.run_agent import run_agent, AgentRunConfig

                agent_config = AgentRunConfig(
                    agent_id=identity.agent_id,
                    agent_type=config.agent_type or "general-purpose",
                    prompt=current_prompt,
                    description=config.description or f"{identity.agent_name} task",
                    model=config.model,
                    tools=config.allowed_tools or ["*"],
                    disallowed_tools=config.disallowed_tools or [],
                    run_in_background=False,
                    cwd=config.tool_use_context.get_cwd(),  # Use cwd from tool context
                    max_turns=config.max_turns,
                    system_prompt=config.system_prompt,
                    abort_controller=work_abort,  # Pass work abort controller for Escape key
                )

                _debug_print(f"   AgentRunConfig: agent_id='{agent_config.agent_id}', agent_type='{agent_config.agent_type}'")

                try:
                    # Use streaming to process each message in real-time.
                    # Matches TypeScript runAgent.ts: for-await-of + yield,
                    # where each message is written to disk (sidechain) before
                    # being yielded to the caller.
                    from claude_code_py.tools.agent_tool.run_agent import run_agent_stream

                    _debug_print("   → Starting streaming agent execution...")
                    iteration_messages = []
                    async for msg in run_agent_stream(agent_config):
                        iteration_messages.append(msg)
                        all_messages.append(msg)

                        # Update progress (tool counting, spinner verb)
                        if hasattr(msg, "message"):
                            content = msg.message.get("content", [])
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "tool_use":
                                        progress_tracker.total_tool_calls += 1
                                        tool_name = block.get("name", "tool")
                                        update_task_state(
                                            config.task_id,
                                            lambda task: {
                                                **task,
                                                "spinner_verb": f"using {tool_name}",
                                            } if isinstance(task, dict) else replace(
                                                task,
                                                spinner_verb=f"using {tool_name}",
                                            ),
                                            set_app_state,
                                        )
                        update_progress_from_message(progress_tracker, msg)

                        # Write to sidechain JSONL — disk-write-before-yield
                        msg_uuid = getattr(msg, "uuid", None)
                        msg_type = getattr(msg, "type", None)
                        try:
                            session_storage.insert_message_chain(
                                [msg],
                                is_sidechain=True,
                                agent_id=identity.agent_id,
                                starting_parent_uuid=last_recorded_uuid,
                            )
                        except Exception:
                            pass

                        # Append to task.messages so transcript view sees it immediately
                        append_teammate_message(config.task_id, msg, set_app_state)

                        # Track last recorded UUID for parent chain (skip progress)
                        if msg_uuid and msg_type != "progress":
                            last_recorded_uuid = msg_uuid

                    _debug_print(f"   ← Streaming loop completed: {len(iteration_messages)} messages")
                    _debug_print(f"   Total tool calls: {progress_tracker.total_tool_calls}")

                except Exception as e:
                    _debug_print(f"   ❌ Agent execution error: {type(e).__name__}: {e}")
                    logger.debug(f"Agent execution error: {e}")
                    last_error = f"{type(e).__name__}: {e}"

                # Check lifecycle abort (kills whole teammate)
                if abort_controller.signal.aborted:
                    _debug_print("⚠️ Lifecycle abort detected - exiting loop")
                    break

                # Check work abort (stops current turn only, teammate continues)
                if work_abort.signal.aborted:
                    _debug_print("⚠️ Work abort detected (Escape pressed) - skipping to next turn")
                    # Clear the work abort controller for next iteration
                    update_task_state(
                        config.task_id,
                        lambda task: {
                            **task,
                            "current_work_abort_controller": None,
                        } if isinstance(task, dict) else replace(
                            task,
                            current_work_abort_controller=None,
                        ),
                        set_app_state,
                    )
                    continue  # Skip to next iteration

                # Update progress (internal tracking only - no state update needed)
                progress_tracker.update(activity=current_prompt[:50])
                _debug_print("✅ Progress tracked internally")

            except Exception as e:
                if abort_controller.signal.aborted:
                    _debug_print("⚠️ Abort detected during iteration")
                    break
                _debug_print(f"❌ Iteration error: {type(e).__name__}: {e}")
                logger.debug(f"run_in_process_teammate iteration error: {e}")

            # Check if already idle before updating (to skip duplicate notification)
            prev_app_state = tool_use_context.get_app_state()
            prev_task = prev_app_state.tasks.get(config.task_id)
            was_already_idle = False
            if prev_task:
                if hasattr(prev_task, 'is_idle'):
                    was_already_idle = prev_task.is_idle
                elif isinstance(prev_task, dict):
                    was_already_idle = prev_task.get("isIdle", False)

            # Mark as idle and clear work abort controller
            _debug_print("→ Marking task as IDLE...")
            update_task_state(
                config.task_id,
                lambda task: {
                    **task,
                    "is_idle": True,
                    "spinner_verb": "idle",
                    "token_count": estimate_token_count(all_messages),
                    "current_work_abort_controller": None,  # Clear work controller
                } if isinstance(task, dict) else replace(
                    task,
                    is_idle=True,
                    spinner_verb="idle",
                    token_count=estimate_token_count(all_messages),
                    current_work_abort_controller=None,  # Clear work controller
                ),
                set_app_state,
            )
            _debug_print("✅ Task marked as idle")

            # Only send idle notification on transition to idle (not if already idle)
            # This matches TypeScript's behavior to prevent duplicate notifications
            if was_already_idle:
                _debug_print(f"ℹ️ Skipping duplicate idle notification (was already idle)")
            else:
                _debug_print("→ Sending idle notification to leader...")

                # Generate summary from last meaningful message
                summary = None
                if current_task_id:
                    summary = f"Completed task #{current_task_id}"
                if not summary and all_messages:
                    # Scan messages in reverse for usable text content.
                    # Covers assistant, system (e.g. max-turns), and user messages.
                    for msg in reversed(all_messages):
                        content = None
                        if hasattr(msg, "type") and hasattr(msg, "message"):
                            content = msg.message.get("content", "")
                        elif hasattr(msg, "type") and hasattr(msg, "content"):
                            # SystemMessage: content is a direct attribute
                            content = msg.content or ""
                        elif isinstance(msg, dict):
                            inner = msg.get("message", {})
                            if isinstance(inner, dict) and "content" in inner:
                                content = inner["content"]
                            else:
                                content = msg.get("content", "")
                        if not content:
                            continue
                        if isinstance(content, str) and content.strip():
                            if content.startswith("<teammate_message") or content.startswith("<"):
                                continue
                            summary = content[:80] + "..." if len(content) > 80 else content
                            break
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text and not text.startswith("<"):
                                        summary = text[:80] + "..." if len(text) > 80 else text
                                        break
                            if summary:
                                break
                if not summary:
                    summary = "No output produced — send shutdown to this teammate"

                await send_idle_notification(
                    identity.agent_name,
                    identity.color,
                    identity.team_name,
                    options={
                        "idle_reason": "available",
                        "summary": summary,
                        "completed_task_id": current_task_id,
                        "completed_status": "resolved" if current_task_id else None,
                    },
                )
                _debug_print(f"✅ Idle notification sent (task_id={current_task_id})")

            # Check for compaction
            _debug_print("→ Checking for compaction...")
            all_messages = await check_and_compact(
                all_messages,
                abort_controller,
                config.system_prompt,
            )
            _debug_print(f"✅ Compaction check done, messages count: {len(all_messages)}")

            # Wait for next prompt
            _debug_print("→ Waiting for next prompt...")
            result = await wait_for_next_prompt(
                identity,
                abort_controller,
                config.task_id,
                tool_use_context.get_app_state,
                set_app_state,
                task_list_id,
            )
            _debug_print(f"← wait_for_next_prompt returned: type='{result.type}'")

            if result.type == "shutdown_request":
                _debug_print("⚠️ Shutdown request received!")
                _debug_print(f"   from_agent: '{result.from_agent}'")
                _debug_print(f"   request_id: '{result.request.get('requestId')}'")

                # Send shutdown_response (approve=True) to leader
                request_id = result.request.get("requestId", "") if result.request else ""

                # Write shutdown_approved message to leader's mailbox
                from claude_code_py.utils.teammate_mailbox import (
                    create_shutdown_approved_message,
                )

                shutdown_approved = create_shutdown_approved_message(
                    request_id=request_id,
                    from_agent=identity.agent_name,
                    pane_id=None,
                    backend_type="in-process",
                )

                await write_to_mailbox(
                    TEAM_LEAD_NAME,
                    TeammateMessage(
                        from_agent=identity.agent_name,
                        text=json.dumps({
                            "type": shutdown_approved.type,
                            "requestId": shutdown_approved.request_id,
                            "from": shutdown_approved.from_agent,
                            "timestamp": shutdown_approved.timestamp,
                            "paneId": shutdown_approved.pane_id,
                            "backendType": shutdown_approved.backend_type,
                        }),
                        timestamp=datetime.now().isoformat(),
                        color=identity.color,
                    ),
                    identity.team_name,
                )
                _debug_print("✅ Shutdown approved sent to leader")

                # Abort and exit
                abort_controller.abort()
                should_exit = True
                _debug_print("→ Setting should_exit=True, aborting...")
                continue  # Skip rest of loop and exit

            elif result.type == "new_message":
                _debug_print(f"✅ New message received")
                _debug_print(f"   from_agent: '{result.from_agent}'")
                _debug_print(f"   message preview: '{str(result.message)[:100]}{'...' if len(str(result.message)) > 100 else ''}'")
                if result.from_agent == "user":
                    current_prompt = result.message or ""
                else:
                    current_prompt = format_teammate_xml(
                        result.from_agent,
                        result.message or "",
                        result.color,
                        result.summary,
                    )
                # Clear task tracking when processing non-task messages
                current_task_id = None

            elif result.type == "task_claimed":
                _debug_print(f"✅ Task claimed!")
                _debug_print(f"   task_id: #{result.task_id}")
                _debug_print(f"   message preview: '{str(result.message)[:100]}{'...' if len(str(result.message)) > 100 else ''}'")
                current_prompt = result.message or ""
                current_task_id = result.task_id  # Track the new task

            elif result.type == "aborted":
                _debug_print("⚠️ Wait aborted, setting should_exit=True")
                should_exit = True

        # Final state update
        _debug_print("")
        _debug_print("=" * 70)
        _debug_print("run_in_process_teammate: EXITING")
        _debug_print(f"  Total iterations: {iteration_count}")
        _debug_print(f"  Total messages: {len(all_messages)}")
        _debug_print(f"  Total tool calls: {progress_tracker.total_tool_calls}")
        _debug_print(f"  Final status: COMPLETED")
        _debug_print("=" * 70)

        update_task_state(
            config.task_id,
            lambda task: replace(task, status=TaskStatus.COMPLETED, end_time=time.time()) if hasattr(task, '__dataclass_fields__') else {**task, "status": TaskStatus.COMPLETED.value, "end_time": time.time()},
            set_app_state,
        )

        return InProcessRunnerResult(success=True, messages=all_messages)

    try:
        # Run with teammate context
        _debug_print("→ Running with teammate context...")
        result = await run_with_teammate_context_async(teammate_context, run_loop)
        _debug_print(f"← run_with_teammate_context_async returned: success={result.success}")
        return result

    except Exception as e:
        _debug_print("=" * 70)
        _debug_print("❌ run_in_process_teammate: EXCEPTION")
        _debug_print(f"  {type(e).__name__}: {e}")
        _debug_print("=" * 70)

        update_task_state(
            config.task_id,
            lambda task: replace(task, status=TaskStatus.FAILED, error=str(e), end_time=time.time()) if hasattr(task, '__dataclass_fields__') else {**task, "status": TaskStatus.FAILED.value, "error": str(e), "end_time": time.time()},
            set_app_state,
        )
        return InProcessRunnerResult(success=False, messages=all_messages, error=str(e))


def start_in_process_teammate(config: InProcessRunnerConfig) -> "threading.Thread":
    """Start teammate in a dedicated thread with its own event loop.

    Each teammate runs in an isolated thread with its own asyncio event loop.
    This provides better isolation than a shared background loop - one teammate
    blocking doesn't affect others.

    Returns:
        threading.Thread - caller can check thread.is_alive() to track execution.
    """
    import threading

    agent_id = config.identity.agent_id

    _debug_print("=" * 70)
    _debug_print("start_in_process_teammate: STARTING")
    _debug_print(f"  agent_id: '{agent_id}'")
    _debug_print(f"  task_id: '{config.task_id}'")
    _debug_print("=" * 70)

    def run_in_thread():
        """Run teammate in thread with its own event loop."""
        _debug_print(f"→ run_in_thread() started for '{agent_id}'")
        try:
            # Create new event loop for this thread
            asyncio.run(run_in_process_teammate(config))
            _debug_print(f"✅ run_in_process_teammate() completed for '{agent_id}'")
        except Exception as e:
            _debug_print(f"❌ Error in thread for '{agent_id}': {type(e).__name__}: {e}")
            logger.debug(f"in_process_runner thread error {agent_id}: {e}")

    thread = threading.Thread(
        target=run_in_thread,
        name=f"teammate-{agent_id}",
        daemon=True,  # Don't block process exit
    )
    thread.start()

    _debug_print(f"✅ Thread started: {thread.name}")
    _debug_print(f"   thread.is_alive()={thread.is_alive()}")
    logger.debug(f"Started teammate {agent_id} in thread {thread.name}")

    return thread


__all__ = [
    "ProgressTracker",
    "create_progress_tracker",
    "update_progress_from_message",
    "get_progress_update",
    "InProcessRunnerConfig",
    "InProcessRunnerResult",
    "WaitResult",
    "ClaimedTask",
    "update_task_state",
    "append_teammate_message",
    "run_in_process_teammate",
    "start_in_process_teammate",
    "send_idle_notification",
    "try_claim_next_task",
    "wait_for_next_prompt",
    "format_teammate_xml",
    "check_and_compact",
    "estimate_token_count",
    "find_available_task",
    "format_task_as_prompt",
]