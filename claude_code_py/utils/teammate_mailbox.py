"""Teammate Mailbox - File-based messaging system for agent swarms.

Each teammate has an inbox file at .claude/teams/{team_name}/inboxes/{agent_name}.json.
Other teammates can write messages to it, and the recipient sees them as attachments.

Note: Inboxes are keyed by agent name within a team.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable

from claude_code_py.utils.teammate_context import (
    get_current_agent_name,
    get_current_team_name,
    is_team_lead,
    TEAM_LEAD_NAME,
)
from claude_code_py.memory.paths import get_memory_base

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


@dataclass
class TeammateMessage:
    """A message in a teammate's inbox."""

    from_agent: str  # Sender agent name
    text: str  # Message content
    timestamp: str  # ISO timestamp
    read: bool = False  # Whether message has been read
    color: Optional[str] = None  # Sender's assigned color
    summary: Optional[str] = None  # 5-10 word summary for preview


@dataclass
class IdleNotificationMessage:
    """Message sent when a teammate becomes idle."""

    type: str = "idle_notification"
    from_agent: str = ""
    timestamp: str = ""
    idle_reason: Optional[str] = None  # 'available', 'interrupted', 'failed'
    summary: Optional[str] = None
    completed_task_id: Optional[str] = None
    completed_status: Optional[str] = None  # 'resolved', 'blocked', 'failed'
    failure_reason: Optional[str] = None


@dataclass
class PermissionRequestMessage:
    """Permission request message from worker to leader."""

    type: str = "permission_request"
    request_id: str = ""
    agent_id: str = ""
    tool_name: str = ""
    tool_use_id: str = ""
    description: str = ""
    input: dict = field(default_factory=dict)
    permission_suggestions: list = field(default_factory=list)


@dataclass
class PermissionResponseMessage:
    """Permission response message from leader to worker."""

    type: str = "permission_response"
    request_id: str = ""
    subtype: str = "success"  # 'success' or 'error'
    error: Optional[str] = None
    response: Optional[dict] = None


@dataclass
class ShutdownRequestMessage:
    """Shutdown request message from leader to teammate."""

    type: str = "shutdown_request"
    request_id: str = ""
    from_agent: str = ""
    reason: Optional[str] = None
    timestamp: str = ""


@dataclass
class ShutdownApprovedMessage:
    """Shutdown approved message from teammate to leader."""

    type: str = "shutdown_approved"
    request_id: str = ""
    from_agent: str = ""
    timestamp: str = ""
    pane_id: Optional[str] = None
    backend_type: Optional[str] = None


@dataclass
class ShutdownRejectedMessage:
    """Shutdown rejected message from teammate to leader."""

    type: str = "shutdown_rejected"
    request_id: str = ""
    from_agent: str = ""
    reason: str = ""
    timestamp: str = ""


@dataclass
class PlanApprovalRequestMessage:
    """Plan approval request from teammate to leader."""

    type: str = "plan_approval_request"
    from_agent: str = ""
    timestamp: str = ""
    plan_file_path: str = ""
    plan_content: str = ""
    request_id: str = ""


@dataclass
class PlanApprovalResponseMessage:
    """Plan approval response from leader to teammate."""

    type: str = "plan_approval_response"
    request_id: str = ""
    approved: bool = False
    feedback: Optional[str] = None
    timestamp: str = ""
    permission_mode: Optional[str] = None


@dataclass
class SandboxPermissionRequestMessage:
    """Sandbox permission request from worker to leader."""

    type: str = "sandbox_permission_request"
    request_id: str = ""
    worker_id: str = ""
    worker_name: str = ""
    worker_color: Optional[str] = None
    host_pattern: dict = field(default_factory=dict)
    created_at: int = 0


@dataclass
class SandboxPermissionResponseMessage:
    """Sandbox permission response from leader to worker."""

    type: str = "sandbox_permission_response"
    request_id: str = ""
    host: str = ""
    allow: bool = False
    timestamp: str = ""


@dataclass
class TaskAssignmentMessage:
    """Task assignment message from leader to teammate."""

    type: str = "task_assignment"
    task_id: str = ""
    subject: str = ""
    description: str = ""
    assigned_by: str = ""
    timestamp: str = ""


@dataclass
class TeamPermissionUpdateMessage:
    """Team permission update from leader to teammates."""

    type: str = "team_permission_update"
    permission_update: dict = field(default_factory=dict)
    directory_path: str = ""
    tool_name: str = ""


@dataclass
class ModeSetRequestMessage:
    """Mode set request from leader to teammate."""

    type: str = "mode_set_request"
    mode: str = ""
    from_agent: str = ""


@dataclass
class QuestionRequestMessage:
    """Question request from teammate to leader (AskUserQuestion tool)."""

    type: str = "question_request"
    request_id: str = ""
    from_agent: str = ""
    questions: list = field(default_factory=list)  # List of question dicts
    timestamp: str = ""


@dataclass
class QuestionResponseMessage:
    """Question response from leader to teammate."""

    type: str = "question_response"
    request_id: str = ""
    subtype: str = "success"  # 'success' or 'error'
    answers: dict = field(default_factory=dict)  # question text -> answer
    annotations: dict = field(default_factory=dict)  # optional annotations
    error: Optional[str] = None


# =============================================================================
# Path Helpers
# =============================================================================


def sanitize_component(name: str) -> str:
    """Sanitize a path component.

    Args:
        name: Name to sanitize

    Returns:
        Sanitized name safe for filesystem
    """
    # Replace problematic characters
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name)


def get_teams_dir() -> Path:
    """Get the teams directory.

    Returns:
        Path to ~/.claude/teams
    """
    return get_memory_base() / "teams"


def get_inbox_path(agent_name: str, team_name: Optional[str] = None) -> str:
    """Get the path to a teammate's inbox file.

    Structure: ~/.claude/teams/{team_name}/inboxes/{agent_name}.json

    Args:
        agent_name: Agent name (not UUID)
        team_name: Optional team name (defaults to current)

    Returns:
        Path to inbox file
    """
    team = team_name or get_current_team_name() or "default"
    safe_team = sanitize_component(team)
    safe_agent = sanitize_component(agent_name)

    inbox_dir = get_teams_dir() / safe_team / "inboxes"
    return str(inbox_dir / f"{safe_agent}.json")


async def ensure_inbox_dir(team_name: Optional[str] = None) -> None:
    """Ensure the inbox directory exists for a team.

    Args:
        team_name: Optional team name
    """
    team = team_name or get_current_team_name() or "default"
    safe_team = sanitize_component(team)
    inbox_dir = get_teams_dir() / safe_team / "inboxes"
    inbox_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Mailbox Operations
# =============================================================================


async def read_mailbox(
    agent_name: str,
    team_name: Optional[str] = None,
) -> list[TeammateMessage]:
    """Read all messages from a teammate's inbox.

    Args:
        agent_name: Agent name to read inbox for
        team_name: Optional team name

    Returns:
        List of TeammateMessage
    """
    inbox_path = get_inbox_path(agent_name, team_name)

    try:
        with open(inbox_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = [
            TeammateMessage(
                from_agent=msg.get("from", ""),
                text=msg.get("text", ""),
                timestamp=msg.get("timestamp", ""),
                read=msg.get("read", False),
                color=msg.get("color"),
                summary=msg.get("summary"),
            )
            for msg in data
        ]
        return result
    except FileNotFoundError:
        # Inbox not created yet - return empty list
        return []
    except Exception as e:
        logger.debug(f"read_mailbox error: {e}")
        return []


async def read_unread_messages(
    agent_name: str,
    team_name: Optional[str] = None,
) -> list[TeammateMessage]:
    """Read only unread messages from a teammate's inbox.

    Args:
        agent_name: Agent name
        team_name: Optional team name

    Returns:
        List of unread TeammateMessage
    """
    messages = await read_mailbox(agent_name, team_name)
    return [m for m in messages if not m.read]


async def write_to_mailbox(
    recipient_name: str,
    message: TeammateMessage,
    team_name: Optional[str] = None,
) -> None:
    """Write a message to a teammate's inbox.

    Uses simple file locking to prevent race conditions.

    Args:
        recipient_name: Recipient agent name
        message: Message to write
        team_name: Optional team name
    """
    await ensure_inbox_dir(team_name)

    inbox_path = get_inbox_path(recipient_name, team_name)
    logger.debug(f"write_to_mailbox: recipient='{recipient_name}', team='{team_name}', inbox_path='{inbox_path}'")
    lock_path = inbox_path + ".lock"

    # Create inbox file if not exists
    Path(inbox_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(inbox_path).exists():
        with open(inbox_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    # Simple file lock
    try:
        # Try to create lock file
        while Path(lock_path).exists():
            await asyncio.sleep(0.05)

        Path(lock_path).write_text(str(os.getpid()))

        # Read current messages
        messages = await read_mailbox(recipient_name, team_name)

        # Add new message
        messages.append(message)

        # Write back
        data = [
            {
                "from": m.from_agent,
                "text": m.text,
                "timestamp": m.timestamp,
                "read": m.read,
                "color": m.color,
                "summary": m.summary,
            }
            for m in messages
        ]

        with open(inbox_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"write_to_mailbox SUCCESS: wrote message from '{message.from_agent}' to '{inbox_path}', total messages: {len(messages)}")

    finally:
        # Release lock
        try:
            Path(lock_path).unlink()
        except Exception:
            pass


async def mark_messages_as_read(
    agent_name: str,
    team_name: Optional[str] = None,
) -> None:
    """Mark all messages in a teammate's inbox as read.

    Args:
        agent_name: Agent name
        team_name: Optional team name
    """
    inbox_path = get_inbox_path(agent_name, team_name)
    lock_path = inbox_path + ".lock"

    if not Path(inbox_path).exists():
        return

    try:
        # Simple lock
        while Path(lock_path).exists():
            await asyncio.sleep(0.05)

        Path(lock_path).write_text(str(os.getpid()))

        messages = await read_mailbox(agent_name, team_name)

        if not messages:
            return

        # Mark all as read
        data = [
            {
                "from": m.from_agent,
                "text": m.text,
                "timestamp": m.timestamp,
                "read": True,
                "color": m.color,
                "summary": m.summary,
            }
            for m in messages
        ]

        with open(inbox_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    finally:
        try:
            Path(lock_path).unlink()
        except Exception:
            pass


async def mark_message_as_read_by_index(
    agent_name: str,
    index: int,
    team_name: Optional[str] = None,
) -> None:
    """Mark a single message as read by its index.

    This is used to mark only one message as read, leaving others unread.
    This prevents message loss when processing one message at a time.

    Args:
        agent_name: Agent name
        index: Index of message to mark as read
        team_name: Optional team name
    """
    inbox_path = get_inbox_path(agent_name, team_name)
    lock_path = inbox_path + ".lock"

    if not Path(inbox_path).exists():
        return

    try:
        # Simple lock
        while Path(lock_path).exists():
            await asyncio.sleep(0.05)

        Path(lock_path).write_text(str(os.getpid()))

        messages = await read_mailbox(agent_name, team_name)

        if not messages or index < 0 or index >= len(messages):
            return

        # Mark only the specified message as read
        data = []
        for i, m in enumerate(messages):
            data.append({
                "from": m.from_agent,
                "text": m.text,
                "timestamp": m.timestamp,
                "read": i == index or m.read,  # Only mark the specified index
                "color": m.color,
                "summary": m.summary,
            })

        with open(inbox_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    finally:
        try:
            Path(lock_path).unlink()
        except Exception:
            pass


async def clear_mailbox(
    agent_name: str,
    team_name: Optional[str] = None,
) -> None:
    """Clear a teammate's inbox.

    Args:
        agent_name: Agent name
        team_name: Optional team name
    """
    inbox_path = get_inbox_path(agent_name, team_name)

    try:
        with open(inbox_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    except FileNotFoundError:
        pass


# =============================================================================
# Message Formatting
# =============================================================================

TEAMMATE_MESSAGE_TAG = "teammate_message"


def format_teammate_messages(messages: list[TeammateMessage]) -> str:
    """Format teammate messages as XML for attachment display.

    Args:
        messages: List of messages

    Returns:
        XML formatted string
    """
    return "\n\n".join(
        f'<{TEAMMATE_MESSAGE_TAG} teammate_id="{m.from_agent}"'
        + (f' color="{m.color}"' if m.color else "")
        + (f' summary="{m.summary}"' if m.summary else "")
        + f">\n{m.text}\n</{TEAMMATE_MESSAGE_TAG}>"
        for m in messages
    )


# =============================================================================
# Message Creation Helpers
# =============================================================================


def create_idle_notification(
    agent_id: str,
    options: Optional[dict] = None,
) -> IdleNotificationMessage:
    """Create an idle notification message.

    Args:
        agent_id: Agent ID
        options: Optional fields

    Returns:
        IdleNotificationMessage
    """
    opts = options or {}
    return IdleNotificationMessage(
        type="idle_notification",
        from_agent=agent_id,
        timestamp=datetime.now().isoformat(),
        idle_reason=opts.get("idle_reason"),
        summary=opts.get("summary"),
        completed_task_id=opts.get("completed_task_id"),
        completed_status=opts.get("completed_status"),
        failure_reason=opts.get("failure_reason"),
    )


def is_idle_notification(message_text: str) -> Optional[IdleNotificationMessage]:
    """Check if a message is an idle notification.

    Args:
        message_text: Message text to parse

    Returns:
        IdleNotificationMessage or None
    """
    try:
        data = json.loads(message_text)
        if data.get("type") == "idle_notification":
            return IdleNotificationMessage(
                type=data.get("type", ""),
                from_agent=data.get("from", ""),
                timestamp=data.get("timestamp", ""),
                idle_reason=data.get("idle_reason"),
                summary=data.get("summary"),
                completed_task_id=data.get("completed_task_id"),
                completed_status=data.get("completed_status"),
                failure_reason=data.get("failure_reason"),
            )
    except Exception:
        pass
    return None


def is_permission_request(message_text: str) -> Optional[PermissionRequestMessage]:
    """Check if a message is a permission request.

    Args:
        message_text: Message text to parse

    Returns:
        PermissionRequestMessage or None
    """
    try:
        data = json.loads(message_text)
        if data.get("type") == "permission_request":
            return PermissionRequestMessage(
                type="permission_request",
                request_id=data.get("request_id", ""),
                agent_id=data.get("agent_id", ""),
                tool_name=data.get("tool_name", ""),
                tool_use_id=data.get("tool_use_id", ""),
                description=data.get("description", ""),
                input=data.get("input", {}),
                permission_suggestions=data.get("permission_suggestions", []),
            )
    except Exception:
        pass
    return None


def is_permission_response(message_text: str) -> Optional[PermissionResponseMessage]:
    """Check if a message is a permission response.

    Args:
        message_text: Message text to parse

    Returns:
        PermissionResponseMessage or None
    """
    try:
        data = json.loads(message_text)
        if data.get("type") == "permission_response":
            return PermissionResponseMessage(
                type="permission_response",
                request_id=data.get("request_id", ""),
                subtype=data.get("subtype", "success"),
                error=data.get("error"),
                response=data.get("response"),
            )
    except Exception:
        pass
    return None


def is_structured_protocol_message(message_text: str) -> bool:
    """Check if a message is a structured protocol message.

    These should be routed by inbox poller rather than consumed as raw LLM context.

    Args:
        message_text: Message text to check

    Returns:
        True if structured protocol message
    """
    try:
        data = json.loads(message_text)
        if not isinstance(data, dict) or "type" not in data:
            return False
        msg_type = data.get("type")
        return msg_type in [
            "permission_request",
            "permission_response",
            "idle_notification",
            "shutdown_request",
            "shutdown_approved",
            "shutdown_rejected",
            "team_permission_update",
            "mode_set_request",
            "plan_approval_request",
            "plan_approval_response",
            "sandbox_permission_request",
            "sandbox_permission_response",
            "task_assignment",
        ]
    except Exception:
        return False


# =============================================================================
# Shutdown Message Helpers
# =============================================================================


def create_shutdown_request_message(
    request_id: str,
    from_agent: str,
    reason: Optional[str] = None,
) -> ShutdownRequestMessage:
    """Create a shutdown request message."""
    return ShutdownRequestMessage(
        type="shutdown_request",
        request_id=request_id,
        from_agent=from_agent,
        reason=reason,
        timestamp=datetime.now().isoformat(),
    )


def create_shutdown_approved_message(
    request_id: str,
    from_agent: str,
    pane_id: Optional[str] = None,
    backend_type: Optional[str] = None,
) -> ShutdownApprovedMessage:
    """Create a shutdown approved message."""
    return ShutdownApprovedMessage(
        type="shutdown_approved",
        request_id=request_id,
        from_agent=from_agent,
        timestamp=datetime.now().isoformat(),
        pane_id=pane_id,
        backend_type=backend_type,
    )


def create_shutdown_rejected_message(
    request_id: str,
    from_agent: str,
    reason: str,
) -> ShutdownRejectedMessage:
    """Create a shutdown rejected message."""
    return ShutdownRejectedMessage(
        type="shutdown_rejected",
        request_id=request_id,
        from_agent=from_agent,
        reason=reason,
        timestamp=datetime.now().isoformat(),
    )


def is_shutdown_request(message_text: str) -> Optional[ShutdownRequestMessage]:
    """Check if a message is a shutdown request."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "shutdown_request":
            return ShutdownRequestMessage(
                type=data.get("type", ""),
                request_id=data.get("requestId", ""),
                from_agent=data.get("from", ""),
                reason=data.get("reason"),
                timestamp=data.get("timestamp", ""),
            )
    except Exception:
        pass
    return None


def is_shutdown_approved(message_text: str) -> Optional[ShutdownApprovedMessage]:
    """Check if a message is a shutdown approved."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "shutdown_approved":
            return ShutdownApprovedMessage(
                type=data.get("type", ""),
                request_id=data.get("requestId", ""),
                from_agent=data.get("from", ""),
                timestamp=data.get("timestamp", ""),
                pane_id=data.get("paneId"),
                backend_type=data.get("backendType"),
            )
    except Exception:
        pass
    return None


def is_shutdown_rejected(message_text: str) -> Optional[ShutdownRejectedMessage]:
    """Check if a message is a shutdown rejected."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "shutdown_rejected":
            return ShutdownRejectedMessage(
                type=data.get("type", ""),
                request_id=data.get("requestId", ""),
                from_agent=data.get("from", ""),
                reason=data.get("reason", ""),
                timestamp=data.get("timestamp", ""),
            )
    except Exception:
        pass
    return None


# =============================================================================
# Plan Approval Helpers
# =============================================================================


def create_plan_approval_request_message(
    from_agent: str,
    plan_file_path: str,
    plan_content: str,
    request_id: str,
) -> PlanApprovalRequestMessage:
    """Create a plan approval request message."""
    return PlanApprovalRequestMessage(
        type="plan_approval_request",
        from_agent=from_agent,
        timestamp=datetime.now().isoformat(),
        plan_file_path=plan_file_path,
        plan_content=plan_content,
        request_id=request_id,
    )


def create_plan_approval_response_message(
    request_id: str,
    approved: bool,
    feedback: Optional[str] = None,
    permission_mode: Optional[str] = None,
) -> PlanApprovalResponseMessage:
    """Create a plan approval response message."""
    return PlanApprovalResponseMessage(
        type="plan_approval_response",
        request_id=request_id,
        approved=approved,
        feedback=feedback,
        timestamp=datetime.now().isoformat(),
        permission_mode=permission_mode,
    )


def is_plan_approval_request(message_text: str) -> Optional[PlanApprovalRequestMessage]:
    """Check if a message is a plan approval request."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "plan_approval_request":
            return PlanApprovalRequestMessage(
                type=data.get("type", ""),
                from_agent=data.get("from", ""),
                timestamp=data.get("timestamp", ""),
                plan_file_path=data.get("planFilePath", ""),
                plan_content=data.get("planContent", ""),
                request_id=data.get("requestId", ""),
            )
    except Exception:
        pass
    return None


def is_plan_approval_response(message_text: str) -> Optional[PlanApprovalResponseMessage]:
    """Check if a message is a plan approval response."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "plan_approval_response":
            return PlanApprovalResponseMessage(
                type=data.get("type", ""),
                request_id=data.get("requestId", ""),
                approved=data.get("approved", False),
                feedback=data.get("feedback"),
                timestamp=data.get("timestamp", ""),
                permission_mode=data.get("permissionMode"),
            )
    except Exception:
        pass
    return None


# =============================================================================
# Sandbox Permission Helpers
# =============================================================================


def is_sandbox_permission_request(message_text: str) -> Optional[SandboxPermissionRequestMessage]:
    """Check if a message is a sandbox permission request."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "sandbox_permission_request":
            return SandboxPermissionRequestMessage(
                type=data.get("type", ""),
                request_id=data.get("requestId", ""),
                worker_id=data.get("workerId", ""),
                worker_name=data.get("workerName", ""),
                worker_color=data.get("workerColor"),
                host_pattern=data.get("hostPattern", {}),
                created_at=data.get("createdAt", 0),
            )
    except Exception:
        pass
    return None


def is_sandbox_permission_response(message_text: str) -> Optional[SandboxPermissionResponseMessage]:
    """Check if a message is a sandbox permission response."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "sandbox_permission_response":
            return SandboxPermissionResponseMessage(
                type=data.get("type", ""),
                request_id=data.get("requestId", ""),
                host=data.get("host", ""),
                allow=data.get("allow", False),
                timestamp=data.get("timestamp", ""),
            )
    except Exception:
        pass
    return None


# =============================================================================
# Task Assignment Helpers
# =============================================================================


def create_task_assignment_message(
    task_id: str,
    subject: str,
    description: str,
    assigned_by: str,
) -> TaskAssignmentMessage:
    """Create a task assignment message."""
    return TaskAssignmentMessage(
        type="task_assignment",
        task_id=task_id,
        subject=subject,
        description=description,
        assigned_by=assigned_by,
        timestamp=datetime.now().isoformat(),
    )


def is_task_assignment(message_text: str) -> Optional[TaskAssignmentMessage]:
    """Check if a message is a task assignment."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "task_assignment":
            return TaskAssignmentMessage(
                type=data.get("type", ""),
                task_id=data.get("taskId", ""),
                subject=data.get("subject", ""),
                description=data.get("description", ""),
                assigned_by=data.get("assignedBy", ""),
                timestamp=data.get("timestamp", ""),
            )
    except Exception:
        pass
    return None


# =============================================================================
# Team Permission Update Helpers
# =============================================================================


def is_team_permission_update(message_text: str) -> Optional[TeamPermissionUpdateMessage]:
    """Check if a message is a team permission update."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "team_permission_update":
            return TeamPermissionUpdateMessage(
                type=data.get("type", ""),
                permission_update=data.get("permissionUpdate", {}),
                directory_path=data.get("directoryPath", ""),
                tool_name=data.get("toolName", ""),
            )
    except Exception:
        pass
    return None


# =============================================================================
# Mode Set Helpers
# =============================================================================


def create_mode_set_request_message(mode: str, from_agent: str) -> ModeSetRequestMessage:
    """Create a mode set request message."""
    return ModeSetRequestMessage(
        type="mode_set_request",
        mode=mode,
        from_agent=from_agent,
    )


def is_mode_set_request(message_text: str) -> Optional[ModeSetRequestMessage]:
    """Check if a message is a mode set request."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "mode_set_request":
            return ModeSetRequestMessage(
                type=data.get("type", ""),
                mode=data.get("mode", ""),
                from_agent=data.get("from", ""),
            )
    except Exception:
        pass
    return None


# =============================================================================
# Question Request/Response Helpers (AskUserQuestion tool)
# =============================================================================


def create_question_request_message(
    request_id: str,
    from_agent: str,
    questions: list,
) -> QuestionRequestMessage:
    """Create a question request message."""
    return QuestionRequestMessage(
        type="question_request",
        request_id=request_id,
        from_agent=from_agent,
        questions=questions,
        timestamp=datetime.now().isoformat(),
    )


def create_question_response_message(
    request_id: str,
    answers: dict,
    annotations: Optional[dict] = None,
    error: Optional[str] = None,
) -> QuestionResponseMessage:
    """Create a question response message."""
    return QuestionResponseMessage(
        type="question_response",
        request_id=request_id,
        subtype="success" if not error else "error",
        answers=answers,
        annotations=annotations or {},
        error=error,
    )


def is_question_request(message_text: str) -> Optional[QuestionRequestMessage]:
    """Check if a message is a question request."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "question_request":
            return QuestionRequestMessage(
                type=data.get("type", ""),
                request_id=data.get("request_id", ""),
                from_agent=data.get("from_agent", ""),
                questions=data.get("questions", []),
                timestamp=data.get("timestamp", ""),
            )
    except Exception:
        pass
    return None


def is_question_response(message_text: str) -> Optional[QuestionResponseMessage]:
    """Check if a message is a question response."""
    try:
        data = json.loads(message_text)
        if data.get("type") == "question_response":
            return QuestionResponseMessage(
                type=data.get("type", ""),
                request_id=data.get("request_id", ""),
                subtype=data.get("subtype", "success"),
                answers=data.get("answers", {}),
                annotations=data.get("annotations", {}),
                error=data.get("error"),
            )
    except Exception:
        pass
    return None


__all__ = [
    # Message types
    "TeammateMessage",
    "IdleNotificationMessage",
    "PermissionRequestMessage",
    "PermissionResponseMessage",
    "ShutdownRequestMessage",
    "ShutdownApprovedMessage",
    "ShutdownRejectedMessage",
    "PlanApprovalRequestMessage",
    "PlanApprovalResponseMessage",
    "SandboxPermissionRequestMessage",
    "SandboxPermissionResponseMessage",
    "TaskAssignmentMessage",
    "TeamPermissionUpdateMessage",
    "ModeSetRequestMessage",
    # Constants
    "TEAMMATE_MESSAGE_TAG",
    # Path helpers
    "sanitize_component",
    "get_teams_dir",
    "get_inbox_path",
    "ensure_inbox_dir",
    # Mailbox operations
    "read_mailbox",
    "read_unread_messages",
    "write_to_mailbox",
    "mark_messages_as_read",
    "mark_message_as_read_by_index",
    "clear_mailbox",
    # Formatting
    "format_teammate_messages",
    # Idle notification helpers
    "create_idle_notification",
    "is_idle_notification",
    # Permission helpers
    "is_permission_request",
    "is_permission_response",
    "is_structured_protocol_message",
    # Shutdown helpers
    "create_shutdown_request_message",
    "create_shutdown_approved_message",
    "create_shutdown_rejected_message",
    "is_shutdown_request",
    "is_shutdown_approved",
    "is_shutdown_rejected",
    # Plan approval helpers
    "create_plan_approval_request_message",
    "create_plan_approval_response_message",
    "is_plan_approval_request",
    "is_plan_approval_response",
    # Sandbox permission helpers
    "is_sandbox_permission_request",
    "is_sandbox_permission_response",
    # Task assignment helpers
    "create_task_assignment_message",
    "is_task_assignment",
    # Team permission helpers
    "is_team_permission_update",
    # Mode set helpers
    "create_mode_set_request_message",
    "is_mode_set_request",
    # Question request/response helpers
    "QuestionRequestMessage",
    "QuestionResponseMessage",
    "create_question_request_message",
    "create_question_response_message",
    "is_question_request",
    "is_question_response",
]