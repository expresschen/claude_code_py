"""SendMessage Tool implementation.

Sends messages to teammates via the mailbox system, or continues/resumes agents.
Handles shutdown_request, shutdown_response, and other structured messages.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional, Union, TYPE_CHECKING

from pydantic import BaseModel

# Debug flag - controlled by environment variable CLAUDE_CODE_DEBUG_TEAMMATE
DEBUG_SEND_MSG = os.environ.get("CLAUDE_CODE_DEBUG_TEAMMATE", "").lower() in ("1", "true", "yes")

# Import unified debug logger
from claude_code_py.utils.debug_log import debug_log

def _debug_print(msg: str) -> None:
    """Print debug message to console and log file."""
    debug_log("[SEND_MESSAGE]", msg, DEBUG_SEND_MSG)

from claude_code_py.tool.base import Tool, build_tool, ValidationResult
from claude_code_py.tool.result import ToolResult, ToolError
from claude_code_py.task.types import TaskStatus
from claude_code_py.utils.teammate_context import (
    get_current_agent_name,
    get_current_agent_id,
    get_current_team_name,
    get_teammate_context,
    is_team_lead,
    TEAM_LEAD_NAME,
)
from claude_code_py.utils.teammate_mailbox import (
    write_to_mailbox,
    TeammateMessage,
    format_teammate_messages,
    create_shutdown_approved_message,
    create_shutdown_rejected_message,
    create_shutdown_request_message,
    is_shutdown_request,
)
from claude_code_py.utils.agent_resume import (
    is_agent_id_format,
    to_agent_id,
    find_agent_for_send,
    queue_pending_message,
    resume_agent_background,
)
from claude_code_py.task.manager import find_task_by_agent_id
from claude_code_py.utils.team.team_file import read_team_file, BackendType
from claude_code_py.utils.swarm.spawn_in_process import generate_task_id
from claude_code_py.engine.coordinator_mode import is_coordinator_mode

from .constants import SEND_MESSAGE_TOOL_NAME

if TYPE_CHECKING:
    from claude_code_py.tool.context import ToolUseContext
    from claude_code_py.core_types.message import AssistantMessage
    from claude_code_py.core_types.permissions import PermissionResult


# =============================================================================
# Input Schema
# =============================================================================


class StructuredMessage(BaseModel):
    """Structured message types."""

    type: str  # "shutdown_request", "shutdown_response", "plan_approval_response"
    reason: Optional[str] = None
    request_id: Optional[str] = None
    approve: Optional[bool] = None
    feedback: Optional[str] = None


class SendMessageInput(BaseModel):
    """Input for SendMessageTool."""

    to: str  # Recipient: teammate name, "*", or "lead"
    summary: Optional[str] = None  # 5-10 word summary
    message: Union[str, dict]  # Plain text or structured message


class SendMessageOutput(BaseModel):
    """Output for SendMessageTool."""

    message: str
    recipient: str
    sender: str


class RequestOutput(BaseModel):
    """Output for request-type messages."""

    success: bool
    message: str
    request_id: str
    target: Optional[str] = None


# =============================================================================
# Shutdown Handlers
# =============================================================================


async def handle_shutdown_request(
    target_name: str,
    reason: Optional[str],
    context: "ToolUseContext",
) -> dict:
    """Handle shutdown request from leader to teammate.

    Creates shutdown_request message and writes to teammate's mailbox.

    Args:
        target_name: Target teammate name
        reason: Shutdown reason
        context: Tool use context

    Returns:
        Result dict with request_id
    """
    sender_name = get_current_agent_name() or TEAM_LEAD_NAME

    # Get team_name: first try teammate context, then AppState.team_context
    team_name = get_current_team_name()
    if not team_name:
        app_state = context.get_app_state()
        team_context = app_state.team_context if app_state else None
        team_name = team_context.get("teamName") if team_context else None
    if not team_name:
        team_name = "default"

    request_id = generate_task_id("shutdown")

    _debug_print("→ handle_shutdown_request:")
    _debug_print(f"   target_name: '{target_name}'")
    _debug_print(f"   sender_name: '{sender_name}'")
    _debug_print(f"   team_name: '{team_name}'")
    _debug_print(f"   request_id: '{request_id}'")
    _debug_print(f"   reason: '{reason}'")

    # Create shutdown_request message
    shutdown_message = create_shutdown_request_message(
        request_id=request_id,
        from_agent=sender_name,
        reason=reason,
    )

    # Write to teammate's mailbox
    await write_to_mailbox(
        target_name,
        TeammateMessage(
            from_agent=sender_name,
            text=json.dumps({
                "type": shutdown_message.type,
                "requestId": shutdown_message.request_id,
                "from": shutdown_message.from_agent,
                "reason": shutdown_message.reason,
                "timestamp": shutdown_message.timestamp,
            }),
            timestamp=datetime.now().isoformat(),
        ),
        team_name,
    )

    _debug_print("✅ Shutdown request sent to mailbox")

    return {
        "success": True,
        "message": f"Shutdown request sent to {target_name}. Request ID: {request_id}",
        "request_id": request_id,
        "target": target_name,
    }


async def handle_shutdown_approval(
    request_id: str,
    context: "ToolUseContext",
) -> dict:
    """Handle shutdown approval from teammate.

    Sends shutdown_approved message to leader and aborts own controller.

    Args:
        request_id: Shutdown request ID
        context: Tool use context

    Returns:
        Result dict with success status
    """
    team_name = get_current_team_name()
    agent_id = get_current_agent_id()
    agent_name = get_current_agent_name() or "teammate"

    # Get own pane info from team file
    own_pane_id = None
    own_backend_type = None
    if team_name:
        team_file = read_team_file(team_name)
        if team_file and agent_id:
            for member in team_file.members:
                if member.agent_id == agent_id:
                    own_pane_id = member.tmux_pane_id
                    own_backend_type = member.backend_type
                    break

    # Create shutdown_approved message
    approved_message = create_shutdown_approved_message(
        request_id=request_id,
        from_agent=agent_name,
        pane_id=own_pane_id,
        backend_type=own_backend_type.value if own_backend_type else "in-process",
    )

    # Send to leader mailbox
    await write_to_mailbox(
        TEAM_LEAD_NAME,
        TeammateMessage(
            from_agent=agent_name,
            text=json.dumps({
                "type": approved_message.type,
                "requestId": approved_message.request_id,
                "from": approved_message.from_agent,
                "timestamp": approved_message.timestamp,
                "paneId": approved_message.pane_id,
                "backendType": approved_message.backend_type,
            }),
            timestamp=datetime.now().isoformat(),
        ),
        team_name,
    )

    # Abort own controller for in-process teammate
    if own_backend_type == BackendType.IN_PROCESS or agent_id:
        app_state = context.get_app_state()
        task = find_task_by_agent_id(agent_id, context.get_app_state)
        if task and task.abort_controller:
            task.abort_controller.abort()

    return {
        "success": True,
        "message": f"Shutdown approved. Agent {agent_name} is now exiting.",
        "request_id": request_id,
    }


async def handle_shutdown_rejection(
    request_id: str,
    reason: str,
    context: "ToolUseContext",
) -> dict:
    """Handle shutdown rejection from teammate.

    Sends shutdown_rejected message to leader.

    Args:
        request_id: Shutdown request ID
        reason: Rejection reason
        context: Tool use context

    Returns:
        Result dict with success status
    """
    team_name = get_current_team_name()
    agent_name = get_current_agent_name() or "teammate"

    # Create shutdown_rejected message
    rejected_message = create_shutdown_rejected_message(
        request_id=request_id,
        from_agent=agent_name,
        reason=reason,
    )

    # Send to leader mailbox
    await write_to_mailbox(
        TEAM_LEAD_NAME,
        TeammateMessage(
            from_agent=agent_name,
            text=json.dumps({
                "type": rejected_message.type,
                "requestId": rejected_message.request_id,
                "from": rejected_message.from_agent,
                "reason": rejected_message.reason,
                "timestamp": rejected_message.timestamp,
            }),
            timestamp=datetime.now().isoformat(),
        ),
        team_name,
    )

    return {
        "success": True,
        "message": f"Shutdown rejected by {agent_name}: {reason}",
        "request_id": request_id,
    }


# =============================================================================
# Tool Implementation
# =============================================================================


class SendMessageToolClass(Tool[SendMessageInput, SendMessageOutput, None]):
    """Tool for sending messages to teammates."""

    name = SEND_MESSAGE_TOOL_NAME
    search_hint = "send a message to a teammate"
    input_schema = SendMessageInput
    output_schema = SendMessageOutput

    async def call(
        self,
        args: SendMessageInput,
        context: "ToolUseContext",
        can_use_tool: Any,
        parent_message: "AssistantMessage",
        on_progress: Optional[Any] = None,
    ) -> ToolResult[SendMessageOutput]:
        """Execute the tool.

        Handles three routing modes:
        1. Agent continuation: to matches agent ID or name → resume/queue
        2. Swarm teammate: to matches teammate name → mailbox
        3. Broadcast: to == "*" → broadcast to team

        Args:
            args: Tool input
            context: Tool context
            can_use_tool: Permission check function
            parent_message: Parent assistant message
            on_progress: Progress callback

        Returns:
            ToolResult with message info
        """
        recipient = args.to
        message = args.message

        _debug_print("=" * 60)
        _debug_print("SendMessageTool.call:")
        _debug_print(f"  recipient: '{recipient}'")
        _debug_print(f"  summary: '{args.summary}'")
        _debug_print(f"  message type: {type(message).__name__}")
        if isinstance(message, str):
            _debug_print(f"  message preview: '{message[:100]}{'...' if len(message) > 100 else ''}'")
        elif isinstance(message, dict):
            _debug_print(f"  message dict type: '{message.get('type', 'unknown')}'")
        _debug_print("=" * 60)

        # Route 1: Agent continuation (coordinator mode only)
        # Only used in coordinator mode to continue/resume worker agents spawned via AgentTool.
        # In swarm mode, teammate messages should go through mailbox (Route 3/4).
        if isinstance(message, str) and recipient != "*" and is_coordinator_mode():
            # Exclude team-lead - leader messages go through mailbox
            if recipient.lower() != TEAM_LEAD_NAME.lower():
                # Check if recipient looks like an agent ID or name
                _debug_print("→ Route 1: Checking agent continuation (coordinator mode)...")
                agent_task = find_agent_for_send(
                    recipient,
                    context.get_app_state,
                )

                if agent_task:
                    # Found a matching agent task
                    agent_id = agent_task.identity.agent_id
                    _debug_print(f"   Found agent task: agent_id='{agent_id}'")
                    _debug_print(f"   agent status: '{agent_task.status}'")

                    # Check agent status
                    if agent_task.status == TaskStatus.RUNNING:
                        # Agent running - queue message
                        _debug_print("   Agent is RUNNING - queueing message")
                        queue_pending_message(
                            agent_id,
                            message,
                            context.set_app_state_for_tasks or context.set_app_state,
                        )
                        _debug_print("✅ Message queued")
                        return ToolResult(
                            data=SendMessageOutput(
                                message=f"Message queued for delivery to {recipient} at its next tool round.",
                                recipient=recipient,
                                sender=get_current_agent_name() or TEAM_LEAD_NAME,
                            )
                        )

                    # Agent stopped - resume it
                    _debug_print(f"   Agent is NOT RUNNING ({agent_task.status}) - attempting resume")
                    try:
                        result = await resume_agent_background(
                            agent_id,
                            message,
                            context,
                            invoking_request_id=parent_message.request_id if parent_message else None,
                        )
                        output_file = result.get("output_file", "")
                        _debug_print(f"✅ Agent resumed: output_file='{output_file}'")
                        return ToolResult(
                            data=SendMessageOutput(
                                message=f"Agent '{recipient}' was stopped ({agent_task.status}); resumed it in background with your message. Output: {output_file}",
                                recipient=recipient,
                                sender=get_current_agent_name() or TEAM_LEAD_NAME,
                            )
                        )
                    except Exception as e:
                        _debug_print(f"❌ Resume failed: {type(e).__name__}: {e}")
                        return ToolResult(
                            data=SendMessageOutput(
                                message=f"Agent '{recipient}' is stopped ({agent_task.status}) and could not be resumed: {str(e)}",
                                recipient=recipient,
                                sender=get_current_agent_name() or TEAM_LEAD_NAME,
                            )
                        )
                else:
                    _debug_print("   No matching agent task found")

        # Route 2: Handle structured messages (shutdown_request, shutdown_response, plan_approval_response)
        if isinstance(message, dict):
            msg_type = message.get("type")
            _debug_print(f"→ Route 2: Structured message type='{msg_type}'")

            # Shutdown request from leader to teammate
            if msg_type == "shutdown_request":
                _debug_print("   Processing shutdown_request")
                reason = message.get("reason")
                result = await handle_shutdown_request(recipient, reason, context)
                return ToolResult(
                    data=SendMessageOutput(
                        message=result.get("message", "Shutdown request sent"),
                        recipient=recipient,
                        sender=get_current_agent_name() or TEAM_LEAD_NAME,
                    )
                )

            # Shutdown response from teammate to leader
            if msg_type == "shutdown_response":
                request_id = message.get("request_id", "")
                approve = message.get("approve", False)
                reason = message.get("reason")
                _debug_print(f"   Processing shutdown_response: approve={approve}, request_id='{request_id}'")

                if approve:
                    _debug_print("   → Handling approval")
                    result = await handle_shutdown_approval(request_id, context)
                    return ToolResult(
                        data=SendMessageOutput(
                            message=result.get("message", "Shutdown approved"),
                            recipient=TEAM_LEAD_NAME,
                            sender=get_current_agent_name() or "teammate",
                        )
                    )
                else:
                    _debug_print("   → Handling rejection")
                    result = await handle_shutdown_rejection(request_id, reason or "Rejected", context)
                    return ToolResult(
                        data=SendMessageOutput(
                            message=result.get("message", "Shutdown rejected"),
                            recipient=TEAM_LEAD_NAME,
                            sender=get_current_agent_name() or "teammate",
                        )
                    )

        # Route 3 & 4: Swarm teammate or broadcast
        _debug_print("→ Route 3/4: Swarm teammate or broadcast")
        # Get sender info from context
        sender_name = get_current_agent_name() or TEAM_LEAD_NAME

        # Get team_name: first try teammate context, then AppState.team_context
        team_name = get_current_team_name()
        if not team_name:
            # Leader context: get from AppState.team_context
            app_state = context.get_app_state()
            team_context = app_state.team_context if app_state else None
            team_name = team_context.get("teamName") if team_context else None
        if not team_name:
            team_name = "default"

        _debug_print(f"   sender_name: '{sender_name}'")
        _debug_print(f"   team_name: '{team_name}' (from {'teammate_context' if get_current_team_name() else 'AppState'})")

        # Get sender color from context
        sender_color = None
        teammate_ctx = get_teammate_context()
        if teammate_ctx:
            sender_color = teammate_ctx.color
            _debug_print(f"   sender_color: '{sender_color}'")

        # Build message content
        if isinstance(message, dict):
            # Structured message
            message_text = json.dumps(message)
        else:
            # Plain text message
            message_text = message

        # Create mailbox message
        mailbox_message = TeammateMessage(
            from_agent=sender_name,
            text=message_text,
            timestamp=datetime.now().isoformat(),
            read=False,
            color=sender_color,
            summary=args.summary,
        )

        # Handle broadcast vs direct message
        if recipient == "*":
            # Broadcast: write to all team members
            _debug_print("   Broadcast mode (*)")
            # For simplicity, just acknowledge broadcast
            return ToolResult(
                data=SendMessageOutput(
                    message=f"Broadcast message sent to team {team_name}",
                    recipient="*",
                    sender=sender_name,
                )
            )
        elif recipient.lower() == TEAM_LEAD_NAME.lower():
            # Send to lead
            recipient_name = TEAM_LEAD_NAME
            _debug_print(f"   Sending to TEAM_LEAD: '{recipient_name}'")
        else:
            # Send to specific teammate
            recipient_name = recipient
            _debug_print(f"   Sending to teammate: '{recipient_name}'")

        # Write to mailbox
        _debug_print(f"   → Writing to mailbox...")
        await write_to_mailbox(recipient_name, mailbox_message, team_name)
        _debug_print(f"✅ Message written to mailbox")

        return ToolResult(
            data=SendMessageOutput(
                message=f"Message sent to {recipient_name}",
                recipient=recipient_name,
                sender=sender_name,
            )
        )

    async def description(self, input: SendMessageInput, options: dict) -> str:
        """Generate tool description."""
        to = input.to
        summary = input.summary or "message"
        return f"Sending to {to}: {summary}"

    async def prompt(self, options: dict) -> str:
        """Generate tool prompt."""
        from .prompt import get_send_message_tool_prompt
        return get_send_message_tool_prompt()

    def is_concurrency_safe(self, input: SendMessageInput) -> bool:
        return True  # Writing to mailbox is safe

    def is_read_only(self, input: SendMessageInput) -> bool:
        return False  # Writes to mailbox

    def is_destructive(self, input: SendMessageInput) -> bool:
        return False


# Create the tool instance
SendMessageTool = SendMessageToolClass()