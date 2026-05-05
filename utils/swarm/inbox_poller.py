"""Inbox Poller for Leader - Polls mailbox for messages from teammates.

The leader periodically checks its mailbox for:
- Permission requests (teammate needs approval)
- Permission responses (teammate received approval)
- Sandbox permission requests/responses
- Shutdown requests/approvals
- Team permission updates
- Mode set requests
- Plan approval requests
- Regular messages (including idle_notification)

Ported from: src/hooks/useInboxPoller.ts
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Optional, List

# Debug flag - controlled by environment variable
DEBUG_POLLER = os.environ.get("CLAUDE_CODE_DEBUG_TEAMMATE", "").lower() in ("1", "true", "yes")

from claude_code_py.utils.debug_log import debug_log

def _debug_print(msg: str) -> None:
    """Print debug message (controlled by env var)."""
    debug_log("[INBOX_POLLER]", msg, DEBUG_POLLER)

from claude_code_py.utils.teammate_mailbox import (
    TeammateMessage,
    read_unread_messages,
    mark_messages_as_read,
    is_permission_request,
    is_permission_response,
    is_sandbox_permission_request,
    is_sandbox_permission_response,
    is_shutdown_request,
    is_shutdown_approved,
    is_shutdown_rejected,
    is_team_permission_update,
    is_mode_set_request,
    is_plan_approval_request,
    is_plan_approval_response,
    is_question_request,
    PermissionRequestMessage,
    PermissionResponseMessage,
    SandboxPermissionRequestMessage,
    SandboxPermissionResponseMessage,
    ShutdownApprovedMessage,
    PlanApprovalRequestMessage,
    TeamPermissionUpdateMessage,
    ModeSetRequestMessage,
    TEAMMATE_MESSAGE_TAG,
)
from claude_code_py.utils.teammate_mailbox import write_to_mailbox
from claude_code_py.utils.swarm.constants import (
    INBOX_POLL_INTERVAL_S,
    TEAM_LEAD_NAME,
)
from claude_code_py.utils.swarm.permission_sync import (
    send_permission_response_via_mailbox,
)
from claude_code_py.utils.team.team_file import remove_member_by_agent_id

logger = logging.getLogger(__name__)


# =============================================================================
# InboxPollerConfig
# =============================================================================


@dataclass
class InboxPollerConfig:
    """Configuration for the InboxPoller.

    Args:
        team_name: Team name for mailbox routing
        get_app_state: Function to get current AppState
        set_app_state: Function to update AppState
        submit_message_fn: Optional callback to submit message to LLM (returns bool)
        interrupt_fn: Optional callback to interrupt REPL input for permission handling
    """

    team_name: str
    get_app_state: Callable[[], Any]
    set_app_state: Callable[[Any], None]
    submit_message_fn: Optional[Callable[[str], bool]] = None
    interrupt_fn: Optional[Callable[[], None]] = None
    show_permission_dialog: Optional[Callable[[], Any]] = None  # Async callback to show permission dialog


# =============================================================================
# InboxPoller
# =============================================================================


class InboxPoller:
    """Polls the leader's mailbox for messages from teammates.

    Runs in the leader session, periodically checking TEAM_LEAD_NAME mailbox
    for messages and routing them appropriately.

    Matches TypeScript's useInboxPoller hook behavior.
    """

    def __init__(self, config: InboxPollerConfig) -> None:
        """Initialize the InboxPoller.

        Args:
            config: Configuration dataclass
        """
        self.config = config
        self._poll_task: Optional[asyncio.Task] = None
        self._running: bool = False

    def start(self) -> None:
        """Start polling the inbox.

        Uses background event loop for reliable execution.
        """
        if self._running:
            logger.warning("InboxPoller already running")
            return

        self._running = True

        from claude_code_py.utils.async_helpers import get_background_loop
        import asyncio

        loop = get_background_loop()
        future = asyncio.run_coroutine_threadsafe(self._poll_loop(), loop)

        logger.info(f"InboxPoller started for team: {self.config.team_name}")

    async def stop(self) -> None:
        """Stop polling the inbox."""
        if not self._running:
            return

        self._running = False
        self._poll_task = None

        logger.info(f"InboxPoller stopped for team: {self.config.team_name}")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        _debug_print(f"_poll_loop started, running={self._running}")
        while self._running:
            try:
                await self._process_inbox()
                await self._deliver_pending_messages()
            except Exception as e:
                logger.error(f"Error processing inbox: {e}")
                _debug_print(f"Error processing inbox: {e}")

            await asyncio.sleep(INBOX_POLL_INTERVAL_S)
            _debug_print(f"Poll cycle complete, sleeping for {INBOX_POLL_INTERVAL_S}s")

    async def _process_inbox(self) -> None:
        """Process unread messages from the leader's mailbox.

        Matches TypeScript's useInboxPoller.poll() logic.
        Classifies messages into categories and routes them appropriately.
        """
        # Get team_name from AppState
        app_state = self.config.get_app_state()
        team_context = app_state.team_context
        team_name = team_context.get("teamName") if team_context else None

        _debug_print(f"_process_inbox: team_name from AppState = {team_name}")

        # Skip if no team yet
        if not team_name:
            logger.debug("No team_name in AppState, skipping inbox poll")
            _debug_print("No team_name in AppState, skipping inbox poll")
            return

        # Read unread messages
        _debug_print(f"Reading unread messages for TEAM_LEAD_NAME='{TEAM_LEAD_NAME}', team_name='{team_name}'")
        unread = await read_unread_messages(
            TEAM_LEAD_NAME,
            team_name,
        )

        if not unread:
            return

        if unread:
            _debug_print(f"Got {len(unread)} unread messages")

        logger.debug(f"Processing {len(unread)} unread messages")

        # Helper to mark messages as read
        async def mark_read():
            await mark_messages_as_read(TEAM_LEAD_NAME, team_name)

        # Classify messages into categories (matches TS lines 216-248)
        permission_requests: List[TeammateMessage] = []
        permission_responses: List[TeammateMessage] = []
        sandbox_permission_requests: List[TeammateMessage] = []
        sandbox_permission_responses: List[TeammateMessage] = []
        shutdown_requests: List[TeammateMessage] = []
        shutdown_approvals: List[TeammateMessage] = []
        team_permission_updates: List[TeammateMessage] = []
        mode_set_requests: List[TeammateMessage] = []
        plan_approval_requests: List[TeammateMessage] = []
        question_requests: List[TeammateMessage] = []
        regular_messages: List[TeammateMessage] = []

        for m in unread:
            text = m.text

            perm_req = is_permission_request(text)
            perm_resp = is_permission_response(text)
            sandbox_req = is_sandbox_permission_request(text)
            sandbox_resp = is_sandbox_permission_response(text)
            shutdown_req = is_shutdown_request(text)
            shutdown_approval = is_shutdown_approved(text)
            team_perm_update = is_team_permission_update(text)
            mode_set_req = is_mode_set_request(text)
            plan_approval_req = is_plan_approval_request(text)
            question_req = is_question_request(text)

            if perm_req:
                permission_requests.append(m)
            elif perm_resp:
                permission_responses.append(m)
            elif sandbox_req:
                sandbox_permission_requests.append(m)
            elif sandbox_resp:
                sandbox_permission_responses.append(m)
            elif shutdown_req:
                shutdown_requests.append(m)
            elif shutdown_approval:
                shutdown_approvals.append(m)
            elif team_perm_update:
                team_permission_updates.append(m)
            elif mode_set_req:
                mode_set_requests.append(m)
            elif plan_approval_req:
                plan_approval_requests.append(m)
            elif question_req:
                question_requests.append(m)
            else:
                # Unknown type or regular message (including idle_notification)
                regular_messages.append(m)

        # Log classification results
        if unread:
            logger.info(f"[InboxPoller] Classified {len(unread)} messages: "
                       f"perm_req={len(permission_requests)}, perm_resp={len(permission_responses)}, "
                       f"regular={len(regular_messages)}")

        # Handle permission requests (leader side) - queue for user approval
        if permission_requests:
            _debug_print(f"Found {len(permission_requests)} permission request(s)")
            await self._handle_permission_requests(permission_requests, team_name, app_state)

        # Handle question requests (leader side) - queue for user input
        if question_requests:
            _debug_print(f"Found {len(question_requests)} question request(s)")
            await self._handle_question_requests(question_requests, app_state)

        # Handle permission responses (worker side) - invoke callbacks
        # Note: Leader doesn't usually receive these, but handle anyway
        if permission_responses:
            _debug_print(f"Found {len(permission_responses)} permission response(s)")
            # TODO: Implement permission response callback handling if needed

        # Handle sandbox permission requests (leader side)
        if sandbox_permission_requests:
            _debug_print(f"Found {len(sandbox_permission_requests)} sandbox permission request(s)")
            await self._handle_sandbox_permission_requests(sandbox_permission_requests, app_state)

        # Handle sandbox permission responses (worker side)
        if sandbox_permission_responses:
            _debug_print(f"Found {len(sandbox_permission_responses)} sandbox permission response(s)")
            # TODO: Implement sandbox permission response handling if needed

        # Handle team permission updates (teammate side - not for leader)
        # Leader receives these but typically doesn't need to act on them
        if team_permission_updates:
            _debug_print(f"Found {len(team_permission_updates)} team permission update(s)")

        # Handle mode set requests (teammate side - not for leader)
        if mode_set_requests:
            _debug_print(f"Found {len(mode_set_requests)} mode set request(s)")

        # Handle shutdown requests - pass to LLM as regular message
        # (matches TS lines 664-675)
        if shutdown_requests:
            _debug_print(f"Found {len(shutdown_requests)} shutdown request(s)")
            for m in shutdown_requests:
                regular_messages.append(m)

        # Handle shutdown approvals (leader side)
        # (matches TS lines 677-800)
        if shutdown_approvals:
            _debug_print(f"Found {len(shutdown_approvals)} shutdown approval(s)")
            for m in shutdown_approvals:
                parsed = is_shutdown_approved(m.text)
                if parsed:
                    await self._handle_shutdown_approval(parsed, m, app_state, team_name)
                # Pass through as regular message for UI rendering (TS line 798)
                regular_messages.append(m)

        # Handle plan approval requests (leader side)
        # (matches TS lines 599-662)
        if plan_approval_requests:
            _debug_print(f"Found {len(plan_approval_requests)} plan approval request(s)")
            for m in plan_approval_requests:
                parsed = is_plan_approval_request(m.text)
                if parsed:
                    await self._handle_plan_approval_request(parsed, m, team_name, app_state)
                # Still pass through as regular message (TS line 660)
                regular_messages.append(m)

        # Process regular messages (matches TS lines 802-864)
        if regular_messages:
            logger.info(f"[InboxPoller] Processing {len(regular_messages)} regular message(s)")
            self._queue_messages(regular_messages)

        # Mark messages as read after processing all types
        await mark_read()

        # Show permission dialog if any items are pending.
        # This handles both mailbox-enqueued items (from _handle_permission_requests
        # above) and Bridge-path items (enqueued directly to pending_permissions
        # by permission_queue_setter between poll cycles).
        if self.config.show_permission_dialog:
            app_state = self.config.get_app_state()
            if app_state.pending_permissions:
                await self.config.show_permission_dialog()

    async def _handle_permission_requests(
        self,
        permission_requests: List[TeammateMessage],
        team_name: str,
        app_state: Any,
    ) -> None:
        """Handle permission requests from teammates.

        Queue them for user approval in pending_permissions.
        Matches TS lines 250-364.
        """
        pending_items: List[Dict[str, Any]] = []

        for m in permission_requests:
            parsed = is_permission_request(m.text)
            if not parsed:
                continue

            tool_name = parsed.tool_name
            request_id = parsed.request_id
            from_agent = parsed.agent_id or m.from_agent

            # Read-only tools that can be auto-approved
            READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "TaskList", "TaskGet"}

            if tool_name in READ_ONLY_TOOLS:
                # Auto-approve read-only tools
                _debug_print(f"Auto-approving read-only tool: {tool_name} for {from_agent}")
                await send_permission_response_via_mailbox(
                    request_id=request_id,
                    team_name=team_name,
                    recipient_name=from_agent,
                    approved=True,
                )
            else:
                # Queue for user approval
                _debug_print(f"Queueing permission request: {tool_name} from {from_agent}")
                pending_items.append({
                    "id": str(uuid.uuid4()),
                    "request_id": request_id,
                    "from_agent": from_agent,
                    "team_name": team_name,
                    "tool_name": tool_name,
                    "tool_use_id": parsed.tool_use_id,
                    "description": parsed.description,
                    "input": parsed.input,
                    "timestamp": datetime.now().isoformat(),
                    "status": "pending",
                })

        # Batch-initialize pending_permissions from all queued items.
        # The dialog is shown later in _process_inbox after ALL message
        # types are handled — this way Bridge-path items (enqueued directly
        # to pending_permissions) are also included in one dialog session.
        if pending_items:
            self.config.set_app_state(lambda prev: replace(
                prev,
                pending_permissions=(prev.pending_permissions or []) + pending_items,
            ))

            # Interrupt fallback (only when show_permission_dialog unavailable)
            if self.config.show_permission_dialog is None and self.config.interrupt_fn:
                _debug_print(f"Triggering interrupt for {len(pending_items)} permission request(s)")
                self.config.interrupt_fn()

    async def _handle_question_requests(
        self,
        question_requests: List[TeammateMessage],
        app_state: Any,
    ) -> None:
        """Handle question requests from teammates (AskUserQuestion).

        Queue them for user input in pending_questions.
        """
        for m in question_requests:
            parsed = is_question_request(m.text)
            if not parsed:
                continue

            request_id = parsed.request_id
            from_agent = parsed.from_agent
            questions = parsed.questions

            _debug_print(f"Queueing question request: {len(questions)} questions from {from_agent}")

            pending_question = {
                "id": str(uuid.uuid4()),
                "request_id": request_id,
                "from_agent": from_agent,
                "questions": questions,
                "timestamp": datetime.now().isoformat(),
                "status": "pending",
            }

            self.config.set_app_state(lambda prev: replace(
                prev,
                pending_questions=(prev.pending_questions or []) + [pending_question],
            ))

            # Trigger interrupt to notify REPL
            if self.config.interrupt_fn:
                _debug_print(f"Triggering interrupt for question request")
                self.config.interrupt_fn()

    async def _handle_sandbox_permission_requests(
        self,
        sandbox_requests: List[TeammateMessage],
        app_state: Any,
    ) -> None:
        """Handle sandbox permission requests.

        Queue them in workerSandboxPermissions.
        Matches TS lines 399-463.
        """
        new_requests = []
        for m in sandbox_requests:
            parsed = is_sandbox_permission_request(m.text)
            if not parsed:
                continue

            # Validate required fields
            if not parsed.host_pattern or not parsed.host_pattern.get("host"):
                _debug_print("Invalid sandbox permission request: missing host")
                continue

            new_requests.append({
                "requestId": parsed.request_id,
                "workerId": parsed.worker_id,
                "workerName": parsed.worker_name,
                "workerColor": parsed.worker_color,
                "host": parsed.host_pattern["host"],
                "createdAt": parsed.created_at,
            })

        if new_requests:
            self.config.set_app_state(lambda prev: replace(
                prev,
                worker_sandbox_permissions={
                    **prev.worker_sandbox_permissions,
                    "queue": (prev.worker_sandbox_permissions.get("queue", []) + new_requests),
                },
            ))

    async def _handle_shutdown_approval(
        self,
        parsed: ShutdownApprovedMessage,
        original_msg: TeammateMessage,
        app_state: Any,
        team_name: str,
    ) -> None:
        """Handle shutdown approval from teammate.

        Remove teammate from teamContext and mark task completed.
        Matches TS lines 677-800.
        """
        from_agent = parsed.from_agent
        _debug_print(f"Processing shutdown approval from {from_agent}")

        # Remove teammate from teamContext
        team_context = app_state.team_context
        if team_context and "teammates" in team_context:
            teammates = team_context["teammates"]
            # Find teammate by name
            teammate_id = None
            for tid, info in teammates.items():
                if info.get("name") == from_agent:
                    teammate_id = tid
                    break

            if teammate_id:
                # Remove from teammates dict
                updated_teammates = {k: v for k, v in teammates.items() if k != teammate_id}

                # Mark any in-process task as completed
                updated_tasks = dict(app_state.tasks)
                for tid, task in updated_tasks.items():
                    if hasattr(task, 'identity') and task.identity.agent_id == teammate_id:
                        updated_tasks[tid] = replace(task, status="completed", end_time=datetime.now().timestamp())

                # Add system notification
                notification_msg = {
                    "id": str(uuid.uuid4()),
                    "from": "system",
                    "text": json.dumps({
                        "type": "teammate_terminated",
                        "message": f"{from_agent} has shut down.",
                    }),
                    "timestamp": datetime.now().isoformat(),
                    "status": "pending",
                }

                self.config.set_app_state(lambda prev: replace(
                    prev,
                    team_context={
                        **prev.team_context,
                        "teammates": updated_teammates,
                    },
                    tasks=updated_tasks,
                    inbox={
                        "messages": prev.inbox.get("messages", []) + [notification_msg],
                    },
                ))

                # Also update the persistent team file
                remove_member_by_agent_id(team_name, teammate_id)

                _debug_print(f"Removed {from_agent} ({teammate_id}) from teamContext and team file")

    async def _handle_plan_approval_request(
        self,
        parsed: PlanApprovalRequestMessage,
        original_msg: TeammateMessage,
        team_name: str,
        app_state: Any,
    ) -> None:
        """Handle plan approval request from teammate.

        Auto-approve and write response to teammate's mailbox.
        Matches TS lines 599-662.
        """
        from_agent = original_msg.from_agent
        request_id = parsed.request_id

        _debug_print(f"Auto-approving plan from {from_agent} (request {request_id})")

        # Get leader's permission mode to inherit
        perm_context = getattr(app_state, 'tool_permission_context', {})
        leader_mode = perm_context.get('mode', 'default') if isinstance(perm_context, dict) else 'default'
        mode_to_inherit = 'default' if leader_mode == 'plan' else leader_mode

        # Write approval response to teammate's mailbox
        approval_response = {
            "type": "plan_approval_response",
            "requestId": request_id,
            "approved": True,
            "timestamp": datetime.now().isoformat(),
            "permissionMode": mode_to_inherit,
        }

        await write_to_mailbox(
            from_agent,
            TeammateMessage(
                from_agent=TEAM_LEAD_NAME,
                text=json.dumps(approval_response),
                timestamp=datetime.now().isoformat(),
            ),
            team_name,
        )

        _debug_print(f"Plan approval sent to {from_agent}")

    def _format_messages(self, messages: List[TeammateMessage]) -> str:
        """Format messages with XML wrapper.

        Matches TS lines 810-820.
        """
        formatted_parts = []
        for m in messages:
            color_attr = f' color="{m.color}"' if m.color else ""
            summary_attr = f' summary="{m.summary}"' if m.summary else ""
            formatted_parts.append(
                f'<{TEAMMATE_MESSAGE_TAG} teammate_id="{m.from_agent}"{color_attr}{summary_attr}>\n{m.text}\n</{TEAMMATE_MESSAGE_TAG}>'
            )
        return "\n\n".join(formatted_parts)

    def _queue_messages(self, messages: List[TeammateMessage]) -> None:
        """Queue messages in AppState.inbox for later delivery.

        Matches TS lines 823-841.
        """
        inbox_msgs = []
        for m in messages:
            inbox_msgs.append({
                "id": str(uuid.uuid4()),
                "from": m.from_agent,
                "text": m.text,
                "timestamp": m.timestamp or datetime.now().isoformat(),
                "status": "pending",
                "color": m.color,
                "summary": m.summary,
            })

        self.config.set_app_state(lambda prev: replace(
            prev,
            inbox={
                "messages": prev.inbox.get("messages", []) + inbox_msgs,
            },
        ))
        _debug_print(f"Queued {len(inbox_msgs)} messages")
        logger.info(f"[InboxPoller] Queued {len(inbox_msgs)} message(s) for later delivery")

    def _format_inbox_message(self, msg_dict: dict[str, Any]) -> str:
        """Format an inbox message dict with XML wrapper."""
        color_attr = f' color="{msg_dict.get("color")}"' if msg_dict.get("color") else ""
        summary_attr = f' summary="{msg_dict.get("summary")}"' if msg_dict.get("summary") else ""
        return (
            f'<{TEAMMATE_MESSAGE_TAG} teammate_id="{msg_dict["from"]}"{color_attr}{summary_attr}>\n'
            f'{msg_dict["text"]}\n'
            f'</{TEAMMATE_MESSAGE_TAG}>'
        )

    async def _deliver_pending_messages(self) -> None:
        """Deliver pending messages when session becomes idle.

        This is the SINGLE entry point for submitting messages to the LLM.
        _process_inbox only queues messages; this method handles actual submission.

        Matches TS lines 875-941.
        Called from poll loop after _process_inbox, and externally via deliver_pending_now()
        when is_loading becomes False.
        """
        app_state = self.config.get_app_state()
        is_loading = getattr(app_state, 'is_loading', False)

        logger.debug(f"[InboxPoller._deliver] is_loading={is_loading}")
        _debug_print(f"_deliver_pending_messages: is_loading={is_loading}")

        # Skip if busy (LLM processing)
        if is_loading:
            _debug_print(f"Skipping delivery: session busy (loading)")
            return

        inbox_messages = app_state.inbox.get("messages", [])
        pending_messages = [m for m in inbox_messages if m.get("status") == "pending"]
        processed_messages = [m for m in inbox_messages if m.get("status") == "processed"]

        # Log inbox state for debugging
        logger.debug(f"[InboxPoller._deliver] inbox: {len(inbox_messages)} total, {len(pending_messages)} pending")
        _debug_print(f"inbox has {len(inbox_messages)} total, {len(pending_messages)} pending")

        # Clean up processed messages
        if processed_messages:
            _debug_print(f"Cleaning up {len(processed_messages)} processed message(s)")
            processed_ids = {m["id"] for m in processed_messages}
            self.config.set_app_state(lambda prev: replace(
                prev,
                inbox={
                    "messages": [m for m in prev.inbox.get("messages", []) if m["id"] not in processed_ids]
                },
            ))

        # No pending messages to deliver
        if not pending_messages:
            return

        logger.info(f"[InboxPoller] Session idle, delivering {len(pending_messages)} pending message(s)")
        _debug_print(f"Session idle, delivering {len(pending_messages)} pending message(s)")

        # Format and submit
        formatted = "\n\n".join([self._format_inbox_message(m) for m in pending_messages])

        if self.config.submit_message_fn:
            submitted = self.config.submit_message_fn(formatted)
            if submitted:
                # Clear submitted messages
                submitted_ids = {m["id"] for m in pending_messages}
                self.config.set_app_state(lambda prev: replace(
                    prev,
                    inbox={
                        "messages": [m for m in prev.inbox.get("messages", []) if m["id"] not in submitted_ids]
                    },
                ))
                logger.info(f"[InboxPoller] Delivered {len(submitted_ids)} pending messages")
                _debug_print(f"Delivered {len(submitted_ids)} pending messages")
            else:
                _debug_print("Submission rejected, keeping messages queued")
        else:
            logger.warning("No submit_message_fn configured, cannot deliver pending messages")

    def deliver_pending_now(self) -> None:
        """Synchronously trigger pending message delivery.

        Called externally when is_loading becomes False to immediately
        deliver queued messages without waiting for next poll cycle.
        Matches TypeScript's useEffect behavior (lines 876-941).
        """
        if not self._running:
            return

        # Schedule delivery on the background event loop
        from claude_code_py.utils.async_helpers import get_background_loop
        import asyncio

        loop = get_background_loop()
        asyncio.run_coroutine_threadsafe(self._deliver_pending_messages(), loop)
        logger.debug("[InboxPoller] Triggered immediate pending message delivery")


# =============================================================================
# Factory Function
# =============================================================================


def create_inbox_poller(
    team_name: str,
    get_app_state: Callable[[], Any],
    set_app_state: Callable[[Any], None],
    submit_message_fn: Optional[Callable[[str], bool]] = None,
    interrupt_fn: Optional[Callable[[], None]] = None,
    show_permission_dialog: Optional[Callable[[], Any]] = None,
) -> InboxPoller:
    """Create an InboxPoller instance.

    Args:
        team_name: Team name for mailbox routing
        get_app_state: Function to get current AppState
        set_app_state: Function to update AppState
        submit_message_fn: Optional callback to submit message to LLM
        interrupt_fn: Optional callback to interrupt REPL input for permission handling

    Returns:
        InboxPoller instance
    """
    config = InboxPollerConfig(
        team_name=team_name,
        get_app_state=get_app_state,
        set_app_state=set_app_state,
        submit_message_fn=submit_message_fn,
        interrupt_fn=interrupt_fn,
        show_permission_dialog=show_permission_dialog,
    )
    return InboxPoller(config)


__all__ = [
    "InboxPollerConfig",
    "InboxPoller",
    "create_inbox_poller",
]