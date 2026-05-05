"""Leader permission request handler.

This module handles permission requests from worker teammates on the leader side.
When a worker needs permission for a tool use, it sends a request via mailbox.
The leader polls for these requests and displays them to the user.

Ported from: src/hooks/useInboxPoller.ts (permission request handling section)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from claude_code_py.utils.swarm.permission_sync import (
    is_team_leader,
)
from claude_code_py.utils.swarm.constants import TEAM_LEAD_NAME
from claude_code_py.utils.teammate_mailbox import (
    read_mailbox,
    mark_messages_as_read,
    TeammateMessage,
)
from claude_code_py.utils.teammate_context import (
    get_current_team_name,
    get_current_agent_name,
)

if TYPE_CHECKING:
    from claude_code_py.tool.base import Tool
    from claude_code_py.tool.context import ToolUseContext
    from claude_code_py.state.app_state import AppState

logger = logging.getLogger(__name__)


# =============================================================================
# Message Parsing
# =============================================================================


def is_permission_request_message(text: str) -> Optional[Dict[str, Any]]:
    """Parse a permission request message from mailbox.

    Args:
        text: Message text to parse

    Returns:
        Parsed permission request dict or None if not a permission request
    """
    try:
        data = json.loads(text)
        if data.get("type") == "permission_request":
            return data
    except json.JSONDecodeError:
        pass
    return None


def is_permission_response_message(text: str) -> Optional[Dict[str, Any]]:
    """Parse a permission response message from mailbox.

    Args:
        text: Message text to parse

    Returns:
        Parsed permission response dict or None if not a permission response
    """
    try:
        data = json.loads(text)
        if data.get("type") == "permission_response":
            return data
    except json.JSONDecodeError:
        pass
    return None


def is_sandbox_permission_request_message(text: str) -> Optional[Dict[str, Any]]:
    """Parse a sandbox permission request message from mailbox.

    Args:
        text: Message text to parse

    Returns:
        Parsed sandbox permission request dict or None
    """
    try:
        data = json.loads(text)
        if data.get("type") == "sandbox_permission_request":
            return data
    except json.JSONDecodeError:
        pass
    return None


def is_shutdown_request_message(text: str) -> Optional[Dict[str, Any]]:
    """Parse a shutdown request message from mailbox.

    Args:
        text: Message text to parse

    Returns:
        Parsed shutdown request dict or None
    """
    try:
        data = json.loads(text)
        if data.get("type") == "shutdown_request":
            return data
    except json.JSONDecodeError:
        pass
    return None


# =============================================================================
# Permission Request Processing
# =============================================================================


@dataclass
class ProcessedPermissionRequest:
    """A processed permission request ready for UI display."""

    request_id: str
    tool_name: str
    tool_use_id: str
    worker_name: str
    worker_color: Optional[str]
    description: str
    input: Dict[str, Any]
    team_name: str
    created_at: int


async def process_permission_requests(
    team_name: str,
    app_state: Optional["AppState"] = None,
    tools: Optional[List["Tool"]] = None,
    set_app_state: Optional[Callable] = None,
) -> List[ProcessedPermissionRequest]:
    """Process pending permission requests from worker teammates.

    Reads the leader's mailbox for permission requests and adds them
    directly to pending_permissions in AppState. The actual dialog display
    is handled by the REPL loop or _check_worker_permission_requests.

    Args:
        team_name: Team name to check requests for
        app_state: Current app state (optional, for context)
        tools: Available tools list (optional, for tool lookup)
        set_app_state: Function to update app state (optional)

    Returns:
        List of processed permission requests
    """
    if not is_team_leader(team_name):
        logger.debug("Not a team leader, skipping permission request check")
        return []

    if not set_app_state:
        logger.debug("No set_app_state, skipping permission request processing")
        return []

    # Read leader's mailbox
    try:
        messages = await read_mailbox(TEAM_LEAD_NAME, team_name)
    except Exception as e:
        logger.debug(f"Failed to read mailbox: {e}")
        return []

    # Find unread permission requests
    permission_requests: List[TeammateMessage] = []
    for msg in messages:
        if msg and not msg.read:
            parsed = is_permission_request_message(msg.text)
            if parsed:
                permission_requests.append(msg)

    if not permission_requests:
        return []

    logger.debug(f"Found {len(permission_requests)} permission request(s)")

    processed: List[ProcessedPermissionRequest] = []
    new_items: List[Dict[str, Any]] = []

    for msg in permission_requests:
        parsed = is_permission_request_message(msg.text)
        if not parsed:
            continue

        # Validate required fields
        if not parsed.get("tool_name") or not parsed.get("request_id"):
            logger.debug("Invalid permission request: missing required fields")
            continue

        request_id = parsed["request_id"]
        tool_name = parsed["tool_name"]
        worker_name = parsed.get("agent_id", msg.from_agent)

        request = ProcessedPermissionRequest(
            request_id=request_id,
            tool_name=tool_name,
            tool_use_id=parsed.get("tool_use_id", ""),
            worker_name=worker_name,
            worker_color=msg.color,
            description=parsed.get("description", ""),
            input=parsed.get("input", {}),
            team_name=team_name,
            created_at=int(time.time() * 1000),
        )
        processed.append(request)

        # Build pending_permission item for AppState
        pending_item = {
            "id": str(uuid.uuid4()),
            "request_id": request_id,
            "from_agent": worker_name,
            "team_name": team_name,
            "tool_name": tool_name,
            "tool_use_id": parsed.get("tool_use_id", ""),
            "description": parsed.get("description", ""),
            "input": parsed.get("input", {}),
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
        }
        new_items.append(pending_item)
        logger.debug(f"Added permission request {request_id} from {worker_name}")

    if new_items:
        # Directly append to pending_permissions in AppState
        set_app_state(lambda prev: replace(
            prev,
            pending_permissions=(prev.pending_permissions or []) + new_items,
        ))

    # Mark messages as read after processing
    await mark_messages_as_read(TEAM_LEAD_NAME, team_name)

    return processed


# =============================================================================
# Sandbox Permission Request Processing
# =============================================================================


@dataclass
class ProcessedSandboxPermissionRequest:
    """A processed sandbox permission request ready for UI display."""

    request_id: str
    worker_id: str
    worker_name: str
    worker_color: Optional[str]
    host: str
    created_at: int


async def process_sandbox_permission_requests(
    team_name: str,
    set_app_state: Optional[Callable] = None,
) -> List[ProcessedSandboxPermissionRequest]:
    """Process pending sandbox permission requests from worker teammates.

    Args:
        team_name: Team name to check requests for
        set_app_state: Function to update app state

    Returns:
        List of processed sandbox permission requests
    """
    if not is_team_leader(team_name):
        return []

    try:
        messages = await read_mailbox(TEAM_LEAD_NAME, team_name)
    except Exception as e:
        logger.debug(f"Failed to read mailbox: {e}")
        return []

    sandbox_requests: List[TeammateMessage] = []
    for msg in messages:
        if msg and not msg.read:
            parsed = is_sandbox_permission_request_message(msg.text)
            if parsed:
                sandbox_requests.append(msg)

    if not sandbox_requests:
        return []

    logger.debug(f"Found {len(sandbox_requests)} sandbox permission request(s)")

    processed: List[ProcessedSandboxPermissionRequest] = []

    for msg in sandbox_requests:
        parsed = is_sandbox_permission_request_message(msg.text)
        if not parsed:
            continue

        # Validate hostPattern.host
        host_pattern = parsed.get("hostPattern", {})
        host = host_pattern.get("host")
        if not host:
            logger.debug("Invalid sandbox request: missing hostPattern.host")
            continue

        request = ProcessedSandboxPermissionRequest(
            request_id=parsed.get("requestId", ""),
            worker_id=parsed.get("workerId", ""),
            worker_name=parsed.get("workerName", msg.from_agent),
            worker_color=parsed.get("workerColor", msg.color),
            host=host,
            created_at=parsed.get("createdAt", int(time.time() * 1000)),
        )
        processed.append(request)

    # Mark messages as read
    await mark_messages_as_read(TEAM_LEAD_NAME, team_name)

    return processed


# =============================================================================
# Leader Polling Helper
# =============================================================================


async def check_leader_permission_requests(
    get_app_state: Callable[[], "AppState"],
    set_app_state: Callable,
    tools: List["Tool"],
    is_idle: bool = False,
) -> List[ProcessedPermissionRequest]:
    """Check for permission requests on the leader side.

    This is the main entry point for the leader to poll for
    worker permission requests. Call this periodically from
    the main loop.

    Args:
        get_app_state: Function to get current app state
        set_app_state: Function to update app state
        tools: Available tools
        is_idle: Whether the session is idle (not processing)

    Returns:
        List of processed permission requests
    """
    app_state = get_app_state()

    # Check if we're a team leader
    team_context = getattr(app_state, "team_context", None)
    if not team_context:
        return []

    team_name = getattr(team_context, "team_name", None)
    if not team_name:
        return []

    # Check if this session is the leader
    lead_agent_id = getattr(team_context, "lead_agent_id", None)
    current_agent_id = getattr(app_state, "agent_id", None)

    if current_agent_id and current_agent_id != lead_agent_id:
        # Not the leader
        return []

    # Process permission requests
    return await process_permission_requests(
        team_name=team_name,
        app_state=app_state,
        tools=tools,
        set_app_state=set_app_state,
    )


# =============================================================================
# Shutdown Request Processing
# =============================================================================


async def process_shutdown_requests(
    team_name: str,
    app_state: Optional["AppState"] = None,
    set_app_state: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Process pending shutdown requests from worker teammates.

    Args:
        team_name: Team name to check requests for
        app_state: Current app state
        set_app_state: Function to update app state

    Returns:
        List of processed shutdown requests
    """
    if not is_team_leader(team_name):
        return []

    try:
        messages = await read_mailbox(TEAM_LEAD_NAME, team_name)
    except Exception:
        return []

    shutdown_requests: List[Dict[str, Any]] = []

    for msg in messages:
        if msg and not msg.read:
            parsed = is_shutdown_request_message(msg.text)
            if parsed:
                shutdown_requests.append({
                    "from_agent": msg.from_agent,
                    "request": parsed,
                    "timestamp": msg.timestamp,
                })

    if shutdown_requests:
        await mark_messages_as_read(TEAM_LEAD_NAME, team_name)

    return shutdown_requests


__all__ = [
    "is_permission_request_message",
    "is_permission_response_message",
    "is_sandbox_permission_request_message",
    "is_shutdown_request_message",
    "ProcessedPermissionRequest",
    "ProcessedSandboxPermissionRequest",
    "process_permission_requests",
    "process_sandbox_permission_requests",
    "check_leader_permission_requests",
    "process_shutdown_requests",
]