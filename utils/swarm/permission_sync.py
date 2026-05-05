"""Permission Sync System for Agent Swarms.

This module provides infrastructure for coordinating permission prompts across
multiple agents in a swarm. When a worker agent needs permission for a tool use,
it forwards the request to the team leader, who can then approve or deny it.

The system uses the teammate mailbox for message passing:
- Workers send permission requests to the leader's mailbox
- Leaders send permission responses to the worker's mailbox

Ported from: src/utils/swarm/permissionSync.ts
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from claude_code_py.utils.team.team_file import (
    get_team_dir,
    read_team_file,
    read_team_file_async,
)
from claude_code_py.utils.teammate_context import (
    get_current_agent_id,
    get_current_agent_name,
    get_current_team_name,
    get_current_teammate_color,
)
from claude_code_py.utils.teammate_mailbox import (
    write_to_mailbox,
    TeammateMessage,
)


# =============================================================================
# Enums
# =============================================================================


class PermissionStatus(str, Enum):
    """Status value of a permission request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PermissionResolver(str, Enum):
    """Who resolved the permission request."""

    WORKER = "worker"
    LEADER = "leader"


# =============================================================================
# Simple PermissionRequest Dataclass (Mailbox Protocol)
# =============================================================================


@dataclass
class PermissionRequest:
    """A simple permission request for mailbox-based protocol.

    This is the lightweight message format used between workers and leaders.
    """

    id: str  # Unique request ID: "perm-{timestamp}-{random}"
    tool_name: str  # Tool requiring permission (e.g., "Bash", "Edit")
    tool_use_id: str  # Original toolUseID from worker's context
    input: Dict[str, Any]  # Serialized tool input
    description: str  # Human-readable description of the tool use
    team_name: str  # Team name for routing
    worker_id: str  # Worker's full agent ID
    worker_name: str  # Worker's display name
    worker_color: Optional[str] = None  # Worker's color for UI display
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    status: str = "pending"  # "pending", "approved", "denied"


# =============================================================================
# PermissionRequestStatus - In-Memory Pending Request Tracker
# =============================================================================


class PermissionRequestStatus:
    """Class for tracking pending permission requests in memory.

    Used by workers to track their pending requests while waiting for responses.
    """

    _pending_requests: Dict[str, PermissionRequest] = {}

    @classmethod
    def add(cls, request: PermissionRequest) -> None:
        """Add a request to the pending dict."""
        cls._pending_requests[request.id] = request

    @classmethod
    def get(cls, request_id: str) -> Optional[PermissionRequest]:
        """Get a pending request by ID."""
        return cls._pending_requests.get(request_id)

    @classmethod
    def remove(cls, request_id: str) -> None:
        """Remove a request from pending."""
        cls._pending_requests.pop(request_id, None)

    @classmethod
    def resolve(cls, request_id: str, approved: bool) -> Optional[PermissionRequest]:
        """Mark a request as resolved.

        Args:
            request_id: The request ID
            approved: True for approved, False for denied

        Returns:
            The resolved request, or None if not found
        """
        request = cls._pending_requests.get(request_id)
        if request:
            request.status = "approved" if approved else "denied"
            cls._pending_requests.pop(request_id, None)
        return request

    @classmethod
    def clear(cls) -> None:
        """Clear all pending requests."""
        cls._pending_requests.clear()

    @classmethod
    def all_pending(cls) -> List[PermissionRequest]:
        """Get all pending requests."""
        return list(cls._pending_requests.values())


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SwarmPermissionRequest:
    """A permission request from a worker to the leader.

    Stored in ~/.claude/teams/{team_name}/permissions/pending/{request_id}.json
    """

    # Required fields (no defaults) - must come first
    id: str  # Unique request ID: "perm-{timestamp}-{random}"
    worker_id: str  # Worker's CLAUDE_CODE_AGENT_ID
    worker_name: str  # Worker's CLAUDE_CODE_AGENT_NAME
    team_name: str  # Team name for routing

    # Permission details
    tool_name: str  # Tool requiring permission (e.g., "Bash", "Edit")
    tool_use_id: str  # Original toolUseID from worker's context
    description: str  # Human-readable description of the tool use

    # Optional fields (with defaults) - must come after required
    worker_color: Optional[str] = None  # Worker's CLAUDE_CODE_AGENT_COLOR
    input: Dict[str, Any] = field(default_factory=dict)  # Serialized tool input
    permission_suggestions: List[Any] = field(default_factory=list)

    # Status
    status: PermissionStatus = PermissionStatus.PENDING
    resolved_by: Optional[PermissionResolver] = None
    resolved_at: Optional[int] = None  # Unix timestamp (milliseconds)

    # Resolution data
    feedback: Optional[str] = None  # Rejection feedback message
    updated_input: Optional[Dict[str, Any]] = None  # Modified input if changed
    permission_updates: Optional[List[Any]] = None  # "Always allow" rules

    # Timestamp
    created_at: int = 0  # Unix timestamp (milliseconds)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "id": self.id,
            "workerId": self.worker_id,
            "workerName": self.worker_name,
            "workerColor": self.worker_color,
            "teamName": self.team_name,
            "toolName": self.tool_name,
            "toolUseId": self.tool_use_id,
            "description": self.description,
            "input": self.input,
            "permissionSuggestions": self.permission_suggestions,
            "status": self.status.value,
            "resolvedBy": self.resolved_by.value if self.resolved_by else None,
            "resolvedAt": self.resolved_at,
            "feedback": self.feedback,
            "updatedInput": self.updated_input,
            "permissionUpdates": self.permission_updates,
            "createdAt": self.created_at,
        }


@dataclass
class PermissionResolution:
    """Resolution data returned when leader/worker resolves a request."""

    decision: str  # "approved" or "rejected"
    resolved_by: str  # "worker" or "leader"
    feedback: Optional[str] = None
    updated_input: Optional[Dict[str, Any]] = None
    permission_updates: Optional[List[Any]] = None


@dataclass
class PermissionResponse:
    """Legacy response type for worker polling."""

    request_id: str
    decision: str
    timestamp: str
    feedback: Optional[str] = None
    updated_input: Optional[Dict[str, Any]] = None
    permission_updates: Optional[List[Any]] = None


# =============================================================================
# Path Helpers
# =============================================================================


def get_permission_dir(team_name: str) -> Path:
    """Get the base directory for a team's permission requests."""
    return Path(get_team_dir(team_name)) / "permissions"


def get_pending_dir(team_name: str) -> Path:
    """Get the pending directory for a team."""
    return get_permission_dir(team_name) / "pending"


def get_resolved_dir(team_name: str) -> Path:
    """Get the resolved directory for a team."""
    return get_permission_dir(team_name) / "resolved"


def get_pending_request_path(team_name: str, request_id: str) -> Path:
    """Get the path to a pending request file."""
    return get_pending_dir(team_name) / f"{request_id}.json"


def get_resolved_request_path(team_name: str, request_id: str) -> Path:
    """Get the path to a resolved request file."""
    return get_resolved_dir(team_name) / f"{request_id}.json"


async def ensure_permission_dirs_async(team_name: str) -> None:
    """Ensure the permissions directory structure exists."""
    for dir_path in [
        get_permission_dir(team_name),
        get_pending_dir(team_name),
        get_resolved_dir(team_name),
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


def ensure_permission_dirs_sync(team_name: str) -> None:
    """Ensure the permissions directory structure exists (sync)."""
    for dir_path in [
        get_permission_dir(team_name),
        get_pending_dir(team_name),
        get_resolved_dir(team_name),
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ID Generation
# =============================================================================


def generate_request_id() -> str:
    """Generate a unique request ID."""
    import random
    import string

    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=7))
    return f"perm-{int(time.time() * 1000)}-{random_str}"


def generate_sandbox_request_id() -> str:
    """Generate a unique sandbox permission request ID."""
    import random
    import string

    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=7))
    return f"sandbox-{int(time.time() * 1000)}-{random_str}"


# =============================================================================
# Create Permission Request
# =============================================================================


def create_permission_request(
    tool_name: str,
    tool_use_id: str,
    input: Dict[str, Any],
    description: str,
    team_name: str,
    worker_id: str,
    worker_name: str,
    worker_color: Optional[str] = None,
    *,
    permission_suggestions: Optional[List[Any]] = None,
) -> PermissionRequest:
    """Create a new PermissionRequest object for mailbox protocol.

    Args:
        tool_name: Tool requiring permission
        tool_use_id: Original toolUseID from worker's context
        input: Serialized tool input
        description: Human-readable description
        team_name: Team name for routing
        worker_id: Worker's full agent ID
        worker_name: Worker's display name
        worker_color: Optional worker's color for UI
        permission_suggestions: Optional permission suggestions (keyword-only)

    Returns:
        PermissionRequest for mailbox-based protocol
    """
    return PermissionRequest(
        id=generate_request_id(),
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        input=input,
        description=description,
        team_name=team_name,
        worker_id=worker_id,
        worker_name=worker_name,
        worker_color=worker_color,
    )


def create_swarm_permission_request(
    tool_name: str,
    tool_use_id: str,
    input: Dict[str, Any],
    description: str,
    team_name: Optional[str] = None,
    worker_id: Optional[str] = None,
    worker_name: Optional[str] = None,
    worker_color: Optional[str] = None,
    permission_suggestions: Optional[List[Any]] = None,
) -> SwarmPermissionRequest:
    """Create a SwarmPermissionRequest for file-based storage.

    This is the full permission request format with to_dict() serialization
    used for file-based storage and leader polling.

    Args:
        tool_name: Tool requiring permission
        tool_use_id: Original toolUseID from worker's context
        input: Serialized tool input
        description: Human-readable description
        team_name: Team name (defaults to current team)
        worker_id: Worker ID (defaults to current agent)
        worker_name: Worker name (defaults to current agent)
        worker_color: Worker color (defaults to current color)
        permission_suggestions: Permission suggestions

    Returns:
        SwarmPermissionRequest for file-based storage
    """
    resolved_team_name = team_name or get_current_team_name()
    resolved_worker_id = worker_id or get_current_agent_id()
    resolved_worker_name = worker_name or get_current_agent_name()
    resolved_worker_color = worker_color or get_current_teammate_color()

    if not resolved_team_name:
        raise ValueError("Team name is required for permission requests")
    if not resolved_worker_id:
        raise ValueError("Worker ID is required for permission requests")
    if not resolved_worker_name:
        raise ValueError("Worker name is required for permission requests")

    return SwarmPermissionRequest(
        id=generate_request_id(),
        worker_id=resolved_worker_id,
        worker_name=resolved_worker_name,
        worker_color=resolved_worker_color,
        team_name=resolved_team_name,
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        description=description,
        input=input,
        permission_suggestions=permission_suggestions or [],
        status=PermissionStatus.PENDING,
        created_at=int(time.time() * 1000),
    )


# =============================================================================
# CRUD Operations (Sync)
# =============================================================================


def write_permission_request_sync(request: SwarmPermissionRequest) -> bool:
    """Write a permission request to the pending directory."""
    ensure_permission_dirs_sync(request.team_name)

    pending_path = get_pending_request_path(request.team_name, request.id)
    lock_path = pending_path.with_suffix(".lock")

    try:
        while lock_path.exists():
            time.sleep(0.05)
        lock_path.write_text(str(os.getpid()))
        pending_path.write_text(json.dumps(request.to_dict(), indent=2))
        return True
    except Exception:
        return False
    finally:
        try:
            lock_path.unlink()
        except Exception:
            pass


def read_pending_permissions_sync(team_name: Optional[str] = None) -> List[SwarmPermissionRequest]:
    """Read all pending permission requests for a team."""
    resolved_team = team_name or get_current_team_name()
    if not resolved_team:
        return []

    pending_dir = get_pending_dir(resolved_team)

    try:
        json_files = [f for f in pending_dir.iterdir() if f.suffix == ".json" and f.name != ".lock"]
    except FileNotFoundError:
        return []

    requests = []
    for file_path in json_files:
        try:
            content = file_path.read_text()
            data = json.loads(content)
            requests.append(dict_to_permission_request(data))
        except Exception:
            continue

    requests.sort(key=lambda r: r.created_at)
    return requests


def read_resolved_permission_sync(
    request_id: str, team_name: Optional[str] = None
) -> Optional[SwarmPermissionRequest]:
    """Read a resolved permission request by ID."""
    resolved_team = team_name or get_current_team_name()
    if not resolved_team:
        return None

    resolved_path = get_resolved_request_path(resolved_team, request_id)

    try:
        content = resolved_path.read_text()
        data = json.loads(content)
        return dict_to_permission_request(data)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def resolve_permission_sync(
    request_id: str,
    resolution: PermissionResolution,
    team_name: Optional[str] = None,
) -> bool:
    """Resolve a permission request."""
    resolved_team = team_name or get_current_team_name()
    if not resolved_team:
        return False

    ensure_permission_dirs_sync(resolved_team)

    pending_path = get_pending_request_path(resolved_team, request_id)
    resolved_path = get_resolved_request_path(resolved_team, request_id)
    lock_path = pending_path.with_suffix(".lock")

    try:
        while lock_path.exists():
            time.sleep(0.05)
        lock_path.write_text(str(os.getpid()))

        try:
            content = pending_path.read_text()
        except FileNotFoundError:
            return False

        data = json.loads(content)
        request = dict_to_permission_request(data)

        request.status = (
            PermissionStatus.APPROVED
            if resolution.decision == "approved"
            else PermissionStatus.REJECTED
        )
        request.resolved_by = PermissionResolver(resolution.resolved_by)
        request.resolved_at = int(time.time() * 1000)
        request.feedback = resolution.feedback
        request.updated_input = resolution.updated_input
        request.permission_updates = resolution.permission_updates

        resolved_path.write_text(json.dumps(request.to_dict(), indent=2))
        pending_path.unlink()

        return True
    except Exception:
        return False
    finally:
        try:
            lock_path.unlink()
        except Exception:
            pass


def delete_resolved_permission_sync(request_id: str, team_name: Optional[str] = None) -> bool:
    """Delete a resolved permission file."""
    resolved_team = team_name or get_current_team_name()
    if not resolved_team:
        return False

    resolved_path = get_resolved_request_path(resolved_team, request_id)

    try:
        resolved_path.unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


# =============================================================================
# CRUD Operations (Async)
# =============================================================================


async def write_permission_request(request: SwarmPermissionRequest) -> SwarmPermissionRequest:
    """Write a permission request to the pending directory."""
    await ensure_permission_dirs_async(request.team_name)

    pending_path = get_pending_request_path(request.team_name, request.id)
    lock_path = pending_path.with_suffix(".lock")

    loop = asyncio.get_event_loop()

    try:
        while lock_path.exists():
            await asyncio.sleep(0.05)

        await loop.run_in_executor(None, lambda: lock_path.write_text(str(os.getpid())))

        await loop.run_in_executor(
            None, lambda: pending_path.write_text(json.dumps(request.to_dict(), indent=2))
        )

        return request
    except Exception:
        raise
    finally:
        try:
            await loop.run_in_executor(None, lambda: lock_path.unlink())
        except Exception:
            pass


async def read_pending_permissions(team_name: Optional[str] = None) -> List[SwarmPermissionRequest]:
    """Read all pending permission requests for a team."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, read_pending_permissions_sync, team_name)


async def read_resolved_permission(
    request_id: str, team_name: Optional[str] = None
) -> Optional[SwarmPermissionRequest]:
    """Read a resolved permission request by ID."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, read_resolved_permission_sync, request_id, team_name)


async def resolve_permission(
    request_id: str,
    resolution: PermissionResolution,
    team_name: Optional[str] = None,
) -> bool:
    """Resolve a permission request."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, resolve_permission_sync, request_id, resolution, team_name)


async def delete_resolved_permission(request_id: str, team_name: Optional[str] = None) -> bool:
    """Delete a resolved permission file."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, delete_resolved_permission_sync, request_id, team_name)


# =============================================================================
# Mailbox-Based Permission System
# =============================================================================


async def get_leader_name(team_name: Optional[str] = None) -> Optional[str]:
    """Get the leader's name from the team file."""
    resolved_team = team_name or get_current_team_name()
    if not resolved_team:
        return None

    team_file = await read_team_file_async(resolved_team)
    if not team_file:
        return None

    for member in team_file.members:
        if member.agent_id == team_file.lead_agent_id:
            return member.name

    return "team-lead"


async def send_permission_request_via_mailbox(request: PermissionRequest) -> None:
    """Send a permission request to the leader via mailbox.

    Args:
        request: PermissionRequest to send
    """
    from claude_code_py.utils.swarm.constants import TEAM_LEAD_NAME

    message_data = {
        "type": "permission_request",
        "request_id": request.id,
        "agent_id": request.worker_name,
        "tool_name": request.tool_name,
        "tool_use_id": request.tool_use_id,
        "description": request.description,
        "input": request.input,
    }

    await write_to_mailbox(
        TEAM_LEAD_NAME,
        TeammateMessage(
            from_agent=request.worker_name,
            text=json.dumps(message_data),
            timestamp=datetime.now().isoformat(),
            color=request.worker_color,
        ),
        request.team_name,
    )


async def send_permission_response_via_mailbox(
    request_id: str,
    team_name: str,
    recipient_name: str,
    approved: bool,
    updated_input: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Send a permission response to a worker via mailbox.

    Args:
        request_id: The request ID being responded to
        team_name: Team name
        recipient_name: Worker name to send response to
        approved: Whether permission was approved
        updated_input: Optional modified input
        error: Optional error message for denials
    """
    from claude_code_py.utils.swarm.constants import TEAM_LEAD_NAME

    message_data = {
        "type": "permission_response",
        "request_id": request_id,
        "approved": approved,
        "updated_input": updated_input,
        "error": error,
    }

    await write_to_mailbox(
        recipient_name,
        TeammateMessage(
            from_agent=TEAM_LEAD_NAME,
            text=json.dumps(message_data),
            timestamp=datetime.now().isoformat(),
        ),
        team_name,
    )


# =============================================================================
# Sandbox Permission Mailbox System
# =============================================================================


async def send_sandbox_permission_request_via_mailbox(
    host: str,
    request_id: str,
    team_name: Optional[str] = None,
) -> bool:
    """Send a sandbox permission request to the leader via mailbox."""
    resolved_team = team_name or get_current_team_name()
    if not resolved_team:
        return False

    leader_name = await get_leader_name(resolved_team)
    if not leader_name:
        return False

    worker_id = get_current_agent_id()
    worker_name = get_current_agent_name()
    worker_color = get_current_teammate_color()

    if not worker_id or not worker_name:
        return False

    message_data = {
        "type": "sandbox_permission_request",
        "requestId": request_id,
        "workerId": worker_id,
        "workerName": worker_name,
        "workerColor": worker_color,
        "hostPattern": {"host": host},
        "createdAt": int(time.time() * 1000),
    }

    await write_to_mailbox(
        leader_name,
        TeammateMessage(
            from_agent=worker_name,
            text=json.dumps(message_data),
            timestamp=datetime.now().isoformat(),
            color=worker_color,
        ),
        resolved_team,
    )

    return True


async def send_sandbox_permission_response_via_mailbox(
    worker_name: str,
    request_id: str,
    host: str,
    allow: bool,
    team_name: Optional[str] = None,
) -> bool:
    """Send a sandbox permission response to a worker via mailbox."""
    resolved_team = team_name or get_current_team_name()
    if not resolved_team:
        return False

    message_data = {
        "type": "sandbox_permission_response",
        "requestId": request_id,
        "host": host,
        "allow": allow,
        "timestamp": datetime.now().isoformat(),
    }

    sender_name = get_current_agent_name() or "team-lead"

    await write_to_mailbox(
        worker_name,
        TeammateMessage(
            from_agent=sender_name,
            text=json.dumps(message_data),
            timestamp=datetime.now().isoformat(),
        ),
        resolved_team,
    )

    return True


# =============================================================================
# Polling Functions
# =============================================================================


async def poll_for_response(
    request_id: str,
    team_name: Optional[str] = None,
) -> Optional[PermissionResponse]:
    """Poll for a permission response (worker-side convenience)."""
    resolved = await read_resolved_permission(request_id, team_name)
    if not resolved:
        return None

    timestamp = (
        datetime.fromtimestamp(resolved.resolved_at / 1000).isoformat()
        if resolved.resolved_at
        else datetime.fromtimestamp(resolved.created_at / 1000).isoformat()
    )

    return PermissionResponse(
        request_id=resolved.id,
        decision="approved" if resolved.status == PermissionStatus.APPROVED else "denied",
        timestamp=timestamp,
        feedback=resolved.feedback,
        updated_input=resolved.updated_input,
        permission_updates=resolved.permission_updates,
    )


async def remove_worker_response(request_id: str, team_name: Optional[str] = None) -> None:
    """Remove a worker's response after processing."""
    await delete_resolved_permission(request_id, team_name)


# =============================================================================
# Cleanup Functions
# =============================================================================


async def cleanup_old_resolutions(
    team_name: Optional[str] = None,
    max_age_ms: int = 3600000,
) -> int:
    """Clean up old resolved permission files."""
    resolved_team = team_name or get_current_team_name()
    if not resolved_team:
        return 0

    resolved_dir = get_resolved_dir(resolved_team)

    try:
        json_files = [f for f in resolved_dir.iterdir() if f.suffix == ".json"]
    except FileNotFoundError:
        return 0

    now = int(time.time() * 1000)
    cleaned_count = 0

    for file_path in json_files:
        try:
            content = file_path.read_text()
            data = json.loads(content)
            request = dict_to_permission_request(data)

            resolved_at = request.resolved_at or request.created_at
            if now - resolved_at >= max_age_ms:
                file_path.unlink()
                cleaned_count += 1
        except Exception:
            try:
                file_path.unlink()
                cleaned_count += 1
            except Exception:
                pass

    return cleaned_count


# =============================================================================
# Deserialization
# =============================================================================


def dict_to_permission_request(data: Dict[str, Any]) -> SwarmPermissionRequest:
    """Convert a dict to a SwarmPermissionRequest."""
    status_str = data.get("status", "pending")
    try:
        status = PermissionStatus(status_str)
    except ValueError:
        status = PermissionStatus.PENDING

    resolved_by_str = data.get("resolvedBy")
    resolved_by = None
    if resolved_by_str:
        try:
            resolved_by = PermissionResolver(resolved_by_str)
        except ValueError:
            pass

    return SwarmPermissionRequest(
        id=data.get("id", ""),
        worker_id=data.get("workerId", ""),
        worker_name=data.get("workerName", ""),
        worker_color=data.get("workerColor"),
        team_name=data.get("teamName", ""),
        tool_name=data.get("toolName", ""),
        tool_use_id=data.get("toolUseId", ""),
        description=data.get("description", ""),
        input=data.get("input", {}),
        permission_suggestions=data.get("permissionSuggestions", []),
        status=status,
        resolved_by=resolved_by,
        resolved_at=data.get("resolvedAt"),
        feedback=data.get("feedback"),
        updated_input=data.get("updatedInput"),
        permission_updates=data.get("permissionUpdates"),
        created_at=data.get("createdAt", 0),
    )


# =============================================================================
# Utility Functions
# =============================================================================


def is_team_leader(team_name: Optional[str] = None) -> bool:
    """Check if the current agent is a team leader."""
    resolved_team = team_name or get_current_team_name()
    if not resolved_team:
        return False

    agent_id = get_current_agent_id()
    return not agent_id or agent_id == "team-lead"


def is_swarm_worker() -> bool:
    """Check if the current agent is a worker in a swarm."""
    team_name = get_current_team_name()
    agent_id = get_current_agent_id()

    return bool(team_name) and bool(agent_id) and not is_team_leader(team_name)


# =============================================================================
# Worker Permission Request (Bubble to Leader)
# =============================================================================


async def request_permission_from_leader(
    tool_name: str,
    tool_use_id: str,
    tool_input: Dict[str, Any],
    description: str,
    timeout_ms: int = 60000,
) -> Dict[str, Any]:
    """Request permission from leader for a dangerous operation.

    This is the main entry point for swarm workers to bubble permission
    requests to the team leader. Tries Bridge path (direct UI queue) first,
    falls back to mailbox system when Bridge is unavailable.

    Args:
        tool_name: Name of the tool requiring permission
        tool_use_id: Unique ID for this tool use
        tool_input: The tool input parameters
        description: Human-readable description of the operation
        timeout_ms: Maximum time to wait for response (default 60s)

    Returns:
        Dict with keys:
            - "behavior": "allow" or "reject"
            - "updated_input": Modified input if allowed
            - "message": Error message if rejected
            - "timeout": True if timed out
    """
    from claude_code_py.utils.teammate_mailbox import (
        read_mailbox,
        mark_messages_as_read,
        TeammateMessage,
    )
    from claude_code_py.utils.swarm.permission_bridge import (
        get_leader_permission_queue,
    )

    team_name = get_current_team_name()
    worker_id = get_current_agent_id()
    worker_name = get_current_agent_name()
    worker_color = get_current_teammate_color()

    if not team_name or not worker_id or not worker_name:
        return {
            "behavior": "reject",
            "message": "Not in a swarm context",
            "timeout": False,
        }

    # Create permission request (shared between bridge and mailbox paths)
    request = create_permission_request(
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        input=tool_input,
        description=description,
        team_name=team_name,
        worker_id=worker_id,
        worker_name=worker_name,
        worker_color=worker_color,
    )

    # Track pending request
    PermissionRequestStatus.add(request)

    # ─────────────────────────────────────────────────────
    # Path 1: Bridge (direct UI queue)
    # ─────────────────────────────────────────────────────
    queue_setter = get_leader_permission_queue()

    if queue_setter:
        # Create future for async callback resolution
        response_future: asyncio.Future = asyncio.get_event_loop().create_future()

        def on_allow(updated_input=None, permission_updates=None, feedback=None):
            if not response_future.done():
                asyncio.create_task(
                    send_permission_response_via_mailbox(
                        request_id=request.id,
                        team_name=team_name,
                        recipient_name=worker_name,
                        approved=True,
                        updated_input=updated_input or tool_input,
                    )
                )
                response_future.set_result({
                    "behavior": "allow",
                    "updated_input": updated_input or tool_input,
                    "message": None,
                    "timeout": False,
                })

        def on_reject(feedback=None):
            if not response_future.done():
                asyncio.create_task(
                    send_permission_response_via_mailbox(
                        request_id=request.id,
                        team_name=team_name,
                        recipient_name=worker_name,
                        approved=False,
                        error=feedback or "Permission denied",
                    )
                )
                response_future.set_result({
                    "behavior": "reject",
                    "message": feedback or "Permission denied",
                    "timeout": False,
                })

        def on_abort():
            if not response_future.done():
                response_future.set_result({
                    "behavior": "reject",
                    "message": "Permission request aborted",
                    "timeout": False,
                })

        # Enqueue to Leader UI — unified dict format (same shape as mailbox path)
        pending_item = {
            "id": f"perm-bridge-{request.id}",
            "request_id": request.id,
            "from_agent": worker_name,
            "team_name": team_name,
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "description": description,
            "input": tool_input,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "_bridge_callbacks": {
                "on_allow": on_allow,
                "on_reject": on_reject,
                "on_abort": on_abort,
            },
        }
        queue_setter(lambda prev: prev + [pending_item])

        # Wait for response via future (Bridge callbacks set the future)
        try:
            result = await asyncio.wait_for(
                response_future,
                timeout=timeout_ms / 1000,
            )
            PermissionRequestStatus.remove(request.id)
            return result
        except asyncio.TimeoutError:
            PermissionRequestStatus.remove(request.id)
            return {
                "behavior": "reject",
                "message": "Permission request timed out",
                "timeout": True,
            }

    # ─────────────────────────────────────────────────────
    # Path 2: Mailbox fallback (no Bridge available)
    # ─────────────────────────────────────────────────────
    await send_permission_request_via_mailbox(request)

    # Poll for response in teammate mailbox
    start_time = time.time()
    poll_interval = 0.5  # 500ms

    while (time.time() - start_time) * 1000 < timeout_ms:
        await asyncio.sleep(poll_interval)

        messages = await read_mailbox(worker_name, team_name)
        for msg in messages:
            if msg and not msg.read:
                try:
                    data = json.loads(msg.text)
                    if data.get("type") == "permission_response":
                        response_id = data.get("request_id")
                        if response_id == request.id:
                            await mark_messages_as_read(worker_name, team_name)
                            PermissionRequestStatus.remove(request.id)

                            approved = data.get("approved", False)
                            if approved:
                                return {
                                    "behavior": "allow",
                                    "updated_input": data.get("updated_input", tool_input),
                                    "message": None,
                                    "timeout": False,
                                }
                            else:
                                return {
                                    "behavior": "reject",
                                    "message": data.get("error", "Permission denied by leader"),
                                    "timeout": False,
                                }
                except json.JSONDecodeError:
                    continue

    # Timeout - no response received
    PermissionRequestStatus.remove(request.id)
    return {
        "behavior": "reject",
        "message": "Permission request timed out",
        "timeout": True,
    }


__all__ = [
    "PermissionRequest",
    "PermissionRequestStatus",
    "PermissionStatus",
    "PermissionResolver",
    "SwarmPermissionRequest",
    "PermissionResolution",
    "PermissionResponse",
    "get_permission_dir",
    "get_pending_dir",
    "get_resolved_dir",
    "get_pending_request_path",
    "get_resolved_request_path",
    "ensure_permission_dirs_async",
    "ensure_permission_dirs_sync",
    "generate_request_id",
    "generate_sandbox_request_id",
    "create_permission_request",
    "create_swarm_permission_request",
    "write_permission_request_sync",
    "read_pending_permissions_sync",
    "read_resolved_permission_sync",
    "resolve_permission_sync",
    "delete_resolved_permission_sync",
    "write_permission_request",
    "read_pending_permissions",
    "read_resolved_permission",
    "resolve_permission",
    "delete_resolved_permission",
    "get_leader_name",
    "send_permission_request_via_mailbox",
    "send_permission_response_via_mailbox",
    "send_sandbox_permission_request_via_mailbox",
    "send_sandbox_permission_response_via_mailbox",
    "poll_for_response",
    "remove_worker_response",
    "cleanup_old_resolutions",
    "dict_to_permission_request",
    "is_team_leader",
    "is_swarm_worker",
    "request_permission_from_leader",
]