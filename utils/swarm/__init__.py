"""Swarm utilities for in-process teammates.

Ported from: src/utils/swarm/
"""

from .permission_bridge import (
    register_leader_permission_queue,
    unregister_leader_permission_queue,
    register_leader_permission_context_setter,
    unregister_leader_permission_context_setter,
    get_leader_permission_queue,
    get_leader_permission_context_setter,
    PermissionQueueItem,
    WorkerBadge,
    enqueue_permission_request,
)

from .constants import (
    TEAM_LEAD_NAME,
    INBOX_POLL_INTERVAL_S,
    PERMISSION_POLL_INTERVAL_S,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_KILLED,
    BACKEND_IN_PROCESS,
    BACKEND_TMUX,
    BACKEND_ITERM2,
    LOCK_OPTIONS,
    is_agent_teams_enabled,
)

# Backward compatibility aliases
IN_PROCESS_BACKEND_TYPE = BACKEND_IN_PROCESS
POLL_INTERVAL_MS = int(INBOX_POLL_INTERVAL_S * 1000)

from .in_process_runner import (
    InProcessRunnerConfig,
    InProcessRunnerResult,
    run_in_process_teammate,
    start_in_process_teammate,
    send_idle_notification,
    try_claim_next_task,
    wait_for_next_prompt,
    format_teammate_xml,
    ProgressTracker,
    create_progress_tracker,
    update_progress_from_message,
    get_progress_update,
    WaitResult,
    update_task_state,
    append_teammate_message,
    check_and_compact,
    estimate_token_count,
    find_available_task,
    format_task_as_prompt,
)

from .spawn_in_process import (
    SpawnContext,
    generate_task_id,
    format_agent_id,
    set_session_id,
    get_session_id,
    generate_session_id,
    register_cleanup,
    run_cleanup_handlers,
    STOPPED_DISPLAY_MS,
)

# Import from task/manager to avoid duplication
from claude_code_py.task.manager import (
    spawn_in_process_teammate,
    InProcessSpawnConfig,
    InProcessSpawnOutput,
)

from .permission_sync import (
    PermissionRequest,
    PermissionRequestStatus,
    PermissionResolver,
    SwarmPermissionRequest,
    PermissionResolution,
    PermissionResponse,
    create_permission_request,
    write_permission_request,
    read_pending_permissions,
    read_resolved_permission,
    resolve_permission,
    send_permission_request_via_mailbox,
    send_permission_response_via_mailbox,
    is_swarm_worker,
    is_team_leader,
    generate_request_id,
    request_permission_from_leader,
)

from .inbox_poller import (
    InboxPollerConfig,
    InboxPoller,
    create_inbox_poller,
)

from .leader_permission_handler import (
    is_permission_request_message,
    is_permission_response_message,
    is_sandbox_permission_request_message,
    is_shutdown_request_message,
    ProcessedPermissionRequest,
    ProcessedSandboxPermissionRequest,
    process_permission_requests,
    process_sandbox_permission_requests,
    check_leader_permission_requests,
    process_shutdown_requests,
)


__all__ = [
    # Permission bridge
    "register_leader_permission_queue",
    "unregister_leader_permission_queue",
    "register_leader_permission_context_setter",
    "unregister_leader_permission_context_setter",
    "get_leader_permission_queue",
    "get_leader_permission_context_setter",
    "PermissionQueueItem",
    "WorkerBadge",
    "enqueue_permission_request",
    # Constants
    "TEAM_LEAD_NAME",
    "INBOX_POLL_INTERVAL_S",
    "PERMISSION_POLL_INTERVAL_S",
    "TASK_STATUS_COMPLETED",
    "TASK_STATUS_FAILED",
    "TASK_STATUS_KILLED",
    "BACKEND_IN_PROCESS",
    "BACKEND_TMUX",
    "BACKEND_ITERM2",
    "LOCK_OPTIONS",
    "is_agent_teams_enabled",
    # Backward compatibility
    "IN_PROCESS_BACKEND_TYPE",
    "POLL_INTERVAL_MS",
    # Runner
    "InProcessRunnerConfig",
    "InProcessRunnerResult",
    "run_in_process_teammate",
    "start_in_process_teammate",
    "send_idle_notification",
    "try_claim_next_task",
    "wait_for_next_prompt",
    "format_teammate_xml",
    "ProgressTracker",
    "create_progress_tracker",
    "update_progress_from_message",
    "get_progress_update",
    "WaitResult",
    "update_task_state",
    "append_teammate_message",
    "handle_permission_request",
    "check_and_compact",
    "estimate_token_count",
    "find_available_task",
    "format_task_as_prompt",
    # Spawn helpers (from spawn_in_process.py)
    "SpawnContext",
    "generate_task_id",
    "format_agent_id",
    "set_session_id",
    "get_session_id",
    "generate_session_id",
    "register_cleanup",
    "run_cleanup_handlers",
    "STOPPED_DISPLAY_MS",
    # Spawn core (from task/manager.py)
    "spawn_in_process_teammate",
    "InProcessSpawnConfig",
    "InProcessSpawnOutput",
    # Permission sync
    "PermissionRequest",
    "PermissionRequestStatus",
    "PermissionResolver",
    "SwarmPermissionRequest",
    "PermissionResolution",
    "PermissionResponse",
    "create_permission_request",
    "write_permission_request",
    "read_pending_permissions",
    "read_resolved_permission",
    "resolve_permission",
    "send_permission_request_via_mailbox",
    "send_permission_response_via_mailbox",
    "is_swarm_worker",
    "is_team_leader",
    "generate_request_id",
    "request_permission_from_leader",
    # Inbox poller
    "InboxPollerConfig",
    "InboxPoller",
    "create_inbox_poller",
    # Leader permission handler
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
