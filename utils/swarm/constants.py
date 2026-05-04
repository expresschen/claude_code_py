"""Swarm constants for team-based multi-agent.

Ported from: src/utils/swarm/constants.ts
"""

from __future__ import annotations

# Team lead name (used in agent IDs and mailbox)
TEAM_LEAD_NAME = "team-lead"

# Polling intervals (milliseconds -> seconds)
INBOX_POLL_INTERVAL_S = 0.5  # 500ms
PERMISSION_POLL_INTERVAL_S = 0.5  # 500ms

# Task notification status
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_KILLED = "killed"

# Backend types
BACKEND_IN_PROCESS = "in-process"
BACKEND_TMUX = "tmux"
BACKEND_ITERM2 = "iterm2"

# Lock options for file-based coordination
LOCK_OPTIONS = {
    "retries": 10,
    "min_timeout_ms": 5,
    "max_timeout_ms": 100,
}

# Experimental flag
def is_agent_teams_enabled() -> bool:
    """Check if agent teams feature is enabled."""
    import os
    return os.environ.get("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS", "").lower() in ("1", "true", "yes")

__all__ = [
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
]