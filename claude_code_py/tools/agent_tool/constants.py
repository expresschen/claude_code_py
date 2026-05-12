"""Agent types and constants."""

from __future__ import annotations

# Tool names
AGENT_TOOL_NAME = "Agent"
LEGACY_AGENT_TOOL_NAME = "Task"
VERIFICATION_AGENT_TYPE = "verification"

# Built-in agents that run once and return a report
ONE_SHOT_BUILTIN_AGENT_TYPES: frozenset[str] = frozenset([
    "Explore",
    "Plan",
])

# Agent status constants
AGENT_STATUS_COMPLETED = "completed"
AGENT_STATUS_ASYNC_LAUNCHED = "async_launched"
AGENT_STATUS_ERROR = "error"

# Teammate spawn parameters
TEAM_NAME_PARAM = "team_name"
NAME_PARAM = "name"
SPAWN_MODE_DEFAULT = "default"
SPAWN_MODE_PLAN = "plan"

__all__ = [
    "AGENT_TOOL_NAME",
    "LEGACY_AGENT_TOOL_NAME",
    "VERIFICATION_AGENT_TYPE",
    "ONE_SHOT_BUILTIN_AGENT_TYPES",
    "AGENT_STATUS_COMPLETED",
    "AGENT_STATUS_ASYNC_LAUNCHED",
    "AGENT_STATUS_ERROR",
    "TEAM_NAME_PARAM",
    "NAME_PARAM",
    "SPAWN_MODE_DEFAULT",
    "SPAWN_MODE_PLAN",
]