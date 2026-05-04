"""Agent definition types.

This mirrors the TypeScript AgentDefinition interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union


class AgentSource(str, Enum):
    """Source of agent definition."""

    BUILTIN = "built-in"
    USER_SETTINGS = "userSettings"
    PROJECT_SETTINGS = "projectSettings"
    POLICY_SETTINGS = "policySettings"
    FLAG_SETTINGS = "flagSettings"
    PLUGIN = "plugin"


class AgentMemoryScope(str, Enum):
    """Memory scope for agent."""

    USER = "user"
    PROJECT = "project"
    LOCAL = "local"


@dataclass
class BaseAgentDefinition:
    """Base type with common fields for all agents."""

    agent_type: str
    when_to_use: str
    tools: Optional[list[str]] = None
    disallowed_tools: Optional[list[str]] = None
    model: Optional[str] = None  # 'sonnet', 'opus', 'haiku', or 'inherit'
    permission_mode: Optional[str] = None
    max_turns: Optional[int] = None
    effort: Optional[Union[str, int]] = None
    required_mcp_servers: Optional[list[str]] = None
    mcp_servers: Optional[list[Union[str, dict[str, Any]]]] = None
    hooks: Optional[dict[str, Any]] = None
    skills: Optional[list[str]] = None
    background: Optional[bool] = None
    isolation: Optional[str] = None  # 'worktree' or 'remote'
    memory: Optional[AgentMemoryScope] = None
    color: Optional[str] = None


@dataclass
class BuiltInAgentDefinition(BaseAgentDefinition):
    """Built-in agents with dynamic prompts."""

    source: AgentSource = field(default=AgentSource.BUILTIN, init=False)
    base_dir: str = "built-in"
    callback: Optional[Callable[[], None]] = None
    get_system_prompt: Optional[Callable[[], str]] = None


@dataclass
class CustomAgentDefinition(BaseAgentDefinition):
    """Custom agents from user/project/policy settings."""

    get_system_prompt: Optional[Callable[[], str]] = None
    source: AgentSource = AgentSource.USER_SETTINGS
    base_dir: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class PluginAgentDefinition(BaseAgentDefinition):
    """Plugin agents with plugin metadata."""

    get_system_prompt: Optional[Callable[[], str]] = None
    source: AgentSource = field(default=AgentSource.PLUGIN, init=False)
    filename: Optional[str] = None


# Union type for all agent types
AgentDefinition = Union[BuiltInAgentDefinition, CustomAgentDefinition, PluginAgentDefinition]


@dataclass
class AgentDefinitionsResult:
    """Result of loading agent definitions."""

    active_agents: list[AgentDefinition]
    all_agents: list[AgentDefinition]
    failed_files: Optional[list[dict[str, str]]] = None
    allowed_agent_types: Optional[list[str]] = None


# Type guards
def is_built_in_agent(agent: AgentDefinition) -> bool:
    """Check if agent is built-in."""
    return isinstance(agent, BuiltInAgentDefinition)


def is_custom_agent(agent: AgentDefinition) -> bool:
    """Check if agent is custom (not built-in or plugin)."""
    return isinstance(agent, CustomAgentDefinition)


def is_plugin_agent(agent: AgentDefinition) -> bool:
    """Check if agent is plugin-provided."""
    return isinstance(agent, PluginAgentDefinition)


def get_active_agents_from_list(all_agents: list[AgentDefinition]) -> list[AgentDefinition]:
    """Get active agents prioritizing built-in > plugin > user > project > managed."""
    built_in_agents = [a for a in all_agents if is_built_in_agent(a)]
    plugin_agents = [a for a in all_agents if is_plugin_agent(a)]
    user_agents = [a for a in all_agents if a.source == AgentSource.USER_SETTINGS]
    project_agents = [a for a in all_agents if a.source == AgentSource.PROJECT_SETTINGS]
    managed_agents = [a for a in all_agents if a.source in (
        AgentSource.POLICY_SETTINGS,
        AgentSource.FLAG_SETTINGS,
    )]

    # Priority order: built-in, plugin, managed, user, project
    agent_groups = [
        built_in_agents,
        plugin_agents,
        managed_agents,
        user_agents,
        project_agents,
    ]

    agent_map: dict[str, AgentDefinition] = {}

    for agents in agent_groups:
        for agent in agents:
            # First occurrence wins (higher priority)
            if agent.agent_type not in agent_map:
                agent_map[agent.agent_type] = agent

    return list(agent_map.values())


def has_required_mcp_servers(
    agent: AgentDefinition,
    available_servers: list[str],
) -> bool:
    """Check if agent's required MCP servers are available."""
    if not agent.required_mcp_servers:
        return True
    return all(s in available_servers for s in agent.required_mcp_servers)


def filter_agents_by_mcp_requirements(
    agents: list[AgentDefinition],
    available_servers: list[str],
) -> list[AgentDefinition]:
    """Filter agents by MCP requirements."""
    return [a for a in agents if has_required_mcp_servers(a, available_servers)]


def get_tools_description(agent: AgentDefinition) -> str:
    """Get tools description for agent listing."""
    tools = agent.tools
    disallowed_tools = agent.disallowed_tools

    has_allowlist = tools and len(tools) > 0
    has_denylist = disallowed_tools and len(disallowed_tools) > 0

    if has_allowlist and has_denylist:
        # Both defined: filter allowlist by denylist
        deny_set = set(disallowed_tools)
        effective_tools = [t for t in tools if t not in deny_set]
        if not effective_tools:
            return "None"
        return ", ".join(effective_tools)
    elif has_allowlist:
        return ", ".join(tools)
    elif has_denylist:
        return f"All tools except {', '.join(disallowed_tools)}"
    else:
        return "All tools"


def format_agent_line(agent: AgentDefinition) -> str:
    """Format one agent line for listing: `- type: whenToUse (Tools: ...)`."""
    tools_desc = get_tools_description(agent)
    return f"- {agent.agent_type}: {agent.when_to_use} (Tools: {tools_desc})"