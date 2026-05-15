"""Agent tool for launching subagents.

This implements the core Agent tool from TypeScript AgentTool.tsx.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError
from .constants import (
    AGENT_TOOL_NAME,
    AGENT_STATUS_COMPLETED,
    AGENT_STATUS_ASYNC_LAUNCHED,
    TEAM_NAME_PARAM,
    NAME_PARAM,
    SPAWN_MODE_DEFAULT,
    SPAWN_MODE_PLAN,
)
from .prompt import get_prompt
from .types import AgentDefinition, get_active_agents_from_list
from .run_agent import run_agent, AgentRunConfig, AgentProgress
from .builtin_agents import get_built_in_agents


__all__ = [
    "AgentTool",
    "AgentInput",
    "AgentOutput",
    "agent_tool",
    "AGENT_TOOL_NAME",
    "get_built_in_agents",
    "get_all_agent_tools",
]


class AgentInput(BaseModel):
    """Input for Agent tool."""

    description: str = Field(
        description="A short (3-5 word) description of the task"
    )
    prompt: str = Field(
        description="The task for the agent to perform"
    )
    subagent_type: Optional[str] = Field(
        default=None,
        description="The type of specialized agent to use for this task"
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model override for this agent (sonnet, opus, haiku)"
    )
    run_in_background: Optional[bool] = Field(
        default=False,
        description="Set to true to run this agent in the background"
    )
    isolation: Optional[str] = Field(
        default=None,
        description="Isolation mode: 'worktree' creates a temporary git worktree"
    )
    cwd: Optional[str] = Field(
        default=None,
        description="Absolute path to run the agent in"
    )
    # Teammate spawn parameters
    team_name: Optional[str] = Field(
        default=None,
        description="Team name for teammate spawn"
    )
    name: Optional[str] = Field(
        default=None,
        description="Teammate name when spawning into team"
    )
    mode: Optional[str] = Field(
        default=None,
        description="Spawn mode: 'default' or 'plan'"
    )


@dataclass
class AgentOutput:
    """Output from Agent tool."""

    status: str
    agent_id: Optional[str] = None
    description: Optional[str] = None
    prompt: Optional[str] = None
    output: Optional[str] = None
    duration_ms: int = 0


class AgentTool(Tool[AgentInput, AgentOutput, dict[str, Any]]):
    """Tool for launching subagents."""

    name = AGENT_TOOL_NAME
    aliases = ["Task"]  # Legacy name
    input_schema = AgentInput
    max_result_size_chars = 100_000
    search_hint = "launch a specialized agent to handle complex tasks"

    def __init__(self):
        self._agent_definitions: Optional[list[AgentDefinition]] = None

    def get_agent_definitions(self) -> list[AgentDefinition]:
        """Get available agent definitions."""
        if self._agent_definitions is None:
            # Load built-in agents
            all_agents = get_built_in_agents()
            # Get active agents (priority ordering)
            self._agent_definitions = get_active_agents_from_list(all_agents)
        return self._agent_definitions

    async def call(
        self,
        args: AgentInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[AgentOutput]:
        """Launch the agent.

        Args:
            args: Agent arguments
            context: Execution context
            can_use_tool: Permission function
            parent_message: Parent message
            on_progress: Progress callback

        Returns:
            Tool result with agent output
        """
        # Recursion guard: prevent agent nesting.
        # When context.agent_id is set, we are already inside a subagent.
        # Allowing further Agent calls leads to unbounded cascading spawns.
        if context.agent_id is not None:
            raise ToolError(
                "Cannot spawn a subagent from within another subagent. "
                "Complete the current task using your own tools directly."
            )

        # Teammate spawn path - when team_name and name are provided
        if args.team_name and args.name:
            from .spawn_teammate import spawn_teammate, SpawnTeammateInput

            # Build spawn input
            spawn_input = SpawnTeammateInput(
                name=args.name,
                prompt=args.prompt,
                description=args.description,
                team_name=args.team_name,
                model=args.model,
                plan_mode_required=args.mode == SPAWN_MODE_PLAN,
                agent_type=args.subagent_type,
                tool_use_id=getattr(parent_message, "request_id", None) if hasattr(parent_message, "request_id") else None,
            )

            # Spawn teammate
            result = await spawn_teammate(spawn_input, context)

            if not result.success:
                raise ToolError(result.error or "Failed to spawn teammate")

            # Return async launched status
            return ToolResult(
                data=AgentOutput(
                    status=AGENT_STATUS_ASYNC_LAUNCHED,
                    agent_id=result.agent_id,
                    description=args.description,
                    prompt=args.prompt,
                )
            )

        # Normal subagent path (existing code)
        # Get agent definition
        agent_type = args.subagent_type or "general-purpose"
        agent_def = self._find_agent_definition(agent_type)

        if not agent_def:
            raise ToolError(f"Unknown agent type: {agent_type}")

        # Create agent ID
        import uuid
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"

        # Build system prompt
        system_prompt = self._build_system_prompt(agent_def)

        # Build run config
        # Use args.cwd if provided, otherwise inherit from context
        config = AgentRunConfig(
            agent_id=agent_id,
            agent_type=agent_type,
            prompt=args.prompt,
            description=args.description,
            model=args.model or agent_def.model,
            tools=agent_def.tools,
            disallowed_tools=agent_def.disallowed_tools,
            run_in_background=args.run_in_background or False,
            isolation=args.isolation,
            cwd=args.cwd or context.get_cwd(),  # Inherit cwd from context
            max_turns=agent_def.max_turns,
            system_prompt=system_prompt,
        )

        # Progress callback wrapper
        def progress_callback(progress: AgentProgress) -> None:
            if on_progress:
                on_progress({
                    "tool_use_id": agent_id,
                    "data": {
                        "type": "agent_progress",
                        "agent_id": progress.agent_id,
                        "message": progress.message,
                        "tool_use": progress.tool_use,
                    }
                })

        # Run agent
        result = await run_agent(config, progress_callback)

        # Build output
        output = AgentOutput(
            status=result.status,
            agent_id=result.agent_id,
            description=args.description,
            prompt=args.prompt,
            output=result.output,
            duration_ms=result.duration_ms,
        )

        return ToolResult(data=output)

    def _find_agent_definition(self, agent_type: str) -> Optional[AgentDefinition]:
        """Find agent definition by type."""
        for agent in self.get_agent_definitions():
            if agent.agent_type == agent_type:
                return agent
        return None

    def _build_system_prompt(self, agent_def: AgentDefinition) -> str:
        """Build system prompt for agent."""
        # Get agent's system prompt
        if hasattr(agent_def, 'get_system_prompt') and agent_def.get_system_prompt:
            return agent_def.get_system_prompt()

        # Default system prompt
        return """You are an agent for Claude Code. Given the user's message, you should use the tools available to complete the task. Complete the task fully—don't gold-plate, but don't leave it half-done.

When you complete the task, respond with a concise report covering what was done and any key findings."""

    async def description(
        self,
        input: AgentInput,
        options: dict[str, Any],
    ) -> str:
        """Get description."""
        agent_type = input.subagent_type or "general-purpose"
        return f"Launch {agent_type} agent: {input.description}"

    async def prompt(self, options: dict[str, Any]) -> str:
        """Get tool prompt."""
        agents = self.get_agent_definitions()
        is_coordinator = options.get("is_coordinator", False)
        allowed_agent_types = options.get("allowed_agent_types")
        return get_prompt(agents, is_coordinator, allowed_agent_types)

    def is_concurrency_safe(self, input: AgentInput) -> bool:
        """Launching agents is concurrency safe."""
        return True

    def is_read_only(self, input: AgentInput) -> bool:
        """Launching agents is read-only from parent's perspective."""
        return True

    def user_facing_name(self, input: Optional[AgentInput]) -> str:
        """Get user-facing name."""
        if input:
            return input.description
        return "Agent"


# Create instance
agent_tool = AgentTool()


def get_all_agent_tools() -> list[Tool]:
    """Get all agent-related tools."""
    return [agent_tool]