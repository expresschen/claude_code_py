"""TeamCreate Tool - Create a multi-agent swarm team.

Creates a team file, task list directory, and registers the leader.

Environment variable: CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, Field

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError
from claude_code_py.utils.team.team_file import (
    write_team_file,
    read_team_file,
    get_team_file_path,
    ensure_team_dir,
    sanitize_team_name,
    TeamFile,
    TeamMember,
    BackendType,
)
from claude_code_py.utils.swarm.constants import TEAM_LEAD_NAME
from claude_code_py.utils.task.file_storage import ensure_tasks_dir

if TYPE_CHECKING:
    from claude_code_py.utils.swarm.inbox_poller import create_inbox_poller


# =============================================================================
# Experimental Flag Check
# =============================================================================


def is_agent_teams_enabled() -> bool:
    """Check if agent teams feature is enabled."""
    return os.environ.get("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS", "").lower() in ("1", "true", "yes")


# =============================================================================
# Input/Output Types
# =============================================================================


class TeamCreateInput(BaseModel):
    """Input for TeamCreate tool."""

    team_name: str = Field(description="Name for the new team to create")
    description: Optional[str] = Field(default=None, description="Team description/purpose")
    agent_type: Optional[str] = Field(default=None, description="Type/role of the team lead")


class TeamCreateOutput(BaseModel):
    """Output for TeamCreate tool."""

    team_name: str
    team_file_path: str
    lead_agent_id: str


# =============================================================================
# Color Assignment
# =============================================================================


_TEAM_COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
_color_index = 0

# Module-level poller reference (can't be stored in AppState - contains asyncio.Task)
_active_inbox_poller: Optional[Any] = None


def assign_teammate_color(agent_id: str) -> str:
    """Assign a color to a teammate."""
    global _color_index
    color = _TEAM_COLORS[_color_index % len(_TEAM_COLORS)]
    _color_index += 1
    return color


def clear_teammate_colors() -> None:
    """Clear color assignments."""
    global _color_index
    _color_index = 0


def get_active_inbox_poller() -> Optional[Any]:
    """Get the active inbox poller instance."""
    return _active_inbox_poller


def stop_active_inbox_poller() -> None:
    """Stop the active inbox poller if running."""
    global _active_inbox_poller
    if _active_inbox_poller:
        try:
            # Try async stop, but if no event loop running, just cancel
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_active_inbox_poller.stop())
            except RuntimeError:
                # No event loop running, just mark as not running
                _active_inbox_poller._running = False
        except Exception:
            pass
        _active_inbox_poller = None


# =============================================================================
# TeamCreate Tool
# =============================================================================


class TeamCreateTool(Tool[TeamCreateInput, TeamCreateOutput, Dict[str, Any]]):
    """Tool to create a multi-agent team."""

    name = "TeamCreate"
    description = "Create a new team for coordinating multiple agents"
    input_schema = TeamCreateInput

    def is_read_only(self, args: TeamCreateInput) -> bool:
        return False

    def is_concurrency_safe(self, args: TeamCreateInput) -> bool:
        return False

    def is_enabled(self) -> bool:
        """Only enabled when experimental flag is set."""
        return is_agent_teams_enabled()

    async def prompt(self, options: Dict[str, Any]) -> str:
        """Get tool prompt."""
        return """# TeamCreate

## When to Use

Use this tool proactively whenever:
- The user explicitly asks to use a team, swarm, or group of agents
- The user mentions wanting agents to work together, coordinate, or collaborate
- A task is complex enough that it would benefit from parallel work by multiple agents (e.g., building a full-stack feature with frontend and backend work, refactoring a codebase while keeping tests passing, implementing a multi-step project with research, planning, and coding phases)

When in doubt about whether a task warrants a team, prefer spawning a team.

## Choosing Agent Types for Teammates

When spawning teammates via the Agent tool, choose the `subagent_type` based on what tools the agent needs for its task. Each agent type has a different set of available tools — match the agent to the work:

- **Read-only agents** (e.g., Explore, Plan) cannot edit or write files. Only assign them research, search, or planning tasks. Never assign them implementation work.
- **Full-capability agents** (e.g., general-purpose) have access to all tools including file editing, writing, and bash. Use these for tasks that require making changes.
- **Custom agents** defined in `.claude/agents/` may have their own tool restrictions. Check their descriptions to understand what they can and cannot do.

Always review the agent type descriptions and their available tools listed in the Agent tool prompt before selecting a `subagent_type` for a teammate.

Create a new team to coordinate multiple agents working on a project. Teams have a 1:1 correspondence with task lists (Team = TaskList).

```
{
  "team_name": "my-project",
  "description": "Working on feature X"
}
```

This creates:
- A team file at `~/.claude/teams/{team-name}/config.json`
- A corresponding task list directory at `~/.claude/tasks/{team-name}/`

## Team Workflow

1. **Create a team** with TeamCreate - this creates both the team and its task list
2. **Create tasks** using the Task tools (TaskCreate, TaskList, etc.) - they automatically use the team's task list
3. **Spawn teammates** using the Agent tool with `team_name` and `name` parameters to create teammates that join the team
4. **Assign tasks** using TaskUpdate with `owner` to give tasks to idle teammates
5. **Teammates work on assigned tasks** and mark them completed via TaskUpdate
6. **Teammates go idle between turns** - after each turn, teammates automatically go idle and send a notification. IMPORTANT: Be patient with idle teammates! Don't comment on their idleness until it actually impacts your work.
7. **Shutdown your team** - when the task is completed, gracefully shut down your teammates via SendMessage with `message: {type: "shutdown_request"}`.

## Task Ownership

Tasks are assigned using TaskUpdate with the `owner` parameter. Any agent can set or change task ownership via TaskUpdate.

## Automatic Message Delivery

**IMPORTANT**: Messages from teammates are automatically delivered to you. You do NOT need to manually check your inbox.

When you spawn teammates:
- They will send you messages when they complete tasks or need help
- These messages appear automatically as new conversation turns (like user messages)
- If you're busy (mid-turn), messages are queued and delivered when your turn ends
- The UI shows a brief notification with the sender's name when messages are waiting

Messages will be delivered automatically.

When reporting on teammate messages, you do NOT need to quote the original message—it's already rendered to the user.

## Teammate Idle State

Teammates go idle after every turn—this is completely normal and expected. A teammate going idle immediately after sending you a message does NOT mean they are done or unavailable. Idle simply means they are waiting for input.

- **Idle teammates can receive messages.** Sending a message to an idle teammate wakes them up and they will process it normally.
- **Idle notifications are automatic.** The system sends an idle notification whenever a teammate's turn ends. You do not need to react to idle notifications unless you want to assign new work or send a follow-up message.
- **Do not treat idle as an error.** A teammate sending a message and then going idle is the normal flow—they sent their message and are now waiting for a response.
- **Peer DM visibility.** When a teammate sends a DM to another teammate, a brief summary is included in their idle notification. This gives you visibility into peer collaboration without the full message content. You do not need to respond to these summaries — they are informational.

## Discovering Team Members

Teammates can read the team config file to discover other team members:
- **Team config location**: `~/.claude/teams/{team-name}/config.json`

The config file contains a `members` array with each teammate's:
- `name`: Human-readable name (**always use this** for messaging and task assignment)
- `agentId`: Unique identifier (for reference only - do not use for communication)
- `agentType`: Role/type of the agent

**IMPORTANT**: Always refer to teammates by their NAME (e.g., "team-lead", "researcher", "tester"). Names are used for:
- `to` when sending messages
- Identifying task owners

Example of reading team config:
```
Use the Read tool to read ~/.claude/teams/{team-name}/config.json
```

## Task List Coordination

Teams share a task list that all teammates can access at `~/.claude/tasks/{team-name}/`.

Teammates should:
1. Check TaskList periodically, **especially after completing each task**, to find available work or see newly unblocked tasks
2. Claim unassigned, unblocked tasks with TaskUpdate (set `owner` to your name). **Prefer tasks in ID order** (lowest ID first) when multiple tasks are available, as earlier tasks often set up context for later ones
3. Create new tasks with `TaskCreate` when identifying additional work
4. Mark tasks as completed with `TaskUpdate` when done, then check TaskList for next work
5. Coordinate with other teammates by reading the task list status
6. If all available tasks are blocked, notify the team lead or help resolve blocking tasks

**IMPORTANT notes for communication with your team**:
- Do not use terminal tools to view your team's activity; always send a message to your teammates (and remember, refer to them by name).
- Your team cannot hear you if you do not use the SendMessage tool. Always send a message to your teammates if you are responding to them.
- Do NOT send structured JSON status messages like `{"type":"idle",...}` or `{"type":"task_completed",...}`. Just communicate in plain text when you need to message teammates.
- Use TaskUpdate to mark tasks completed.
- If you are an agent in the team, the system will automatically send idle notifications to the team lead when you stop.

## IMPORTANT: Output File Paths

When teammates write output files, they inherit the working directory (cwd) from the team context:
- The cwd is set when TeamCreate is called and comes from `context.get_cwd()`
- Teammates should write output files relative to this cwd, NOT to `~/.claude/tasks/`
- Specify relative paths like `./output.txt` or `docs/architecture.md` instead of absolute paths to internal directories

Example: If cwd is `/home/user/my-project`, a teammate should write to `./report.md` (creates `/home/user/my-project/report.md`), NOT to `~/.claude/tasks/team-name/report.md`.

## When to Shutdown Team

The team should be shutdown when all work is complete:
1. **Check TaskList** - All tasks should have `status: "completed"`
2. **Verify outputs** - Confirm the expected output files exist in the cwd
3. **Send shutdown_request** - Use SendMessage with `message: {type: "shutdown_request"}` to each teammate
4. **Wait for shutdown_response** - Teammates will respond with `{type: "shutdown_response", approve: true}`
5. **Call TeamDelete** - After all teammates confirm shutdown, remove the team

Do NOT shutdown prematurely. Ensure all tasks are truly complete before sending shutdown requests."""

    async def call(
        self,
        args: TeamCreateInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Any = None,
    ) -> TeamCreateOutput:
        """Create a new team."""
        set_app_state = context.set_app_state
        get_app_state = context.get_app_state

        # Check if already in a team
        app_state = get_app_state()
        team_context = app_state.team_context
        existing_team = team_context.get("teamName") if team_context else None

        if existing_team:
            raise ToolError(
                f"Already leading team '{existing_team}'. "
                "Use TeamDelete to end the current team before creating a new one."
            )

        # Generate unique team name if exists
        team_name = sanitize_team_name(args.team_name)
        if read_team_file(team_name):
            # Generate a random slug name
            import random
            import string
            random_slug = "".join(random.choices(string.ascii_lowercase, k=8))
            team_name = f"team-{random_slug}"

        # Ensure team directory
        ensure_team_dir(team_name)

        # Create task list
        ensure_tasks_dir(team_name)

        # Format lead agent ID
        lead_agent_id = f"{TEAM_LEAD_NAME}@{team_name}"

        # Get working directory from context
        cwd = context.get_cwd()

        # Create team file
        team_file = TeamFile(
            name=team_name,
            created_at=int(time.time() * 1000),
            lead_agent_id=lead_agent_id,
            description=args.description,
            members=[
                TeamMember(
                    agent_id=lead_agent_id,
                    name=TEAM_LEAD_NAME,
                    agent_type=args.agent_type or TEAM_LEAD_NAME,
                    color=assign_teammate_color(lead_agent_id),
                    joined_at=int(time.time() * 1000),
                    cwd=cwd,
                    backend_type=BackendType.IN_PROCESS,
                    is_active=True,
                )
            ],
        )

        write_team_file(team_name, team_file)

        # Set leader team name (module-level variable for task list resolution)
        from claude_code_py.utils.task.file_storage import set_leader_team_name
        set_leader_team_name(team_name)

        # Note: Inbox poller is started in REPL class when agent teams is enabled,
        # similar to TypeScript's useInboxPoller hook in REPL.tsx.
        # TeamCreate only creates team file and sets state.

        # Get team file path
        team_file_path = str(get_team_file_path(team_name))

        # Update AppState with team context
        team_context = {
            "teamName": team_name,
            "teamFilePath": team_file_path,
            "leadAgentId": lead_agent_id,
            "teammates": {
                lead_agent_id: {
                    "name": TEAM_LEAD_NAME,
                    "agentType": args.agent_type or TEAM_LEAD_NAME,
                    "color": assign_teammate_color(lead_agent_id),
                    "cwd": cwd,
                    "spawnedAt": int(time.time() * 1000),
                }
            },
        }
        set_app_state(lambda prev: replace(prev, team_context=team_context))

        return ToolResult(data=TeamCreateOutput(
            team_name=team_name,
            team_file_path=team_file_path,
            lead_agent_id=lead_agent_id,
        ))


__all__ = [
    "TeamCreateTool",
    "TeamCreateInput",
    "TeamCreateOutput",
    "is_agent_teams_enabled",
    "assign_teammate_color",
    "clear_teammate_colors",
    "get_active_inbox_poller",
    "stop_active_inbox_poller",
]