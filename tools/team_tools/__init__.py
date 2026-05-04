"""Team Tools - TeamCreate and TeamDelete.

These tools manage multi-agent swarm teams.

Environment variable required: CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
"""

from .team_create import (
    TeamCreateTool,
    TeamCreateInput,
    TeamCreateOutput,
    is_agent_teams_enabled,
    assign_teammate_color,
    clear_teammate_colors,
)

from .team_delete import (
    TeamDeleteTool,
    TeamDeleteInput,
    TeamDeleteOutput,
    cleanup_team_directories,
)

__all__ = [
    # TeamCreate
    "TeamCreateTool",
    "TeamCreateInput",
    "TeamCreateOutput",
    # TeamDelete
    "TeamDeleteTool",
    "TeamDeleteInput",
    "TeamDeleteOutput",
    # Helpers
    "is_agent_teams_enabled",
    "assign_teammate_color",
    "clear_teammate_colors",
    "cleanup_team_directories",
]