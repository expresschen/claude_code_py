"""Team file management for agent swarms.

Team files store team configuration including members, lead agent ID,
and team-wide permission rules.

Directory structure:
    ~/.claude/teams/{team_name}/
        config.json           # TeamFile
        inboxes/              # Mailbox directory (managed by teammate_mailbox)
"""

from .team_file import (
    TeamFile,
    TeamMember,
    TeamAllowedPath,
    BackendType,
    is_pane_backend,
    get_teams_dir,
    get_team_dir,
    get_team_file_path,
    ensure_team_dir,
    read_team_file,
    read_team_file_async,
    write_team_file,
    write_team_file_async,
    add_member_to_team,
    remove_member_by_agent_id,
    remove_member_from_team,
    set_member_mode,
    set_member_active,
    set_member_active_async,
    sanitize_team_name,
    sanitize_agent_name,
    format_agent_id,
    dict_to_team_member,
    dict_to_team_file,
    TEAM_LEAD_NAME,
)

from .team_helpers import (
    cleanup_team_directories,
    cleanup_team_directories_sync,
    cleanup_session_teams,
    register_team_for_session_cleanup,
    unregister_team_for_session_cleanup,
    destroy_worktree,
)

__all__ = [
    # Constants
    "TEAM_LEAD_NAME",
    # Enums
    "BackendType",
    "is_pane_backend",
    # Types
    "TeamFile",
    "TeamMember",
    "TeamAllowedPath",
    # Path utilities
    "get_teams_dir",
    "get_team_dir",
    "get_team_file_path",
    "ensure_team_dir",
    # CRUD (sync)
    "read_team_file",
    "write_team_file",
    "add_member_to_team",
    "remove_member_by_agent_id",
    "remove_member_from_team",
    "set_member_mode",
    "set_member_active",
    # CRUD (async)
    "read_team_file_async",
    "write_team_file_async",
    "set_member_active_async",
    # Cleanup
    "cleanup_team_directories",
    "cleanup_team_directories_sync",
    "cleanup_session_teams",
    "register_team_for_session_cleanup",
    "unregister_team_for_session_cleanup",
    "destroy_worktree",
    # Sanitization
    "sanitize_team_name",
    "sanitize_agent_name",
    # ID formatting
    "format_agent_id",
    # Deserialization
    "dict_to_team_member",
    "dict_to_team_file",
]