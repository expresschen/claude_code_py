"""Storage module for session persistence and search."""

from .session import (
    SessionStorage,
    SessionMeta,
    SessionLog,
    LogOption,
    AgentMetadata,
    WorktreeSession,
    get_session_dir,
    get_session_path,
    get_session_subagents_dir,
    get_agent_transcript_path,
    get_agent_metadata_path,
    get_session_env_dir,
    list_sessions,
    delete_session,
    extract_message_text,
    write_agent_metadata,
    read_agent_metadata,
    save_worktree_state,
    load_worktree_state,
    clear_worktree_state,
    switch_session,
)
from .session_search import (
    agentic_session_search,
    simple_session_search,
    SESSION_SEARCH_SYSTEM_PROMPT,
)

__all__ = [
    # Session Storage
    "SessionStorage",
    "SessionMeta",
    "SessionLog",
    "LogOption",
    "AgentMetadata",
    "WorktreeSession",
    "get_session_dir",
    "get_session_path",
    "get_session_subagents_dir",
    "get_agent_transcript_path",
    "get_agent_metadata_path",
    "get_session_env_dir",
    "list_sessions",
    "delete_session",
    "extract_message_text",
    "write_agent_metadata",
    "read_agent_metadata",
    "save_worktree_state",
    "load_worktree_state",
    "clear_worktree_state",
    "switch_session",
    # Session Search
    "agentic_session_search",
    "simple_session_search",
    "SESSION_SEARCH_SYSTEM_PROMPT",
]