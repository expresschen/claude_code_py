"""Utility modules for Claude Code."""

from .abort_controller import (
    AbortController,
    AbortSignal,
    AbortError,
    AbortControllerPair,
    create_abort_controller,
    create_abort_controller_pair,
    check_abort,
)
from .generators import all, merge_generators, async_generator_to_list
from .json_utils import json_parse, json_stringify

# Teammate context
from .teammate_context import (
    TeammateContext,
    get_teammate_context,
    run_with_teammate_context,
    run_with_teammate_context_async,
    is_in_process_teammate,
    create_teammate_context,
    format_agent_id,
    parse_agent_id,
    get_current_agent_id,
    get_current_agent_name,
    get_current_team_name,
    get_current_parent_session_id,
    is_team_lead,
    is_teammate,
    TEAM_LEAD_NAME,
)

# Teammate mailbox
from .teammate_mailbox import (
    TeammateMessage,
    IdleNotificationMessage,
    PermissionRequestMessage,
    PermissionResponseMessage,
    get_teams_dir,
    get_inbox_path,
    read_mailbox,
    read_unread_messages,
    write_to_mailbox,
    mark_messages_as_read,
    clear_mailbox,
    format_teammate_messages,
    create_idle_notification,
    is_idle_notification,
    is_permission_request,
    is_permission_response,
    is_structured_protocol_message,
    TEAMMATE_MESSAGE_TAG,
)

# Worktree utilities
from .worktree import (
    WorktreeSession,
    WorktreeCreateResult,
    validate_worktree_slug,
    worktree_branch_name,
    worktree_path_for,
    find_git_root,
    find_canonical_git_root,
    get_current_branch,
    get_default_branch,
    get_head_commit,
    get_or_create_worktree,
    create_agent_worktree,
    remove_agent_worktree,
    has_worktree_changes,
    create_worktree_for_session,
    get_current_worktree_session,
    restore_worktree_session,
    keep_worktree,
    cleanup_worktree,
    cleanup_stale_agent_worktrees,
)

# Context management
from .context import (
    TokenWarningState,
    TokenWarningLevel,
    calculate_token_warning_state,
    get_auto_compact_threshold,
    get_effective_context_window,
    get_context_window_for_model,
    rough_token_count_estimation,
    rough_token_count_estimation_for_messages,
    token_count_from_last_api_response,
    analyze_context,
    should_auto_compact,
    is_auto_compact_enabled,
)

# Side query for model-based operations
try:
    from .side_query import (
        side_query,
        SideQueryOptions,
        SideQueryResult,
        QuerySource,
        get_default_sonnet_model,
        get_default_haiku_model,
        get_small_fast_model,
        select_relevant_memories_with_model,
        SELECT_MEMORIES_SYSTEM_PROMPT,
    )
    _has_side_query = True
except ImportError:
    _has_side_query = False
    side_query = None  # type: ignore
    SideQueryOptions = None  # type: ignore
    SideQueryResult = None  # type: ignore
    QuerySource = None  # type: ignore
    get_default_sonnet_model = None  # type: ignore
    get_default_haiku_model = None  # type: ignore
    get_small_fast_model = None  # type: ignore
    select_relevant_memories_with_model = None  # type: ignore
    SELECT_MEMORIES_SYSTEM_PROMPT = None  # type: ignore

# Cache optimization
try:
    from .cache import (
        CacheScope,
        split_system_prompt_for_cache,
        add_cache_control_to_last_message,
        add_cache_control_to_tools,
        build_cached_api_request,
        estimate_cache_savings,
        CacheReferenceTracker,
        SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    )
    _has_cache = True
except ImportError:
    _has_cache = False

# Prompt category utilities
from .prompt_category import (
    get_query_source_for_repl,
    get_query_source_for_agent,
    is_foreground_query_source,
)

__all__ = [
    "AbortController",
    "AbortSignal",
    "AbortError",
    "AbortControllerPair",
    "create_abort_controller",
    "create_abort_controller_pair",
    "check_abort",
    "all",
    "merge_generators",
    "async_generator_to_list",
    "json_parse",
    "json_stringify",
    # Teammate Context
    "TeammateContext",
    "get_teammate_context",
    "run_with_teammate_context",
    "run_with_teammate_context_async",
    "is_in_process_teammate",
    "create_teammate_context",
    "format_agent_id",
    "parse_agent_id",
    "get_current_agent_id",
    "get_current_agent_name",
    "get_current_team_name",
    "get_current_parent_session_id",
    "is_team_lead",
    "is_teammate",
    "TEAM_LEAD_NAME",
    # Teammate Mailbox
    "TeammateMessage",
    "IdleNotificationMessage",
    "PermissionRequestMessage",
    "PermissionResponseMessage",
    "get_teams_dir",
    "get_inbox_path",
    "read_mailbox",
    "read_unread_messages",
    "write_to_mailbox",
    "mark_messages_as_read",
    "clear_mailbox",
    "format_teammate_messages",
    "create_idle_notification",
    "is_idle_notification",
    "is_permission_request",
    "is_permission_response",
    "is_structured_protocol_message",
    "TEAMMATE_MESSAGE_TAG",
    # Worktree
    "WorktreeSession",
    "WorktreeCreateResult",
    "validate_worktree_slug",
    "worktree_branch_name",
    "worktree_path_for",
    "find_git_root",
    "find_canonical_git_root",
    "get_current_branch",
    "get_default_branch",
    "get_head_commit",
    "get_or_create_worktree",
    "create_agent_worktree",
    "remove_agent_worktree",
    "has_worktree_changes",
    "create_worktree_for_session",
    "get_current_worktree_session",
    "restore_worktree_session",
    "keep_worktree",
    "cleanup_worktree",
    "cleanup_stale_agent_worktrees",
    # Context Management
    "TokenWarningState",
    "TokenWarningLevel",
    "calculate_token_warning_state",
    "get_auto_compact_threshold",
    "get_effective_context_window",
    "get_context_window_for_model",
    "rough_token_count_estimation",
    "rough_token_count_estimation_for_messages",
    "token_count_from_last_api_response",
    "analyze_context",
    "should_auto_compact",
    "is_auto_compact_enabled",
    # Side Query (conditional)
    "side_query",
    "SideQueryOptions",
    "SideQueryResult",
    "QuerySource",
    "get_default_sonnet_model",
    "get_default_haiku_model",
    "get_small_fast_model",
    "select_relevant_memories_with_model",
    "SELECT_MEMORIES_SYSTEM_PROMPT",
    # Cache Optimization (conditional)
    "CacheScope",
    "split_system_prompt_for_cache",
    "add_cache_control_to_last_message",
    "add_cache_control_to_tools",
    "build_cached_api_request",
    "estimate_cache_savings",
    "CacheReferenceTracker",
    "SYSTEM_PROMPT_DYNAMIC_BOUNDARY",
    # Prompt Category
    "get_query_source_for_repl",
    "get_query_source_for_agent",
    "is_foreground_query_source",
]