"""Constants and enumerations.

This defines the QuerySource enum matching TypeScript's querySource.ts
with proper support for repl_main_thread and output style variants.
"""

from enum import Enum


class QuerySource(str, Enum):
    """Source of a query for analytics and behavior control.

    This matches the TypeScript QuerySource type defined in the constants.
    The repl_main_thread values are used for the main REPL conversation,
    with variants for different output styles.
    """

    # Main REPL thread queries
    REPL_MAIN_THREAD = "repl_main_thread"
    # Output style variants (prefix match)
    REPL_MAIN_THREAD_OUTPUT_STYLE_EXPLANATORY = "repl_main_thread:outputStyle:Explanatory"
    REPL_MAIN_THREAD_OUTPUT_STYLE_LEARNING = "repl_main_thread:outputStyle:Learning"
    REPL_MAIN_THREAD_OUTPUT_STYLE_CUSTOM = "repl_main_thread:outputStyle:custom"

    # SDK and CLI queries
    SDK = "sdk"
    CLI = "cli"

    # Remote/Bridge
    REMOTE = "remote"
    BRIDGE = "bridge"

    # Forked agents (subagents)
    AGENT_BUILTIN = "agent:builtin"
    AGENT_CUSTOM = "agent:custom"
    AGENT_DEFAULT = "agent:default"

    # Background tasks
    COMPACT = "compact"
    SESSION_MEMORY = "session_memory"
    PROMPT_SUGGESTION = "prompt_suggestion"
    AUTO_MODE = "auto_mode"
    SIDE_QUESTION = "side_question"
    MEMDIR_RELEVANCE = "memdir_relevance"
    MEMORY_EXTRACTION = "memory_extraction"
    VERIFICATION_AGENT = "verification_agent"
    HOOK_AGENT = "hook_agent"

    # Legacy (for compatibility)
    REPL = "repl"


def is_main_thread_source(query_source: str) -> bool:
    """Check if query source is from the main REPL thread.

    This implements prefix matching like TypeScript's isMainThreadSource():
    - 'repl_main_thread' (default style)
    - 'repl_main_thread:outputStyle:*' (non-default styles)
    - undefined/None (treated as main thread for backward compatibility)

    Args:
        query_source: Query source string or None

    Returns:
        True if this is a main thread source
    """
    if not query_source:
        return True  # undefined treated as main thread (backward compat)
    return query_source == QuerySource.REPL_MAIN_THREAD or query_source.startswith("repl_main_thread")


class PermissionMode(str, Enum):
    """Permission mode enumeration - re-exported from types for convenience."""

    DEFAULT = "default"
    AUTO = "auto"
    PLAN = "plan"
    BYPASS = "bypass"
    INTERACTIVE = "interactive"


class FeatureFlag(str, Enum):
    """Feature flags for conditional functionality."""

    MCP_SKILLS = "mcp_skills"
    HISTORY_SNIP = "history_snip"
    REACTIVE_COMPACT = "reactive_compact"
    CONTEXT_COLLAPSE = "context_collapse"
    TOKEN_BUDGET = "token_budget"
    VOICE_MODE = "voice_mode"
    BUDDY = "buddy"
    COORDINATOR_MODE = "coordinator_mode"
    CACHED_MICROCOMPACT = "cached_microcompact"


# Output styles (matching TypeScript OUTPUT_STYLE_CONFIG)
class OutputStyle(str, Enum):
    """Output style for response formatting."""

    DEFAULT = "default"
    EXPLANATORY = "Explanatory"
    LEARNING = "Learning"
    CUSTOM = "custom"


# XML-style tags used in message content
COMMAND_NAME_TAG = "local-command-name"
TICK_TAG = "tick"
LOCAL_COMMAND_STDOUT_TAG = "local-command-stdout"
LOCAL_COMMAND_STDERR_TAG = "local-command-stderr"

# Tool limits
TOOL_SUMMARY_MAX_LENGTH = 100
MAX_TOOL_RESULT_SIZE_CHARS = 25000
PREVIEW_SIZE_BYTES = 4096

# Concurrency limits
DEFAULT_MAX_TOOL_USE_CONCURRENCY = 10