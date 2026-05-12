"""
Claude Code Python Implementation

A Python reimplementation of the core architecture from claude-code.

Architecture layers:
    CLI Layer -> Query/Agent Engine -> Tool/Permission Layer -> Memory/Persistence Layer -> MCP/Remote Extension Layer
"""

__version__ = "0.1.0"

from .core_types import (
    AgentId,
    SessionId,
    TaskId,
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ProgressMessage,
    AttachmentMessage,
    PermissionMode,
    PermissionResult,
)
from .tool import Tool, ToolResult, build_tool
from .state import Store, AppState
from .engine import (
    QueryEngine,
    process_user_input,
    ProcessUserInputOptions,
    ProcessUserInputResult,
    PromptInputMode,
    QuerySource,
)
from .memory import (
    ENTRYPOINT_NAME,
    MAX_ENTRYPOINT_LINES,
    MAX_ENTRYPOINT_BYTES,
    build_memory_prompt,
    is_auto_memory_enabled,
    get_auto_mem_path,
    AgentMemoryScope,
    get_agent_memory_dir,
    SessionMemory,
    should_extract_memory,
    find_relevant_memories,
    RelevantMemory,
    execute_extract_memories,
    drain_pending_extraction,
)
from .storage import (
    SessionStorage,
    SessionMeta,
    LogOption,
    list_sessions,
    delete_session,
    agentic_session_search,
    simple_session_search,
)
from .utils.context import (
    TokenWarningState,
    TokenWarningLevel,
    calculate_token_warning_state,
    get_auto_compact_threshold,
    get_effective_context_window,
    rough_token_count_estimation,
    analyze_context,
)
from .services import (
    compact_conversation,
    auto_compact_if_needed,
    CompactResult,
    get_compact_stats,
)

__all__ = [
    # Types
    "AgentId",
    "SessionId",
    "TaskId",
    "Message",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ProgressMessage",
    "AttachmentMessage",
    "PermissionMode",
    "PermissionResult",
    # Tool
    "Tool",
    "ToolResult",
    "build_tool",
    # State
    "Store",
    "AppState",
    # Engine
    "QueryEngine",
    "process_user_input",
    "ProcessUserInputOptions",
    "ProcessUserInputResult",
    "PromptInputMode",
    "QuerySource",
    # Memory
    "ENTRYPOINT_NAME",
    "MAX_ENTRYPOINT_LINES",
    "MAX_ENTRYPOINT_BYTES",
    "build_memory_prompt",
    "is_auto_memory_enabled",
    "get_auto_mem_path",
    "AgentMemoryScope",
    "get_agent_memory_dir",
    "SessionMemory",
    "should_extract_memory",
    "find_relevant_memories",
    "RelevantMemory",
    "execute_extract_memories",
    "drain_pending_extraction",
    # Storage
    "SessionStorage",
    "SessionMeta",
    "LogOption",
    "list_sessions",
    "delete_session",
    "agentic_session_search",
    "simple_session_search",
    # Context Management
    "TokenWarningState",
    "TokenWarningLevel",
    "calculate_token_warning_state",
    "get_auto_compact_threshold",
    "get_effective_context_window",
    "rough_token_count_estimation",
    "analyze_context",
    # Compaction
    "compact_conversation",
    "auto_compact_if_needed",
    "CompactResult",
    "get_compact_stats",
]