"""Memory system module.

This implements the multi-layer file-based memory system:
- Auto Memory: User/project long-term memory
- Session Memory: Current session summary
- Agent Memory: Agent-specific persistent memory
- Team Memory: Shared team knowledge
"""

from .memdir import (
    ENTRYPOINT_NAME,
    MAX_ENTRYPOINT_LINES,
    MAX_ENTRYPOINT_BYTES,
    build_memory_prompt,
    build_memory_lines,
    truncate_entrypoint_content,
    ensure_memory_dir_exists,
)
from .paths import (
    get_auto_mem_path,
    is_auto_memory_enabled,
    is_auto_mem_path,
    get_memory_base,
    get_project_slug,
    get_session_memory_dir,
    get_session_memory_path,
)
from .memory_types import (
    MemoryType,
    MemoryFrontmatter,
    parse_memory_file,
)
from .session_memory import (
    SessionMemory,
    should_extract_memory,
    extract_session_memory,
    manually_extract_session_memory,
    init_session_memory,
    get_session_memory_content,
    is_session_memory_gate_enabled,
    wait_for_session_memory_extraction,
    MINIMUM_MESSAGE_TOKENS_TO_INIT,
    MINIMUM_TOKENS_BETWEEN_UPDATE,
    TOOL_CALLS_BETWEEN_UPDATES,
)
from .agent_memory import (
    AgentMemoryScope,
    get_agent_memory_dir,
    get_agent_memory_entrypoint,
    is_agent_memory_path,
    load_agent_memory_prompt,
)
from .find_relevant import (
    find_relevant_memories,
    get_memory_files_to_attachments,
    scan_memory_files,
    RelevantMemory,
    format_memory_manifest,
    collect_surfaced_memories,
    extract_recent_tools,
    SurfacedMemoriesInfo,
    MAX_RECENT_TOOLS,
    RECENT_MESSAGE_WINDOW,
)
from .extract import (
    execute_extract_memories,
    drain_pending_extraction,
    trigger_memory_extraction,
    build_extraction_user_prompt,
    EXTRACTION_SYSTEM_PROMPT_FALLBACK,
)
from .session_memory_prompts import (
    DEFAULT_SESSION_MEMORY_TEMPLATE,
    MAX_SECTION_LENGTH,
    MAX_TOTAL_SESSION_MEMORY_TOKENS,
    load_session_memory_template,
    get_default_update_prompt,
    load_session_memory_prompt,
    substitute_variables,
    analyze_section_sizes,
    generate_section_reminders,
    build_session_memory_update_prompt,
    build_session_memory_init_prompt,
    is_session_memory_empty,
    truncate_session_memory_for_compact,
)

__all__ = [
    # Constants
    "ENTRYPOINT_NAME",
    "MAX_ENTRYPOINT_LINES",
    "MAX_ENTRYPOINT_BYTES",
    # Core functions
    "build_memory_prompt",
    "build_memory_lines",
    "truncate_entrypoint_content",
    "ensure_memory_dir_exists",
    # Paths
    "get_auto_mem_path",
    "is_auto_memory_enabled",
    "is_auto_mem_path",
    "get_memory_base",
    "get_project_slug",
    "get_session_memory_dir",
    "get_session_memory_path",
    # Types
    "MemoryType",
    "MemoryFrontmatter",
    "parse_memory_file",
    # Session Memory
    "SessionMemory",
    "should_extract_memory",
    "extract_session_memory",
    "manually_extract_session_memory",
    "init_session_memory",
    "get_session_memory_content",
    "is_session_memory_gate_enabled",
    "wait_for_session_memory_extraction",
    "MINIMUM_MESSAGE_TOKENS_TO_INIT",
    "MINIMUM_TOKENS_BETWEEN_UPDATE",
    "TOOL_CALLS_BETWEEN_UPDATES",
    # Agent Memory
    "AgentMemoryScope",
    "get_agent_memory_dir",
    "get_agent_memory_entrypoint",
    "is_agent_memory_path",
    "load_agent_memory_prompt",
    # Find Relevant
    "find_relevant_memories",
    "get_memory_files_to_attachments",
    "scan_memory_files",
    "RelevantMemory",
    "format_memory_manifest",
    "collect_surfaced_memories",
    "extract_recent_tools",
    "SurfacedMemoriesInfo",
    "MAX_RECENT_TOOLS",
    "RECENT_MESSAGE_WINDOW",
    # Extraction
    "execute_extract_memories",
    "drain_pending_extraction",
    "trigger_memory_extraction",
    "build_extraction_user_prompt",
    "EXTRACTION_SYSTEM_PROMPT_FALLBACK",
    # Session Memory Prompts
    "DEFAULT_SESSION_MEMORY_TEMPLATE",
    "MAX_SECTION_LENGTH",
    "MAX_TOTAL_SESSION_MEMORY_TOKENS",
    "load_session_memory_template",
    "get_default_update_prompt",
    "load_session_memory_prompt",
    "substitute_variables",
    "analyze_section_sizes",
    "generate_section_reminders",
    "build_session_memory_update_prompt",
    "build_session_memory_init_prompt",
    "is_session_memory_empty",
    "truncate_session_memory_for_compact",
]