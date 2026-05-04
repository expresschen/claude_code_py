"""Memory Extraction - Background agent for extracting memories.

This implements the forked agent pattern from extractMemories.ts
for automatically extracting durable memories from conversations.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from claude_code_py.memory import (
    is_auto_memory_enabled,
    get_auto_mem_path,
    build_memory_prompt,
    ensure_memory_dir_exists,
    scan_memory_files,
    format_memory_manifest,
    ENTRYPOINT_NAME,
)
from claude_code_py.memory.paths import is_auto_mem_path
from claude_code_py.core_types.message import Message, UserMessage, AssistantMessage
from claude_code_py.core_types.permissions import PermissionResult, PermissionBehavior
from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext, CanUseToolFn

# Import session memory compact integration
from claude_code_py.services.session_memory_compact import (
    mark_extraction_started,
    mark_extraction_complete,
)


# =============================================================================
# Constants
# =============================================================================

MAX_EXTRACTION_TURNS = 5
MIN_MESSAGES_FOR_EXTRACTION = 10


# =============================================================================
# System Prompt for Extraction (UNUSED - kept for fallback)
# =============================================================================
# NOTE: TypeScript uses user prompt for extraction instructions, not system prompt.
# This is kept for the case when no last_cache_params is available.

EXTRACTION_SYSTEM_PROMPT_FALLBACK = """You are Claude. Follow the user's instructions carefully."""

# =============================================================================
# Extraction Prompt Builder (matches TypeScript buildExtractAutoOnlyPrompt)
# =============================================================================


def build_extraction_user_prompt(
    new_message_count: int,
    existing_memories: str,
) -> str:
    """Build the extraction prompt for auto-only memory.

    Matches TypeScript's buildExtractAutoOnlyPrompt from prompts.ts.
    Extraction instructions are passed via user message, not system prompt,
    to preserve prompt cache sharing with the parent conversation.

    Args:
        new_message_count: Number of new messages since last extraction
        existing_memories: Manifest of existing memory files

    Returns:
        Complete extraction prompt for user message
    """
    manifest = ""
    if existing_memories and len(existing_memories) > 0:
        manifest = f"""

## Existing memory files

{existing_memories}

Check this list before writing — update an existing file rather than creating a duplicate."""

    return f"""You are now acting as the memory extraction subagent. Analyze the most recent ~{new_message_count} messages above and use them to update your persistent memory systems.

Available tools: Read, Grep, Glob, read-only Bash (ls/find/cat/stat/wc/head/tail and similar), and Edit/Write for paths inside the memory directory only. Bash rm is not permitted. All other tools — MCP, Agent, write-capable Bash, etc — will be denied.

You have a limited turn budget. Edit requires a prior Read of the same file, so the efficient strategy is: turn 1 — issue all Read calls in parallel for every file you might update; turn 2 — issue all Write/Edit calls in parallel. Do not interleave reads and writes across multiple turns.

You MUST only use content from the last ~{new_message_count} messages to update your persistent memories. Do not waste any turns attempting to investigate or verify that content further — no grepping source files, no reading code to confirm a pattern exists, no git commands.{manifest}

If the user explicitly asks you to remember something, save it immediately as whichever type fits best. If they ask you to forget something, find and remove the relevant entry.

## Memory Types

- **user**: User's role, goals, responsibilities, and knowledge. Save when you learn about the user's profile or preferences.
- **feedback**: Guidance about how to approach work — both what to avoid and what to keep doing. Save when user corrects your approach or confirms a non-obvious approach worked.
- **project**: Information about ongoing work, goals, initiatives. Save when you learn who is doing what, why, or by when.
- **reference**: Pointers to where information can be found in external systems. Save when you learn about resources in external systems.

## What NOT to save

- Code patterns, conventions, architecture, file paths, project structure
- Git history, recent changes, or who-changed-what
- Debugging solutions or fix recipes
- Anything already documented in CLAUDE.md files
- Ephemeral task details: in-progress work, temporary state, current conversation context

## How to save memories

Saving a memory is a two-step process:

**Step 1** — write the memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) using this frontmatter format:

---
name: {{memory name}}
description: {{one-line description — used to decide relevance}}
type: user | feedback | project | reference
---

{{memory content — for feedback/project types, structure as: rule/fact, then **Why:** and **How to apply:** lines}}

**Step 2** — add a pointer to that file in `MEMORY.md`. `MEMORY.md` is an index, not a memory — each entry should be one line, under ~150 characters: `- [Title](file.md) — one-line hook`. It has no frontmatter. Never write memory content directly into `MEMORY.md`.

- `MEMORY.md` is always loaded into your system prompt — lines after 200 will be truncated, so keep the index concise
- Organize memory semantically by topic, not chronologically
- Update or remove memories that turn out to be wrong or outdated
- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one."""


# =============================================================================
# Extraction State
# =============================================================================

@dataclass
class ExtractionState:
    """State for memory extraction tracking."""

    last_memory_message_uuid: Optional[str] = None
    in_progress: bool = False
    pending_context: Optional[dict[str, Any]] = None
    turns_since_last_extraction: int = 0


# Global state (module-level singleton)
_extraction_state = ExtractionState()


def is_extraction_in_progress() -> bool:
    """Check if extraction is currently in progress.

    Returns:
        True if extraction is running
    """
    return _extraction_state.in_progress


def get_extraction_state() -> ExtractionState:
    """Get the extraction state for external access.

    Returns:
        ExtractionState instance
    """
    return _extraction_state


# =============================================================================
# Tool Permissions for Auto Memory
# =============================================================================


def create_auto_mem_can_use_tool(memory_dir: str) -> CanUseToolFn:
    """Create a can_use_tool function for auto memory extraction.

    Allows:
    - Read, Glob, Grep (unrestricted - read-only)
    - Write/Edit only for paths within the auto-memory directory

    Args:
        memory_dir: Path to auto memory directory

    Returns:
        Permission check function
    """
    read_only_tools = {"Read", "Glob", "Grep"}

    async def can_use_tool(
        tool: Tool,
        input: Any,
        context: ToolUseContext,
        assistant_message: AssistantMessage,
        tool_use_id: Optional[str] = None,
        force_decision: Optional[str] = None,
    ) -> Any:
        """Permission check for auto memory extraction."""
        tool_name = getattr(tool, "name", "")

        # Allow Read/Glob/Grep unrestricted - all inherently read-only
        if tool_name in read_only_tools:
            return PermissionResult.allow(updated_input=input)

        # Allow Write/Edit only for auto-memory paths
        if tool_name in ("Write", "Edit"):
            if hasattr(input, "file_path"):
                file_path = str(input.file_path)
            elif isinstance(input, dict):
                file_path = input.get("file_path", "")
            else:
                file_path = ""

            # Use memory_dir directly for path check (avoid cwd mismatch)
            if file_path and str(file_path).startswith(memory_dir):
                return PermissionResult.allow(updated_input=input)

        # Deny everything else
        return PermissionResult.deny(
            reason=f"Only Read, Glob, Grep, and Write/Edit within {memory_dir} are allowed"
        )

    return can_use_tool


# =============================================================================
# Extract Written Paths from Agent Output
# =============================================================================


def extract_written_paths(agent_messages: list[Message]) -> list[str]:
    """Extract file paths that were written by the agent.

    Args:
        agent_messages: Messages from the forked agent

    Returns:
        List of unique written file paths
    """
    paths: list[str] = []

    for msg in agent_messages:
        if msg.type != "assistant":
            continue

        content = msg.message.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_name = block.get("name", "")
                if tool_name in ("Write", "Edit"):
                    tool_input = block.get("input", {})
                    if isinstance(tool_input, dict):
                        file_path = tool_input.get("file_path")
                        if file_path and isinstance(file_path, str):
                            paths.append(file_path)

    # Return unique paths
    return list(set(paths))


# =============================================================================
# Main Extraction Functions
# =============================================================================


async def execute_extract_memories(
    messages: list[Message],
    context: Optional[dict[str, Any]] = None,
    append_system_message: Optional[Callable[[Message], None]] = None,
) -> None:
    """Execute memory extraction from conversation.

    This runs in the background (fire-and-forget) and does not block
    the main conversation.

    Args:
        messages: Current conversation messages
        context: Optional context (tool use context, etc.)
        append_system_message: Optional callback to append system message
    """
    global _extraction_state

    # Check if enabled
    if not is_auto_memory_enabled():
        return

    # Skip if already in progress (stash for trailing run)
    if _extraction_state.in_progress:
        _extraction_state.pending_context = {
            "messages": messages,
            "context": context,
            "append_system_message": append_system_message,
        }
        return

    _extraction_state.in_progress = True

    try:
        await _run_extraction(messages, context, append_system_message)
    except Exception as e:
        import logging
        logging.debug(f"[extractMemories] error: {e}")
    finally:
        _extraction_state.in_progress = False

        # Run trailing extraction if stashed
        if _extraction_state.pending_context:
            pending = _extraction_state.pending_context
            _extraction_state.pending_context = None
            await execute_extract_memories(
                pending["messages"],
                pending.get("context"),
                pending.get("append_system_message"),
            )


async def _run_extraction(
    messages: list[Message],
    context: Optional[dict[str, Any]] = None,
    append_system_message: Optional[Callable[[Message], None]] = None,
) -> None:
    """Run the actual memory extraction.

    Args:
        messages: Conversation messages
        context: Optional tool use context
        append_system_message: Callback for system messages
    """
    global _extraction_state

    # Mark extraction started (for session memory compact)
    mark_extraction_started()

    # Check minimum messages
    new_message_count = _count_messages_since(
        messages,
        _extraction_state.last_memory_message_uuid,
    )

    if new_message_count < MIN_MESSAGES_FOR_EXTRACTION:
        mark_extraction_complete()
        return

    # Get cwd from context (needed for memory path checks)
    cwd = None
    if context and "tool_use_context" in context:
        tool_use_context = context["tool_use_context"]
        if hasattr(tool_use_context, "get_cwd"):
            cwd = tool_use_context.get_cwd()

    # Check if main agent already wrote memories (mutual exclusion)
    if _has_memory_writes_since(messages, _extraction_state.last_memory_message_uuid, cwd):
        # Advance cursor past this range
        if messages:
            _extraction_state.last_memory_message_uuid = messages[-1].uuid
        mark_extraction_complete()
        return

    # Get memory directory
    memory_dir = get_auto_mem_path(cwd)
    await ensure_memory_dir_exists(memory_dir)

    # Scan existing memories
    existing = scan_memory_files(memory_dir)
    existing_manifest = format_memory_manifest(existing)

    # Build extraction prompt (TypeScript style: instructions in user message)
    user_prompt = build_extraction_user_prompt(
        new_message_count=new_message_count,
        existing_memories=existing_manifest or "",
    )

    # Run extraction agent using forked agent pattern
    try:
        result = await _run_extraction_agent(
            user_prompt,
            memory_dir,
            messages,
            context,
        )

        # Advance cursor
        if messages:
            _extraction_state.last_memory_message_uuid = messages[-1].uuid

        # Extract written paths
        written_paths = extract_written_paths(result.get("messages", []))

        # Filter out MEMORY.md index file (the "memory" is the topic file itself)
        memory_paths = [
            p for p in written_paths
            if Path(p).name != ENTRYPOINT_NAME
        ]

        # Notify if memories were saved
        if memory_paths and append_system_message:
            from claude_code_py.core_types.message import SystemMessage
            msg = SystemMessage(
                content=f"Memory saved: {', '.join(memory_paths)}",
            )
            append_system_message(msg)

        # Mark extraction complete (for session memory compact)
        mark_extraction_complete()

    except Exception as e:
        import logging
        logging.debug(f"[extractMemories] extraction failed: {e}")
        # Still mark complete on failure
        mark_extraction_complete()


async def _run_extraction_agent(
    user_prompt: str,
    memory_dir: str,
    messages: list[Message],
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the extraction agent using forked agent pattern.

    Args:
        user_prompt: User prompt for extraction
        memory_dir: Memory directory path
        messages: Parent conversation messages
        context: Optional tool use context

    Returns:
        Dict with extraction results (messages, written paths)
    """
    from claude_code_py.utils.forked_agent import (
        run_forked_agent,
        ForkedAgentParams,
        CacheSafeParams,
        create_cache_safe_params,
        create_user_message,
        create_subagent_context,
        get_last_cache_safe_params,
    )

    # Get cache safe params - either from context or use last saved
    cache_params: Optional[CacheSafeParams] = None

    if context and "tool_use_context" in context:
        tool_use_context = context["tool_use_context"]

        # Try to get cache params from the last saved ones (for prompt cache sharing)
        last_cache_params = get_last_cache_safe_params()

        if last_cache_params:
            # Use the last saved cache params for prompt cache sharing
            # Keep system_prompt unchanged for cache sharing
            # Update the fork_context_messages to current messages
            cache_params = CacheSafeParams(
                system_prompt=last_cache_params.system_prompt,
                user_context=last_cache_params.user_context,
                system_context=last_cache_params.system_context,
                tool_use_context=last_cache_params.tool_use_context,
                fork_context_messages=messages,
            )
        else:
            # Create isolated context for extraction
            isolated_context = create_subagent_context(tool_use_context)

            # Build minimal cache params with fallback system prompt
            cache_params = CacheSafeParams(
                system_prompt=EXTRACTION_SYSTEM_PROMPT_FALLBACK,
                user_context={},
                system_context={},
                tool_use_context=isolated_context,
                fork_context_messages=[],
            )

    if not cache_params:
        # No context available - create minimal setup
        return {"messages": [], "written_paths": []}

    # Create permission handler
    can_use_tool = create_auto_mem_can_use_tool(memory_dir)

    # Create forked agent params
    fork_params = ForkedAgentParams(
        prompt_messages=[create_user_message(user_prompt)],
        cache_safe_params=cache_params,
        can_use_tool=can_use_tool,
        query_source="extract_memories",
        fork_label="extract_memories",
        max_turns=MAX_EXTRACTION_TURNS,
        skip_transcript=True,  # Don't record to transcript
    )

    # Run forked agent
    result = await run_forked_agent(fork_params)

    return {
        "messages": result.messages,
        "written_paths": extract_written_paths(result.messages),
        "usage": result.total_usage,
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _count_messages_since(
    messages: list[Message],
    since_uuid: Optional[str],
) -> int:
    """Count model-visible messages since a UUID.

    Args:
        messages: Message list
        since_uuid: UUID to count after

    Returns:
        Message count
    """
    if since_uuid is None:
        return sum(1 for m in messages if m.type in ("user", "assistant"))

    found = False
    count = 0
    for msg in messages:
        if not found:
            if msg.uuid == since_uuid:
                found = True
            continue
        if msg.type in ("user", "assistant"):
            count += 1

    # If since_uuid was not found (e.g., removed by compaction),
    # fall back to counting all model-visible messages
    if not found:
        return sum(1 for m in messages if m.type in ("user", "assistant"))

    return count


def _has_memory_writes_since(
    messages: list[Message],
    since_uuid: Optional[str],
    cwd: Optional[str] = None,
) -> bool:
    """Check if main agent wrote to memory files since UUID.

    Args:
        messages: Message list
        since_uuid: UUID to check after
        cwd: Working directory (defaults to current)

    Returns:
        True if memory writes detected
    """
    if not is_auto_memory_enabled():
        return False

    found = since_uuid is None
    memory_dir = get_auto_mem_path(cwd)

    for msg in messages:
        if not found:
            if msg.uuid == since_uuid:
                found = True
            continue

        if msg.type != "assistant":
            continue

        # Check for Write/Edit tool calls to memory paths
        content = getattr(msg, "message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_name = block.get("name", "")
                    if tool_name in ("Edit", "Write"):
                        tool_input = block.get("input", {})
                        file_path = tool_input.get("file_path", "")
                        if file_path and str(file_path).startswith(memory_dir):
                            return True

    return False


# =============================================================================
# Drain for Clean Shutdown
# =============================================================================


async def drain_pending_extraction(timeout_ms: int = 60000) -> None:
    """Wait for pending extractions to complete.

    Called during shutdown to ensure extractions finish.

    Args:
        timeout_ms: Maximum time to wait
    """
    global _extraction_state

    if not _extraction_state.in_progress:
        return

    # Wait with timeout
    start = asyncio.get_event_loop().time()
    while _extraction_state.in_progress:
        if (asyncio.get_event_loop().time() - start) * 1000 > timeout_ms:
            break
        await asyncio.sleep(0.1)


# =============================================================================
# Manual Trigger
# =============================================================================


async def trigger_memory_extraction(
    messages: list[Message],
) -> dict[str, Any]:
    """Manually trigger memory extraction.

    Use this when user explicitly asks to save memories.

    Args:
        messages: Conversation messages

    Returns:
        Extraction result
    """
    await execute_extract_memories(messages)
    return {"status": "extraction_triggered"}