from __future__ import annotations

"""Memory directory management.

This implements the core memory system from memdir.ts.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Constants
ENTRYPOINT_NAME = "MEMORY.md"
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000

# Guidance text
DIR_EXISTS_GUIDANCE = (
    "This directory already exists — write to it directly with the Write tool "
    "(do not run mkdir or check for its existence)."
)
DIRS_EXIST_GUIDANCE = (
    "Both directories already exist — write to them directly with the Write tool "
    "(do not run mkdir or check for their existence)."
)


@dataclass
class EntrypointTruncation:
    """Result of truncating MEMORY.md content."""

    content: str
    line_count: int
    byte_count: int
    was_line_truncated: bool
    was_byte_truncated: bool


def truncate_entrypoint_content(raw: str) -> EntrypointTruncation:
    """Truncate MEMORY.md content to line AND byte caps.

    Args:
        raw: Raw content to truncate

    Returns:
        Truncation result with metadata
    """
    trimmed = raw.strip()
    content_lines = trimmed.split("\n")
    line_count = len(content_lines)
    byte_count = len(trimmed)

    was_line_truncated = line_count > MAX_ENTRYPOINT_LINES
    was_byte_truncated = byte_count > MAX_ENTRYPOINT_BYTES

    if not was_line_truncated and not was_byte_truncated:
        return EntrypointTruncation(
            content=trimmed,
            line_count=line_count,
            byte_count=byte_count,
            was_line_truncated=False,
            was_byte_truncated=False,
        )

    # First truncate by lines
    truncated = (
        "\n".join(content_lines[:MAX_ENTRYPOINT_LINES])
        if was_line_truncated
        else trimmed
    )

    # Then truncate by bytes if needed
    if len(truncated) > MAX_ENTRYPOINT_BYTES:
        cut_at = truncated.rfind("\n", 0, MAX_ENTRYPOINT_BYTES)
        truncated = truncated[: cut_at if cut_at > 0 else MAX_ENTRYPOINT_BYTES]

    # Build warning message
    if was_byte_truncated and not was_line_truncated:
        reason = f"{_format_size(byte_count)} (limit: {_format_size(MAX_ENTRYPOINT_BYTES)}) — index entries are too long"
    elif was_line_truncated and not was_byte_truncated:
        reason = f"{line_count} lines (limit: {MAX_ENTRYPOINT_LINES})"
    else:
        reason = f"{line_count} lines and {_format_size(byte_count)}"

    return EntrypointTruncation(
        content=(
            truncated
            + f"\n\n> WARNING: {ENTRYPOINT_NAME} is {reason}. "
            "Only part of it was loaded. Keep index entries to one line under ~200 chars; "
            "move detail into topic files."
        ),
        line_count=line_count,
        byte_count=byte_count,
        was_line_truncated=was_line_truncated,
        was_byte_truncated=was_byte_truncated,
    )


def _format_size(bytes_count: int) -> str:
    """Format byte count as human-readable size."""
    if bytes_count < 1024:
        return f"{bytes_count}B"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    else:
        return f"{bytes_count / (1024 * 1024):.1f}MB"


async def ensure_memory_dir_exists(memory_dir: str) -> None:
    """Ensure a memory directory exists.

    Args:
        memory_dir: Path to memory directory
    """
    path = Path(memory_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Log for debugging but continue - the Write tool will handle it
        import logging
        logging.debug(f"ensure_memory_dir_exists failed for {memory_dir}: {e}")


def build_memory_lines(
    display_name: str,
    memory_dir: str,
    extra_guidelines: Optional[list[str]] = None,
    skip_index: bool = False,
) -> list[str]:
    """Build the typed-memory behavioral instructions.

    Args:
        display_name: Name to display in prompt
        memory_dir: Path to memory directory
        extra_guidelines: Additional guidelines to include
        skip_index: If True, skip the index step

    Returns:
        List of prompt lines
    """
    from .memory_types import (
        MEMORY_FRONTMATTER_EXAMPLE,
        TYPES_SECTION,
        WHAT_NOT_TO_SAVE_SECTION,
        WHEN_TO_ACCESS_SECTION,
        TRUSTING_RECALL_SECTION,
    )

    extra_guidelines = extra_guidelines or []

    # Build "how to save" section
    if skip_index:
        how_to_save = [
            "## How to save memories",
            "",
            "Write each memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) using this frontmatter format:",
            "",
            *MEMORY_FRONTMATTER_EXAMPLE,
            "",
            "- Keep the name, description, and type fields in memory files up-to-date with the content",
            "- Organize memory semantically by topic, not chronologically",
            "- Update or remove memories that turn out to be wrong or outdated",
            "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
        ]
    else:
        how_to_save = [
            "## How to save memories",
            "",
            "Saving a memory is a two-step process:",
            "",
            "**Step 1** — write the memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) using this frontmatter format:",
            "",
            *MEMORY_FRONTMATTER_EXAMPLE,
            "",
            f"**Step 2** — add a pointer to that file in `{ENTRYPOINT_NAME}`. `{ENTRYPOINT_NAME}` is an index, not a memory — each entry should be one line, under ~150 characters: `- [Title](file.md) — one-line hook`. It has no frontmatter. Never write memory content directly into `{ENTRYPOINT_NAME}`.",
            "",
            f"- `{ENTRYPOINT_NAME}` is always loaded into your conversation context — lines after {MAX_ENTRYPOINT_LINES} will be truncated, so keep the index concise",
            "- Keep the name, description, and type fields in memory files up-to-date with the content",
            "- Organize memory semantically by topic, not chronologically",
            "- Update or remove memories that turn out to be wrong or outdated",
            "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
        ]

    lines = [
        f"# {display_name}",
        "",
        f"You have a persistent, file-based memory system at `{memory_dir}`. {DIR_EXISTS_GUIDANCE}",
        "",
        "You should build up this memory system over time so that future conversations can have a complete picture of who the user is, how they'd like to collaborate with you, what behaviors to avoid or repeat, and the context behind the work the user gives you.",
        "",
        "If the user explicitly asks you to remember something, save it immediately as whichever type fits best. If they ask you to forget something, find and remove the relevant entry.",
        "",
        *TYPES_SECTION,
        *WHAT_NOT_TO_SAVE_SECTION,
        "",
        *how_to_save,
        "",
        *WHEN_TO_ACCESS_SECTION,
        "",
        *TRUSTING_RECALL_SECTION,
        "",
        *extra_guidelines,
    ]

    return lines


def build_memory_prompt(
    display_name: str,
    memory_dir: str,
    extra_guidelines: Optional[list[str]] = None,
) -> str:
    """Build the typed-memory prompt with MEMORY.md content included.

    Args:
        display_name: Name to display in prompt
        memory_dir: Path to memory directory
        extra_guidelines: Additional guidelines to include

    Returns:
        Complete memory prompt string
    """
    lines = build_memory_lines(display_name, memory_dir, extra_guidelines)

    # Read existing entrypoint
    entrypoint_path = Path(memory_dir) / ENTRYPOINT_NAME
    entrypoint_content = ""

    try:
        entrypoint_content = entrypoint_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass

    if entrypoint_content.strip():
        truncated = truncate_entrypoint_content(entrypoint_content)
        lines.extend([
            f"## {ENTRYPOINT_NAME}",
            "",
            truncated.content,
        ])
    else:
        lines.extend([
            f"## {ENTRYPOINT_NAME}",
            "",
            f"Your {ENTRYPOINT_NAME} is currently empty. When you save new memories, they will appear here.",
        ])

    return "\n".join(lines)


async def load_memory_prompt(cwd: Optional[str] = None) -> Optional[str]:
    """Load the unified memory prompt for inclusion in system prompt.

    Args:
        cwd: Working directory (defaults to current)

    Returns:
        Memory prompt string or None if memory is disabled
    """
    from .paths import is_auto_memory_enabled, get_auto_mem_path

    if not is_auto_memory_enabled():
        return None

    auto_dir = get_auto_mem_path(cwd)
    await ensure_memory_dir_exists(auto_dir)

    return "\n".join(build_memory_lines("auto memory", auto_dir))