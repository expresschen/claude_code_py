"""Memory Manager - Four-layer memory system.

This implements the complete memory system with four layers:
1. Auto Memory - User-level, persistent across all projects
2. Session Memory - Project-level, per-session context
3. Agent Memory - Agent-specific memory for subagents
4. Team Memory - Shared memory for team/swarm agents

Key features:
- Memory file creation with proper frontmatter
- MEMORY.md index management
- Memory extraction from conversations
- Memory restoration on resume
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .memory_types import (
    MemoryType,
    MemoryFrontmatter,
    parse_memory_file,
    format_memory_index_entry,
    MEMORY_DRIFT_CAVEAT,
)
from .memdir import (
    ENTRYPOINT_NAME,
    MAX_ENTRYPOINT_LINES,
    MAX_ENTRYPOINT_BYTES,
    truncate_entrypoint_content,
    build_memory_prompt,
    ensure_memory_dir_exists,
)


# =============================================================================
# Memory Layer Types
# =============================================================================


class MemoryLayer(str):
    """Memory layer types."""

    AUTO = "auto"
    SESSION = "session"
    AGENT = "agent"
    TEAM = "team"


# =============================================================================
# Memory Paths
# =============================================================================


def get_auto_memory_dir() -> Path:
    """Get the auto memory directory.

    Returns:
        Path to auto memory directory
    """
    from .paths import get_memory_base

    return get_memory_base() / "memory"


def get_session_memory_dir(cwd: Optional[str] = None) -> Path:
    """Get the session memory directory for current project.

    Args:
        cwd: Working directory

    Returns:
        Path to session memory directory
    """
    from .paths import get_memory_base, get_project_slug

    memory_base = get_memory_base()
    project_slug = get_project_slug(cwd)
    return memory_base / "projects" / project_slug / "memory"


def get_team_memory_dir(team_name: str) -> Path:
    """Get the team memory directory.

    Args:
        team_name: Team name

    Returns:
        Path to team memory directory
    """
    from .paths import get_memory_base

    return get_memory_base() / "teams" / team_name / "memory"


def get_memory_entrypoint(memory_dir: Path) -> Path:
    """Get the MEMORY.md entrypoint path.

    Args:
        memory_dir: Memory directory

    Returns:
        Path to MEMORY.md
    """
    return memory_dir / ENTRYPOINT_NAME


# =============================================================================
# Memory File Operations
# =============================================================================


def create_memory_file(
    memory_dir: Path,
    name: str,
    memory_type: MemoryType,
    content: str,
    description: Optional[str] = None,
) -> Path:
    """Create a new memory file with proper frontmatter.

    Args:
        memory_dir: Memory directory
        name: Memory name (used for filename)
        memory_type: Type of memory
        content: Memory body content
        description: Optional description (auto-generated if not provided)

    Returns:
        Path to created memory file
    """
    # Sanitize name for filename
    safe_name = name.replace(" ", "_").replace("/", "-").lower()
    filename = f"{safe_name}.md"
    file_path = memory_dir / filename

    # Generate description from first line if not provided
    if not description:
        first_line = content.split("\n")[0][:100]
        description = first_line.strip() or name

    # Create frontmatter
    frontmatter = MemoryFrontmatter(
        name=name,
        description=description,
        type=memory_type,
        created=datetime.now(),
        updated=datetime.now(),
    )

    # Write file
    full_content = frontmatter.to_yaml() + "\n" + content
    memory_dir.mkdir(parents=True, exist_ok=True)
    file_path.write_text(full_content, encoding="utf-8")

    # Update index
    update_memory_index(memory_dir)

    return file_path


def update_memory_file(
    memory_dir: Path,
    filename: str,
    new_content: Optional[str] = None,
    new_description: Optional[str] = None,
) -> bool:
    """Update an existing memory file.

    Args:
        memory_dir: Memory directory
        filename: Memory filename
        new_content: New content (optional)
        new_description: New description (optional)

    Returns:
        True if updated successfully
    """
    file_path = memory_dir / filename

    if not file_path.exists():
        return False

    # Read existing
    existing_content = file_path.read_text(encoding="utf-8")
    frontmatter, body = parse_memory_file(existing_content)

    if not frontmatter:
        return False

    # Update fields
    if new_description:
        frontmatter.description = new_description
    frontmatter.updated = datetime.now()

    # Build new content
    updated_body = new_content if new_content else body
    full_content = frontmatter.to_yaml() + "\n" + updated_body

    file_path.write_text(full_content, encoding="utf-8")

    # Update index
    update_memory_index(memory_dir)

    return True


def delete_memory_file(memory_dir: Path, filename: str) -> bool:
    """Delete a memory file.

    Args:
        memory_dir: Memory directory
        filename: Memory filename

    Returns:
        True if deleted successfully
    """
    file_path = memory_dir / filename

    if not file_path.exists():
        return False

    file_path.unlink()

    # Update index
    update_memory_index(memory_dir)

    return True


# =============================================================================
# Memory Index Management
# =============================================================================


def update_memory_index(memory_dir: Path) -> None:
    """Update the MEMORY.md index from all memory files.

    Args:
        memory_dir: Memory directory
    """
    if not memory_dir.exists():
        return

    entrypoint = get_memory_entrypoint(memory_dir)

    # Collect all memory files
    memory_files: list[tuple[str, str, str]] = []

    for md_file in memory_dir.glob("*.md"):
        if md_file.name == ENTRYPOINT_NAME:
            continue

        content = md_file.read_text(encoding="utf-8")
        frontmatter, body = parse_memory_file(content)

        if frontmatter:
            memory_files.append((
                frontmatter.name,
                md_file.name,
                frontmatter.description[:150],  # Truncate description
            ))

    # Sort by name
    memory_files.sort(key=lambda x: x[0])

    # Build index content
    lines = [
        "# Memory Index",
        "",
        f"Project memories and context for {memory_dir.parent.name} work.",
        "",
        "## Entries",
        "",
    ]

    for name, filename, description in memory_files:
        lines.append(format_memory_index_entry(name, filename, description))

    # Add currentDate for timestamp reference
    lines.extend([
        "",
        "# currentDate",
        f"Today's date is {datetime.now().strftime('%Y/%m/%d')}.",
        "",
        f"      IMPORTANT: This context may or may not be relevant to your tasks unless it is extremely relevant to the current work. You should ignore it unless highly relevant.",
    ])

    # Check truncation
    index_content = "\n".join(lines)
    truncated = truncate_entrypoint_content(index_content)

    # Write index
    entrypoint.write_text(truncated.content, encoding="utf-8")


def read_memory_index(memory_dir: Path) -> list[dict[str, str]]:
    """Read the MEMORY.md index entries.

    Args:
        memory_dir: Memory directory

    Returns:
        List of index entries (name, filename, hook)
    """
    entrypoint = get_memory_entrypoint(memory_dir)

    if not entrypoint.exists():
        return []

    content = entrypoint.read_text(encoding="utf-8")
    entries: list[dict[str, str]] = []

    # Parse index entries
    for line in content.split("\n"):
        # Match format: - [Title](filename) — description
        match = re.match(r"- \[([^\]]+)\]\(([^\)]+)\) — (.+)", line)
        if match:
            entries.append({
                "name": match.group(1),
                "filename": match.group(2),
                "hook": match.group(3),
            })

    return entries


# =============================================================================
# Memory Extraction
# =============================================================================


async def extract_memory_from_message(
    message: dict[str, Any],
    model: str,
) -> Optional[dict[str, Any]]:
    """Extract memory content from a conversation message.

    Args:
        message: Message to extract from
        model: Model to use for extraction

    Returns:
        Extracted memory dict or None
    """
    # Only extract from user messages that contain key info
    if message.get("type") != "user":
        return None

    content = message.get("message", {}).get("content", "")
    if isinstance(content, list):
        # Extract text from content blocks
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        content = "\n".join(texts)

    if not content or len(content) < 50:
        return None

    # Use LLM to extract memory-worthy content
    from claude_code_py.utils.side_query import side_query, SideQueryOptions, QuerySource

    extract_prompt = """Analyze the following user message and extract any memory-worthy information.

Memory-worthy information includes:
- User preferences or role information
- Feedback on how to approach work
- Project-specific context not in code
- References to external systems

If memory-worthy content exists, respond with:
MEMORY_TYPE: <user|feedback|project|reference>
MEMORY_NAME: <short name>
MEMORY_DESCRIPTION: <one-line description>
MEMORY_CONTENT: <memory body>

If no memory-worthy content, respond with: NO_MEMORY

User message:
""" + content

    opts = SideQueryOptions(
        model=model,
        messages=[{"role": "user", "content": extract_prompt}],
        max_tokens=500,
        query_source=QuerySource.MEMORY_EXTRACTION,
    )

    result = await side_query(opts)

    # Parse result
    for block in result.content:
        if block.get("type") == "text":
            text = block.get("text", "")
            if "NO_MEMORY" in text:
                return None

            # Extract fields
            memory_type_match = re.search(r"MEMORY_TYPE: (\w+)", text)
            name_match = re.search(r"MEMORY_NAME: (.+)", text)
            desc_match = re.search(r"MEMORY_DESCRIPTION: (.+)", text)
            content_match = re.search(r"MEMORY_CONTENT: (.+)", text)

            if memory_type_match and name_match and content_match:
                try:
                    memory_type = MemoryType(memory_type_match.group(1).lower())
                except ValueError:
                    memory_type = MemoryType.USER

                return {
                    "type": memory_type,
                    "name": name_match.group(1).strip(),
                    "description": desc_match.group(1).strip() if desc_match else "",
                    "content": content_match.group(1).strip(),
                }

    return None


async def extract_memories_from_conversation(
    messages: list[dict[str, Any]],
    model: str,
    max_memories: int = 5,
) -> list[dict[str, Any]]:
    """Extract memories from a conversation history.

    Args:
        messages: Conversation messages
        model: Model to use
        max_memories: Maximum memories to extract

    Returns:
        List of extracted memories
    """
    memories: list[dict[str, Any]] = []

    # Extract from recent user messages only
    recent_users = [
        m for m in messages[-20:]
        if m.get("type") == "user"
    ]

    for msg in reversed(recent_users):
        if len(memories) >= max_memories:
            break

        memory = await extract_memory_from_message(msg, model)
        if memory:
            memories.append(memory)

    return memories


# =============================================================================
# Memory Restoration
# =============================================================================


def restore_memories_from_attachments(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore memories from attachment messages.

    Args:
        messages: Message list

    Returns:
        List of restored memories
    """
    restored: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("type") != "attachment":
            continue

        attachment = msg.get("attachment", {})
        if attachment.get("type") == "relevant_memories":
            for memory in attachment.get("memories", []):
                if memory.get("content"):
                    restored.append(memory)

    return restored


def create_memory_attachment(
    memories: list[dict[str, Any]],
    memory_dir: Path,
) -> dict[str, Any]:
    """Create a relevant_memories attachment for conversation.

    Args:
        memories: Memory entries
        memory_dir: Memory directory path

    Returns:
        Attachment dict
    """
    memory_entries: list[dict[str, str]] = []

    for entry in memories:
        file_path = memory_dir / entry.get("filename", "")
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            frontmatter, body = parse_memory_file(content)

            if frontmatter:
                memory_entries.append({
                    "name": frontmatter.name,
                    "type": frontmatter.type.value,
                    "content": body,
                    "filename": entry.get("filename", ""),
                })

    return {
        "type": "attachment",
        "attachment": {
            "type": "relevant_memories",
            "memories": memory_entries,
            "memory_dir": str(memory_dir),
        },
    }


# =============================================================================
# Unified Memory Prompt Builder
# =============================================================================


async def build_unified_memory_prompt(
    cwd: Optional[str] = None,
    agent_type: Optional[str] = None,
    team_name: Optional[str] = None,
) -> str:
    """Build unified memory prompt for all layers.

    Args:
        cwd: Working directory
        agent_type: Agent type (for agent memory)
        team_name: Team name (for team memory)

    Returns:
        Unified memory prompt
    """
    prompts: list[str] = []

    # Auto memory (always)
    auto_dir = get_auto_memory_dir()
    await ensure_memory_dir_exists(str(auto_dir))
    auto_prompt = build_memory_prompt("Auto Memory", str(auto_dir))
    prompts.append(auto_prompt)

    # Session memory (project-level)
    session_dir = get_session_memory_dir(cwd)
    await ensure_memory_dir_exists(str(session_dir))
    session_prompt = build_memory_prompt("Session Memory", str(session_dir))
    prompts.append(session_prompt)

    # Agent memory (if specified)
    if agent_type:
        from .agent_memory import load_agent_memory_prompt, AgentMemoryScope
        agent_prompt = await load_agent_memory_prompt(
            agent_type,
            AgentMemoryScope.PROJECT,
            cwd,
        )
        prompts.append(agent_prompt)

    # Team memory (if specified)
    if team_name:
        team_dir = get_team_memory_dir(team_name)
        await ensure_memory_dir_exists(str(team_dir))
        team_prompt = build_memory_prompt("Team Memory", str(team_dir))
        prompts.append(team_prompt)

    return "\n\n".join(prompts)


# =============================================================================
# Memory Statistics
# =============================================================================


def get_memory_stats(memory_dir: Path) -> dict[str, Any]:
    """Get statistics for a memory directory.

    Args:
        memory_dir: Memory directory

    Returns:
        Stats dict
    """
    if not memory_dir.exists():
        return {
            "exists": False,
            "file_count": 0,
            "index_entries": 0,
            "total_bytes": 0,
        }

    # Count files
    md_files = list(memory_dir.glob("*.md"))
    file_count = len(md_files) - 1 if memory_dir / ENTRYPOINT_NAME in md_files else len(md_files)

    # Count index entries
    entries = read_memory_index(memory_dir)

    # Total bytes
    total_bytes = sum(f.stat().st_size for f in md_files)

    return {
        "exists": True,
        "file_count": file_count,
        "index_entries": len(entries),
        "total_bytes": total_bytes,
        "entrypoint_exists": (memory_dir / ENTRYPOINT_NAME).exists(),
    }