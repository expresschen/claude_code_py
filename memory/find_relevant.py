"""Find relevant memories.

This implements the memory recall system from findRelevantMemories.ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from claude_code_py.core_types.message import Message

# Import side query for model-based selection
try:
    from claude_code_py.utils.side_query import (
        select_relevant_memories_with_model,
        get_default_sonnet_model,
    )
    HAS_SIDE_QUERY = True
except ImportError:
    HAS_SIDE_QUERY = False


@dataclass
class MemoryManifest:
    """Manifest of available memory files."""

    files: list[dict[str, Any]]
    total_count: int


@dataclass
class RelevantMemory:
    """A relevant memory file."""

    path: str
    name: str
    description: str
    mtime_ms: float


@dataclass
class SurfacedMemoriesInfo:
    """Information about already surfaced memories from messages."""

    paths: set[str]
    total_bytes: int


# Constants
MAX_MEMORY_FILES = 200
FRONTMATTER_MAX_LINES = 30
MAX_RECENT_TOOLS = 5  # Maximum recent tools to track
RECENT_MESSAGE_WINDOW = 10  # Number of recent messages to scan for tools


# =============================================================================
# Message Analysis Functions
# =============================================================================


def collect_surfaced_memories(messages: list["Message"]) -> SurfacedMemoriesInfo:
    """Extract already surfaced memory paths from messages.

    Scans messages for past relevant_memories attachments to prevent
    re-injecting the same memory files. This deduplication ensures the
    selector spends its budget on fresh candidates.

    Args:
        messages: List of conversation messages

    Returns:
        SurfacedMemoriesInfo with paths set and total byte count
    """
    paths: set[str] = set()
    total_bytes = 0

    for msg in messages:
        # Check for AttachmentMessage with relevant_memories type
        if hasattr(msg, "type") and msg.type == "attachment":
            attachment = getattr(msg, "attachment", None)
            if attachment and isinstance(attachment, dict):
                if attachment.get("type") == "relevant_memories":
                    for mem in attachment.get("memories", []):
                        path = mem.get("path", "")
                        if path:
                            paths.add(path)
                            total_bytes += len(mem.get("content", ""))
                elif attachment.get("type") == "nested_memory":
                    # Also track nested_memory attachments
                    path = attachment.get("path", "")
                    if path:
                        paths.add(path)
                        total_bytes += len(attachment.get("content", ""))

    return SurfacedMemoriesInfo(paths=paths, total_bytes=total_bytes)


def extract_recent_tools(messages: list["Message"]) -> list[str]:
    """Extract recently used tool names from messages.

    Identifies tools actively being used so the memory selector can
    exclude their reference/documentation memories (avoiding noise
    when Claude Code is already exercising those tools).

    Args:
        messages: List of conversation messages

    Returns:
        List of recent tool names (up to MAX_RECENT_TOOLS)
    """
    tools: list[str] = []

    # Scan recent messages (last RECENT_MESSAGE_WINDOW)
    recent_messages = messages[-RECENT_MESSAGE_WINDOW:] if len(messages) > RECENT_MESSAGE_WINDOW else messages

    for msg in recent_messages:
        # Check AssistantMessage for tool_use blocks
        if hasattr(msg, "type") and msg.type == "assistant":
            message_content = getattr(msg, "message", None)
            if message_content and isinstance(message_content, dict):
                content = message_content.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_name = block.get("name", "")
                            if tool_name and tool_name not in tools:
                                tools.append(tool_name)

    # Return most recent tools (up to limit)
    return tools[-MAX_RECENT_TOOLS:] if len(tools) > MAX_RECENT_TOOLS else tools


# =============================================================================
# Memory Directory Scanning
# =============================================================================


def scan_memory_files(memory_dir: str) -> list[dict[str, Any]]:
    """Scan memory directory for .md files.

    Args:
        memory_dir: Path to memory directory

    Returns:
        List of file info dicts
    """
    dir_path = Path(memory_dir)
    if not dir_path.exists():
        return []

    files = []
    for md_file in dir_path.glob("*.md"):
        if md_file.name == "MEMORY.md":
            continue  # Skip index file

        # Read header for metadata
        header = _read_file_header(md_file)

        # Get modification time
        try:
            mtime_ms = md_file.stat().st_mtime * 1000
        except OSError:
            mtime_ms = 0

        files.append({
            "path": str(md_file),
            "filename": md_file.name,
            "name": md_file.stem,
            "description": header.get("description", ""),
            "type": header.get("type", "user"),
            "mtime_ms": mtime_ms,
        })

    # Sort by modification time (newest first)
    files.sort(key=lambda x: -x.get("mtime_ms", 0))

    # Cap at MAX_MEMORY_FILES
    return files[:MAX_MEMORY_FILES]


def _read_file_header(path: Path, max_lines: int = 20) -> dict[str, str]:
    """Read frontmatter header from a memory file.

    Args:
        path: Path to file
        max_lines: Maximum lines to read

    Returns:
        Header metadata
    """
    metadata: dict[str, str] = {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            in_frontmatter = False
            for i, line in enumerate(f):
                if i >= max_lines:
                    break

                line = line.rstrip("\n")

                if line == "---":
                    if in_frontmatter:
                        break  # End of frontmatter
                    in_frontmatter = True
                    continue

                if in_frontmatter and ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

    except Exception:
        pass

    return metadata


def format_memory_manifest(files: list[dict[str, Any]], include_timestamp: bool = False) -> str:
    """Format memory files as a manifest string.

    Args:
        files: List of file info
        include_timestamp: Whether to include modification timestamp

    Returns:
        Manifest string
    """
    from datetime import datetime

    lines = []

    for f in files:
        tag = f.get("type", "")
        tag_str = f"[{tag}] " if tag else ""

        filename = f.get("name") or f.get("filename", "unknown")
        description = f.get("description", "")

        if include_timestamp:
            mtime_ms = f.get("mtime_ms", 0)
            ts = datetime.fromtimestamp(mtime_ms / 1000).isoformat()
            if description:
                lines.append(f"- {tag_str}{filename} ({ts}): {description}")
            else:
                lines.append(f"- {tag_str}{filename} ({ts})")
        else:
            if description:
                lines.append(f"- {tag_str}{filename}: {description}")
            else:
                lines.append(f"- {tag_str}{filename}")

    if not lines:
        return "No memory files found."

    return "\n".join(lines)


async def find_relevant_memories(
    memory_dir: str,
    query: str,
    already_surfaced: Optional[set[str]] = None,
    max_results: int = 5,
    recent_tools: Optional[list[str]] = None,
    use_model_selection: bool = True,
) -> list[RelevantMemory]:
    """Find relevant memory files for a query.

    This uses Sonnet model selection when available, with keyword
    matching as fallback.

    Args:
        memory_dir: Path to memory directory
        query: Search query
        already_surfaced: Set of already shown file paths
        max_results: Maximum results to return
        recent_tools: Recently used tools (to exclude their docs)
        use_model_selection: Whether to use model-based selection

    Returns:
        List of relevant memories
    """
    already_surfaced = already_surfaced or set()

    # Scan files
    files = scan_memory_files(memory_dir)

    if not files:
        return []

    # Filter out already surfaced
    candidates = [f for f in files if f["path"] not in already_surfaced]

    if not candidates:
        return []

    # Select relevant memories
    selected_filenames: list[str] = []

    if use_model_selection and HAS_SIDE_QUERY:
        # Use Sonnet model for selection
        try:
            selected_filenames = await select_relevant_memories_with_model(
                query,
                candidates,
                recent_tools,
            )
        except Exception:
            # Fallback to keyword matching on error
            selected_filenames = _select_by_keywords(
                query,
                candidates,
                max_results,
            )
    else:
        # Use keyword matching (fallback or when side_query unavailable)
        selected_filenames = _select_by_keywords(
            query,
            candidates,
            max_results,
        )

    # Map filenames to RelevantMemory objects
    by_filename = {f.get("name"): f for f in candidates}

    results = []
    for filename in selected_filenames[:max_results]:
        f = by_filename.get(filename)
        if f:
            results.append(RelevantMemory(
                path=f["path"],
                name=f["name"],
                description=f.get("description", ""),
                mtime_ms=f.get("mtime_ms", 0),
            ))

    return results


def _select_by_keywords(
    query: str,
    memories: list[dict[str, Any]],
    max_results: int = 5,
) -> list[str]:
    """Select relevant memories using simple keyword matching.

    This is a fallback when the API is not available.

    Args:
        query: User query
        memories: List of memory headers
        max_results: Maximum results

    Returns:
        List of selected memory filenames (stem names)
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Filter out very short words
    query_words = {w for w in query_words if len(w) > 2}

    scored = []
    for m in memories:
        score = 0
        name = (m.get("name") or m.get("filename", "")).lower()
        desc = m.get("description", "").lower()

        for word in query_words:
            if word in name:
                score += 3  # Name match is more important
            if word in desc:
                score += 2

        # Include low-score results to ensure we have something
        if score > 0 or len(scored) < max_results:
            scored.append((score, m.get("name") or m.get("filename", "unknown")))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])

    return [name for score, name in scored[:max_results]]


def get_memory_files_to_attachments(
    memory_files: list[RelevantMemory],
    loaded_paths: set[str],
) -> list[dict[str, Any]]:
    """Convert memory files to attachment format.

    Args:
        memory_files: List of relevant memories
        loaded_paths: Set of already loaded paths

    Returns:
        List of attachment dicts
    """
    attachments = []

    for mem in memory_files:
        if mem.path in loaded_paths:
            continue

        path = Path(mem.path)
        if not path.exists():
            continue

        try:
            content = path.read_text(encoding="utf-8")

            attachments.append({
                "type": "nested_memory",
                "path": mem.path,
                "content": content,
                "name": mem.name,
                "description": mem.description,
            })

            loaded_paths.add(mem.path)

        except Exception:
            pass

    return attachments