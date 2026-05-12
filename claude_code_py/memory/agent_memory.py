"""Agent Memory implementation.

This implements agent-specific persistent memory.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from .memdir import build_memory_prompt, ensure_memory_dir_exists
from .paths import get_memory_base, get_project_slug


class AgentMemoryScope(str, Enum):
    """Scope for agent memory."""

    USER = "user"
    PROJECT = "project"
    LOCAL = "local"


def get_agent_memory_dir(
    agent_type: str,
    scope: AgentMemoryScope,
    cwd: Optional[str] = None,
) -> str:
    """Get the memory directory for an agent.

    Args:
        agent_type: Agent type identifier
        scope: Memory scope
        cwd: Working directory (for project/local scope)

    Returns:
        Path to agent memory directory
    """
    # Sanitize agent type (replace : with -)
    safe_agent_type = agent_type.replace(":", "-")

    if scope == AgentMemoryScope.USER:
        # User scope: shared across projects
        memory_base = get_memory_base()
        return str(memory_base / "agent-memory" / safe_agent_type)

    elif scope == AgentMemoryScope.PROJECT:
        # Project scope: within current project
        project_slug = get_project_slug(cwd)
        cwd_path = Path(cwd or ".")
        return str(cwd_path / ".claude" / "agent-memory" / safe_agent_type)

    else:  # LOCAL
        # Local scope: current machine/environment
        cwd_path = Path(cwd or ".")

        # Check for remote memory directory
        import os
        remote_dir = os.environ.get("CLAUDE_CODE_REMOTE_MEMORY_DIR")

        if remote_dir:
            project_slug = get_project_slug(cwd)
            return str(
                Path(remote_dir)
                / "projects"
                / project_slug
                / "agent-memory-local"
                / safe_agent_type
            )
        else:
            return str(cwd_path / ".claude" / "agent-memory-local" / safe_agent_type)


def get_agent_memory_entrypoint(
    agent_type: str,
    scope: AgentMemoryScope,
    cwd: Optional[str] = None,
) -> str:
    """Get the MEMORY.md path for an agent.

    Args:
        agent_type: Agent type identifier
        scope: Memory scope
        cwd: Working directory

    Returns:
        Path to agent's MEMORY.md
    """
    memory_dir = get_agent_memory_dir(agent_type, scope, cwd)
    return f"{memory_dir}/MEMORY.md"


def is_agent_memory_path(path: str, cwd: Optional[str] = None) -> bool:
    """Check if a path is within an agent memory directory.

    Args:
        path: Path to check
        cwd: Working directory

    Returns:
        True if path is an agent memory path
    """
    path = str(Path(path).resolve())

    for scope in AgentMemoryScope:
        # Check all possible agent memory directories
        # This is a simplified check - in production, would normalize and compare
        if "agent-memory" in path or "agent-memory-local" in path:
            return True

    return False


def get_scope_note(scope: AgentMemoryScope) -> str:
    """Get the scope note for a memory scope.

    Args:
        scope: Memory scope

    Returns:
        Scope note string
    """
    if scope == AgentMemoryScope.USER:
        return (
            "This memory is shared across all projects. Write memories that are "
            "generally applicable to how you work with this user, not specific to "
            "any single project."
        )
    elif scope == AgentMemoryScope.PROJECT:
        return (
            "This memory is specific to the current project. Write memories that "
            "are relevant to this project's context and requirements."
        )
    else:  # LOCAL
        return (
            "This memory is specific to the current machine or environment. "
            "Write memories that are relevant to this specific setup, such as "
            "local tools, paths, or environment-specific configurations."
        )


async def load_agent_memory_prompt(
    agent_type: str,
    scope: AgentMemoryScope,
    cwd: Optional[str] = None,
) -> str:
    """Load the agent memory prompt.

    Args:
        agent_type: Agent type identifier
        scope: Memory scope
        cwd: Working directory

    Returns:
        Agent memory prompt string
    """
    memory_dir = get_agent_memory_dir(agent_type, scope, cwd)
    scope_note = get_scope_note(scope)

    # Ensure directory exists (fire-and-forget)
    import asyncio
    asyncio.create_task(ensure_memory_dir_exists(memory_dir))

    # Build prompt with scope note
    extra_guidelines = [
        "",
        "## Memory scope",
        "",
        scope_note,
    ]

    return build_memory_prompt(
        display_name="Persistent Agent Memory",
        memory_dir=memory_dir,
        extra_guidelines=extra_guidelines,
    )


# =============================================================================
# Agent Memory Snapshot
# =============================================================================

def get_agent_memory_snapshot_dir(agent_type: str, cwd: Optional[str] = None) -> Path:
    """Get the snapshot directory for an agent.

    Args:
        agent_type: Agent type identifier
        cwd: Working directory

    Returns:
        Path to snapshot directory
    """
    safe_agent_type = agent_type.replace(":", "-")
    cwd_path = Path(cwd or ".")
    return cwd_path / ".claude" / "agent-memory-snapshots" / safe_agent_type


class SnapshotAction(str, Enum):
    """Actions to take for snapshot."""

    NONE = "none"
    INITIALIZE = "initialize"
    PROMPT_UPDATE = "prompt-update"


def check_agent_memory_snapshot(
    agent_type: str,
    scope: AgentMemoryScope,
    cwd: Optional[str] = None,
) -> SnapshotAction:
    """Check what action to take for agent memory snapshot.

    Args:
        agent_type: Agent type identifier
        scope: Memory scope
        cwd: Working directory

    Returns:
        Action to take
    """
    snapshot_dir = get_agent_memory_snapshot_dir(agent_type, cwd)
    snapshot_file = snapshot_dir / "snapshot.json"
    synced_file = snapshot_dir / ".snapshot-synced.json"

    # No snapshot exists
    if not snapshot_file.exists():
        return SnapshotAction.NONE

    # Get memory directory
    memory_dir = Path(get_agent_memory_dir(agent_type, scope, cwd))

    # Check if local memory is empty (no .md files)
    md_files = list(memory_dir.glob("*.md")) if memory_dir.exists() else []

    if not md_files:
        return SnapshotAction.INITIALIZE

    # Check if snapshot is newer than synced
    if not synced_file.exists():
        return SnapshotAction.PROMPT_UPDATE

    # Compare timestamps
    import json
    try:
        snapshot_data = json.loads(snapshot_file.read_text())
        synced_data = json.loads(synced_file.read_text())

        snapshot_updated = snapshot_data.get("updatedAt", 0)
        synced_from = synced_data.get("syncedFrom", 0)

        if snapshot_updated > synced_from:
            return SnapshotAction.PROMPT_UPDATE
    except Exception:
        pass

    return SnapshotAction.NONE


async def initialize_from_snapshot(
    agent_type: str,
    scope: AgentMemoryScope,
    cwd: Optional[str] = None,
) -> bool:
    """Initialize agent memory from snapshot.

    Args:
        agent_type: Agent type identifier
        scope: Memory scope
        cwd: Working directory

    Returns:
        True if initialized successfully
    """
    import shutil
    import json

    snapshot_dir = get_agent_memory_snapshot_dir(agent_type, cwd)
    snapshot_file = snapshot_dir / "snapshot.json"

    if not snapshot_file.exists():
        return False

    memory_dir = Path(get_agent_memory_dir(agent_type, scope, cwd))
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files except snapshot.json
    for file in snapshot_dir.iterdir():
        if file.name not in ("snapshot.json", ".snapshot-synced.json"):
            dest = memory_dir / file.name
            if file.is_file():
                shutil.copy2(file, dest)

    # Write synced metadata
    snapshot_data = json.loads(snapshot_file.read_text())
    synced_file = snapshot_dir / ".snapshot-synced.json"
    synced_file.write_text(json.dumps({
        "syncedFrom": snapshot_data.get("updatedAt", 0),
    }))

    return True