"""Session Storage - Persist and retrieve conversation history.

This implements the session storage system from sessionStorage.ts for
persisting conversations and enabling session search/resume.

Key features:
- JSONL append-only storage with parentUuid chain
- Message deduplication via UUID tracking
- Session metadata with timestamps
- Agent transcript sidechain support
- Resume chain building and recovery
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Set

from claude_code_py.core_types.message import Message, UserMessage, AssistantMessage
from claude_code_py.memory.paths import get_memory_base, get_project_slug


# =============================================================================
# Types
# =============================================================================


@dataclass
class SessionMeta:
    """Metadata for a session."""

    session_id: str
    created_at: float
    updated_at: float
    first_prompt: Optional[str] = None
    summary: Optional[str] = None
    tag: Optional[str] = None
    git_branch: Optional[str] = None
    custom_title: Optional[str] = None
    message_count: int = 0


@dataclass
class AgentMetadata:
    """Metadata for a subagent."""

    agent_type: str
    worktree_path: Optional[str] = None
    description: Optional[str] = None


@dataclass
class WorktreeSession:
    """Worktree session state for persistence."""

    original_cwd: str
    worktree_path: str
    worktree_name: str
    session_id: str
    worktree_branch: Optional[str] = None
    original_branch: Optional[str] = None
    original_head_commit: Optional[str] = None
    hook_based: bool = False


@dataclass
class SessionLog:
    """A session log entry."""

    session_id: str
    meta: SessionMeta
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LogOption:
    """Option for session selection."""

    session_id: str
    display_name: str
    first_prompt: Optional[str] = None
    summary: Optional[str] = None
    tag: Optional[str] = None
    git_branch: Optional[str] = None
    custom_title: Optional[str] = None
    created_at: float = 0
    messages: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Paths
# =============================================================================


def get_session_dir(cwd: Optional[str] = None) -> Path:
    """Get the directory for session storage.

    Args:
        cwd: Working directory (defaults to current)

    Returns:
        Path to sessions directory
    """
    memory_base = get_memory_base()
    project_slug = get_project_slug(cwd)
    return memory_base / "projects" / project_slug / "sessions"


def get_session_path(session_id: str, cwd: Optional[str] = None) -> Path:
    """Get the path to a session file.

    Args:
        session_id: Session identifier
        cwd: Working directory (defaults to current)

    Returns:
        Path to session file
    """
    return get_session_dir(cwd) / f"{session_id}.jsonl"


def get_session_subagents_dir(session_id: str, cwd: Optional[str] = None) -> Path:
    """Get the directory for subagent transcripts.

    Args:
        session_id: Session identifier
        cwd: Working directory (defaults to current)

    Returns:
        Path to subagents directory
    """
    return get_session_dir(cwd) / session_id / "subagents"


def get_agent_transcript_path(session_id: str, agent_id: str, cwd: Optional[str] = None) -> Path:
    """Get the path to an agent's transcript file.

    Args:
        session_id: Session identifier
        agent_id: Agent identifier
        cwd: Working directory (defaults to current)

    Returns:
        Path to agent transcript file
    """
    return get_session_subagents_dir(session_id, cwd) / f"agent-{agent_id}.jsonl"


def get_agent_metadata_path(session_id: str, agent_id: str, cwd: Optional[str] = None) -> Path:
    """Get the path to an agent's metadata file.

    Args:
        session_id: Session identifier
        agent_id: Agent identifier
        cwd: Working directory (defaults to current)

    Returns:
        Path to agent metadata file (.meta.json)
    """
    return get_agent_transcript_path(session_id, agent_id, cwd).with_suffix(".meta.json")


def read_agent_transcript(
    session_id: str,
    agent_id: str,
    cwd: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Read an agent's JSONL transcript from disk.

    Each line is a JSON object representing one message.
    Returns empty list if the transcript file doesn't exist or is unreadable.

    Args:
        session_id: Session identifier
        agent_id: Agent identifier
        cwd: Working directory

    Returns:
        List of message dicts from the transcript file
    """
    path = get_agent_transcript_path(session_id, agent_id, cwd)
    messages: list[dict[str, Any]] = []
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return messages


def bootstrap_agent_messages(
    live_messages: list[dict[str, Any]],
    disk_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge disk transcript with live messages, deduplicating by UUID.

    Disk-write-before-yield guarantees live messages are always a suffix of
    disk, so disk messages come first. UUID-based dedup prevents duplicates
    from messages that were written to disk before being streamed.

    Matches TypeScript disk bootstrap logic in REPL.tsx (lines 655-664).

    Args:
        live_messages: Messages already in task.messages
        disk_messages: Messages read from disk JSONL

    Returns:
        Merged list: disk-only messages first, then all live messages
    """
    # Helper to safely get uuid from either Pydantic or dict messages
    def _get_uuid(m: Any) -> str:
        if hasattr(m, "uuid"):
            return m.uuid or ""
        elif isinstance(m, dict):
            return m.get("uuid", "")
        return ""

    live_uuids: set[str] = set()
    for m in live_messages:
        uid = _get_uuid(m)
        if uid:
            live_uuids.add(uid)

    disk_only = [m for m in disk_messages if _get_uuid(m) not in live_uuids]
    return disk_only + live_messages


def get_session_env_dir(session_id: str) -> Path:
    """Get the directory for session environment scripts.

    Args:
        session_id: Session identifier

    Returns:
        Path to session-env directory
    """
    memory_base = get_memory_base()
    return memory_base / "session-env" / session_id


# =============================================================================
# Session Storage Class
# =============================================================================


class SessionStorage:
    """Manages session persistence and retrieval.

    Features:
    - JSONL append-only storage with parentUuid chain
    - Message UUID tracking for deduplication
    - Last parent UUID tracking for chain building
    - Agent transcript support for sidechains
    """

    def __init__(self, session_id: Optional[str] = None, cwd: Optional[str] = None):
        """Initialize session storage.

        Args:
            session_id: Optional session ID (generates new one if not provided)
            cwd: Working directory (defaults to current)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self._cwd = cwd
        self._session_dir = get_session_dir(cwd)
        self._session_path = get_session_path(self.session_id, cwd)
        self._meta: Optional[SessionMeta] = None

        # UUID tracking for deduplication
        self._message_uuids: Set[str] = set()

        # Last parent UUID for chain building
        self._last_parent_uuid: Optional[str] = None

        # Write buffer for async writes
        self._write_buffer: list[dict[str, Any]] = []
        self._flush_interval: float = 0.1  # 100ms
        self._flush_task: Optional[asyncio.Task] = None

    def _ensure_dir(self) -> None:
        """Ensure session directory exists."""
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def _load_existing_uuids(self) -> None:
        """Load existing message UUIDs for deduplication."""
        if self._message_uuids:
            return  # Already loaded

        if not self._session_path.exists():
            return

        try:
            with open(self._session_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        msg_uuid = entry.get("uuid")
                        if msg_uuid:
                            self._message_uuids.add(msg_uuid)
                            # Track last UUID for parent chain
                            self._last_parent_uuid = msg_uuid
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    def get_meta(self) -> SessionMeta:
        """Get or create session metadata.

        Returns:
            SessionMeta
        """
        if self._meta:
            return self._meta

        # Try to load existing meta
        meta_path = self._session_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._meta = SessionMeta(**data)
                return self._meta
            except Exception:
                pass

        # Create new meta
        now = datetime.now().timestamp()
        self._meta = SessionMeta(
            session_id=self.session_id,
            created_at=now,
            updated_at=now,
        )
        return self._meta

    def update_meta(self, **kwargs: Any) -> None:
        """Update session metadata.

        Args:
            **kwargs: Fields to update
        """
        meta = self.get_meta()
        for key, value in kwargs.items():
            if hasattr(meta, key):
                setattr(meta, key, value)
        meta.updated_at = datetime.now().timestamp()

        # Save meta
        self._save_meta()

    def _save_meta(self) -> None:
        """Save metadata to disk."""
        self._ensure_dir()
        meta_path = self._session_path.with_suffix(".meta.json")

        meta = self.get_meta()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": meta.session_id,
                "created_at": meta.created_at,
                "updated_at": meta.updated_at,
                "first_prompt": meta.first_prompt,
                "summary": meta.summary,
                "tag": meta.tag,
                "git_branch": meta.git_branch,
                "custom_title": meta.custom_title,
                "message_count": meta.message_count,
            }, f, indent=2)

    def append_message(self, message: Message, parent_uuid: Optional[str] = None) -> None:
        """Append a message to the session with parentUuid chain.

        Args:
            message: Message to append
            parent_uuid: Optional parent UUID (uses last message if not provided)
        """
        self._ensure_dir()
        self._load_existing_uuids()

        # Convert message to dict
        if hasattr(message, "model_dump"):
            msg_dict = message.model_dump()
        else:
            msg_dict = {
                "type": getattr(message, "type", "unknown"),
                "uuid": getattr(message, "uuid", str(uuid.uuid4())),
                "message": getattr(message, "message", {}),
            }

        msg_uuid = msg_dict.get("uuid")

        # Deduplication check
        if msg_uuid and msg_uuid in self._message_uuids:
            return  # Already recorded

        # Set parentUuid for chain building
        # Only user/assistant/attachment/system participate in chain
        msg_type = msg_dict.get("type")
        if msg_type in ("user", "assistant", "attachment", "system"):
            if parent_uuid:
                msg_dict["parentUuid"] = parent_uuid
            elif self._last_parent_uuid:
                msg_dict["parentUuid"] = self._last_parent_uuid

            # Update last parent for next message
            if msg_uuid:
                self._last_parent_uuid = msg_uuid

        # Add sessionId to message
        msg_dict["sessionId"] = self.session_id

        # Track UUID
        if msg_uuid:
            self._message_uuids.add(msg_uuid)

        # Append to JSONL
        with open(self._session_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg_dict) + "\n")

        # Update meta
        meta = self.get_meta()
        meta.message_count += 1

        # Update first prompt if this is the first user message
        if msg_dict.get("type") == "user" and not meta.first_prompt:
            content = msg_dict.get("message", {}).get("content", "")
            if isinstance(content, str):
                meta.first_prompt = content[:300]
            elif isinstance(content, list):
                # Extract text from content blocks
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                if texts:
                    meta.first_prompt = " ".join(texts)[:300]

        self._save_meta()

    def append_entry(self, entry: dict[str, Any]) -> None:
        """Append a raw entry dict to the session.

        Args:
            entry: Raw entry dict to append
        """
        self._ensure_dir()
        self._load_existing_uuids()

        entry_uuid = entry.get("uuid")

        # Deduplication check
        if entry_uuid and entry_uuid in self._message_uuids:
            return

        # Set parentUuid for chain participants
        entry_type = entry.get("type")
        if entry_type in ("user", "assistant", "attachment", "system"):
            if "parentUuid" not in entry:
                if self._last_parent_uuid:
                    entry["parentUuid"] = self._last_parent_uuid

            # Update last parent
            if entry_uuid:
                self._last_parent_uuid = entry_uuid

        # Add sessionId
        entry["sessionId"] = self.session_id

        # Track UUID
        if entry_uuid:
            self._message_uuids.add(entry_uuid)

        # Append to JSONL
        with open(self._session_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def insert_message_chain(
        self,
        messages: list[Message],
        is_sidechain: bool = False,
        agent_id: Optional[str] = None,
        starting_parent_uuid: Optional[str] = None,
    ) -> None:
        """Insert a chain of messages with proper parentUuid linking.

        Args:
            messages: Messages to insert
            is_sidechain: Whether this is a sidechain transcript
            agent_id: Agent ID for sidechain
            starting_parent_uuid: Starting parent UUID
        """
        self._ensure_dir()
        self._load_existing_uuids()

        # Determine target file
        if is_sidechain and agent_id:
            target_path = get_agent_transcript_path(self.session_id, agent_id, self._cwd)
            target_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            target_path = self._session_path

        parent_uuid = starting_parent_uuid

        for msg in messages:
            msg_uuid = getattr(msg, "uuid", str(uuid.uuid4()))

            # Skip if already recorded (for main chain)
            if not is_sidechain and msg_uuid in self._message_uuids:
                continue

            # Convert to dict
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump()
            else:
                msg_dict = {
                    "type": getattr(msg, "type", "unknown"),
                    "uuid": msg_uuid,
                    "message": getattr(msg, "message", {}),
                }

            # Set parentUuid for chain participants
            msg_type = msg_dict.get("type")
            if msg_type in ("user", "assistant", "attachment", "system"):
                if parent_uuid:
                    msg_dict["parentUuid"] = parent_uuid
                parent_uuid = msg_uuid

            # Add metadata
            msg_dict["sessionId"] = self.session_id
            if is_sidechain:
                msg_dict["isSidechain"] = True
                if agent_id:
                    msg_dict["agentId"] = agent_id

            # Track UUID
            if not is_sidechain:
                self._message_uuids.add(msg_uuid)

            # Write to target file
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg_dict) + "\n")

        # Update last parent
        if parent_uuid and not is_sidechain:
            self._last_parent_uuid = parent_uuid

    def get_last_parent_uuid(self) -> Optional[str]:
        """Get the last parent UUID for chain building.

        Returns:
            Last parent UUID or None
        """
        self._load_existing_uuids()
        return self._last_parent_uuid

    def get_message_uuids(self) -> Set[str]:
        """Get all recorded message UUIDs.

        Returns:
            Set of UUIDs
        """
        self._load_existing_uuids()
        return self._message_uuids.copy()

    def load_messages(self) -> list[dict[str, Any]]:
        """Load all messages from the session.

        Returns:
            List of message dicts
        """
        if not self._session_path.exists():
            return []

        messages = []
        try:
            with open(self._session_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))
        except Exception:
            pass

        return messages

    def load_transcript(
        self,
        max_messages: int = 100,
        max_chars: int = 2000,
    ) -> str:
        """Load a truncated transcript for search.

        Args:
            max_messages: Maximum messages to include
            max_chars: Maximum characters in transcript

        Returns:
            Truncated transcript string
        """
        messages = self.load_messages()
        if not messages:
            return ""

        # Take from start and end
        if len(messages) > max_messages:
            half = max_messages // 2
            messages = messages[:half] + messages[-half:]

        # Extract text
        texts = []
        for msg in messages:
            text = extract_message_text(msg)
            if text:
                texts.append(text)

        transcript = " ".join(texts).strip()

        # Truncate
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "…"

        return transcript


# =============================================================================
# Session List Operations
# =============================================================================


def list_sessions(cwd: Optional[str] = None) -> list[LogOption]:
    """List all available sessions.

    Args:
        cwd: Working directory (defaults to current)

    Returns:
        List of LogOption for each session
    """
    session_dir = get_session_dir(cwd)
    if not session_dir.exists():
        return []

    logs = []
    for meta_file in session_dir.glob("*.meta.json"):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            session_id = data.get("session_id", meta_file.stem)
            first_prompt = data.get("first_prompt", "No prompt")

            # Create display name
            display_name = (
                data.get("custom_title")
                or first_prompt[:50]
                or "Untitled Session"
            )

            logs.append(LogOption(
                session_id=session_id,
                display_name=display_name,
                first_prompt=first_prompt,
                summary=data.get("summary"),
                tag=data.get("tag"),
                git_branch=data.get("git_branch"),
                custom_title=data.get("custom_title"),
                created_at=data.get("created_at", 0),
            ))
        except Exception:
            continue

    # Sort by created_at descending (newest first)
    logs.sort(key=lambda x: -x.created_at)

    return logs


def delete_session(session_id: str, cwd: Optional[str] = None) -> bool:
    """Delete a session.

    Args:
        session_id: Session to delete
        cwd: Working directory (defaults to current)

    Returns:
        True if deleted successfully
    """
    session_path = get_session_path(session_id, cwd)
    meta_path = session_path.with_suffix(".meta.json")

    deleted = False

    try:
        if session_path.exists():
            session_path.unlink()
            deleted = True
    except Exception:
        pass

    try:
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
    except Exception:
        pass

    return deleted


# =============================================================================
# Agent Metadata Operations
# =============================================================================


async def write_agent_metadata(
    session_id: str,
    agent_id: str,
    metadata: AgentMetadata,
    cwd: Optional[str] = None,
) -> None:
    """Write agent metadata to disk.

    Args:
        session_id: Session identifier
        agent_id: Agent identifier
        metadata: Agent metadata to write
        cwd: Working directory (defaults to current)
    """
    meta_path = get_agent_metadata_path(session_id, agent_id, cwd)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "agent_type": metadata.agent_type,
            "worktree_path": metadata.worktree_path,
            "description": metadata.description,
        }, f, indent=2)


def read_agent_metadata(
    session_id: str,
    agent_id: str,
    cwd: Optional[str] = None,
) -> Optional[AgentMetadata]:
    """Read agent metadata from disk.

    Args:
        session_id: Session identifier
        agent_id: Agent identifier
        cwd: Working directory (defaults to current)

    Returns:
        AgentMetadata or None if not found
    """
    meta_path = get_agent_metadata_path(session_id, agent_id, cwd)

    if not meta_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AgentMetadata(
            agent_type=data.get("agent_type", ""),
            worktree_path=data.get("worktree_path"),
            description=data.get("description"),
        )
    except Exception:
        return None


# =============================================================================
# Worktree State Operations
# =============================================================================


def save_worktree_state(
    session_id: str,
    worktree: WorktreeSession,
    cwd: Optional[str] = None,
) -> None:
    """Save worktree state to project config.

    Args:
        session_id: Session identifier
        worktree: Worktree session state
        cwd: Working directory (defaults to current)
    """
    from claude_code_py.memory.paths import get_memory_base, get_project_slug

    memory_base = get_memory_base()
    project_slug = get_project_slug(cwd)
    config_path = memory_base / "projects" / project_slug / "config.json"

    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass

    # Update with worktree state
    config["active_worktree_session"] = {
        "original_cwd": worktree.original_cwd,
        "worktree_path": worktree.worktree_path,
        "worktree_name": worktree.worktree_name,
        "worktree_branch": worktree.worktree_branch,
        "original_branch": worktree.original_branch,
        "original_head_commit": worktree.original_head_commit,
        "session_id": worktree.session_id,
        "hook_based": worktree.hook_based,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_worktree_state(session_id: str, cwd: Optional[str] = None) -> Optional[WorktreeSession]:
    """Load worktree state from project config.

    Args:
        session_id: Session identifier
        cwd: Working directory (defaults to current)

    Returns:
        WorktreeSession or None if not found
    """
    from claude_code_py.memory.paths import get_memory_base, get_project_slug

    memory_base = get_memory_base()
    project_slug = get_project_slug(cwd)
    config_path = memory_base / "projects" / project_slug / "config.json"

    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        worktree_data = config.get("active_worktree_session")
        if not worktree_data:
            return None

        return WorktreeSession(
            original_cwd=worktree_data.get("original_cwd", ""),
            worktree_path=worktree_data.get("worktree_path", ""),
            worktree_name=worktree_data.get("worktree_name", ""),
            worktree_branch=worktree_data.get("worktree_branch"),
            original_branch=worktree_data.get("original_branch"),
            original_head_commit=worktree_data.get("original_head_commit"),
            session_id=worktree_data.get("session_id", session_id),
            hook_based=worktree_data.get("hook_based", False),
        )
    except Exception:
        return None


def clear_worktree_state(session_id: str, cwd: Optional[str] = None) -> None:
    """Clear worktree state from project config.

    Args:
        session_id: Session identifier
        cwd: Working directory (defaults to current)
    """
    from claude_code_py.memory.paths import get_memory_base, get_project_slug

    memory_base = get_memory_base()
    project_slug = get_project_slug(cwd)
    config_path = memory_base / "projects" / project_slug / "config.json"

    if not config_path.exists():
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config.pop("active_worktree_session", None)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass


# =============================================================================
# Session Switching
# =============================================================================


def switch_session(
    new_session_id: str,
    project_dir: Optional[str] = None,
) -> None:
    """Switch to a different session.

    Args:
        new_session_id: New session identifier
        project_dir: Optional project directory (for cross-project resume)
    """
    # This is primarily used for cross-project resume
    # The actual state update should be done via AppState
    pass


# =============================================================================
# Message Text Extraction
# =============================================================================


def extract_message_text(message: dict[str, Any]) -> str:
    """Extract searchable text from a message.

    Args:
        message: Message dict

    Returns:
        Extracted text
    """
    msg_type = message.get("type", "")

    if msg_type == "user":
        return _extract_user_text(message)
    elif msg_type == "assistant":
        return _extract_assistant_text(message)
    elif msg_type == "attachment":
        return _extract_attachment_text(message)

    return ""


def _extract_user_text(message: dict[str, Any]) -> str:
    """Extract text from user message."""
    content = message.get("message", {}).get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    # Extract tool result content
                    result = block.get("content", "")
                    if isinstance(result, str):
                        texts.append(result)
        return " ".join(texts)

    return ""


def _extract_assistant_text(message: dict[str, Any]) -> str:
    """Extract text from assistant message."""
    content = message.get("message", {}).get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    # Extract tool use inputs
                    input_dict = block.get("input", {})
                    for key in ["command", "pattern", "file_path", "prompt", "query"]:
                        if key in input_dict:
                            texts.append(str(input_dict[key]))
        return " ".join(texts)

    return ""


def _extract_attachment_text(message: dict[str, Any]) -> str:
    """Extract text from attachment message."""
    attachment = message.get("attachment", {})

    if attachment.get("type") == "relevant_memories":
        memories = attachment.get("memories", [])
        return " ".join(m.get("content", "") for m in memories)

    return attachment.get("content", "")