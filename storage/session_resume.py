"""Session Resume - Conversation recovery and message chain building.

This implements the session resume functionality from conversationRecovery.ts
for loading and deserializing conversations with proper chain reconstruction.

Key features:
- buildConversationChain: Build chain from leaf to root via parentUuid
- filterUnresolvedToolUses: Remove unmatched tool_use/tool_result pairs
- filterOrphanedThinkingOnlyMessages: Remove orphaned thinking blocks
- detectTurnInterruption: Detect interrupted sessions
- deserializeMessages: Prepare messages for API-valid resume
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union
from uuid import UUID

from claude_code_py.core_types.message import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    AttachmentMessage,
)


# =============================================================================
# Constants
# =============================================================================

NO_RESPONSE_REQUESTED = "[No response requested - conversation was interrupted]"

# Valid permission modes (for deserialization validation)
PERMISSION_MODES = ["accept-all", "plan", "auto", "default"]


# =============================================================================
# Turn Interruption Types
# =============================================================================


class TurnInterruptionKind(str, Enum):
    """Kind of turn interruption."""

    NONE = "none"
    INTERRUPTED_PROMPT = "interrupted_prompt"
    INTERRUPTED_TURN = "interrupted_turn"


@dataclass
class TurnInterruptionState:
    """State of turn interruption."""

    kind: TurnInterruptionKind
    message: Optional[UserMessage] = None


@dataclass
class DeserializeResult:
    """Result of deserializing messages."""

    messages: list[Message]
    turn_interruption_state: TurnInterruptionState


# =============================================================================
# Conversation Chain Building
# =============================================================================


def build_conversation_chain(
    messages: dict[str, Message],
    leaf_message: Message,
) -> list[Message]:
    """Build a conversation chain from leaf message to root.

    Walks the parentUuid chain backwards from the leaf to the root,
    collecting messages along the way.

    Args:
        messages: Map of UUID to message
        leaf_message: The leaf message to start from

    Returns:
        List of messages from root to leaf
    """
    transcript: list[Message] = []
    seen: set[str] = set()
    current_msg: Optional[Message] = leaf_message

    while current_msg:
        msg_uuid = getattr(current_msg, "uuid", None)
        if not msg_uuid:
            break

        if msg_uuid in seen:
            # Cycle detected - return partial transcript
            break

        seen.add(msg_uuid)
        transcript.append(current_msg)

        # Get parent UUID
        parent_uuid = getattr(current_msg, "parentUuid", None)
        if parent_uuid:
            current_msg = messages.get(parent_uuid)
        else:
            current_msg = None

    # Reverse to get root -> leaf order
    transcript.reverse()

    # Recover orphaned parallel tool results
    return recover_orphaned_parallel_tool_results(messages, transcript, seen)


def recover_orphaned_parallel_tool_results(
    all_messages: dict[str, Message],
    chain: list[Message],
    seen: set[str],
) -> list[Message]:
    """Recover sibling assistant blocks and tool_results orphaned by single-parent walk.

    Streaming emits one AssistantMessage per content_block_stop - N parallel
    tool_uses result in N messages with distinct UUIDs but same message.id.
    The single-parent walk only keeps one branch; this pass recovers the others.

    Args:
        all_messages: Map of all messages by UUID
        chain: Current chain (from build_conversation_chain)
        seen: Set of UUIDs already in chain

    Returns:
        Chain with orphaned siblings recovered
    """
    # Find assistant messages in chain with message.id
    chain_assistants = [
        m for m in chain
        if isinstance(m, AssistantMessage) and m.message.get("id")
    ]

    if not chain_assistants:
        return chain

    # Build anchor map: message.id -> last on-chain assistant in group
    anchor_by_msg_id: dict[str, AssistantMessage] = {}
    for a in chain_assistants:
        msg_id = a.message.get("id")
        if msg_id:
            anchor_by_msg_id[msg_id] = a

    # Find sibling groups and tool results by parent
    siblings_by_msg_id: dict[str, list[Message]] = {}
    tool_results_by_parent: dict[str, list[Message]] = {}

    for m in all_messages.values():
        if isinstance(m, AssistantMessage):
            msg_id = m.message.get("id")
            if msg_id:
                if msg_id not in siblings_by_msg_id:
                    siblings_by_msg_id[msg_id] = []
                siblings_by_msg_id[msg_id].append(m)

        elif isinstance(m, UserMessage):
            parent_uuid = getattr(m, "parentUuid", None)
            content = m.message.get("content", [])
            if parent_uuid and isinstance(content, list):
                # Check if it has tool_result blocks
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                )
                if has_tool_result:
                    if parent_uuid not in tool_results_by_parent:
                        tool_results_by_parent[parent_uuid] = []
                    tool_results_by_parent[parent_uuid].append(m)

    # Collect off-chain siblings and their tool results
    inserts: dict[str, list[Message]] = {}
    processed_groups: set[str] = set()

    for msg_id, anchor in anchor_by_msg_id.items():
        if msg_id in processed_groups:
            continue
        processed_groups.add(msg_id)

        # Get off-chain siblings
        siblings = siblings_by_msg_id.get(msg_id, [])
        off_chain = [s for s in siblings if getattr(s, "uuid", None) not in seen]

        if not off_chain:
            continue

        # Collect tool results for ALL members (on-chain + off-chain)
        to_insert: list[Message] = []
        for sib in siblings:
            sib_uuid = getattr(sib, "uuid", None)
            if sib_uuid:
                trs = tool_results_by_parent.get(sib_uuid, [])
                to_insert.extend(trs)

        to_insert.extend(off_chain)

        if to_insert:
            anchor_uuid = getattr(anchor, "uuid", None)
            if anchor_uuid:
                inserts[anchor_uuid] = to_insert

    # Splice inserts into chain
    result: list[Message] = []
    for m in chain:
        result.append(m)
        m_uuid = getattr(m, "uuid", None)
        if m_uuid and m_uuid in inserts:
            result.extend(inserts[m_uuid])

    return result


# =============================================================================
# Message Filtering
# =============================================================================


def filter_unresolved_tool_uses(messages: list[Message]) -> list[Message]:
    """Filter out unresolved tool uses and their synthetic followers.

    Removes assistant messages with tool_use blocks that have no matching
    tool_result in a subsequent user message. Also removes user messages
    that follow an unresolved assistant message (synthetic responses).

    Args:
        messages: Original message list

    Returns:
        Filtered message list
    """
    # Collect all tool_use IDs and tool_result IDs
    tool_use_ids: set[str] = set()
    tool_result_ids: set[str] = set()

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            content = msg.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_use_ids.add(block.get("id", ""))

        elif isinstance(msg, UserMessage):
            content = msg.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_result_ids.add(block.get("tool_use_id", ""))

    # Find unresolved tool_use IDs
    unresolved_ids = tool_use_ids - tool_result_ids

    if not unresolved_ids:
        return messages

    # Filter out messages with unresolved tool_uses
    result: list[Message] = []
    skip_next_user = False

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            content = msg.message.get("content", [])
            if isinstance(content, list):
                # Check if any tool_use is unresolved
                has_unresolved = any(
                    isinstance(b, dict)
                    and b.get("type") == "tool_use"
                    and b.get("id") in unresolved_ids
                    for b in content
                )
                if has_unresolved:
                    skip_next_user = True
                    continue

        elif isinstance(msg, UserMessage):
            if skip_next_user:
                # Check if it's a synthetic response
                content = msg.message.get("content", "")
                if isinstance(content, str) and not content.strip():
                    skip_next_user = False
                    continue
                # Keep meaningful user messages
                skip_next_user = False

        result.append(msg)

    return result


def filter_orphaned_thinking_only_messages(messages: list[Message]) -> list[Message]:
    """Filter orphaned thinking-only assistant messages.

    During streaming, thinking blocks can be emitted as separate messages.
    When loaded for resume, these orphaned thinking-only messages cause
    "thinking blocks cannot be modified" API errors.

    A thinking-only message is orphaned if no other assistant with the
    same message.id contains non-thinking content.

    Args:
        messages: Original message list

    Returns:
        Filtered message list
    """
    # First pass: collect message.ids with non-thinking content
    message_ids_with_non_thinking: set[str] = set()

    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue

        content = msg.message.get("content", [])
        if not isinstance(content, list):
            continue

        has_non_thinking = any(
            isinstance(b, dict)
            and b.get("type") not in ("thinking", "redacted_thinking")
            for b in content
        )

        if has_non_thinking and msg.message.get("id"):
            message_ids_with_non_thinking.add(msg.message.get("id"))

    # Second pass: filter orphaned thinking-only messages
    result: list[Message] = []

    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            result.append(msg)
            continue

        content = msg.message.get("content", [])
        if not isinstance(content, list):
            result.append(msg)
            continue

        # Check if it's thinking-only
        is_thinking_only = all(
            isinstance(b, dict)
            and b.get("type") in ("thinking", "redacted_thinking")
            for b in content
        )

        if is_thinking_only:
            msg_id = msg.message.get("id")
            if msg_id and msg_id not in message_ids_with_non_thinking:
                # Orphaned - skip
                continue

        result.append(msg)

    return result


def filter_whitespace_only_assistant_messages(messages: list[Message]) -> list[Message]:
    """Filter assistant messages with only whitespace text content.

    These can occur when model outputs whitespace before thinking and
    user cancels mid-stream.

    Args:
        messages: Original message list

    Returns:
        Filtered message list
    """
    result: list[Message] = []

    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            result.append(msg)
            continue

        content = msg.message.get("content", [])
        if not isinstance(content, list):
            result.append(msg)
            continue

        # Check if only whitespace text blocks
        has_non_whitespace = False
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text.strip():
                        has_non_whitespace = True
                        break
                elif block.get("type") not in ("thinking", "redacted_thinking"):
                    has_non_whitespace = True
                    break

        if has_non_whitespace:
            result.append(msg)

    return result


# =============================================================================
# Turn Interruption Detection
# =============================================================================


def detect_turn_interruption(messages: list[Message]) -> TurnInterruptionState:
    """Detect whether the conversation was interrupted mid-turn.

    Based on the last message after filtering:
    - Assistant as last: completed turn (no interruption)
    - User as last with isMeta/isCompactSummary: no interruption
    - User as last with tool_result: interrupted_turn (unless terminal tool)
    - User as last with plain text: interrupted_prompt

    Args:
        messages: Filtered message list

    Returns:
        TurnInterruptionState
    """
    if not messages:
        return TurnInterruptionState(kind=TurnInterruptionKind.NONE)

    # Find last turn-relevant message (skip system/progress)
    last_relevant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        msg_type = getattr(msg, "type", None)
        if msg_type not in ("system", "progress"):
            # Skip API error assistant messages
            if isinstance(msg, AssistantMessage) and getattr(msg, "is_api_error_message", False):
                continue
            last_relevant_idx = i
            break

    if last_relevant_idx == -1:
        return TurnInterruptionState(kind=TurnInterruptionKind.NONE)

    last_message = messages[last_relevant_idx]

    if isinstance(last_message, AssistantMessage):
        # Assistant as last message means turn completed
        return TurnInterruptionState(kind=TurnInterruptionKind.NONE)

    if isinstance(last_message, UserMessage):
        # Check for meta or compact summary
        if getattr(last_message, "is_meta", False) or getattr(last_message, "is_compact_summary", False):
            return TurnInterruptionState(kind=TurnInterruptionKind.NONE)

        # Check for tool result message
        content = last_message.message.get("content", [])
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        ):
            # Could be interrupted_turn or terminal tool result
            # For now, treat as interrupted_turn
            return TurnInterruptionState(kind=TurnInterruptionKind.INTERRUPTED_TURN)

        # Plain text user prompt - interrupted before response started
        return TurnInterruptionState(
            kind=TurnInterruptionKind.INTERRUPTED_PROMPT,
            message=last_message,
        )

    if isinstance(last_message, AttachmentMessage):
        # Attachment without response
        return TurnInterruptionState(kind=TurnInterruptionKind.INTERRUPTED_TURN)

    return TurnInterruptionState(kind=TurnInterruptionKind.NONE)


# =============================================================================
# Message Deserialization
# =============================================================================


def create_user_message(
    content: str,
    is_meta: bool = False,
) -> UserMessage:
    """Create a synthetic user message.

    Args:
        content: Message content
        is_meta: Whether this is a meta message

    Returns:
        UserMessage
    """
    from uuid import uuid4

    return UserMessage(
        uuid=str(uuid4()),
        message={
            "role": "user",
            "content": content,
        },
        is_meta=is_meta,
    )


def create_assistant_message(
    content: str,
) -> AssistantMessage:
    """Create a synthetic assistant message.

    Args:
        content: Message content

    Returns:
        AssistantMessage
    """
    from uuid import uuid4

    return AssistantMessage(
        uuid=str(uuid4()),
        message={
            "role": "assistant",
            "content": [{"type": "text", "text": content}],
        },
    )


def deserialize_messages(serialized_messages: list[Message]) -> DeserializeResult:
    """Deserialize messages from a log file for REPL/API use.

    Filters unresolved tool uses, orphaned thinking messages, and
    appends synthetic assistant sentinel when last message is user.

    Args:
        serialized_messages: Messages from session file

    Returns:
        DeserializeResult with messages and interruption state
    """
    # Filter invalid permission modes from user messages
    for msg in serialized_messages:
        if isinstance(msg, UserMessage):
            mode = msg.message.get("permissionMode")
            if mode and mode not in PERMISSION_MODES:
                msg.message.pop("permissionMode", None)

    # Filter unresolved tool uses
    filtered_tool_uses = filter_unresolved_tool_uses(serialized_messages)

    # Filter orphaned thinking-only messages
    filtered_thinking = filter_orphaned_thinking_only_messages(filtered_tool_uses)

    # Filter whitespace-only assistant messages
    filtered_messages = filter_whitespace_only_assistant_messages(filtered_thinking)

    # Detect interruption state
    internal_state = detect_turn_interruption(filtered_messages)

    # Transform interrupted_turn into interrupted_prompt with continuation
    turn_interruption_state: TurnInterruptionState
    if internal_state.kind == TurnInterruptionKind.INTERRUPTED_TURN:
        continuation_message = create_user_message(
            "Continue from where you left off.",
            is_meta=True,
        )
        filtered_messages.append(continuation_message)
        turn_interruption_state = TurnInterruptionState(
            kind=TurnInterruptionKind.INTERRUPTED_PROMPT,
            message=continuation_message,
        )
    else:
        turn_interruption_state = internal_state

    # Append synthetic assistant sentinel after last user message
    # Find last relevant message (skip system/progress)
    last_relevant_idx = -1
    for i in range(len(filtered_messages) - 1, -1, -1):
        msg = filtered_messages[i]
        msg_type = getattr(msg, "type", None)
        if msg_type not in ("system", "progress"):
            last_relevant_idx = i
            break

    if last_relevant_idx != -1:
        last_relevant = filtered_messages[last_relevant_idx]
        if isinstance(last_relevant, UserMessage):
            # Insert sentinel right after user message
            sentinel = create_assistant_message(NO_RESPONSE_REQUESTED)
            filtered_messages.insert(last_relevant_idx + 1, sentinel)

    return DeserializeResult(
        messages=filtered_messages,
        turn_interruption_state=turn_interruption_state,
    )


# =============================================================================
# Skill State Restoration
# =============================================================================


def restore_skill_state_from_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Restore skill state from invoked_skills attachments.

    Args:
        messages: Message list

    Returns:
        List of restored skills
    """
    restored_skills: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, AttachmentMessage):
            continue

        attachment = msg.attachment
        if attachment.get("type") == "invoked_skills":
            for skill in attachment.get("skills", []):
                if skill.get("name") and skill.get("path") and skill.get("content"):
                    restored_skills.append({
                        "name": skill["name"],
                        "path": skill["path"],
                        "content": skill["content"],
                    })

    return restored_skills


# =============================================================================
# Transcript File Loading
# =============================================================================


async def load_messages_from_jsonl_path(path: str) -> dict[str, Any]:
    """Load messages from a JSONL transcript file.

    Args:
        path: Path to JSONL file

    Returns:
        Dict with messages and session_id
    """
    import json
    from pathlib import Path

    file_path = Path(path)
    if not file_path.exists():
        return {"messages": [], "session_id": None}

    # Parse JSONL
    messages_by_uuid: dict[str, Message] = {}
    leaf_uuids: set[str] = set()
    children_by_parent: dict[str, list[str]] = {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Check if it's a transcript message
                msg_type = entry.get("type")
                if msg_type in ("user", "assistant", "attachment", "system"):
                    msg_uuid = entry.get("uuid")
                    if msg_uuid:
                        # Convert to appropriate message type
                        messages_by_uuid[msg_uuid] = _convert_entry_to_message(entry)

                        # Track parent-child relationships
                        parent_uuid = entry.get("parentUuid")
                        if parent_uuid:
                            if parent_uuid not in children_by_parent:
                                children_by_parent[parent_uuid] = []
                            children_by_parent[parent_uuid].append(msg_uuid)

        # Find leaf UUIDs (no children)
        for msg_uuid in messages_by_uuid:
            if msg_uuid not in children_by_parent:
                leaf_uuids.add(msg_uuid)

        # Find newest non-sidechain leaf
        tip_uuid: Optional[str] = None
        tip_ts = 0

        for msg_uuid in leaf_uuids:
            msg = messages_by_uuid.get(msg_uuid)
            if msg and not getattr(msg, "is_sidechain", False):
                ts = getattr(msg, "timestamp", 0)
                if ts > tip_ts:
                    tip_ts = ts
                    tip_uuid = msg_uuid

        if not tip_uuid:
            return {"messages": [], "session_id": None}

        # Build chain
        tip_message = messages_by_uuid.get(tip_uuid)
        if not tip_message:
            return {"messages": [], "session_id": None}

        chain = build_conversation_chain(messages_by_uuid, tip_message)

        # Get session ID from tip message
        session_id = getattr(tip_message, "sessionId", None)

        return {
            "messages": chain,
            "session_id": session_id,
        }

    except Exception:
        return {"messages": [], "session_id": None}


def _convert_entry_to_message(entry: dict[str, Any]) -> Message:
    """Convert a JSONL entry to appropriate Message type.

    Args:
        entry: Raw entry dict

    Returns:
        Appropriate Message subclass
    """
    msg_type = entry.get("type")

    if msg_type == "user":
        return UserMessage(
            uuid=entry.get("uuid", ""),
            message=entry.get("message", {}),
            is_meta=entry.get("isMeta", False),
            is_compact_summary=entry.get("isCompactSummary", False),
        )

    elif msg_type == "assistant":
        return AssistantMessage(
            uuid=entry.get("uuid", ""),
            message=entry.get("message", {}),
            stop_reason=entry.get("stopReason"),
            is_api_error_message=entry.get("isApiErrorMessage", False),
        )

    elif msg_type == "system":
        return SystemMessage(
            uuid=entry.get("uuid", ""),
            subtype=entry.get("subtype"),
            content=entry.get("content"),
            compact_metadata=entry.get("compactMetadata"),
        )

    elif msg_type == "attachment":
        return AttachmentMessage(
            uuid=entry.get("uuid", ""),
            attachment=entry.get("attachment", {}),
        )

    # Default to UserMessage
    return UserMessage(
        uuid=entry.get("uuid", ""),
        message=entry.get("message", {}),
    )


# =============================================================================
# Full Resume Function
# =============================================================================


async def load_conversation_for_resume(
    source: Optional[str] = None,
    source_jsonl_file: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Load a conversation for resume from various sources.

    Centralized function for loading and deserializing conversations.

    Args:
        source: Session ID to load, or None for most recent
        source_jsonl_file: Path to JSONL file (for cross-directory resume)

    Returns:
        Dict with messages, session_id, and metadata, or None if not found
    """
    from claude_code_py.storage.session import (
        get_session_path,
        get_session_dir,
        list_sessions,
        SessionStorage,
    )

    messages: list[Message] = []
    session_id: Optional[str] = None

    if source_jsonl_file:
        # Load from specific JSONL path
        result = await load_messages_from_jsonl_path(source_jsonl_file)
        messages = result.get("messages", [])
        session_id = result.get("session_id")

    elif source is None:
        # Load most recent session
        sessions = list_sessions()
        if not sessions:
            return None

        # Get most recent session
        most_recent = sessions[0]
        session_id = most_recent.session_id

        # Load messages from session
        storage = SessionStorage(session_id)
        raw_messages = storage.load_messages()

        # Convert to Message objects
        messages = [_convert_entry_to_message(m) for m in raw_messages]

    else:
        # Load specific session by ID
        session_id = source
        storage = SessionStorage(session_id)
        raw_messages = storage.load_messages()
        messages = [_convert_entry_to_message(m) for m in raw_messages]

    if not messages:
        return None

    # Deserialize messages
    deserialized = deserialize_messages(messages)

    # Restore skill state
    restored_skills = restore_skill_state_from_messages(deserialized.messages)

    return {
        "messages": deserialized.messages,
        "turn_interruption_state": deserialized.turn_interruption_state,
        "session_id": session_id,
        "restored_skills": restored_skills,
    }