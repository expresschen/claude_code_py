"""Compact service types.

This module contains the type definitions for compact services,
separated to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# =============================================================================
# Constants
# =============================================================================


MAX_COMPACT_TURNS = 20  # Maximum turns to keep after compact
POST_COMPACT_TOKEN_BUDGET = 100_000  # Target token budget after compact
POST_COMPACT_MAX_FILES_TO_RESTORE = 5  # Max files to restore in post-compact
POST_COMPACT_MAX_TOKENS_PER_FILE = 10_000  # Max tokens per restored file


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class CompactMetadata:
    """Metadata about a compact operation."""

    # Token counts
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_removed: int = 0

    # Message counts
    messages_before: int = 0
    messages_after: int = 0
    messages_removed: int = 0

    # Timing
    duration_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    # Compact type
    compact_type: str = "standard"  # standard, session_memory, microcompact, reactive

    # Additional info
    tool_uses_preserved: int = 0
    files_restored: list[str] = field(default_factory=list)


@dataclass
class CompactResult:
    """Result of a compact operation."""

    success: bool
    summary: Optional[str] = None
    compacted_messages: list[Any] = field(default_factory=list)
    original_messages: list[Any] = field(default_factory=list)
    metadata: Optional[CompactMetadata] = None
    error: Optional[str] = None

    # For session memory compact - pre-built messages
    boundary_marker: Optional[Any] = None  # SystemMessage with compact_boundary
    summary_messages: list[Any] = field(default_factory=list)  # UserMessage(s) with summary
    messages_to_keep: list[Any] = field(default_factory=list)  # Messages preserved after compact
    attachments: list[Any] = field(default_factory=list)  # AttachmentMessage(s)
    hook_results: list[Any] = field(default_factory=list)  # HookResultMessage(s)

    # Token counts (for session memory compact)
    pre_compact_token_count: Optional[int] = None
    post_compact_token_count: Optional[int] = None

    # For session memory compact
    memory_file_path: Optional[str] = None


@dataclass
class CompactOptions:
    """Options for compact operation."""

    # Compact type
    compact_type: str = "standard"

    # Thresholds
    max_tokens: Optional[int] = None
    max_messages: Optional[int] = None

    # Flags
    is_reactive: bool = False
    preserve_tool_uses: bool = True

    # Model override
    model: Optional[str] = None

    # Query source
    query_source: str = "sdk"


# =============================================================================
# State Types
# =============================================================================


@dataclass
class CompactState:
    """State tracked during compact operation."""

    turn_count: int = 0
    has_attempted_reactive_compact: bool = False
    consecutive_compact_failures: int = 0
    last_compact_result: Optional[CompactResult] = None