"""Session Search - Find relevant past sessions.

This implements agentic session search from agenticSessionSearch.ts
for finding relevant conversations using model-based semantic search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .session import (
    LogOption,
    list_sessions,
    SessionStorage,
)


# =============================================================================
# Constants
# =============================================================================

MAX_TRANSCRIPT_CHARS = 2000
MAX_MESSAGES_TO_SCAN = 100
MAX_SESSIONS_TO_SEARCH = 100


# =============================================================================
# System Prompt
# =============================================================================

SESSION_SEARCH_SYSTEM_PROMPT = """Your goal is to find relevant sessions based on a user's search query.

You will be given a list of sessions with their metadata and a search query. Identify which sessions are most relevant to the query.

Each session may include:
- Title (display name or custom title)
- Tag (user-assigned category, shown as [tag: name] - users tag sessions with /tag command to categorize them)
- Branch (git branch name, shown as [branch: name])
- Summary (AI-generated summary)
- First message (beginning of the conversation)
- Transcript (excerpt of conversation content)

IMPORTANT: Tags are user-assigned labels that indicate the session's topic or category. If the query matches a tag exactly or partially, those sessions should be highly prioritized.

For each session, consider (in order of priority):
1. Exact tag matches (highest priority - user explicitly categorized this session)
2. Partial tag matches or tag-related terms
3. Title matches (custom titles or first message content)
4. Branch name matches
5. Summary and transcript content matches
6. Semantic similarity and related concepts

CRITICAL: Be VERY inclusive in your matching. Include sessions that:
- Contain the query term anywhere in any field
- Are semantically related to the query (e.g., "testing" matches sessions about "tests", "unit tests", "QA", etc.)
- Discuss topics that could be related to the query
- Have transcripts that mention the concept even in passing

When in doubt, INCLUDE the session. It's better to return too many results than too few. The user can easily scan through results, but missing relevant sessions is frustrating.

Return sessions ordered by relevance (most relevant first). If truly no sessions have ANY connection to the query, return an empty array - but this should be rare.

Respond with ONLY the JSON object, no markdown formatting:
{"relevant_indices": [2, 5, 0]}"""


# =============================================================================
# Search Result
# =============================================================================

@dataclass
class AgenticSearchResult:
    """Result from agentic search."""

    relevant_indices: list[int]


# =============================================================================
# Main Search Functions
# =============================================================================


def log_contains_query(log: LogOption, query_lower: str) -> bool:
    """Check if a log contains the query term.

    Args:
        log: Session log
        query_lower: Lowercase query

    Returns:
        True if log contains query
    """
    # Check title
    if log.display_name.lower().find(query_lower) >= 0:
        return True

    # Check custom title
    if log.custom_title and log.custom_title.lower().find(query_lower) >= 0:
        return True

    # Check tag
    if log.tag and log.tag.lower().find(query_lower) >= 0:
        return True

    # Check branch
    if log.git_branch and log.git_branch.lower().find(query_lower) >= 0:
        return True

    # Check summary
    if log.summary and log.summary.lower().find(query_lower) >= 0:
        return True

    # Check first prompt
    if log.first_prompt and log.first_prompt.lower().find(query_lower) >= 0:
        return True

    # Check transcript
    if log.messages:
        storage = SessionStorage(log.session_id)
        transcript = storage.load_transcript()
        if transcript.lower().find(query_lower) >= 0:
            return True

    return False


async def agentic_session_search(
    query: str,
    logs: Optional[list[LogOption]] = None,
    signal: Any = None,
) -> list[LogOption]:
    """Search for relevant sessions using model-based semantic search.

    Args:
        query: Search query
        logs: Optional list of logs (loads all if not provided)
        signal: Abort signal (not used in Python, for API compatibility)

    Returns:
        List of relevant LogOptions
    """
    if not query.strip():
        return []

    # Load logs if not provided
    if logs is None:
        logs = list_sessions()

    if not logs:
        return []

    query_lower = query.lower()

    # Pre-filter: find sessions that contain the query term
    matching_logs = [log for log in logs if log_contains_query(log, query_lower)]

    # Take up to MAX_SESSIONS_TO_SEARCH
    if len(matching_logs) >= MAX_SESSIONS_TO_SEARCH:
        logs_to_search = matching_logs[:MAX_SESSIONS_TO_SEARCH]
    else:
        non_matching = [log for log in logs if not log_contains_query(log, query_lower)]
        remaining = MAX_SESSIONS_TO_SEARCH - len(matching_logs)
        logs_to_search = matching_logs + non_matching[:remaining]

    if not logs_to_search:
        return []

    # Load transcripts for sessions
    for log in logs_to_search:
        if not log.messages:
            storage = SessionStorage(log.session_id)
            log.messages = storage.load_messages()

    # Build session list for the prompt
    session_list = _build_session_list(logs_to_search)

    # Try model-based search
    try:
        from claude_code_py.utils.side_query import side_query, SideQueryOptions, QuerySource, get_small_fast_model

        user_message = f"""Sessions:
{session_list}

Search query: "{query}"

Find the sessions that are most relevant to this query."""

        opts = SideQueryOptions(
            model=get_small_fast_model(),
            system=SESSION_SEARCH_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            query_source=QuerySource.SESSION_SEARCH,
            skip_system_prompt_prefix=True,
        )

        result = await side_query(opts)

        # Parse response
        text_content = None
        for block in result.content:
            if block.get("type") == "text":
                text_content = block.get("text", "")
                break

        if not text_content:
            return []

        # Extract JSON
        import json
        import re

        json_match = re.search(r"\{[\s\S]*\}", text_content)
        if not json_match:
            return []

        data = json.loads(json_match.group())
        relevant_indices = data.get("relevant_indices", [])

        # Map indices back to logs
        relevant_logs = []
        for idx in relevant_indices:
            if 0 <= idx < len(logs_to_search):
                relevant_logs.append(logs_to_search[idx])

        return relevant_logs

    except Exception:
        # Fallback: return matching logs
        return matching_logs


def _build_session_list(logs: list[LogOption]) -> str:
    """Build session list string for the prompt.

    Args:
        logs: List of session logs

    Returns:
        Formatted session list
    """
    lines = []

    for i, log in enumerate(logs):
        parts = [f"{i}:"]

        # Title
        parts.append(log.display_name)

        # Custom title
        if log.custom_title and log.custom_title != log.display_name:
            parts.append(f"[custom title: {log.custom_title}]")

        # Tag
        if log.tag:
            parts.append(f"[tag: {log.tag}]")

        # Git branch
        if log.git_branch:
            parts.append(f"[branch: {log.git_branch}]")

        # Summary
        if log.summary:
            parts.append(f"- Summary: {log.summary}")

        # First prompt
        if log.first_prompt and log.first_prompt != "No prompt":
            parts.append(f"- First message: {log.first_prompt[:300]}")

        # Transcript
        if log.messages:
            storage = SessionStorage(log.session_id)
            transcript = storage.load_transcript()
            if transcript:
                parts.append(f"- Transcript: {transcript}")

        lines.append(" ".join(parts))

    return "\n".join(lines)


def simple_session_search(query: str, logs: Optional[list[LogOption]] = None) -> list[LogOption]:
    """Simple keyword-based session search (no model).

    Args:
        query: Search query
        logs: Optional list of logs

    Returns:
        List of matching LogOptions
    """
    if not query.strip():
        return []

    if logs is None:
        logs = list_sessions()

    if not logs:
        return []

    query_lower = query.lower()
    return [log for log in logs if log_contains_query(log, query_lower)]