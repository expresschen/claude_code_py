"""Side Query - Lightweight API wrapper for queries outside main conversation.

This implements the sideQuery mechanism from sideQuery.ts for making
lightweight API calls for classification, selection, and validation tasks.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from .api_config import get_api_config


# =============================================================================
# Types
# =============================================================================


class QuerySource(str, Enum):
    """Source of side query."""

    PERMISSION_EXPLAINER = "permission_explainer"
    SESSION_SEARCH = "session_search"
    MODEL_VALIDATION = "model_validation"
    MEMDIR_RELEVANCE = "memdir_relevance"
    MEMORY_EXTRACTION = "memory_extraction"
    COMMAND_CLASSIFIER = "command_classifier"


@dataclass
class SideQueryOptions:
    """Options for sideQuery."""

    model: str
    system: Optional[Union[str, list[dict[str, Any]]]] = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[dict[str, Any]] = None
    output_format: Optional[dict[str, Any]] = None
    max_tokens: int = 1024
    max_retries: int = 2
    temperature: Optional[float] = None
    thinking: Optional[Union[int, bool]] = None
    stop_sequences: Optional[list[str]] = None
    query_source: QuerySource = QuerySource.MEMDIR_RELEVANCE
    skip_system_prompt_prefix: bool = False


@dataclass
class SideQueryResult:
    """Result from sideQuery."""

    content: list[dict[str, Any]]
    usage: dict[str, int]
    model: str
    stop_reason: Optional[str] = None
    request_id: Optional[str] = None


# =============================================================================
# Default Models
# =============================================================================


def get_default_sonnet_model() -> str:
    """Get the default Sonnet model for side queries.

    Returns:
        Default Sonnet model identifier
    """
    config = get_api_config()
    return config.default_sonnet or config.model


def get_default_haiku_model() -> str:
    """Get the default Haiku model for lightweight tasks.

    Returns:
        Default Haiku model identifier
    """
    config = get_api_config()
    return config.default_haiku


def get_small_fast_model() -> str:
    """Get the small fast model.

    Returns:
        Small fast model identifier
    """
    import os
    return os.environ.get("ANTHROPIC_SMALL_FAST_MODEL", get_default_haiku_model())


# =============================================================================
# JSON Schema for Structured Output
# =============================================================================


MEMORY_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_memories": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of memory filenames that are relevant to the query",
        },
    },
    "required": ["selected_memories"],
    "additionalProperties": False,
}


# =============================================================================
# System Prompts
# =============================================================================


SELECT_MEMORIES_SYSTEM_PROMPT = """You are selecting memories that will be useful to Claude Code as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a list of filenames for the memories that will clearly be useful to Claude Code as it processes the user's query (up to 5). Only include memories that you are certain will be helpful based on their name and description.

- If you are unsure if a memory will be useful in processing the user's query, then do not include it in your list. Be selective and discerning.
- If there are no memories in the list that would clearly be useful, feel free to return an empty list.
- If a list of recently-used tools is provided, do not select memories that are usage reference or API documentation for those tools (Claude Code is already exercising them). DO still select memories containing warnings, gotchas, or known issues about those tools — active use is exactly when those matter.
"""


# =============================================================================
# Attribution Headers
# =============================================================================


def get_attribution_header(fingerprint: Optional[str] = None) -> str:
    """Get the attribution header for API calls.

    Args:
        fingerprint: Optional fingerprint for OAuth validation

    Returns:
        Attribution header string
    """
    version = "0.1.0"  # Package version
    entrypoint = "side_query"

    if fingerprint:
        return f"cc_version={version}; cc_entrypoint={entrypoint}; cc_fingerprint={fingerprint}"

    return f"cc_version={version}; cc_entrypoint={entrypoint}"


def compute_fingerprint(text: str, version: str) -> str:
    """Compute a fingerprint for OAuth attribution.

    Args:
        text: Message text
        version: Version string

    Returns:
        Fingerprint string
    """
    import hashlib

    combined = f"{text}:{version}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# =============================================================================
# Main Side Query Function
# =============================================================================


async def side_query(opts: SideQueryOptions) -> SideQueryResult:
    """Execute a lightweight side query to the API.

    This is a wrapper around the Anthropic API for making queries
    outside the main conversation loop. Used for:
    - Memory relevance selection
    - Permission explanation
    - Session search
    - Model validation

    Args:
        opts: Side query options

    Returns:
        SideQueryResult with content blocks and usage

    Raises:
        ImportError: If anthropic package is not installed
    """
    if not HAS_ANTHROPIC:
        raise ImportError(
            "anthropic package is required for sideQuery. "
            "Install with: pip install anthropic"
        )

    # Extract options
    model = opts.model
    system = opts.system
    messages = opts.messages
    max_tokens = opts.max_tokens
    output_format = opts.output_format
    temperature = opts.temperature
    stop_sequences = opts.stop_sequences

    # Compute fingerprint from first user message
    first_user_text = extract_first_user_message_text(messages)
    fingerprint = compute_fingerprint(first_user_text, "0.1.0")
    attribution = get_attribution_header(fingerprint)

    # Build system blocks
    system_blocks = build_system_blocks(
        attribution,
        system,
        opts.skip_system_prompt_prefix,
    )

    # Normalize model (strip [1m] suffix if present)
    normalized_model = normalize_model_string(model)

    # Build request
    request_params: dict[str, Any] = {
        "model": normalized_model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": messages,
    }

    if opts.tools:
        request_params["tools"] = opts.tools

    if opts.tool_choice:
        request_params["tool_choice"] = opts.tool_choice

    if output_format:
        # Use structured output beta
        request_params["betas"] = ["structured-outputs-2025-01-24"]
        request_params["response_format"] = output_format

    if temperature is not None:
        request_params["temperature"] = temperature

    if stop_sequences:
        request_params["stop_sequences"] = stop_sequences

    if opts.thinking is not None:
        if opts.thinking is False:
            request_params["thinking"] = {"type": "disabled"}
        else:
            thinking_budget = min(opts.thinking, max_tokens - 1)
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

    # Create client and make request
    config = get_api_config()
    client = anthropic.AsyncAnthropic(**config.to_anthropic_kwargs())

    retries = 0
    last_error = None

    while retries <= opts.max_retries:
        try:
            response = await client.messages.create(**request_params)

            # Extract result
            content_blocks = [
                {"type": block.type, "text": block.text}
                if hasattr(block, "text")
                else {"type": block.type, "content": block.content}
                for block in response.content
            ]

            return SideQueryResult(
                content=content_blocks,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_input_tokens": getattr(
                        response.usage, "cache_read_input_tokens", 0
                    ),
                    "cache_creation_input_tokens": getattr(
                        response.usage, "cache_creation_input_tokens", 0
                    ),
                },
                model=response.model,
                stop_reason=response.stop_reason,
                request_id=getattr(response, "_request_id", None),
            )

        except Exception as e:
            last_error = e
            retries += 1
            if retries <= opts.max_retries:
                await asyncio.sleep(0.5 * retries)  # Exponential backoff

    raise last_error if last_error else RuntimeError("sideQuery failed")


# =============================================================================
# Helper Functions
# =============================================================================


def extract_first_user_message_text(messages: list[dict[str, Any]]) -> str:
    """Extract text from first user message.

    Args:
        messages: Message list

    Returns:
        First user message text or empty string
    """
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content

            # Array of content blocks
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        return block.get("text", "")

    return ""


def build_system_blocks(
    attribution: str,
    system: Optional[Union[str, list[dict[str, Any]]]],
    skip_prefix: bool = False,
) -> list[dict[str, Any]]:
    """Build system blocks for the request.

    Args:
        attribution: Attribution header
        system: System prompt content
        skip_prefix: Whether to skip CLI prefix

    Returns:
        List of system text blocks
    """
    blocks: list[dict[str, Any]] = []

    # Add attribution header in its own block
    blocks.append({"type": "text", "text": attribution})

    # Add CLI prefix (unless skipped)
    if not skip_prefix:
        blocks.append({
            "type": "text",
            "text": "Claude Code Python Implementation",
        })

    # Add provided system content
    if system:
        if isinstance(system, str):
            blocks.append({"type": "text", "text": system})
        elif isinstance(system, list):
            blocks.extend(system)

    return blocks


def normalize_model_string(model: str) -> str:
    """Normalize model string for API.

    Strips [1m] suffix and other annotations.

    Args:
        model: Model string

    Returns:
        Normalized model string
    """
    # Strip [1m] or similar suffixes
    if "[" in model:
        return model.split("[")[0].strip()

    return model.strip()


# =============================================================================
# Memory Selection via sideQuery
# =============================================================================


async def select_relevant_memories_with_model(
    query: str,
    memories: list[dict[str, Any]],
    recent_tools: Optional[list[str]] = None,
) -> list[str]:
    """Select relevant memories using Sonnet model.

    Args:
        query: User query
        memories: List of memory headers
        recent_tools: Recently used tools (to exclude their docs)

    Returns:
        List of selected memory filenames
    """
    # Format manifest
    manifest = format_memory_manifest(memories)

    # Build tools section
    tools_section = ""
    if recent_tools and len(recent_tools) > 0:
        tools_section = f"\n\nRecently used tools: {', '.join(recent_tools)}"

    # Build message
    user_message = f"Query: {query}\n\nAvailable memories:\n{manifest}{tools_section}"

    # Create options
    opts = SideQueryOptions(
        model=get_default_sonnet_model(),
        system=SELECT_MEMORIES_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=256,
        output_format=MEMORY_SELECTION_SCHEMA,
        query_source=QuerySource.MEMDIR_RELEVANCE,
        skip_system_prompt_prefix=True,  # Use our own prompt
    )

    # Execute query
    result = await side_query(opts)

    # Parse response
    text_block = None
    for block in result.content:
        if block.get("type") == "text":
            text_block = block
            break

    if not text_block:
        return []

    try:
        parsed = json.loads(text_block.get("text", "{}"))
        selected = parsed.get("selected_memories", [])

        # Filter to valid filenames
        valid_filenames = {m.get("name") or m.get("filename") for m in memories}
        return [f for f in selected if f in valid_filenames]

    except json.JSONDecodeError:
        return []


def format_memory_manifest(memories: list[dict[str, Any]]) -> str:
    """Format memory headers as a manifest string.

    Args:
        memories: List of memory headers

    Returns:
        Manifest string
    """
    lines = []

    for m in memories:
        tag = m.get("type", "")
        tag_str = f"[{tag}] " if tag else ""

        filename = m.get("name") or m.get("filename", "unknown")

        description = m.get("description", "")
        if description:
            lines.append(f"- {tag_str}{filename}: {description}")
        else:
            lines.append(f"- {tag_str}{filename}")

    if not lines:
        return "No memory files found."

    return "\n".join(lines)


# =============================================================================
# Fallback: Simple Keyword Selection
# =============================================================================


def select_relevant_memories_by_keywords(
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
        List of selected memory filenames
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

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

        if score > 0 or len(scored) < max_results:
            scored.append((score, m.get("name") or m.get("filename", "unknown")))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])

    return [name for score, name in scored[:max_results]]