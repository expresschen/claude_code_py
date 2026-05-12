"""Dependency injection for query engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from claude_code_py.tool.base import Tool
from claude_code_py.core_types.message import Message


@dataclass
class QueryDeps:
    """Dependencies for the query function.

    This allows injecting different implementations for testing
    or different runtime environments.
    """

    # API client
    api_client: Optional[Any] = None

    # Tool executor
    run_tools: Optional[Callable] = None

    # Message normalizer
    normalize_messages: Optional[Callable] = None

    # Compactor
    compact: Optional[Callable] = None

    # Hook executor
    execute_hooks: Optional[Callable] = None

    # System prompt builder
    build_system_prompt: Optional[Callable] = None


def production_deps() -> QueryDeps:
    """Create production dependencies.

    Returns:
        QueryDeps with production implementations
    """
    from claude_code_py.orchestration.executor import run_tools
    from claude_code_py.engine.query import normalize_messages_for_api

    return QueryDeps(
        run_tools=run_tools,
        normalize_messages=normalize_messages_for_api,
    )


def test_deps(
    api_client: Optional[Any] = None,
    run_tools: Optional[Callable] = None,
    normalize_messages: Optional[Callable] = None,
) -> QueryDeps:
    """Create test dependencies.

    Args:
        api_client: Mock API client
        run_tools: Mock tool executor
        normalize_messages: Mock message normalizer

    Returns:
        QueryDeps with test implementations
    """
    return QueryDeps(
        api_client=api_client,
        run_tools=run_tools,
        normalize_messages=normalize_messages,
    )