"""Prompt category utilities.

This implements the getQuerySourceForREPL() function from TypeScript's
promptCategory.ts to properly determine query source based on output style.
"""

from __future__ import annotations

from typing import Optional

from claude_code_py.constants import QuerySource, OutputStyle, is_main_thread_source


def get_query_source_for_repl(output_style: Optional[str] = None) -> str:
    """Get query source for REPL main thread based on output style.

    This matches TypeScript's getQuerySourceForREPL() implementation:
    - Default style -> 'repl_main_thread'
    - Built-in style -> 'repl_main_thread:outputStyle:<style>'
    - Custom style -> 'repl_main_thread:outputStyle:custom'

    Args:
        output_style: Current output style setting (None uses default)

    Returns:
        Query source string
    """
    if not output_style or output_style == OutputStyle.DEFAULT:
        return QuerySource.REPL_MAIN_THREAD

    # Built-in styles: Explanatory, Learning
    if output_style == OutputStyle.EXPLANATORY:
        return QuerySource.REPL_MAIN_THREAD_OUTPUT_STYLE_EXPLANATORY
    if output_style == OutputStyle.LEARNING:
        return QuerySource.REPL_MAIN_THREAD_OUTPUT_STYLE_LEARNING

    # Any other style is treated as custom
    return QuerySource.REPL_MAIN_THREAD_OUTPUT_STYLE_CUSTOM


def get_query_source_for_agent(
    agent_type: Optional[str],
    is_builtin_agent: bool,
) -> str:
    """Get query source for agent execution.

    This matches TypeScript's getQuerySourceForAgent():
    - Built-in agent with type -> 'agent:builtin:<type>'
    - Built-in agent without type -> 'agent:default'
    - Custom agent -> 'agent:custom'

    Args:
        agent_type: The agent type/name
        is_builtin_agent: Whether this is a built-in agent

    Returns:
        Query source string
    """
    if is_builtin_agent:
        if agent_type:
            return f"agent:builtin:{agent_type}"
        return QuerySource.AGENT_DEFAULT
    return QuerySource.AGENT_CUSTOM


def is_foreground_query_source(query_source: Optional[str]) -> bool:
    """Check if this is a foreground query source that should retry on 529 errors.

    Foreground sources are those where the user is blocking on the result.
    Background tasks (summaries, suggestions, classifiers) bail immediately.

    This matches TypeScript's FOREGROUND_529_RETRY_SOURCES set.

    Args:
        query_source: Query source string or None

    Returns:
        True if this is a foreground source
    """
    if not query_source:
        return True  # undefined treated as foreground (conservative)

    foreground_sources = {
        # Main thread and variants
        QuerySource.REPL_MAIN_THREAD,
        "repl_main_thread:outputStyle:Explanatory",
        "repl_main_thread:outputStyle:Learning",
        "repl_main_thread:outputStyle:custom",
        # SDK
        QuerySource.SDK,
        # Agents
        QuerySource.AGENT_CUSTOM,
        QuerySource.AGENT_DEFAULT,
        QuerySource.AGENT_BUILTIN,
        # Important background tasks
        QuerySource.COMPACT,
        QuerySource.HOOK_AGENT,
        QuerySource.VERIFICATION_AGENT,
        QuerySource.SIDE_QUESTION,
        QuerySource.AUTO_MODE,
    }

    # Check exact match or prefix match for main thread
    if query_source in foreground_sources:
        return True
    if query_source.startswith("repl_main_thread"):
        return True
    if query_source.startswith("agent:builtin:"):
        return True

    return False