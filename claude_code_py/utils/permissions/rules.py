"""Permission rule matching logic.

This implements the rule matching system for tool permissions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel


# =============================================================================
# Rule Types
# =============================================================================


class PermissionBehavior(str, Enum):
    """Behavior for a permission rule."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionRuleSource(str, Enum):
    """Source of a permission rule."""

    USER = "user"
    PROJECT = "project"
    GLOBAL = "global"
    MCP = "mcp"


@dataclass
class PermissionRuleContent:
    """Parsed content of a permission rule.

    Examples:
        "Bash" → tool_name="Bash", rule_content=None
        "Bash(npm run *)" → tool_name="Bash", rule_content="npm run *"
        "Read(*)" → tool_name="Read", rule_content="*"
        "mcp__server__*" → tool_name="mcp__server__*", rule_content=None
    """

    tool_name: str
    rule_content: Optional[str] = None


@dataclass
class PermissionRule:
    """A complete permission rule with behavior and source."""

    source: PermissionRuleSource
    rule_behavior: PermissionBehavior
    rule_value: PermissionRuleContent


# =============================================================================
# Rule Parsing
# =============================================================================


def parse_permission_rule(rule_string: str) -> PermissionRuleContent:
    """Parse a permission rule string into components.

    Args:
        rule_string: Rule string like "Bash", "Bash(npm *)", "mcp__server"

    Returns:
        Parsed rule content
    """
    # Check for rule content in parentheses
    if "(" in rule_string:
        # Split on first '('
        tool_part, content_part = rule_string.split("(", 1)
        tool_name = tool_part.strip()

        # Remove trailing ')'
        if content_part.endswith(")"):
            rule_content = content_part[:-1].strip()
        else:
            rule_content = content_part.strip()

        return PermissionRuleContent(
            tool_name=tool_name,
            rule_content=rule_content,
        )

    # No parentheses - entire string is tool name
    return PermissionRuleContent(
        tool_name=rule_string.strip(),
        rule_content=None,
    )


def rule_to_string(rule: PermissionRuleContent) -> str:
    """Convert a rule content back to string format.

    Args:
        rule: Rule content

    Returns:
        String representation
    """
    if rule.rule_content:
        return f"{rule.tool_name}({rule.rule_content})"
    return rule.tool_name


# =============================================================================
# Rule Matching
# =============================================================================


def get_rule_source_name(source: PermissionRuleSource) -> str:
    """Get display name for a rule source.

    Args:
        source: Rule source

    Returns:
        Display name
    """
    names = {
        PermissionRuleSource.USER: "user settings",
        PermissionRuleSource.PROJECT: "project settings",
        PermissionRuleSource.GLOBAL: "global settings",
        PermissionRuleSource.MCP: "MCP server",
    }
    return names.get(source, str(source))


def get_tool_name_for_permission_check(tool: Any) -> str:
    """Get the tool name to use for permission matching.

    For MCP tools, uses the fully qualified name (mcp__server__tool).
    For builtin tools, uses the tool name directly.

    Args:
        tool: Tool instance

    Returns:
        Tool name for matching
    """
    # Check for MCP info
    if hasattr(tool, "mcp_info") and tool.mcp_info:
        return f"mcp__{tool.mcp_info.server_name}__{tool.mcp_info.tool_name}"

    return tool.name


def tool_matches_rule(
    tool: Any,
    rule: PermissionRule,
) -> bool:
    """Check if a tool matches a permission rule.

    Args:
        tool: Tool instance
        rule: Permission rule

    Returns:
        True if tool matches rule
    """
    # Rule with content must match tool + input
    # (handled by tool_matches_rule_with_input)
    if rule.rule_value.rule_content is not None:
        return False

    tool_name_for_match = get_tool_name_for_permission_check(tool)
    rule_tool_name = rule.rule_value.tool_name

    # Direct tool name match
    if rule_tool_name == tool_name_for_match:
        return True

    # MCP server-level permission: "mcp__server" matches "mcp__server__tool"
    # Also wildcard: "mcp__server__*" matches all tools from that server
    if rule_tool_name.startswith("mcp__"):
        rule_parts = rule_tool_name.split("__")
        tool_parts = tool_name_for_match.split("__")

        # Need at least 2 parts (mcp__server)
        if len(rule_parts) >= 2 and len(tool_parts) >= 3:
            rule_server = rule_parts[1]
            tool_server = tool_parts[1]

            if rule_server == tool_server:
                # Server-level rule (no tool name or wildcard)
                if len(rule_parts) == 2:
                    return True
                # Wildcard: mcp__server__*
                if len(rule_parts) == 3 and rule_parts[2] == "*":
                    return True

    return False


def tool_matches_rule_with_input(
    tool: Any,
    input: Any,
    rule: PermissionRule,
) -> bool:
    """Check if a tool with input matches a permission rule.

    Args:
        tool: Tool instance
        input: Tool input (validated)
        rule: Permission rule

    Returns:
        True if tool + input matches rule
    """
    # First check tool name match
    tool_name_for_match = get_tool_name_for_permission_check(tool)
    rule_tool_name = rule.rule_value.tool_name

    # Tool name must match (or be wildcard)
    if rule_tool_name != tool_name_for_match and rule_tool_name != "*":
        # Check MCP server-level match
        if not tool_matches_rule(tool, PermissionRule(
            source=rule.source,
            rule_behavior=rule.rule_behavior,
            rule_value=PermissionRuleContent(tool_name=rule_tool_name),
        )):
            return False

    # No rule content - tool-level match only
    if rule.rule_value.rule_content is None:
        return True

    # Rule has content - let tool check it
    # Tools implement check_rule_content() for specific matching
    if hasattr(tool, "check_rule_content"):
        return tool.check_rule_content(rule.rule_value.rule_content, input)

    # Default: wildcard match
    rule_content = rule.rule_value.rule_content
    if rule_content == "*":
        return True

    # Unknown content - no match
    return False


# =============================================================================
# Rule Lists
# =============================================================================


@dataclass
class ToolPermissionRulesBySource:
    """Permission rules organized by source."""

    user: list[str] = field(default_factory=list)
    project: list[str] = field(default_factory=list)
    global_: list[str] = field(default_factory=list)
    mcp: dict[str, list[str]] = field(default_factory=dict)

    def get_rules_for_source(self, source: PermissionRuleSource) -> list[str]:
        """Get rules for a specific source.

        Args:
            source: Rule source

        Returns:
            List of rule strings
        """
        if source == PermissionRuleSource.USER:
            return self.user
        elif source == PermissionRuleSource.PROJECT:
            return self.project
        elif source == PermissionRuleSource.GLOBAL:
            return self.global_
        elif source == PermissionRuleSource.MCP:
            # MCP rules are per-server
            all_mcp_rules = []
            for server_rules in self.mcp.values():
                all_mcp_rules.extend(server_rules)
            return all_mcp_rules
        return []


def get_allow_rules(
    context: Any,
) -> list[PermissionRule]:
    """Get all allow rules from context.

    Args:
        context: Tool permission context

    Returns:
        List of parsed allow rules
    """
    if not hasattr(context, "always_allow_rules"):
        return []

    rules_by_source = context.always_allow_rules
    if not isinstance(rules_by_source, ToolPermissionRulesBySource):
        return []

    rules = []
    for source in [PermissionRuleSource.USER, PermissionRuleSource.PROJECT, PermissionRuleSource.GLOBAL]:
        rule_strings = rules_by_source.get_rules_for_source(source)
        for rule_string in rule_strings:
            rules.append(PermissionRule(
                source=source,
                rule_behavior=PermissionBehavior.ALLOW,
                rule_value=parse_permission_rule(rule_string),
            ))

    return rules


def get_deny_rules(
    context: Any,
) -> list[PermissionRule]:
    """Get all deny rules from context.

    Args:
        context: Tool permission context

    Returns:
        List of parsed deny rules
    """
    if not hasattr(context, "always_deny_rules"):
        return []

    rules_by_source = context.always_deny_rules
    if not isinstance(rules_by_source, ToolPermissionRulesBySource):
        return []

    rules = []
    for source in [PermissionRuleSource.USER, PermissionRuleSource.PROJECT, PermissionRuleSource.GLOBAL]:
        rule_strings = rules_by_source.get_rules_for_source(source)
        for rule_string in rule_strings:
            rules.append(PermissionRule(
                source=source,
                rule_behavior=PermissionBehavior.DENY,
                rule_value=parse_permission_rule(rule_string),
            ))

    return rules


def get_ask_rules(
    context: Any,
) -> list[PermissionRule]:
    """Get all ask rules from context.

    Args:
        context: Tool permission context

    Returns:
        List of parsed ask rules
    """
    if not hasattr(context, "always_ask_rules"):
        return []

    rules_by_source = context.always_ask_rules
    if not isinstance(rules_by_source, ToolPermissionRulesBySource):
        return []

    rules = []
    for source in [PermissionRuleSource.USER, PermissionRuleSource.PROJECT, PermissionRuleSource.GLOBAL]:
        rule_strings = rules_by_source.get_rules_for_source(source)
        for rule_string in rule_strings:
            rules.append(PermissionRule(
                source=source,
                rule_behavior=PermissionBehavior.ASK,
                rule_value=parse_permission_rule(rule_string),
            ))

    return rules


# =============================================================================
# Permission Rule Finding
# =============================================================================


def find_matching_rule(
    tool: Any,
    input: Any,
    rules: list[PermissionRule],
) -> Optional[PermissionRule]:
    """Find a matching rule for tool + input.

    Args:
        tool: Tool instance
        input: Tool input
        rules: Rules to check

    Returns:
        Matching rule or None
    """
    for rule in rules:
        if tool_matches_rule_with_input(tool, input, rule):
            return rule
    return None


def find_tool_level_rule(
    tool: Any,
    rules: list[PermissionRule],
) -> Optional[PermissionRule]:
    """Find a tool-level rule (without content).

    Args:
        tool: Tool instance
        rules: Rules to check

    Returns:
        Matching tool-level rule or None
    """
    for rule in rules:
        if tool_matches_rule(tool, rule):
            return rule
    return None


def get_deny_rule_for_tool(
    context: Any,
    tool: Any,
) -> Optional[PermissionRule]:
    """Get deny rule for a tool if one exists.

    Args:
        context: Permission context
        tool: Tool instance

    Returns:
        Deny rule or None
    """
    deny_rules = get_deny_rules(context)
    return find_tool_level_rule(tool, deny_rules)


def get_ask_rule_for_tool(
    context: Any,
    tool: Any,
) -> Optional[PermissionRule]:
    """Get ask rule for a tool if one exists.

    Args:
        context: Permission context
        tool: Tool instance

    Returns:
        Ask rule or None
    """
    ask_rules = get_ask_rules(context)
    return find_tool_level_rule(tool, ask_rules)


def get_allow_rule_for_tool(
    context: Any,
    tool: Any,
    input: Any,
) -> Optional[PermissionRule]:
    """Get allow rule for tool + input if one exists.

    Args:
        context: Permission context
        tool: Tool instance
        input: Tool input

    Returns:
        Allow rule or None
    """
    allow_rules = get_allow_rules(context)
    return find_matching_rule(tool, input, allow_rules)


# =============================================================================
# Bash Tool Special Handling
# =============================================================================


def parse_bash_rule_content(content: str) -> dict[str, Any]:
    """Parse Bash tool rule content.

    Formats:
        "npm run *" → prefix="npm run"
        "git status" → exact="git status"
        "rm -rf *" → prefix="rm -rf" (dangerous)
        "*" → wildcard=True

    Args:
        content: Rule content string

    Returns:
        Parsed dict with match_type, command/prefix
    """
    if content == "*":
        return {"wildcard": True}

    if content.endswith("*"):
        prefix = content[:-1].strip()
        return {"prefix": prefix}

    # Exact match
    return {"exact": content}


def check_bash_command_matches_rule(command: str, rule_content: str) -> bool:
    """Check if a bash command matches a rule content.

    Args:
        command: Bash command to check
        rule_content: Rule content (e.g. "npm run *")

    Returns:
        True if command matches
    """
    parsed = parse_bash_rule_content(rule_content)

    if parsed.get("wildcard"):
        return True

    if parsed.get("exact"):
        # Exact match (command must equal exactly)
        return command.strip() == parsed["exact"]

    if parsed.get("prefix"):
        # Prefix match (command starts with prefix)
        prefix = parsed["prefix"]
        return command.strip().startswith(prefix)

    return False