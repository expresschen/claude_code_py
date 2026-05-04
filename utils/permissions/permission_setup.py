"""Permission Setup - Mode transitions and context preparation.

This implements permission mode transitions from permissionSetup.ts:
- prepareContextForPlanMode: Prepare context when entering plan mode
- transitionPermissionMode: Handle mode transitions
- stripDangerousPermissionsForAutoMode: Remove dangerous auto-allow rules
- restoreDangerousPermissions: Restore permissions when exiting auto/plan mode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .classifier import (
    is_auto_mode_allowlisted_tool,
    AutoModeRules,
    get_default_auto_mode_rules,
)


# =============================================================================
# Permission Mode Types
# =============================================================================


class PermissionMode(str, Enum):
    """Permission mode types."""

    DEFAULT = "default"
    ACCEPT_ALL = "accept-all"
    PLAN = "plan"
    AUTO = "auto"
    BYPASS = "bypassPermissions"


@dataclass
class ToolPermissionContext:
    """Context for tool permission checking."""

    mode: PermissionMode = PermissionMode.DEFAULT
    pre_plan_mode: Optional[PermissionMode] = None
    auto_mode_active: bool = False
    stripped_dangerous_rules: bool = False

    # Permission rules
    allow_rules: list[str] = field(default_factory=list)
    deny_rules: list[str] = field(default_factory=list)
    ask_rules: list[str] = field(default_factory=list)

    # Auto mode rules
    auto_mode_rules: Optional[AutoModeRules] = None

    # Working directories
    additional_working_dirs: list[str] = field(default_factory=list)


# =============================================================================
# Dangerous Permission Patterns
# =============================================================================


# Dangerous Bash patterns that should not be auto-allowed
DANGEROUS_BASH_PATTERNS = frozenset([
    # Script interpreters that can execute arbitrary code
    "python",
    "python3",
    "node",
    "ruby",
    "perl",
    "php",
    "bash",
    "sh",
    "zsh",
    "fish",
    # Cross-platform code execution
    "curl",
    "wget",
    "npx",
    "npm",
    "pip",
    "pip3",
    "cargo",
    "go",
    "javac",
    "java",
])

# Dangerous PowerShell patterns
DANGEROUS_POWERSHELL_PATTERNS = frozenset([
    "pwsh",
    "powershell",
    "cmd",
    "wsl",
    "iex",
    "invoke-expression",
    "icm",
    "invoke-command",
    "start-process",
    "saps",
    "start",
    "start-job",
    "sajb",
    "start-threadjob",
])


# =============================================================================
# Dangerous Permission Checking
# =============================================================================


def is_dangerous_bash_permission(
    tool_name: str,
    rule_content: Optional[str] = None,
) -> bool:
    """Check if a Bash permission rule is dangerous for auto mode.

    Dangerous patterns:
    1. Tool-level allow (Bash with no ruleContent) - allows ALL commands
    2. Prefix rules for script interpreters (python:*, node:*, etc.)
    3. Wildcard rules matching interpreters (python*, node*, etc.)

    Args:
        tool_name: Tool name
        rule_content: Rule content (command pattern)

    Returns:
        True if rule is dangerous
    """
    if tool_name != "Bash":
        return False

    # Tool-level allow (no content) - allows ALL commands
    if rule_content is None or rule_content == "":
        return True

    content = rule_content.strip().lower()

    # Standalone wildcard (*) matches everything
    if content == "*":
        return True

    # Check for dangerous patterns
    for pattern in DANGEROUS_BASH_PATTERNS:
        lower_pattern = pattern.lower()

        # Exact match
        if content == lower_pattern:
            return True

        # Prefix syntax: "python:*"
        if content == f"{lower_pattern}:*":
            return True

        # Wildcard at end: "python*"
        if content == f"{lower_pattern}*":
            return True

        # Wildcard with space: "python *"
        if content == f"{lower_pattern} *":
            return True

        # Patterns like "python -*" matching "python -c 'code'"
        if content.startswith(f"{lower_pattern} -") and content.endswith("*"):
            return True

    return False


def is_dangerous_write_permission(
    tool_name: str,
    rule_content: Optional[str] = None,
) -> bool:
    """Check if a Write permission rule is dangerous.

    Args:
        tool_name: Tool name
        rule_content: Rule content (file path pattern)

    Returns:
        True if rule is dangerous
    """
    if tool_name not in ("Write", "Edit"):
        return False

    if rule_content is None or rule_content == "":
        return False

    content = rule_content.strip()

    # Dangerous system paths
    dangerous_paths = [
        "/etc/",
        "~/.ssh/",
        "~/.bashrc",
        "~/.zshrc",
        "~/.profile",
        "/var/",
        "/usr/",
        "C:\\Windows\\",
        "C:\\Program Files\\",
    ]

    for path in dangerous_paths:
        if content.startswith(path) or content == path.rstrip("/"):
            return True

    return False


# =============================================================================
# Permission Stripping for Plan/Auto Mode
# =============================================================================


def strip_dangerous_permissions_for_auto_mode(
    context: ToolPermissionContext,
) -> ToolPermissionContext:
    """Strip dangerous permissions for auto/plan mode.

    When entering auto mode or plan mode with auto-active, dangerous
    auto-allow rules must be removed to prevent the classifier from
    being bypassed.

    Args:
        context: Current permission context

    Returns:
        Context with dangerous rules stripped
    """
    new_allow_rules: list[str] = []
    stripped: list[str] = []

    for rule in context.allow_rules:
        # Parse rule: Tool(content) or Tool
        if "(" in rule:
            tool_name = rule.split("(")[0].strip()
            rule_content = rule.split("(")[1].rstrip(")").strip() if "(" in rule else None
        else:
            tool_name = rule.strip()
            rule_content = None

        # Check if dangerous
        is_dangerous = (
            is_dangerous_bash_permission(tool_name, rule_content)
            or is_dangerous_write_permission(tool_name, rule_content)
        )

        if is_dangerous:
            stripped.append(rule)
        else:
            new_allow_rules.append(rule)

    return ToolPermissionContext(
        mode=context.mode,
        pre_plan_mode=context.pre_plan_mode,
        auto_mode_active=True,
        stripped_dangerous_rules=True,
        allow_rules=new_allow_rules,
        deny_rules=context.deny_rules,
        ask_rules=context.ask_rules,
        auto_mode_rules=context.auto_mode_rules or get_default_auto_mode_rules(),
        additional_working_dirs=context.additional_working_dirs,
    )


def restore_dangerous_permissions(
    context: ToolPermissionContext,
) -> ToolPermissionContext:
    """Restore dangerous permissions when exiting auto/plan mode.

    Args:
        context: Current permission context

    Returns:
        Context with rules restored (caller must add back stripped rules)
    """
    return ToolPermissionContext(
        mode=context.mode,
        pre_plan_mode=None,
        auto_mode_active=False,
        stripped_dangerous_rules=False,
        allow_rules=context.allow_rules,
        deny_rules=context.deny_rules,
        ask_rules=context.ask_rules,
        auto_mode_rules=None,
        additional_working_dirs=context.additional_working_dirs,
    )


# =============================================================================
# Mode Transitions
# =============================================================================


def prepare_context_for_plan_mode(
    context: ToolPermissionContext,
) -> ToolPermissionContext:
    """Prepare permission context for plan mode entry.

    Centralized plan-mode entry that:
    1. Stashes the current mode as prePlanMode
    2. If user has auto-mode opt-in, keeps auto semantics active
    3. Strips dangerous permissions when auto is active

    Args:
        context: Current permission context

    Returns:
        Context prepared for plan mode
    """
    current_mode = context.mode

    if current_mode == PermissionMode.PLAN:
        return context

    # Handle auto mode transition
    if current_mode == PermissionMode.AUTO:
        # Deactivate auto mode during plan (unless opt-in)
        if should_plan_use_auto_mode(context):
            # Keep auto active with dangerous rules stripped
            return ToolPermissionContext(
                mode=PermissionMode.PLAN,
                pre_plan_mode=PermissionMode.AUTO,
                auto_mode_active=True,
                stripped_dangerous_rules=True,
                allow_rules=strip_dangerous_allow_rules(context.allow_rules),
                deny_rules=context.deny_rules,
                ask_rules=context.ask_rules,
                auto_mode_rules=context.auto_mode_rules,
                additional_working_dirs=context.additional_working_dirs,
            )
        else:
            # Deactivate auto, restore dangerous permissions
            return ToolPermissionContext(
                mode=PermissionMode.PLAN,
                pre_plan_mode=PermissionMode.AUTO,
                auto_mode_active=False,
                stripped_dangerous_rules=False,
                allow_rules=context.allow_rules,  # Caller should restore
                deny_rules=context.deny_rules,
                ask_rules=context.ask_rules,
                auto_mode_rules=None,
                additional_working_dirs=context.additional_working_dirs,
            )

    # Handle non-auto modes
    if should_plan_use_auto_mode(context) and current_mode != PermissionMode.BYPASS:
        # Activate auto mode during plan
        return ToolPermissionContext(
            mode=PermissionMode.PLAN,
            pre_plan_mode=current_mode,
            auto_mode_active=True,
            stripped_dangerous_rules=True,
            allow_rules=strip_dangerous_allow_rules(context.allow_rules),
            deny_rules=context.deny_rules,
            ask_rules=context.ask_rules,
            auto_mode_rules=get_default_auto_mode_rules(),
            additional_working_dirs=context.additional_working_dirs,
        )

    # Plain plan mode entry
    return ToolPermissionContext(
        mode=PermissionMode.PLAN,
        pre_plan_mode=current_mode,
        auto_mode_active=False,
        stripped_dangerous_rules=False,
        allow_rules=context.allow_rules,
        deny_rules=context.deny_rules,
        ask_rules=context.ask_rules,
        auto_mode_rules=None,
        additional_working_dirs=context.additional_working_dirs,
    )


def transition_permission_mode(
    context: ToolPermissionContext,
    to_mode: PermissionMode,
) -> ToolPermissionContext:
    """Transition permission mode with proper handling.

    Args:
        context: Current permission context
        to_mode: Target mode

    Returns:
        New permission context
    """
    from_mode = context.mode

    # Same mode - no change
    if from_mode == to_mode:
        return context

    # Handle plan mode entry
    if to_mode == PermissionMode.PLAN:
        return prepare_context_for_plan_mode(context)

    # Handle plan mode exit
    if from_mode == PermissionMode.PLAN:
        pre_plan_mode = context.pre_plan_mode or PermissionMode.DEFAULT

        # Restore dangerous permissions if they were stripped
        if context.stripped_dangerous_rules:
            new_context = restore_dangerous_permissions(context)
            return ToolPermissionContext(
                mode=to_mode,
                pre_plan_mode=None,
                auto_mode_active=False,
                stripped_dangerous_rules=False,
                allow_rules=new_context.allow_rules,
                deny_rules=new_context.deny_rules,
                ask_rules=new_context.ask_rules,
                auto_mode_rules=None,
                additional_working_dirs=new_context.additional_working_dirs,
            )

        return ToolPermissionContext(
            mode=to_mode,
            pre_plan_mode=None,
            auto_mode_active=False,
            stripped_dangerous_rules=False,
            allow_rules=context.allow_rules,
            deny_rules=context.deny_rules,
            ask_rules=context.ask_rules,
            auto_mode_rules=None,
            additional_working_dirs=context.additional_working_dirs,
        )

    # Handle auto mode entry
    if to_mode == PermissionMode.AUTO:
        return ToolPermissionContext(
            mode=PermissionMode.AUTO,
            pre_plan_mode=None,
            auto_mode_active=True,
            stripped_dangerous_rules=True,
            allow_rules=strip_dangerous_allow_rules(context.allow_rules),
            deny_rules=context.deny_rules,
            ask_rules=context.ask_rules,
            auto_mode_rules=get_default_auto_mode_rules(),
            additional_working_dirs=context.additional_working_dirs,
        )

    # Handle auto mode exit
    if from_mode == PermissionMode.AUTO:
        return ToolPermissionContext(
            mode=to_mode,
            pre_plan_mode=None,
            auto_mode_active=False,
            stripped_dangerous_rules=False,
            allow_rules=context.allow_rules,
            deny_rules=context.deny_rules,
            ask_rules=context.ask_rules,
            auto_mode_rules=None,
            additional_working_dirs=context.additional_working_dirs,
        )

    # Default transition
    return ToolPermissionContext(
        mode=to_mode,
        pre_plan_mode=None,
        auto_mode_active=False,
        stripped_dangerous_rules=False,
        allow_rules=context.allow_rules,
        deny_rules=context.deny_rules,
        ask_rules=context.ask_rules,
        auto_mode_rules=None,
        additional_working_dirs=context.additional_working_dirs,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def strip_dangerous_allow_rules(allow_rules: list[str]) -> list[str]:
    """Strip dangerous rules from allow list.

    Args:
        allow_rules: Original allow rules

    Returns:
        Allow rules with dangerous entries removed
    """
    safe_rules: list[str] = []

    for rule in allow_rules:
        if "(" in rule:
            tool_name = rule.split("(")[0].strip()
            rule_content = rule.split("(")[1].rstrip(")").strip()
        else:
            tool_name = rule.strip()
            rule_content = None

        if not is_dangerous_bash_permission(tool_name, rule_content) and \
           not is_dangerous_write_permission(tool_name, rule_content):
            safe_rules.append(rule)

    return safe_rules


def should_plan_use_auto_mode(context: ToolPermissionContext) -> bool:
    """Check if plan mode should use auto mode semantics.

    Args:
        context: Permission context

    Returns:
        True if auto should be active during plan
    """
    # Check for user opt-in to auto mode
    if context.auto_mode_rules is not None:
        return True

    # Check environment variable
    import os
    if os.environ.get("CLAUDE_CODE_AUTO_MODE_DURING_PLAN", "").lower() == "true":
        return True

    return False


# =============================================================================
# Default Permission Context
# =============================================================================


def get_default_permission_context() -> ToolPermissionContext:
    """Get default permission context.

    Returns:
        Default permission context
    """
    return ToolPermissionContext(
        mode=PermissionMode.DEFAULT,
        pre_plan_mode=None,
        auto_mode_active=False,
        stripped_dangerous_rules=False,
        allow_rules=[],
        deny_rules=[],
        ask_rules=[],
        auto_mode_rules=None,
        additional_working_dirs=[],
    )


def get_bypass_permission_context() -> ToolPermissionContext:
    """Get bypass permission context (accept-all mode).

    Returns:
        Bypass permission context
    """
    return ToolPermissionContext(
        mode=PermissionMode.BYPASS,
        pre_plan_mode=None,
        auto_mode_active=False,
        stripped_dangerous_rules=False,
        allow_rules=["*"],  # Allow all
        deny_rules=[],
        ask_rules=[],
        auto_mode_rules=None,
        additional_working_dirs=[],
    )


def get_auto_permission_context() -> ToolPermissionContext:
    """Get auto mode permission context.

    Returns:
        Auto mode permission context
    """
    default_rules = get_default_auto_mode_rules()
    safe_allow = strip_dangerous_allow_rules(default_rules.allow)

    return ToolPermissionContext(
        mode=PermissionMode.AUTO,
        pre_plan_mode=None,
        auto_mode_active=True,
        stripped_dangerous_rules=True,
        allow_rules=safe_allow,
        deny_rules=default_rules.soft_deny,
        ask_rules=[],
        auto_mode_rules=default_rules,
        additional_working_dirs=[],
    )