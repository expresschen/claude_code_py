"""Permission type definitions."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PermissionMode(str, Enum):
    """Permission mode enumeration."""

    DEFAULT = "default"
    AUTO = "auto"
    PLAN = "plan"
    BYPASS = "bypass"


class PermissionBehavior(str, Enum):
    """Permission decision behavior."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionResult(BaseModel):
    """Result of a permission check."""

    behavior: PermissionBehavior
    updated_input: Optional[Any] = None  # Can be dict or Pydantic model
    reason: Optional[str] = None
    rule_source: Optional[str] = None

    @classmethod
    def allow(cls, updated_input: Optional[Any] = None) -> "PermissionResult":
        """Create an allow result."""
        return cls(behavior=PermissionBehavior.ALLOW, updated_input=updated_input)

    @classmethod
    def deny(cls, reason: Optional[str] = None) -> "PermissionResult":
        """Create a deny result."""
        return cls(behavior=PermissionBehavior.DENY, reason=reason)

    @classmethod
    def ask(cls, reason: Optional[str] = None) -> "PermissionResult":
        """Create an ask result (prompt user)."""
        return cls(behavior=PermissionBehavior.ASK, reason=reason)


class ToolPermissionRules(BaseModel):
    """Permission rules for tools by source."""

    command: Optional[list[str]] = None
    mcp: Optional[dict[str, list[str]]] = None


class ToolPermissionContext(BaseModel):
    """Context for tool permission checks."""

    mode: PermissionMode = PermissionMode.DEFAULT
    cwd: str = "."  # Current working directory
    session_id: Optional[str] = None  # Session ID for worktree operations
    additional_working_directories: dict[str, str] = Field(default_factory=dict)
    always_allow_rules: ToolPermissionRules = Field(default_factory=ToolPermissionRules)
    always_deny_rules: ToolPermissionRules = Field(default_factory=ToolPermissionRules)
    always_ask_rules: ToolPermissionRules = Field(default_factory=ToolPermissionRules)
    is_bypass_permissions_mode_available: bool = False
    is_auto_mode_available: bool = False
    should_avoid_permission_prompts: bool = False
    await_automated_checks_before_dialog: bool = False
    pre_plan_mode: Optional[PermissionMode] = None

    class Config:
        frozen = False  # Allow mutation


def get_empty_tool_permission_context() -> ToolPermissionContext:
    """Create an empty tool permission context."""
    return ToolPermissionContext()