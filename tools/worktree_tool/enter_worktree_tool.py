"""Enter Worktree Tool implementation.

Creates an isolated git worktree and switches the session into it.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel

from claude_code_py.tool.base import Tool, build_tool, ValidationResult
from claude_code_py.tool.result import ToolResult, ToolError
from claude_code_py.utils.worktree import (
    create_worktree_for_session,
    get_current_worktree_session,
    validate_worktree_slug,
)
from claude_code_py.storage.session import WorktreeSession

if TYPE_CHECKING:
    from claude_code_py.tool.context import ToolUseContext
    from claude_code_py.core_types.message import AssistantMessage
    from claude_code_py.core_types.permissions import PermissionResult


# =============================================================================
# Input Schema
# =============================================================================


class EnterWorktreeInput(BaseModel):
    """Input for EnterWorktreeTool."""

    name: Optional[str] = None


class EnterWorktreeOutput(BaseModel):
    """Output for EnterWorktreeTool."""

    worktree_path: str
    worktree_branch: Optional[str] = None
    message: str


# =============================================================================
# Tool Implementation
# =============================================================================


class EnterWorktreeToolClass(Tool[EnterWorktreeInput, EnterWorktreeOutput, None]):
    """Tool for creating and entering a git worktree."""

    name = "EnterWorktree"
    search_hint = "create an isolated git worktree and switch into it"
    input_schema = EnterWorktreeInput
    output_schema = EnterWorktreeOutput
    should_defer = True

    async def call(
        self,
        args: EnterWorktreeInput,
        context: "ToolUseContext",
        can_use_tool: Any,
        parent_message: "AssistantMessage",
        on_progress: Optional[Any] = None,
    ) -> ToolResult[EnterWorktreeOutput]:
        """Execute the tool.

        Args:
            args: Tool input
            context: Tool context
            can_use_tool: Permission check function
            parent_message: Parent assistant message
            on_progress: Progress callback

        Returns:
            ToolResult with worktree info
        """
        # Validate not already in a worktree
        if get_current_worktree_session():
            raise ToolError("Already in a worktree session")

        # Get session ID from context
        session_id = context.options.tool_permission_context.session_id or "default"

        # Generate slug
        if args.name:
            try:
                validate_worktree_slug(args.name)
                slug = args.name
            except ValueError as e:
                raise ToolError(str(e))
        else:
            # Generate random slug
            adjectives = ["swift", "bright", "calm", "keen", "bold"]
            nouns = ["fox", "owl", "elm", "oak", "ray"]
            adj = adjectives[secrets.choice(range(len(adjectives)))]
            noun = nouns[secrets.choice(range(len(nouns)))]
            suffix = secrets.token_hex(2)
            slug = f"{adj}-{noun}-{suffix}"

        # Create worktree
        worktree = await create_worktree_for_session(session_id, slug)

        # Switch cwd
        os.chdir(worktree.worktree_path)

        # Update context cwd if possible
        if hasattr(context, "options") and hasattr(context.options, "cwd"):
            context.options.cwd = worktree.worktree_path

        branch_info = f" on branch {worktree.worktree_branch}" if worktree.worktree_branch else ""

        return ToolResult(
            data=EnterWorktreeOutput(
                worktree_path=worktree.worktree_path,
                worktree_branch=worktree.worktree_branch,
                message=f"Created worktree at {worktree.worktree_path}{branch_info}. "
                f"The session is now working in the worktree. "
                f"Use ExitWorktree to leave mid-session.",
            )
        )

    async def description(self, input: EnterWorktreeInput, options: dict) -> str:
        """Generate tool description."""
        name = input.name or "a new worktree"
        return f"Creating worktree: {name}"

    async def prompt(self, options: dict) -> str:
        """Generate tool prompt."""
        from .prompt import get_enter_worktree_tool_prompt
        return get_enter_worktree_tool_prompt()

    def is_concurrency_safe(self, input: EnterWorktreeInput) -> bool:
        return False

    def is_read_only(self, input: EnterWorktreeInput) -> bool:
        return False

    def is_destructive(self, input: EnterWorktreeInput) -> bool:
        return False


# Create the tool instance
EnterWorktreeTool = EnterWorktreeToolClass()