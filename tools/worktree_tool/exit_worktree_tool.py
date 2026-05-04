"""Exit Worktree Tool implementation.

Exits a worktree session, keeping or removing the worktree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel

from claude_code_py.tool.base import Tool, build_tool, ValidationResult
from claude_code_py.tool.result import ToolResult, ToolError
from claude_code_py.utils.worktree import (
    keep_worktree,
    cleanup_worktree,
    get_current_worktree_session,
    has_worktree_changes,
)
from claude_code_py.storage.session import load_worktree_state

if TYPE_CHECKING:
    from claude_code_py.tool.context import ToolUseContext
    from claude_code_py.core_types.message import AssistantMessage
    from claude_code_py.core_types.permissions import PermissionResult


# =============================================================================
# Input Schema
# =============================================================================


class ExitWorktreeInput(BaseModel):
    """Input for ExitWorktreeTool."""

    action: str = "keep"  # "keep" or "remove"
    discard_changes: bool = False


class ExitWorktreeOutput(BaseModel):
    """Output for ExitWorktreeTool."""

    message: str
    worktree_path: Optional[str] = None
    kept: bool = True


# =============================================================================
# Tool Implementation
# =============================================================================


class ExitWorktreeToolClass(Tool[ExitWorktreeInput, ExitWorktreeOutput, None]):
    """Tool for exiting a git worktree session."""

    name = "ExitWorktree"
    search_hint = "exit the worktree and return to the original directory"
    input_schema = ExitWorktreeInput
    output_schema = ExitWorktreeOutput
    should_defer = True

    async def call(
        self,
        args: ExitWorktreeInput,
        context: "ToolUseContext",
        can_use_tool: Any,
        parent_message: "AssistantMessage",
        on_progress: Optional[Any] = None,
    ) -> ToolResult[ExitWorktreeOutput]:
        """Execute the tool.

        Args:
            args: Tool input
            context: Tool context
            can_use_tool: Permission check function
            parent_message: Parent assistant message
            on_progress: Progress callback

        Returns:
            ToolResult with exit info
        """
        worktree = get_current_worktree_session()

        if not worktree:
            raise RuntimeError("Not in a worktree session")

        worktree_path = worktree.worktree_path

        if args.action == "remove":
            # Check for changes unless discard_changes
            if not args.discard_changes and worktree.original_head_commit:
                has_changes = await has_worktree_changes(
                    worktree_path,
                    worktree.original_head_commit,
                )
                if has_changes:
                    raise RuntimeError(
                        f"Worktree has uncommitted changes or new commits. "
                        f"Use discard_changes=true to force removal, "
                        f"or action='keep' to preserve it."
                    )

            await cleanup_worktree()
            return ToolResult(
                data=ExitWorktreeOutput(
                    message=f"Removed worktree at {worktree_path}",
                    worktree_path=worktree_path,
                    kept=False,
                )
            )
        else:
            await keep_worktree()
            return ToolResult(
                data=ExitWorktreeOutput(
                    message=f"Kept worktree at {worktree_path}. "
                    f"You can continue working there by running: cd {worktree_path}",
                    worktree_path=worktree_path,
                    kept=True,
                )
            )

    async def description(self, input: ExitWorktreeInput, options: dict) -> str:
        """Generate tool description."""
        action = "removing" if input.action == "remove" else "keeping"
        return f"Exiting worktree ({action})"

    async def prompt(self, options: dict) -> str:
        """Generate tool prompt."""
        return """Use this tool to exit the current worktree session.

## Parameters

- `action`: "keep" (default) or "remove"
  - "keep": Leave the worktree intact, return to original directory
  - "remove": Delete the worktree and its branch

- `discard_changes`: If true, remove even if there are uncommitted changes

## Behavior

- Changes working directory back to the original location
- With "keep": Worktree remains for later use
- With "remove": Worktree directory and branch are deleted
"""

    def is_concurrency_safe(self, input: ExitWorktreeInput) -> bool:
        return False

    def is_read_only(self, input: ExitWorktreeInput) -> bool:
        return False

    def is_destructive(self, input: ExitWorktreeInput) -> bool:
        return input.action == "remove"


# Create the tool instance
ExitWorktreeTool = ExitWorktreeToolClass()