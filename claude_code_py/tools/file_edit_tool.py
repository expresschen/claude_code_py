"""File edit tool for modifying files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError


class FileEditInput(BaseModel):
    """Input for FileEdit tool."""

    file_path: str = Field(description="The absolute path to the file to edit")
    old_string: str = Field(
        description="The text to replace. Must be EXACTLY match"
    )
    new_string: str = Field(
        description="The text to replace it with"
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences",
    )

    @field_validator("old_string")
    @classmethod
    def old_string_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("old_string cannot be empty")
        return v


@dataclass
class FileEditOutput:
    """Output from FileEdit tool."""

    path: str
    replacements: int
    content_preview: str

    def __str__(self) -> str:
        return f"Made {self.replacements} replacement(s) in {self.path}"


class FileEditTool(Tool[FileEditInput, FileEditOutput, dict[str, Any]]):
    """Tool for editing files."""

    name = "Edit"
    aliases = ["FileEdit", "Replace"]
    input_schema = FileEditInput

    async def call(
        self,
        args: FileEditInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[FileEditOutput]:
        """Edit the file.

        Args:
            args: Edit arguments
            context: Execution context
            can_use_tool: Permission function
            parent_message: Parent message
            on_progress: Progress callback

        Returns:
            Tool result with edit info
        """
        # Resolve path relative to cwd
        resolved_path = context.resolve_path(args.file_path)
        path = Path(resolved_path)

        # Check if file exists
        if not path.exists():
            raise ToolError(f"File not found: {args.file_path}")

        # Read file
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ToolError(f"Failed to read file: {e}")

        # Check if old_string exists
        if args.old_string not in content:
            raise ToolError(
                f"Could not find old_string in file. "
                f"Make sure it's an exact match including whitespace."
            )

        # Count occurrences
        occurrences = content.count(args.old_string)

        # Check uniqueness if not replace_all
        if not args.replace_all and occurrences > 1:
            raise ToolError(
                f"Found {occurrences} occurrences of old_string. "
                f"Use replace_all=True to replace all, or make old_string more specific."
            )

        # Replace
        if args.replace_all:
            new_content = content.replace(args.old_string, args.new_string)
            replacements = occurrences
        else:
            new_content = content.replace(args.old_string, args.new_string, 1)
            replacements = 1

        # Write back
        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            raise ToolError(f"Failed to write file: {e}")

        # Create preview
        preview_lines = new_content.split("\n")[:10]
        preview = "\n".join(preview_lines)
        if len(preview_lines) < new_content.count("\n") + 1:
            preview += "\n..."

        output = FileEditOutput(
            path=str(path),
            replacements=replacements,
            content_preview=preview,
        )

        return ToolResult(data=output)

    async def description(
        self,
        input: FileEditInput,
        options: dict[str, Any],
    ) -> str:
        """Get description."""
        return f"Edit {input.file_path}"

    async def prompt(self, options: dict[str, Any]) -> str:
        """Get tool prompt."""
        return """Edit a file by replacing text.

Usage notes:
- old_string must EXACTLY match the text in the file (including whitespace)
- If old_string appears multiple times, use replace_all=True or make it more specific
- This is a precise find-and-replace operation"""

    def is_concurrency_safe(self, input: FileEditInput) -> bool:
        """Editing is not concurrency safe."""
        return False

    def is_read_only(self, input: FileEditInput) -> bool:
        """Editing is not read-only."""
        return False

    def is_destructive(self, input: FileEditInput) -> bool:
        """Editing modifies files."""
        return True

    def get_path(self, input: FileEditInput) -> Optional[str]:
        """Get file path."""
        return input.file_path


# Create instance
file_edit_tool = FileEditTool()