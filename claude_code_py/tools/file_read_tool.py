"""File read tool for reading file contents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError


class FileReadInput(BaseModel):
    """Input for FileRead tool."""

    file_path: str = Field(description="The absolute path to the file to read")
    offset: Optional[int] = Field(
        default=None,
        description="The line number to start reading from",
    )
    limit: Optional[int] = Field(
        default=None,
        description="The number of lines to read",
    )


@dataclass
class FileReadOutput:
    """Output from FileRead tool."""

    content: str
    path: str
    lines_read: int

    def __str__(self) -> str:
        return self.content


class FileReadTool(Tool[FileReadInput, FileReadOutput, dict[str, Any]]):
    """Tool for reading file contents."""

    name = "Read"
    aliases = ["FileRead", "Cat"]
    input_schema = FileReadInput
    max_result_size_chars = float("inf")  # Never persist - creates circular Read

    async def call(
        self,
        args: FileReadInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[FileReadOutput]:
        """Read the file.

        Args:
            args: File arguments
            context: Execution context
            can_use_tool: Permission function
            parent_message: Parent message
            on_progress: Progress callback

        Returns:
            Tool result with file contents
        """
        # Resolve path relative to cwd
        resolved_path = context.resolve_path(args.file_path)
        path = Path(resolved_path)

        # Check if file exists
        if not path.exists():
            raise ToolError(f"File not found: {args.file_path}")

        if not path.is_file():
            raise ToolError(f"Not a file: {args.file_path}")

        # Read file
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if args.offset:
                    # Skip to offset
                    for _ in range(args.offset):
                        next(f)

                if args.limit:
                    lines = []
                    for i, line in enumerate(f):
                        lines.append(line)
                        if i >= args.limit - 1:
                            break
                    content = "".join(lines)
                    lines_read = len(lines)
                else:
                    content = f.read()
                    lines_read = content.count("\n") + 1

            output = FileReadOutput(
                content=content,
                path=str(path),
                lines_read=lines_read,
            )

            return ToolResult(data=output)

        except Exception as e:
            raise ToolError(f"Failed to read file: {e}")

    async def description(
        self,
        input: FileReadInput,
        options: dict[str, Any],
    ) -> str:
        """Get description."""
        desc = f"Read {input.file_path}"
        if input.offset:
            desc += f" from line {input.offset}"
        if input.limit:
            desc += f" ({input.limit} lines)"
        return desc

    async def prompt(self, options: dict[str, Any]) -> str:
        """Get tool prompt."""
        return """Read a file from the local filesystem.

Usage notes:
- Provide the absolute path to the file
- Use offset and limit for large files
- The file content is returned as-is"""

    def is_concurrency_safe(self, input: FileReadInput) -> bool:
        """Reading is always concurrency safe."""
        return True

    def is_read_only(self, input: FileReadInput) -> bool:
        """Reading is read-only."""
        return True

    def get_path(self, input: FileReadInput) -> Optional[str]:
        """Get file path."""
        return input.file_path


# Create instance
file_read_tool = FileReadTool()