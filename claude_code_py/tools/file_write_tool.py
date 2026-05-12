"""File write tool for creating files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError


class FileWriteInput(BaseModel):
    """Input for FileWrite tool."""

    file_path: str = Field(description="The absolute path to the file to write")
    content: str = Field(description="The content to write to the file")


@dataclass
class FileWriteOutput:
    """Output from FileWrite tool."""

    path: str
    bytes_written: int

    def __str__(self) -> str:
        return f"Wrote {self.bytes_written} bytes to {self.path}"


class FileWriteTool(Tool[FileWriteInput, FileWriteOutput, dict[str, Any]]):
    """Tool for writing files."""

    name = "Write"
    aliases = ["FileWrite", "CreateFile"]
    input_schema = FileWriteInput

    async def call(
        self,
        args: FileWriteInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[FileWriteOutput]:
        """Write the file.

        Args:
            args: File arguments
            context: Execution context
            can_use_tool: Permission function
            parent_message: Parent message
            on_progress: Progress callback

        Returns:
            Tool result with write info
        """
        # Resolve path relative to cwd
        resolved_path = context.resolve_path(args.file_path)
        path = Path(resolved_path)

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        try:
            with open(path, "w", encoding="utf-8") as f:
                bytes_written = f.write(args.content)

            output = FileWriteOutput(
                path=str(path),
                bytes_written=bytes_written,
            )

            return ToolResult(data=output)

        except Exception as e:
            raise ToolError(f"Failed to write file: {e}")

    async def description(
        self,
        input: FileWriteInput,
        options: dict[str, Any],
    ) -> str:
        """Get description."""
        return f"Write {input.file_path}"

    async def prompt(self, options: dict[str, Any]) -> str:
        """Get tool prompt."""
        return """Write a file to the local filesystem.

Usage notes:
- Provide the absolute path for the file
- The file will be created if it doesn't exist
- Parent directories will be created if needed
- This OVERWRITES existing files"""

    def is_concurrency_safe(self, input: FileWriteInput) -> bool:
        """Writing is not concurrency safe."""
        return False

    def is_read_only(self, input: FileWriteInput) -> bool:
        """Writing is not read-only."""
        return False

    def is_destructive(self, input: FileWriteInput) -> bool:
        """Writing overwrites existing files."""
        return True

    def get_path(self, input: FileWriteInput) -> Optional[str]:
        """Get file path."""
        return input.file_path


# Create instance
file_write_tool = FileWriteTool()