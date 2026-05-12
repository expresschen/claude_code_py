"""Grep tool for searching file contents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError


class GrepInput(BaseModel):
    """Input for Grep tool."""

    pattern: str = Field(description="The regex pattern to search for")
    path: Optional[str] = Field(
        default=None,
        description="The directory or file to search in (default: current)",
    )
    glob: Optional[str] = Field(
        default=None,
        description="Glob pattern to filter files",
    )
    case_insensitive: bool = Field(
        default=False,
        description="Case insensitive search",
    )
    output_mode: str = Field(
        default="content",
        description="Output mode: 'content', 'files_with_matches', or 'count'",
    )


@dataclass
class GrepMatch:
    """A single match from grep."""

    file: str
    line_number: int
    line: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line_number}: {self.line}"


@dataclass
class GrepOutput:
    """Output from Grep tool."""

    matches: list[GrepMatch]
    pattern: str
    files_searched: int

    def __str__(self) -> str:
        if not self.matches:
            return f"No matches for '{self.pattern}'"

        if len(self.matches) <= 50:
            return "\n".join(str(m) for m in self.matches)

        result = [str(m) for m in self.matches[:50]]
        result.append(f"... and {len(self.matches) - 50} more matches")
        return "\n".join(result)


class GrepTool(Tool[GrepInput, GrepOutput, dict[str, Any]]):
    """Tool for searching file contents."""

    name = "Grep"
    aliases = ["Search"]
    input_schema = GrepInput

    async def call(
        self,
        args: GrepInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[GrepOutput]:
        """Search for pattern in files.

        Args:
            args: Grep arguments
            context: Execution context
            can_use_tool: Permission function
            parent_message: Parent message
            on_progress: Progress callback

        Returns:
            Tool result with matches
        """
        # Resolve path relative to cwd (handles None by returning cwd)
        base_path = Path(context.resolve_path(args.path))

        # Compile pattern
        flags = re.MULTILINE
        if args.case_insensitive:
            flags |= re.IGNORECASE

        try:
            regex = re.compile(args.pattern, flags)
        except re.error as e:
            raise ToolError(f"Invalid regex: {e}")

        matches: list[GrepMatch] = []
        files_searched = 0

        # Get files to search
        if base_path.is_file():
            files = [base_path]
        else:
            glob_pattern = args.glob or "**/*"
            files = [
                p
                for p in base_path.glob(glob_pattern)
                if p.is_file()
            ]

        # Search files
        for file_path in files:
            files_searched += 1

            # Apply limit
            limit = context.glob_limits.get("max_results", 1000) if context.glob_limits else 1000
            if len(matches) >= limit:
                break

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")

                for i, line in enumerate(content.split("\n"), 1):
                    if regex.search(line):
                        matches.append(GrepMatch(
                            file=str(file_path.relative_to(base_path))
                            if file_path.is_relative_to(base_path)
                            else str(file_path),
                            line_number=i,
                            line=line[:200],  # Truncate long lines
                        ))

                        if len(matches) >= limit:
                            break

            except Exception:
                continue

        output = GrepOutput(
            matches=matches,
            pattern=args.pattern,
            files_searched=files_searched,
        )

        return ToolResult(data=output)

    async def description(
        self,
        input: GrepInput,
        options: dict[str, Any],
    ) -> str:
        """Get description."""
        return f"Search for '{input.pattern}'"

    async def prompt(self, options: dict[str, Any]) -> str:
        """Get tool prompt."""
        return """Search for a regex pattern in files.

Usage notes:
- Uses Python regex syntax
- Use glob to filter which files to search
- Set case_insensitive=True for case-insensitive search
- output_mode: 'content' shows matching lines, 'files_with_matches' shows file paths"""

    def is_concurrency_safe(self, input: GrepInput) -> bool:
        """Grep is concurrency safe."""
        return True

    def is_read_only(self, input: GrepInput) -> bool:
        """Grep is read-only."""
        return True

    def is_search_or_read_command(self, input: GrepInput) -> dict[str, bool]:
        """Mark as search operation."""
        return {"is_search": True, "is_read": False, "is_list": False}


# Create instance
grep_tool = GrepTool()