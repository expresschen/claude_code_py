"""Glob tool for file pattern matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError


class GlobInput(BaseModel):
    """Input for Glob tool."""

    pattern: str = Field(
        description="The glob pattern to match files against"
    )
    path: Optional[str] = Field(
        default=None,
        description="The directory to search in (default: current)",
    )


@dataclass
class GlobOutput:
    """Output from Glob tool."""

    matches: list[str]
    pattern: str
    base_path: str

    def __str__(self) -> str:
        if not self.matches:
            return f"No files matching '{self.pattern}'"

        result = [f"Found {len(self.matches)} file(s) matching '{self.pattern}':"]
        for match in self.matches[:20]:
            result.append(f"  {match}")
        if len(self.matches) > 20:
            result.append(f"  ... and {len(self.matches) - 20} more")
        return "\n".join(result)


class GlobTool(Tool[GlobInput, GlobOutput, dict[str, Any]]):
    """Tool for glob pattern matching."""

    name = "Glob"
    aliases = ["FindFiles"]
    input_schema = GlobInput

    async def call(
        self,
        args: GlobInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[GlobOutput]:
        """Find files matching the pattern.

        Args:
            args: Glob arguments
            context: Execution context
            can_use_tool: Permission function
            parent_message: Parent message
            on_progress: Progress callback

        Returns:
            Tool result with matching files
        """
        # Resolve path relative to cwd (handles None by returning cwd)
        base_path = Path(context.resolve_path(args.path))

        # Resolve glob pattern
        try:
            matches = sorted(
                str(p.relative_to(base_path)) if p.is_relative_to(base_path) else str(p)
                for p in base_path.glob(args.pattern)
                if p.is_file()
            )

            # Apply limit
            limit = context.glob_limits.get("max_results", 1000) if context.glob_limits else 1000
            matches = matches[:limit]

            output = GlobOutput(
                matches=matches,
                pattern=args.pattern,
                base_path=str(base_path),
            )

            return ToolResult(data=output)

        except Exception as e:
            raise ToolError(f"Glob failed: {e}")

    async def description(
        self,
        input: GlobInput,
        options: dict[str, Any],
    ) -> str:
        """Get description."""
        return f"Find files matching '{input.pattern}'"

    async def prompt(self, options: dict[str, Any]) -> str:
        """Get tool prompt."""
        return """Find files matching a glob pattern.

Pattern examples:
- `**/*.py` - All Python files recursively
- `src/**/*.ts` - TypeScript files in src/
- `*.md` - Markdown files in current directory
- `**/test_*.py` - Test files anywhere"""

    def is_concurrency_safe(self, input: GlobInput) -> bool:
        """Glob is concurrency safe."""
        return True

    def is_read_only(self, input: GlobInput) -> bool:
        """Glob is read-only."""
        return True

    def is_search_or_read_command(self, input: GlobInput) -> dict[str, bool]:
        """Mark as search operation."""
        return {"is_search": True, "is_read": False, "is_list": False}


# Create instance
glob_tool = GlobTool()