"""Bash tool for executing shell commands."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field

from claude_code_py.tool.base import Tool, build_tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError, ShellError, TimeoutError


class BashInput(BaseModel):
    """Input for Bash tool."""

    command: str = Field(description="The bash command to execute")
    description: Optional[str] = Field(
        default=None,
        description="Clear, concise description of what this command does",
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Optional timeout in milliseconds",
    )


@dataclass
class BashOutput:
    """Output from Bash tool."""

    stdout: str
    stderr: str
    exit_code: int
    interrupted: bool = False

    def __str__(self) -> str:
        result = []
        if self.stdout:
            result.append(self.stdout)
        if self.stderr:
            result.append(f"[stderr]\n{self.stderr}")
        result.append(f"\n[Exit code: {self.exit_code}]")
        return "\n".join(result)


class BashTool(Tool[BashInput, BashOutput, dict[str, Any]]):
    """Tool for executing bash commands."""

    name = "Bash"
    aliases = ["Shell", "Execute"]
    input_schema = BashInput
    max_result_size_chars = 25000

    async def call(
        self,
        args: BashInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[BashOutput]:
        """Execute the bash command.

        Args:
            args: Command arguments
            context: Execution context
            can_use_tool: Permission function
            parent_message: Parent message
            on_progress: Progress callback

        Returns:
            Tool result with command output
        """
        # Get timeout
        timeout = args.timeout or self._get_default_timeout()

        # Get working directory from context
        cwd = context.get_cwd()

        # Execute command
        try:
            process = await asyncio.create_subprocess_shell(
                args.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout / 1000 if timeout else None,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(timeout_seconds=timeout / 1000)

            exit_code = process.returncode or 0
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            if exit_code != 0:
                raise ShellError(
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=exit_code,
                )

            output = BashOutput(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=exit_code,
            )

            return ToolResult(data=output)

        except (TimeoutError, ShellError):
            raise
        except Exception as e:
            raise ToolError(f"Command execution failed: {e}")

    async def description(
        self,
        input: BashInput,
        options: dict[str, Any],
    ) -> str:
        """Get description of this tool use."""
        desc = input.description or f"Execute: {input.command[:50]}"
        return desc

    async def prompt(self, options: dict[str, Any]) -> str:
        """Get tool prompt."""
        return """Execute bash commands on the local system.

Usage notes:
- ALWAYS specify a clear, concise description for the command
- Use quotes around paths with spaces
- Commands run in the current working directory
- Timeout defaults to 2 minutes unless specified"""

    def is_concurrency_safe(self, input: BashInput) -> bool:
        """Check if command is safe to run concurrently."""
        # Only pure read operations are concurrency safe
        command = input.command.strip().lower()
        read_commands = {
            "cat", "head", "tail", "less", "ls", "find", "grep",
            "rg", "ag", "wc", "stat", "file", "echo", "pwd",
        }

        first_word = command.split()[0] if command.split() else ""
        return first_word in read_commands

    def is_read_only(self, input: BashInput) -> bool:
        """Check if command is read-only."""
        return self.is_concurrency_safe(input)

    def is_destructive(self, input: BashInput) -> bool:
        """Check if command is destructive."""
        command = input.command.strip().lower()
        destructive = {"rm", "mv", "cp", "chmod", "chown", "dd", "format"}
        first_word = command.split()[0] if command.split() else ""
        return first_word in destructive

    def user_facing_name(self, input: Optional[BashInput]) -> str:
        """Get user-facing name."""
        if input and input.description:
            return input.description
        return "Bash"

    def _get_default_timeout(self) -> int:
        """Get default timeout in ms."""
        return int(os.environ.get("CLAUDE_CODE_DEFAULT_TIMEOUT", "120000"))


# Create instance
bash_tool = BashTool()