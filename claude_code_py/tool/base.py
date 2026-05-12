"""Tool base class and factory.

This implements the core Tool abstraction from Tool.ts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, TypeVar, Union, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from .context import ToolUseContext, CanUseToolFn
    from .result import ToolResult, ToolCallProgress
    from claude_code_py.core_types.message import AssistantMessage
    from claude_code_py.core_types.permissions import PermissionResult

# Type variables for Tool generic
InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT")
ProgressT = TypeVar("ProgressT")


class ValidationResult(BaseModel):
    """Result of input validation."""

    result: bool
    message: Optional[str] = None
    error_code: Optional[int] = None

    @classmethod
    def success(cls) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(result=True)

    @classmethod
    def failure(
        cls,
        message: str,
        error_code: int = 400,
    ) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(result=False, message=message, error_code=error_code)


class Tool(ABC, Generic[InputT, OutputT, ProgressT]):
    """Abstract base class for tools.

    This mirrors the TypeScript Tool interface with all methods.
    """

    name: str
    aliases: list[str] = []
    search_hint: Optional[str] = None
    input_schema: type[InputT]
    output_schema: Optional[Any] = None
    max_result_size_chars: int = 25000
    strict: bool = False
    should_defer: bool = False
    always_load: bool = False
    mcp_info: Optional[dict[str, str]] = None
    is_mcp: bool = False
    is_lsp: bool = False

    @abstractmethod
    async def call(
        self,
        args: InputT,
        context: "ToolUseContext",
        can_use_tool: "CanUseToolFn",
        parent_message: "AssistantMessage",
        on_progress: Optional["ToolCallProgress[ProgressT]"] = None,
    ) -> "ToolResult[OutputT]":
        """Execute the tool.

        Args:
            args: Validated input arguments
            context: Tool execution context
            can_use_tool: Permission check function
            parent_message: The assistant message containing this tool use
            on_progress: Optional progress callback

        Returns:
            Tool result with output data
        """
        ...

    @abstractmethod
    async def description(
        self,
        input: InputT,
        options: dict[str, Any],
    ) -> str:
        """Generate a description of this tool use.

        Args:
            input: Tool input
            options: Options including tool_permission_context, tools, etc.

        Returns:
            Human-readable description
        """
        ...

    @abstractmethod
    async def prompt(self, options: dict[str, Any]) -> str:
        """Generate the tool's prompt for the model.

        Args:
            options: Options including tools, agents, etc.

        Returns:
            Tool prompt string
        """
        ...

    def is_concurrency_safe(self, input: InputT) -> bool:
        """Check if this tool call can run concurrently with others.

        Default: False (assume not safe).

        Args:
            input: Tool input

        Returns:
            True if concurrent execution is safe
        """
        return False

    def is_read_only(self, input: InputT) -> bool:
        """Check if this tool only reads, never writes.

        Default: False (assume writes).

        Args:
            input: Tool input

        Returns:
            True if the tool only reads
        """
        return False

    def is_destructive(self, input: InputT) -> bool:
        """Check if this tool performs destructive operations.

        Default: False.

        Args:
            input: Tool input

        Returns:
            True if the tool is destructive
        """
        return False

    def is_enabled(self) -> bool:
        """Check if this tool is enabled.

        Returns:
            True if enabled
        """
        return True

    async def check_permissions(
        self,
        input: InputT,
        context: "ToolUseContext",
    ) -> "PermissionResult":
        """Check permissions for this tool use.

        Default: Defer to general permission system.

        Args:
            input: Tool input
            context: Tool execution context

        Returns:
            Permission result
        """
        from claude_code_py.core_types.permissions import PermissionResult

        return PermissionResult.allow(updated_input=input.model_dump() if hasattr(input, "model_dump") else input)

    async def validate_input(
        self,
        input: InputT,
        context: "ToolUseContext",
    ) -> ValidationResult:
        """Validate input before execution.

        Args:
            input: Tool input
            context: Tool execution context

        Returns:
            Validation result
        """
        return ValidationResult.success()

    def user_facing_name(self, input: Optional[InputT]) -> str:
        """Get a user-facing name for this tool use.

        Args:
            input: Tool input (partial)

        Returns:
            User-facing name
        """
        return self.name

    def get_path(self, input: InputT) -> Optional[str]:
        """Get the file path this tool operates on, if any.

        Args:
            input: Tool input

        Returns:
            File path or None
        """
        return None

    def interrupt_behavior(self) -> str:
        """What happens when user submits new message while tool runs.

        Returns:
            'cancel' or 'block'
        """
        return "block"

    def is_search_or_read_command(self, input: InputT) -> dict[str, bool]:
        """Check if this is a search/read operation for UI collapsing.

        Args:
            input: Tool input

        Returns:
            Dict with is_search, is_read, is_list flags
        """
        return {"is_search": False, "is_read": False, "is_list": False}

    def to_auto_classifier_input(self, input: InputT) -> Any:
        """Convert tool input for auto-mode security classifier.

        Args:
            input: Tool input

        Returns:
            Classifier input (empty string to skip)
        """
        return ""

    def inputs_equivalent(self, a: InputT, b: InputT) -> bool:
        """Check if two inputs are equivalent.

        Args:
            a: First input
            b: Second input

        Returns:
            True if equivalent
        """
        return a == b

    def backfill_observable_input(self, input: dict[str, Any]) -> None:
        """Mutate input in place to add legacy/derived fields.

        Args:
            input: Input dict to mutate
        """
        pass

    def map_tool_result_to_block_param(
        self,
        output: OutputT,
        tool_use_id: str,
    ) -> "ToolResultBlockParam":
        """Map tool output to a tool_result block parameter.

        This is called after tool execution to create the tool_result
        message content. Override to customize output formatting.

        Args:
            output: Tool output data
            tool_use_id: ID of the tool use block

        Returns:
            ToolResultBlockParam with content for the API
        """
        from .result import ToolResultBlockParam

        # Default: convert output to string
        if isinstance(output, str):
            content = output
        elif hasattr(output, "model_dump"):
            content = str(output.model_dump())
        else:
            content = str(output)

        return ToolResultBlockParam(
            type="tool_result",
            tool_use_id=tool_use_id,
            content=content,
        )


# Type alias for Tool definition (partial tool with defaults)
ToolDef = Union[Tool, dict[str, Any]]


# Default implementations
class ToolDefaults:
    """Default implementations for tool methods."""

    @staticmethod
    def is_enabled() -> bool:
        return True

    @staticmethod
    def is_concurrency_safe(input: Any) -> bool:
        return False

    @staticmethod
    def is_read_only(input: Any) -> bool:
        return False

    @staticmethod
    def is_destructive(input: Any) -> bool:
        return False

    @staticmethod
    def check_permissions(input: Any, context: Any) -> "PermissionResult":
        from claude_code_py.core_types.permissions import PermissionResult

        return PermissionResult.allow(updated_input=input)

    @staticmethod
    def to_auto_classifier_input(input: Any) -> str:
        return ""

    @staticmethod
    def user_facing_name(tool_name: str, input: Any) -> str:
        return tool_name


def build_tool(
    name: str,
    input_schema: type[BaseModel],
    call_fn: Callable,
    *,
    aliases: list[str] = None,
    prompt_fn: Optional[Callable] = None,
    description_fn: Optional[Callable] = None,
    is_concurrency_safe_fn: Optional[Callable] = None,
    is_read_only_fn: Optional[Callable] = None,
    is_destructive_fn: Optional[Callable] = None,
    check_permissions_fn: Optional[Callable] = None,
    user_facing_name_fn: Optional[Callable] = None,
    **kwargs,
) -> Tool:
    """Factory function to build a tool from a partial definition.

    This mirrors the TypeScript buildTool function.

    Args:
        name: Tool name
        input_schema: Pydantic model for input
        call_fn: Tool execution function
        aliases: Optional aliases
        prompt_fn: Optional prompt function
        description_fn: Optional description function
        is_concurrency_safe_fn: Optional concurrency check
        is_read_only_fn: Optional read-only check
        is_destructive_fn: Optional destructive check
        check_permissions_fn: Optional permission check
        user_facing_name_fn: Optional user-facing name function
        **kwargs: Additional attributes

    Returns:
        Complete Tool instance
    """
    from claude_code_py.core_types.permissions import PermissionResult

    class DynamicTool(Tool):
        pass

    # Set required attributes
    DynamicTool.name = name
    DynamicTool.input_schema = input_schema
    DynamicTool.aliases = aliases or []

    # Create instance
    instance = DynamicTool()

    # Set call method
    async def call_wrapper(
        args,
        context,
        can_use_tool,
        parent_message,
        on_progress=None,
    ):
        return await call_fn(args, context, can_use_tool, parent_message, on_progress)

    instance.call = call_wrapper

    # Set optional methods with defaults
    async def default_prompt(options):
        return f"Tool: {name}"

    instance.prompt = prompt_fn or default_prompt

    async def default_description(input, options):
        return f"Using {name}"

    instance.description = description_fn or default_description

    instance.is_concurrency_safe = is_concurrency_safe_fn or ToolDefaults.is_concurrency_safe
    instance.is_read_only = is_read_only_fn or ToolDefaults.is_read_only
    instance.is_destructive = is_destructive_fn or ToolDefaults.is_destructive

    async def default_check_permissions(input, context):
        return PermissionResult.allow(updated_input=input)

    instance.check_permissions = check_permissions_fn or default_check_permissions

    instance.user_facing_name = lambda input: user_facing_name_fn(input) if user_facing_name_fn else name

    # Set additional attributes
    for key, value in kwargs.items():
        setattr(instance, key, value)

    return instance


def find_tool_by_name(tools: list[Tool], name: str) -> Optional[Tool]:
    """Find a tool by name or alias.

    Args:
        tools: List of tools
        name: Tool name to find

    Returns:
        Tool or None if not found
    """
    for tool in tools:
        if tool.name == name or name in (tool.aliases or []):
            return tool
    return None


def tool_matches_name(tool: Tool, name: str) -> bool:
    """Check if a tool matches the given name.

    Args:
        tool: Tool to check
        name: Name to match

    Returns:
        True if tool name or alias matches
    """
    return tool.name == name or name in (tool.aliases or [])