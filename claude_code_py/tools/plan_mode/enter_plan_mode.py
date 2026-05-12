"""EnterPlanMode tool implementation.

Ported from TypeScript src/tools/EnterPlanModeTool/EnterPlanModeTool.ts

Key architecture (matches TypeScript):
1. Tool result: Returns simple message only
2. Attachment message: Injects full workflow instructions via new_messages

The detailed instructions (get_plan_mode_v2_instructions or get_plan_mode_interview_instructions)
are injected through attachment messages, NOT in tool result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError, ToolResultBlockParam
from claude_code_py.core_types.message import AttachmentMessage
from .constants import ENTER_PLAN_MODE_TOOL_NAME, PLAN_PERMISSION_MODE
from .prompt import (
    get_enter_plan_mode_prompt,
    get_plan_mode_v2_instructions,
    get_plan_mode_interview_instructions,
    get_plan_template,
)
from .state import get_plan_mode_manager, PlanModePhase
from .plan_mode_v2 import (
    is_plan_mode_interview_phase_enabled,
    get_plan_mode_v2_agent_count,
    get_plan_mode_v2_explore_agent_count,
)


class EnterPlanModeInput(BaseModel):
    """Input for EnterPlanMode tool."""

    # No parameters needed - just triggers transition
    pass


@dataclass
class EnterPlanModeOutput:
    """Output from EnterPlanMode tool.

    Note: This matches TypeScript outputSchema which only has 'message'.
    Detailed workflow instructions are injected via attachment message.
    """

    message: str


class EnterPlanModeTool(Tool[EnterPlanModeInput, EnterPlanModeOutput, dict[str, Any]]):
    """Tool for entering plan mode."""

    name = ENTER_PLAN_MODE_TOOL_NAME
    aliases: list[str] = []
    input_schema = EnterPlanModeInput
    max_result_size_chars = 100_000
    search_hint = "switch to plan mode to design an approach before coding"
    should_defer = True  # Requires user approval

    async def call(
        self,
        args: EnterPlanModeInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[EnterPlanModeOutput]:
        """Enter plan mode.

        This method:
        1. Checks if already in plan mode
        2. Calls prepareContextForPlanMode to handle permission context
        3. Updates AppState.toolPermissionContext
        4. Creates plan file
        5. Returns attachment message with full workflow instructions

        Args:
            args: Input arguments (empty)
            context: Execution context
            can_use_tool: Permission function
            parent_message: Parent message
            on_progress: Progress callback

        Returns:
            Tool result with plan mode info and attachment message
        """
        # Check if used in agent context (TypeScript throws here)
        if context.options.get("agent_id"):
            raise ToolError("EnterPlanMode tool cannot be used in agent contexts")

        # Check if already in plan mode
        manager = get_plan_mode_manager()
        if manager.is_in_plan_mode():
            raise ToolError("Already in plan mode")

        # Get current working directory
        cwd = context.options.get("cwd", str(Path.cwd()))

        # Critical: Prepare permission context for plan mode
        # This handles auto mode activation, dangerous permission stripping, etc.
        from claude_code_py.utils.permissions.permission_setup import (
            prepare_context_for_plan_mode,
        )
        from claude_code_py.state.app_store import get_app_store, set_app_state
        from claude_code_py.state.app_state import AppState

        # Get current permission context
        current_context = context.options.get("tool_permission_context")
        if current_context:
            # Call prepareContextForPlanMode (TypeScript line 91-94)
            # This already sets mode to PLAN internally
            new_permission_context = prepare_context_for_plan_mode(current_context)

            # Update AppState.toolPermissionContext
            # In TypeScript: context.setAppState(prev => ({...prev, toolPermissionContext: ...}))
            set_app_state(
                lambda prev: AppState(
                    **{
                        k: v
                        for k, v in vars(prev).items()
                        if k != "tool_permission_context"
                    },
                    tool_permission_context=new_permission_context,
                )
            )

        # Enter plan mode state
        state = manager.enter_plan_mode(cwd)

        # Store original permission mode for restoration on exit
        original_mode = context.options.get("permission_mode", "default")
        state.original_permission_mode = original_mode

        # Create initial plan file template
        plan_file_path = state.plan_file_path
        plan_exists = False
        if plan_file_path:
            plan_file_path.write_text(get_plan_template())
            plan_exists = plan_file_path.exists()

        # Build output (only message, per TypeScript outputSchema)
        output = EnterPlanModeOutput(
            message="Entered plan mode. You should now focus on exploring the codebase and designing an implementation approach."
        )

        # Create attachment message with full workflow instructions
        # This is where get_plan_mode_v2_instructions and get_plan_mode_interview_instructions are used
        attachment = self._create_plan_mode_attachment(
            plan_file_path=str(plan_file_path) if plan_file_path else "",
            plan_exists=plan_exists,
            is_sub_agent=bool(context.options.get("agent_id")),
        )

        return ToolResult(
            data=output,
            new_messages=[attachment],
        )

    def _create_plan_mode_attachment(
        self,
        plan_file_path: str,
        plan_exists: bool,
        is_sub_agent: bool = False,
        reminder_type: str = "full",
    ) -> AttachmentMessage:
        """Create plan_mode attachment message with full workflow instructions.

        This matches TypeScript getAttachmentMessages -> plan_mode attachment
        which then goes through messages.ts getPlanModeInstructions.

        Args:
            plan_file_path: Path to plan file
            plan_exists: Whether plan file already exists
            is_sub_agent: Whether in sub-agent context
            reminder_type: 'full' or 'sparse'

        Returns:
            AttachmentMessage with full workflow instructions
        """
        from claude_code_py.core_types.message import AttachmentMessage

        # Get full workflow instructions based on Interview Phase configuration
        if is_plan_mode_interview_phase_enabled():
            # Use interview/iterative workflow instructions
            instructions = get_plan_mode_interview_instructions(
                plan_file_path=plan_file_path,
                plan_exists=plan_exists,
            )
        else:
            # Use 5-phase workflow instructions
            explore_count = get_plan_mode_v2_explore_agent_count()
            plan_count = get_plan_mode_v2_agent_count()
            instructions = get_plan_mode_v2_instructions(
                plan_file_path=plan_file_path,
                plan_exists=plan_exists,
                explore_agent_count=explore_count,
                plan_agent_count=plan_count,
                is_interview_phase=False,
            )

        # Create attachment message
        # TypeScript: messages.ts wraps plan mode instructions in system-reminder
        return AttachmentMessage(
            attachment={
                "type": "plan_mode",
                "reminder_type": reminder_type,
                "plan_file_path": plan_file_path,
                "plan_exists": plan_exists,
                "is_sub_agent": is_sub_agent,
                "content": instructions,
            }
        )

    def map_tool_result_to_block_param(
        self,
        output: EnterPlanModeOutput,
        tool_use_id: str,
    ) -> ToolResultBlockParam:
        """Map tool output to tool_result block.

        This returns the SIMPLE message (matching TypeScript).
        Full instructions are in the attachment message (new_messages).

        Args:
            output: Tool output (only message field)
            tool_use_id: Tool use block ID

        Returns:
            ToolResultBlockParam with simple message
        """
        # TypeScript line 104-118: Returns simple message with short instruction
        # Full workflow instructions come via plan_mode attachment
        if is_plan_mode_interview_phase_enabled():
            content = f"{output.message}\n\nDO NOT write or edit any files except the plan file. Detailed workflow instructions will follow."
        else:
            content = f"{output.message}\n\nIn plan mode, you should explore the codebase and design an approach. Full workflow instructions will follow."

        return ToolResultBlockParam(
            type="tool_result",
            content=content,
            tool_use_id=tool_use_id,
        )

    async def description(
        self,
        input: EnterPlanModeInput,
        options: dict[str, Any],
    ) -> str:
        """Get description."""
        return "Requests permission to enter plan mode for complex tasks requiring exploration and design"

    async def prompt(self, options: dict[str, Any]) -> str:
        """Get tool prompt."""
        return get_enter_plan_mode_prompt()

    def is_concurrency_safe(self, input: EnterPlanModeInput) -> bool:
        """Entering plan mode is concurrency safe."""
        return True

    def is_read_only(self, input: EnterPlanModeInput) -> bool:
        """Entering plan mode is read-only."""
        return True

    def user_facing_name(self, input: Optional[EnterPlanModeInput]) -> str:
        """Get user-facing name."""
        return "Enter Plan Mode"

    def is_enabled(self) -> bool:
        """Check if tool is enabled.

        Disabled when --channels is active (same as TypeScript).
        """
        # TODO: Check channels feature flag when implemented
        return True


# Create instance
enter_plan_mode_tool = EnterPlanModeTool()