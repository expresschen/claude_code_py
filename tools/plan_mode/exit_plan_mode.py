"""ExitPlanMode tool implementation.

Ported from TypeScript src/tools/ExitPlanModeTool/ExitPlanModeV2Tool.ts

Key features:
1. Restores permission context (prePlanMode → restoreMode)
2. Handles auto mode restoration with circuit breaker check
3. Triggers plan_mode_exit attachment
4. Dynamic output based on context (isAgent, planWasEdited)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Literal

from pydantic import BaseModel

from claude_code_py.tool.base import Tool, ValidationResult
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError, ToolResultBlockParam
from .constants import EXIT_PLAN_MODE_TOOL_NAME
from .prompt import get_exit_plan_mode_prompt, get_plan_mode_exit_message
from .state import get_plan_mode_manager, PlanModePhase


# =============================================================================
# Input Schema
# =============================================================================


class AllowedPrompt(BaseModel):
    """Prompt-based permission request."""

    tool: Literal["Bash"] = "Bash"
    prompt: str  # Semantic description like "run tests", "install dependencies"


class ExitPlanModeInput(BaseModel):
    """Input for ExitPlanMode tool."""

    # Prompt-based permissions requested by the plan
    allowed_prompts: Optional[list[AllowedPrompt]] = None

    # SDK-facing fields (injected by normalizeToolInput)
    plan: Optional[str] = None  # Plan content from disk
    plan_file_path: Optional[str] = None  # Plan file path


# =============================================================================
# Output Schema (matches TypeScript)
# =============================================================================


@dataclass
class ExitPlanModeOutput:
    """Output from ExitPlanMode tool.

    Matches TypeScript outputSchema exactly.
    """

    plan: Optional[str]  # The plan that was presented
    is_agent: bool  # Whether in agent context
    file_path: Optional[str] = None  # Plan file path
    has_task_tool: Optional[bool] = None  # Agent tool available
    plan_was_edited: Optional[bool] = None  # User edited plan (CCR/Ctrl+G)
    awaiting_leader_approval: Optional[bool] = None  # Teammate awaiting lead
    request_id: Optional[str] = None  # Approval request ID


# =============================================================================
# Global State Flags (matching TypeScript bootstrap/state.ts)
# =============================================================================

_has_exited_plan_mode: bool = False
_needs_plan_mode_exit_attachment: bool = False
_needs_auto_mode_exit_attachment: bool = False


def has_exited_plan_mode_in_session() -> bool:
    """Check if plan mode has been exited in this session."""
    return _has_exited_plan_mode


def set_has_exited_plan_mode(value: bool) -> None:
    """Set the flag indicating plan mode was exited."""
    global _has_exited_plan_mode
    _has_exited_plan_mode = value


def needs_plan_mode_exit_attachment() -> bool:
    """Check if plan_mode_exit attachment needs to be sent."""
    return _needs_plan_mode_exit_attachment


def set_needs_plan_mode_exit_attachment(value: bool) -> None:
    """Set the flag for plan_mode_exit attachment."""
    global _needs_plan_mode_exit_attachment
    _needs_plan_mode_exit_attachment = value


def needs_auto_mode_exit_attachment() -> bool:
    """Check if auto_mode_exit attachment needs to be sent."""
    return _needs_auto_mode_exit_attachment


def set_needs_auto_mode_exit_attachment(value: bool) -> None:
    """Set the flag for auto_mode_exit attachment."""
    global _needs_auto_mode_exit_attachment
    _needs_auto_mode_exit_attachment = value


# =============================================================================
# Tool Implementation
# =============================================================================


class ExitPlanModeTool(Tool[ExitPlanModeInput, ExitPlanModeOutput, dict[str, Any]]):
    """Tool for exiting plan mode."""

    name = EXIT_PLAN_MODE_TOOL_NAME
    aliases: list[str] = []
    input_schema = ExitPlanModeInput
    max_result_size_chars = 100_000
    search_hint = "present plan for approval and start coding (plan mode only)"
    should_defer = True  # Requires user approval

    async def validate_input(
        self,
        input: ExitPlanModeInput,
        context: ToolUseContext,
    ) -> ValidationResult:
        """Validate input - check if in plan mode.

        Args:
            input: Tool input
            context: Tool context

        Returns:
            ValidationResult
        """
        # For teammates, skip this check
        if context.options.get("agent_id"):
            return ValidationResult.success()

        # Check if in plan mode (TypeScript line 204-218)
        mode = context.options.get("tool_permission_context", {}).get("mode", "default")
        if mode != "plan":
            return ValidationResult.failure(
                message="You are not in plan mode. This tool is only for exiting plan mode after writing a plan. If your plan was already approved, continue with implementation.",
                error_code=1,
            )

        return ValidationResult.success()

    async def check_permissions(
        self,
        input: ExitPlanModeInput,
        context: ToolUseContext,
    ) -> dict[str, Any]:
        """Check permissions for exiting plan mode.

        Args:
            input: Tool input
            context: Tool context

        Returns:
            Permission result dict
        """
        # For teammates, bypass permission UI
        if context.options.get("agent_id"):
            return {"behavior": "allow", "updated_input": input.model_dump()}

        # For non-teammates, require user confirmation
        return {
            "behavior": "ask",
            "message": "Exit plan mode?",
            "updated_input": input.model_dump(),
        }

    async def call(
        self,
        args: ExitPlanModeInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[ExitPlanModeOutput]:
        """Exit plan mode and restore permission context.

        Critical: This method must:
        1. Restore permission mode from prePlanMode
        2. Handle auto mode restoration with circuit breaker
        3. Restore stripped dangerous permissions
        4. Trigger plan_mode_exit attachment
        5. Clear plan mode state

        Args:
            args: Input arguments
            context: Execution context
            can_use_tool: Permission function
            parent_message: Parent message
            on_progress: Progress callback

        Returns:
            Tool result with plan info
        """
        is_agent = bool(context.options.get("agent_id"))

        # Get plan file path
        manager = get_plan_mode_manager()
        state = manager.get_state()
        file_path = str(state.plan_file_path) if state and state.plan_file_path else None

        # Get plan content
        # TypeScript: input.plan from permissionResult.updatedInput (CCR edit)
        # or fallback to disk
        input_plan = args.plan
        if input_plan is None and file_path:
            try:
                input_plan = Path(file_path).read_text()
            except Exception:
                pass

        plan = input_plan

        # Sync disk if user edited plan (CCR web UI)
        if input_plan is not None and file_path:
            try:
                Path(file_path).write_text(input_plan)
            except Exception as e:
                pass  # Log error but continue

        # =============================================================================
# CRITICAL: Permission Restoration Logic (TypeScript line 357-403)
        # =============================================================================

        from claude_code_py.utils.permissions.permission_setup import (
            PermissionMode,
            ToolPermissionContext,
            restore_dangerous_permissions,
        )
        from claude_code_py.state.app_store import set_app_state
        from claude_code_py.state.app_state import AppState

        # Get current permission context
        current_context = context.options.get("tool_permission_context")

        if current_context:
            # Convert dict to ToolPermissionContext if needed
            if isinstance(current_context, dict):
                current_context = ToolPermissionContext(
                    mode=PermissionMode(current_context.get("mode", "default")),
                    pre_plan_mode=PermissionMode(current_context.get("pre_plan_mode")) if current_context.get("pre_plan_mode") else None,
                    auto_mode_active=current_context.get("auto_mode_active", False),
                    stripped_dangerous_rules=current_context.get("stripped_dangerous_rules", False),
                    allow_rules=current_context.get("allow_rules", []),
                    deny_rules=current_context.get("deny_rules", []),
                    ask_rules=current_context.get("ask_rules", []),
                    auto_mode_rules=current_context.get("auto_mode_rules"),
                    additional_working_dirs=current_context.get("additional_working_dirs", []),
                )

            # Determine restore mode from prePlanMode
            pre_plan_mode = current_context.pre_plan_mode or PermissionMode.DEFAULT
            restore_mode = pre_plan_mode

            # Circuit breaker check for auto mode
            # If prePlanMode was 'auto' but gate is now off, fallback to 'default'
            auto_mode_gate_enabled = self._is_auto_mode_gate_enabled()
            if restore_mode == PermissionMode.AUTO and not auto_mode_gate_enabled:
                restore_mode = PermissionMode.DEFAULT

            # Handle stripped dangerous permissions
            restoring_to_auto = restore_mode == PermissionMode.AUTO
            base_context = current_context

            if restoring_to_auto:
                # Keep stripped for auto mode
                pass  # Permissions stay stripped
            elif current_context.stripped_dangerous_rules:
                # Restore dangerous permissions when exiting to non-auto
                base_context = restore_dangerous_permissions(current_context)

            # Auto mode state management
            auto_was_used_during_plan = current_context.auto_mode_active
            if auto_was_used_during_plan and not restoring_to_auto:
                # Need auto_mode_exit attachment
                set_needs_auto_mode_exit_attachment(True)

            # Build new context
            new_context = ToolPermissionContext(
                mode=restore_mode,
                pre_plan_mode=None,  # Clear prePlanMode
                auto_mode_active=restoring_to_auto,
                stripped_dangerous_rules=restoring_to_auto,
                allow_rules=base_context.allow_rules,
                deny_rules=base_context.deny_rules,
                ask_rules=base_context.ask_rules,
                auto_mode_rules=current_context.auto_mode_rules if restoring_to_auto else None,
                additional_working_dirs=base_context.additional_working_dirs,
            )

            # Update AppState with restored context
            set_app_state(
                lambda prev: AppState(
                    **{
                        k: v
                        for k, v in vars(prev).items()
                        if k != "tool_permission_context"
                    },
                    tool_permission_context=new_context,
                )
            )

        # =============================================================================
        # Trigger Attachments
        # =============================================================================

        # Set flags for attachment generation (TypeScript line 359-360)
        set_has_exited_plan_mode(True)
        set_needs_plan_mode_exit_attachment(True)

        # Exit plan mode state
        manager.exit_plan_mode()

        # Check if Agent tool is available
        has_task_tool = self._check_has_agent_tool(context)

        # Build output
        output = ExitPlanModeOutput(
            plan=plan,
            is_agent=is_agent,
            file_path=file_path,
            has_task_tool=has_task_tool,
            plan_was_edited=input_plan is not None,
        )

        return ToolResult(data=output)

    def _is_auto_mode_gate_enabled(self) -> bool:
        """Check if auto mode gate is enabled.

        Returns:
            True if auto mode can be entered
        """
        # Check environment variable
        env = os.environ.get("CLAUDE_CODE_AUTO_MODE_DISABLED", "")
        if env.lower() == "true":
            return False
        return True

    def _check_has_agent_tool(self, context: ToolUseContext) -> bool:
        """Check if Agent tool is available.

        Args:
            context: Tool context

        Returns:
            True if Agent tool exists
        """
        tools = context.options.get("tools", [])
        for tool in tools:
            if hasattr(tool, "name") and tool.name == "Agent":
                return True
        return False

    def map_tool_result_to_block_param(
        self,
        output: ExitPlanModeOutput,
        tool_use_id: str,
    ) -> ToolResultBlockParam:
        """Map tool output to tool_result block.

        Handles multiple cases:
        1. isAgent: Simple "ok" response
        2. Empty plan: "proceed" message
        3. Normal: Include full plan with team hint

        Args:
            output: Tool output
            tool_use_id: Tool use block ID

        Returns:
            ToolResultBlockParam with appropriate content
        """
        # Agent context: simple response
        if output.is_agent:
            return ToolResultBlockParam(
                type="tool_result",
                content="User has approved the plan. There is nothing else needed from you now. Please respond with \"ok\"",
                tool_use_id=tool_use_id,
            )

        # Empty plan
        if not output.plan or output.plan.strip() == "":
            return ToolResultBlockParam(
                type="tool_result",
                content="User has approved exiting plan mode. You can now proceed.",
                tool_use_id=tool_use_id,
            )

        # Team hint if Agent tool available
        team_hint = ""
        if output.has_task_tool:
            team_hint = "\n\nIf this plan can be broken down into multiple independent tasks, consider using the Agent tool to create a team and parallelize the work."

        # Label for edited plans
        plan_label = "Approved Plan (edited by user)" if output.plan_was_edited else "Approved Plan"

        content = f"""User has approved your plan. You can now start coding. Start with updating your todo list if applicable

Your plan has been saved to: {output.file_path}
You can refer back to it if needed during implementation.{team_hint}

## {plan_label}:
{output.plan}"""

        return ToolResultBlockParam(
            type="tool_result",
            content=content,
            tool_use_id=tool_use_id,
        )

    async def description(
        self,
        input: ExitPlanModeInput,
        options: dict[str, Any],
    ) -> str:
        """Get description."""
        return "Prompts the user to exit plan mode and start coding"

    async def prompt(self, options: dict[str, Any]) -> str:
        """Get tool prompt."""
        return get_exit_plan_mode_prompt()

    def is_concurrency_safe(self, input: ExitPlanModeInput) -> bool:
        """Exiting plan mode is concurrency safe."""
        return True

    def is_read_only(self, input: ExitPlanModeInput) -> bool:
        """Exit plan mode is NOT read-only - it writes state changes."""
        return False

    def user_facing_name(self, input: Optional[ExitPlanModeInput]) -> str:
        """Get user-facing name."""
        return "Exit Plan Mode"

    def is_enabled(self) -> bool:
        """Check if tool is enabled.

        Disabled when --channels is active.
        """
        # TODO: Check channels feature
        return True

    def requires_user_interaction(self) -> bool:
        """Requires user approval for non-teammates."""
        return True


# Create instance
exit_plan_mode_tool = ExitPlanModeTool()