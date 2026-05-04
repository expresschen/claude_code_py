"""Plan mode tools package.

This implements EnterPlanMode and ExitPlanMode tools.
"""

from __future__ import annotations

from .constants import (
    ENTER_PLAN_MODE_TOOL_NAME,
    EXIT_PLAN_MODE_TOOL_NAME,
    PLAN_PERMISSION_MODE,
    PLAN_FILE_NAME,
)
from .state import (
    PlanModePhase,
    PlanModeState,
    PlanModeManager,
    PlanContent,
    PlanStep,
    get_plan_mode_manager,
    is_in_plan_mode,
    get_plan_file_path,
)
from .prompt import (
    get_enter_plan_mode_prompt,
    get_exit_plan_mode_prompt,
    get_plan_mode_instructions,
    get_plan_template,
)
from .plan_mode_v2 import (
    is_plan_mode_interview_phase_enabled,
    get_plan_mode_v2_agent_count,
    get_plan_mode_v2_explore_agent_count,
    PewterLedgerVariant,
    get_pewter_ledger_variant,
)
from .enter_plan_mode import (
    EnterPlanModeTool,
    EnterPlanModeInput,
    EnterPlanModeOutput,
    enter_plan_mode_tool,
)
from .exit_plan_mode import (
    ExitPlanModeTool,
    ExitPlanModeInput,
    ExitPlanModeOutput,
    exit_plan_mode_tool,
    # State flags
    has_exited_plan_mode_in_session,
    set_has_exited_plan_mode,
    needs_plan_mode_exit_attachment,
    set_needs_plan_mode_exit_attachment,
    needs_auto_mode_exit_attachment,
    set_needs_auto_mode_exit_attachment,
)

__all__ = [
    # Constants
    "ENTER_PLAN_MODE_TOOL_NAME",
    "EXIT_PLAN_MODE_TOOL_NAME",
    "PLAN_PERMISSION_MODE",
    "PLAN_FILE_NAME",
    # State
    "PlanModePhase",
    "PlanModeState",
    "PlanModeManager",
    "PlanContent",
    "PlanStep",
    "get_plan_mode_manager",
    "is_in_plan_mode",
    "get_plan_file_path",
    # Exit Plan Mode Flags
    "has_exited_plan_mode_in_session",
    "set_has_exited_plan_mode",
    "needs_plan_mode_exit_attachment",
    "set_needs_plan_mode_exit_attachment",
    "needs_auto_mode_exit_attachment",
    "set_needs_auto_mode_exit_attachment",
    # Prompts
    "get_enter_plan_mode_prompt",
    "get_exit_plan_mode_prompt",
    "get_plan_mode_instructions",
    "get_plan_template",
    # Plan Mode V2 Config
    "is_plan_mode_interview_phase_enabled",
    "get_plan_mode_v2_agent_count",
    "get_plan_mode_v2_explore_agent_count",
    "PewterLedgerVariant",
    "get_pewter_ledger_variant",
    # Tools
    "EnterPlanModeTool",
    "EnterPlanModeInput",
    "EnterPlanModeOutput",
    "enter_plan_mode_tool",
    "ExitPlanModeTool",
    "ExitPlanModeInput",
    "ExitPlanModeOutput",
    "exit_plan_mode_tool",
]


def get_plan_mode_tools():
    """Get all plan mode tools."""
    return [
        enter_plan_mode_tool,
        exit_plan_mode_tool,
    ]