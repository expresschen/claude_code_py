"""Plan mode V2 configuration and utilities.

Ported from TypeScript src/utils/planModeV2.ts
"""

from __future__ import annotations

import os
from typing import Optional


def is_plan_mode_interview_phase_enabled() -> bool:
    """Check if plan mode interview phase is enabled.

    Config priority:
    1. USER_TYPE=ant → always enabled
    2. CLAUDE_CODE_PLAN_MODE_INTERVIEW_PHASE env var
    3. Default: False (external users use 5-phase workflow)

    Returns:
        True if interview phase workflow should be used
    """
    # Ant内部用户始终启用
    if os.environ.get("USER_TYPE") == "ant":
        return True

    # 环境变量显式设置
    env = os.environ.get("CLAUDE_CODE_PLAN_MODE_INTERVIEW_PHASE", "")
    env_lower = env.lower()

    if env_lower == "true" or env_lower == "1":
        return True
    if env_lower == "false" or env_lower == "0":
        return False

    # 默认False（外部用户使用5-phase workflow）
    return False


def get_plan_mode_v2_agent_count() -> int:
    """Get the maximum number of Plan agents allowed.

    Environment variable override takes precedence.
    Enterprise/team users get 3, others get 1.

    Returns:
        Maximum number of Plan agents
    """
    # Environment variable override
    if os.environ.get("CLAUDE_CODE_PLAN_V2_AGENT_COUNT"):
        try:
            count = int(os.environ.get("CLAUDE_CODE_PLAN_V2_AGENT_COUNT", ""))
            if 0 < count <= 10:
                return count
        except ValueError:
            pass

    # Default: 1 agent for most users
    # TODO: Add subscription type checking when available
    return 1


def get_plan_mode_v2_explore_agent_count() -> int:
    """Get the maximum number of Explore agents allowed.

    Returns:
        Maximum number of Explore agents (default 3)
    """
    if os.environ.get("CLAUDE_CODE_PLAN_V2_EXPLORE_AGENT_COUNT"):
        try:
            count = int(os.environ.get("CLAUDE_CODE_PLAN_V2_EXPLORE_AGENT_COUNT", ""))
            if 0 < count <= 10:
                return count
        except ValueError:
            pass

    return 3


class PewterLedgerVariant:
    """Plan file structure prompt experiment variants."""

    TRIM = "trim"
    CUT = "cut"
    CAP = "cap"
    NONE = None


def get_pewter_ledger_variant() -> Optional[str]:
    """Get the pewter ledger experiment variant.

    Controls Phase 4 "Final Plan" bullets structure.

    Returns:
        Variant name or None
    """
    # TODO: Add GrowthBook integration when available
    # For now, use environment variable
    env = os.environ.get("CLAUDE_CODE_PEWTER_LEDGER", "")
    if env in ("trim", "cut", "cap"):
        return env
    return None