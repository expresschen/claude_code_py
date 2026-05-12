"""Plan mode state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class PlanModePhase(str, Enum):
    """Phases of plan mode."""

    INTERVIEW = "interview"  # Asking clarifying questions
    EXPLORATION = "exploration"  # Exploring codebase
    PLANNING = "planning"  # Writing plan
    APPROVAL = "approval"  # Waiting for approval


@dataclass
class PlanModeState:
    """State for plan mode session."""

    is_active: bool = False
    phase: PlanModePhase = PlanModePhase.EXPLORATION
    plan_file_path: Optional[Path] = None
    started_at: Optional[datetime] = None
    original_permission_mode: Optional[str] = None
    questions_asked: int = 0
    files_explored: int = 0


@dataclass
class PlanContent:
    """Content of a plan."""

    title: str
    summary: str
    steps: list[PlanStep]
    considerations: Optional[str] = None
    alternatives: Optional[list[str]] = None


@dataclass
class PlanStep:
    """A single step in the plan."""

    description: str
    files_to_modify: Optional[list[str]] = None
    details: Optional[str] = None
    dependencies: Optional[list[str]] = None
    estimated_complexity: Optional[str] = None  # "low", "medium", "high"


class PlanModeManager:
    """Manages plan mode state."""

    def __init__(self):
        self._state: Optional[PlanModeState] = None

    def enter_plan_mode(self, cwd: Optional[str] = None) -> PlanModeState:
        """Enter plan mode.

        Args:
            cwd: Current working directory for plan file

        Returns:
            Plan mode state
        """
        if self._state and self._state.is_active:
            raise RuntimeError("Already in plan mode")

        # Create plan file path
        plan_dir = Path(cwd or ".") / ".claude"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_file_path = plan_dir / "plan.md"

        self._state = PlanModeState(
            is_active=True,
            phase=PlanModePhase.EXPLORATION,
            plan_file_path=plan_file_path,
            started_at=datetime.now(),
        )

        return self._state

    def exit_plan_mode(self) -> Optional[PlanModeState]:
        """Exit plan mode.

        Returns:
            Previous state or None if not in plan mode
        """
        if not self._state or not self._state.is_active:
            return None

        previous_state = self._state
        self._state = PlanModeState(is_active=False)
        return previous_state

    def get_state(self) -> Optional[PlanModeState]:
        """Get current plan mode state."""
        return self._state

    def is_in_plan_mode(self) -> bool:
        """Check if currently in plan mode."""
        return self._state is not None and self._state.is_active

    def set_phase(self, phase: PlanModePhase) -> None:
        """Set current phase."""
        if self._state:
            self._state.phase = phase

    def get_plan_file_path(self) -> Optional[Path]:
        """Get plan file path."""
        if self._state:
            return self._state.plan_file_path
        return None

    def increment_questions_asked(self) -> int:
        """Increment questions asked counter."""
        if self._state:
            self._state.questions_asked += 1
            return self._state.questions_asked
        return 0

    def increment_files_explored(self) -> int:
        """Increment files explored counter."""
        if self._state:
            self._state.files_explored += 1
            return self._state.files_explored
        return 0


# Global plan mode manager
_plan_mode_manager: Optional[PlanModeManager] = None


def get_plan_mode_manager() -> PlanModeManager:
    """Get the global plan mode manager."""
    if _plan_mode_manager is None:
        _plan_mode_manager = PlanModeManager()
    return _plan_mode_manager


def is_in_plan_mode() -> bool:
    """Check if currently in plan mode."""
    return get_plan_mode_manager().is_in_plan_mode()


def get_plan_file_path() -> Optional[Path]:
    """Get current plan file path."""
    return get_plan_mode_manager().get_plan_file_path()