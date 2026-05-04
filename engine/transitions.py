"""State transitions for the query loop."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Terminal:
    """Terminal state - the query loop has ended."""

    reason: str = "complete"

    # Possible reasons:
    # - "complete": Normal completion
    # - "max_turns": Reached maximum turns
    # - "aborted": User aborted
    # - "error": Error occurred
    # - "budget_exceeded": Budget limit reached


@dataclass
class Continue:
    """Continue state - the query loop should continue."""

    reason: str
    # Possible reasons:
    # - "tool_use": Tool was executed, need to process result
    # - "recovery": Recovered from an error
    # - "compact": Context was compacted


# Re-export for convenience
__all__ = ["Terminal", "Continue"]