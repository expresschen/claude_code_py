"""Type-safe ID definitions.

These are newtypes that provide type safety at runtime and compile time.
"""

from dataclasses import dataclass
from typing import NewType, TypeVar

# NewType creates a distinct type that's still just a string at runtime
AgentId = NewType("AgentId", str)
SessionId = NewType("SessionId", str)
TaskId = NewType("TaskId", str)

T = TypeVar("T")


def as_agent_id(value: str) -> AgentId:
    """Convert a string to an AgentId with validation."""
    if not value:
        raise ValueError("AgentId cannot be empty")
    return AgentId(value)


def as_session_id(value: str) -> SessionId:
    """Convert a string to a SessionId with validation."""
    if not value:
        raise ValueError("SessionId cannot be empty")
    return SessionId(value)


def as_task_id(value: str) -> TaskId:
    """Convert a string to a TaskId with validation."""
    if not value:
        raise ValueError("TaskId cannot be empty")
    return TaskId(value)


@dataclass(frozen=True)
class IdPrefixes:
    """ID prefixes for different entity types."""

    AGENT: str = "a"
    SESSION: str = "s"
    TASK_BASH: str = "b"
    TASK_AGENT: str = "a"
    TASK_REMOTE: str = "r"
    TASK_TEAMMATE: str = "t"