"""Tests for spawn_in_process.py"""
import pytest

from claude_code_py.utils.swarm.spawn_in_process import (
    SpawnContext,
    generate_task_id,
    format_agent_id,
    set_session_id,
    get_session_id,
    generate_session_id,
)
from claude_code_py.task.in_process_teammate import (
    InProcessSpawnConfig,
    InProcessSpawnOutput,
)


class TestIDGeneration:
    """Test ID generation functions."""

    def test_generate_task_id(self):
        """Should generate unique task IDs."""
        id1 = generate_task_id("test")
        id2 = generate_task_id("test")

        assert id1.startswith("test-")
        assert id2.startswith("test-")
        assert id1 != id2

    def test_format_agent_id(self):
        """Should format agent ID correctly."""
        agent_id = format_agent_id("researcher", "my-team")
        assert agent_id == "researcher@my-team"

    def test_format_agent_id_with_at(self):
        """Should handle @ in name."""
        agent_id = format_agent_id("worker@test", "team")
        assert agent_id == "worker-test@team"

    def test_generate_session_id(self):
        """Should generate unique session IDs."""
        id1 = generate_session_id()
        id2 = generate_session_id()

        assert id1.startswith("session-")
        assert id2.startswith("session-")
        assert id1 != id2


class TestSessionManagement:
    """Test session ID management."""

    def test_set_and_get_session_id(self):
        """Should set and get session ID."""
        session_id = "test-session-123"
        set_session_id(session_id)

        assert get_session_id() == session_id

        # Reset
        set_session_id(None)
        assert get_session_id() is None


class TestSpawnConfig:
    """Test spawn configuration."""

    def test_in_process_spawn_config(self):
        """Should create valid config."""
        config = InProcessSpawnConfig(
            name="researcher",
            team_name="my-team",
            prompt="Search for information",
            color="blue",
            plan_mode_required=False,
            model="claude-sonnet-4-6",
        )

        assert config.name == "researcher"
        assert config.team_name == "my-team"
        assert config.prompt == "Search for information"
        assert config.color == "blue"
        assert config.model == "claude-sonnet-4-6"

    def test_spawn_context(self):
        """Should create valid context."""
        context = SpawnContext(
            set_app_state=lambda fn: None,
            tool_use_id="tool_001",
        )

        assert context.tool_use_id == "tool_001"

    def test_spawn_output_success(self):
        """Should create success output."""
        output = InProcessSpawnOutput(
            success=True,
            agent_id="researcher@my-team",
            task_id="in_process_teammate-001",
            error=None,
        )

        assert output.success is True
        assert output.agent_id == "researcher@my-team"
        assert output.error is None

    def test_spawn_output_failure(self):
        """Should create failure output."""
        output = InProcessSpawnOutput(
            success=False,
            agent_id="worker@team",
            error="Failed to spawn",
        )

        assert output.success is False
        assert output.error == "Failed to spawn"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])