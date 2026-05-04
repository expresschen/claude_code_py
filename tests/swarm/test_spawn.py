"""Tests for teammate spawning."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from claude_code_py.task.manager import (
    spawn_in_process_teammate_v2,
    SpawnTeammateConfig,
    SpawnTeammateResult,
    register_task,
    unregister_task,
    get_task_by_id,
    find_task_by_agent_id,
    _TASK_REGISTRY,
)
from claude_code_py.task.in_process_teammate import (
    TeammateIdentity,
    InProcessTeammateTaskState,
    create_in_process_teammate_state,
    generate_agent_id,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear task registry before each test."""
    _TASK_REGISTRY.clear()


@pytest.fixture
def temp_config_home():
    """Set temporary CLAUDE_CONFIG_HOME for tests."""
    temp_dir = tempfile.mkdtemp()
    old_env = os.environ.get("CLAUDE_CONFIG_HOME")
    os.environ["CLAUDE_CONFIG_HOME"] = temp_dir
    yield temp_dir
    if old_env:
        os.environ["CLAUDE_CONFIG_HOME"] = old_env
    else:
        os.environ.pop("CLAUDE_CONFIG_HOME", None)
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestSpawnTeammateConfig:
    """Test SpawnTeammateConfig creation."""

    def test_spawn_config_creation(self):
        """Test SpawnTeammateConfig creation."""
        config = SpawnTeammateConfig(
            name="worker",
            team_name="test-team",
            prompt="Do something",
        )
        assert config.name == "worker"
        assert config.team_name == "test-team"
        assert config.prompt == "Do something"
        assert config.description is None
        assert config.model is None
        assert config.color is None

    def test_spawn_config_with_optional_fields(self):
        """Test SpawnTeammateConfig with optional fields."""
        config = SpawnTeammateConfig(
            name="researcher",
            team_name="my-team",
            prompt="Research topic X",
            description="Research task",
            model="claude-3-opus",
            color="blue",
            plan_mode_required=True,
        )
        assert config.name == "researcher"
        assert config.team_name == "my-team"
        assert config.description == "Research task"
        assert config.model == "claude-3-opus"
        assert config.color == "blue"
        assert config.plan_mode_required == True


class TestGenerateAgentId:
    """Test agent ID generation."""

    def test_generate_agent_id_basic(self):
        """Test basic agent ID generation."""
        agent_id = generate_agent_id("worker", "test-team")
        assert agent_id == "worker@test-team"

    def test_generate_agent_id_with_special_chars(self):
        """Test agent ID with special characters."""
        # Note: generate_agent_id does not sanitize - it just formats
        agent_id = generate_agent_id("research-agent", "my-team")
        assert agent_id == "research-agent@my-team"


class TestTeammateIdentity:
    """Test TeammateIdentity creation."""

    def test_teammate_identity_basic(self):
        """Test basic TeammateIdentity."""
        identity = TeammateIdentity(
            agent_id="worker@test-team",
            agent_name="worker",
            team_name="test-team",
            parent_session_id="session-123",
        )
        assert identity.agent_id == "worker@test-team"
        assert identity.agent_name == "worker"
        assert identity.team_name == "test-team"
        assert identity.parent_session_id == "session-123"
        assert identity.color is None
        assert identity.plan_mode_required == False

    def test_teammate_identity_with_optional_fields(self):
        """Test TeammateIdentity with optional fields."""
        identity = TeammateIdentity(
            agent_id="researcher@team",
            agent_name="researcher",
            team_name="team",
            parent_session_id="parent-session",
            color="green",
            plan_mode_required=True,
        )
        assert identity.color == "green"
        assert identity.plan_mode_required == True


class TestTaskRegistration:
    """Test task registration functions."""

    def test_task_registration(self):
        """Test task registration."""
        identity = TeammateIdentity(
            agent_id="test@team",
            agent_name="test",
            team_name="team",
            parent_session_id="session",
        )

        task = create_in_process_teammate_state("task-1", identity, "prompt")
        register_task(task)

        assert "task-1" in _TASK_REGISTRY
        assert _TASK_REGISTRY["task-1"].identity.agent_id == "test@team"

    def test_task_unregistration(self):
        """Test task unregistration."""
        identity = TeammateIdentity(
            agent_id="test@team",
            agent_name="test",
            team_name="team",
            parent_session_id="session",
        )

        task = create_in_process_teammate_state("task-2", identity, "prompt")
        register_task(task)

        assert "task-2" in _TASK_REGISTRY

        unregister_task("task-2")
        assert "task-2" not in _TASK_REGISTRY

    def test_get_task_by_id(self):
        """Test get_task_by_id function."""
        identity = TeammateIdentity(
            agent_id="test@team",
            agent_name="test",
            team_name="team",
            parent_session_id="session",
        )

        task = create_in_process_teammate_state("task-3", identity, "prompt")
        register_task(task)

        found = get_task_by_id("task-3")
        assert found is not None
        assert found.identity.agent_id == "test@team"

        not_found = get_task_by_id("nonexistent")
        assert not_found is None

    def test_find_task_by_agent_id(self):
        """Test find_task_by_agent_id function."""
        identity = TeammateIdentity(
            agent_id="worker@test-team",
            agent_name="worker",
            team_name="test-team",
            parent_session_id="session",
        )

        task = create_in_process_teammate_state("task-4", identity, "prompt")
        register_task(task)

        found = find_task_by_agent_id("worker@test-team")
        assert found is not None
        assert found.id == "task-4"

        not_found = find_task_by_agent_id("unknown@team")
        assert not_found is None


@pytest.mark.asyncio
class TestSpawnInProcessTeammateV2:
    """Test spawn_in_process_teammate_v2 function."""

    async def test_spawn_in_process_teammate(self):
        """Test spawning an in-process teammate."""
        mock_set = MagicMock()
        mock_get = MagicMock(return_value={"tasks": {}})

        config = SpawnTeammateConfig(
            name="worker",
            team_name="test-team",
            prompt="Test task",
        )

        # Patch start_in_process_teammate where it's used (inside manager.py imports it from in_process_runner)
        with patch("claude_code_py.utils.swarm.in_process_runner.start_in_process_teammate"):
            result = await spawn_in_process_teammate_v2(config, mock_set, mock_get)

            assert result.success
            assert result.agent_id == "worker@test-team"
            assert result.task_id is not None
            assert result.error is None

    async def test_spawn_multiple_teammates(self):
        """Test spawning multiple teammates."""
        mock_set = MagicMock()
        mock_get = MagicMock(return_value={"tasks": {}})

        configs = [
            SpawnTeammateConfig(name="w1", team_name="team", prompt="T1"),
            SpawnTeammateConfig(name="w2", team_name="team", prompt="T2"),
        ]

        results = []
        with patch("claude_code_py.utils.swarm.in_process_runner.start_in_process_teammate"):
            for c in configs:
                results.append(await spawn_in_process_teammate_v2(c, mock_set, mock_get))

        assert all(r.success for r in results)
        assert results[0].agent_id != results[1].agent_id
        assert results[0].agent_id == "w1@team"
        assert results[1].agent_id == "w2@team"

    async def test_spawn_with_no_callbacks(self):
        """Test spawning without AppState callbacks (uses global registry)."""
        config = SpawnTeammateConfig(
            name="test_worker",
            team_name="test-team",
            prompt="Do something",
        )

        with patch("claude_code_py.utils.swarm.in_process_runner.start_in_process_teammate"):
            result = await spawn_in_process_teammate_v2(config)

            assert result.success
            assert result.agent_id == "test_worker@test-team"

            # Should be registered in global registry
            found = find_task_by_agent_id("test_worker@test-team")
            assert found is not None


class TestCreateInProcessTeammateState:
    """Test create_in_process_teammate_state function."""

    def test_create_state_basic(self, temp_config_home):
        """Test basic state creation."""
        identity = TeammateIdentity(
            agent_id="agent@team",
            agent_name="agent",
            team_name="team",
            parent_session_id="parent",
        )

        state = create_in_process_teammate_state(
            task_id="test-task",
            identity=identity,
            prompt="Test prompt",
        )

        assert state.id == "test-task"
        assert state.identity.agent_id == "agent@team"
        assert state.prompt == "Test prompt"
        assert state.status.name == "RUNNING"

    def test_create_state_with_model(self, temp_config_home):
        """Test state creation with model override."""
        identity = TeammateIdentity(
            agent_id="agent@team",
            agent_name="agent",
            team_name="team",
            parent_session_id="parent",
        )

        state = create_in_process_teammate_state(
            task_id="test-task-2",
            identity=identity,
            prompt="Prompt",
            model="claude-3-opus",
        )

        assert state.model == "claude-3-opus"


class TestSpawnTeammateResult:
    """Test SpawnTeammateResult."""

    def test_success_result(self):
        """Test successful spawn result."""
        result = SpawnTeammateResult(
            success=True,
            agent_id="worker@team",
            task_id="task-123",
        )
        assert result.success
        assert result.error is None

    def test_failure_result(self):
        """Test failed spawn result."""
        result = SpawnTeammateResult(
            success=False,
            agent_id="worker@team",
            error="Failed to spawn",
        )
        assert not result.success
        assert result.error == "Failed to spawn"
        assert result.task_id is None