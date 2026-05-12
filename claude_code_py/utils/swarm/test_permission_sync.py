"""Tests for permission_sync.py"""
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
import pytest

from claude_code_py.utils.swarm.permission_sync import (
    SwarmPermissionRequest,
    PermissionResolution,
    PermissionStatus,
    PermissionResolver,
    create_permission_request,
    create_swarm_permission_request,
    write_permission_request,
    read_pending_permissions,
    read_resolved_permission,
    resolve_permission,
    delete_resolved_permission,
    send_permission_request_via_mailbox,
    send_permission_response_via_mailbox,
    generate_request_id,
    is_swarm_worker,
    is_team_leader,
    dict_to_permission_request,
)


class TestPermissionRequestCreation:
    """Test permission request creation."""

    def test_generate_request_id(self):
        """Request ID should have correct format."""
        id1 = generate_request_id()
        id2 = generate_request_id()

        assert id1.startswith("perm-")
        assert id2.startswith("perm-")
        assert id1 != id2  # Unique IDs

    def test_create_permission_request(self):
        """Should create a valid request."""
        request = create_permission_request(
            tool_name="Bash",
            tool_use_id="tool_123",
            input={"command": "ls"},
            description="List files",
            team_name="test-team",
            worker_id="worker@test-team",
            worker_name="worker",
            worker_color="blue",
        )

        assert request.tool_name == "Bash"
        assert request.tool_use_id == "tool_123"
        assert request.input == {"command": "ls"}
        assert request.team_name == "test-team"
        assert request.worker_name == "worker"
        assert request.status == PermissionStatus.PENDING

    def test_request_to_dict(self):
        """Should serialize to dict correctly."""
        request = SwarmPermissionRequest(
            id="perm-123",
            worker_id="worker@test",
            worker_name="worker",
            team_name="test-team",
            tool_name="Edit",
            tool_use_id="tool_456",
            description="Edit file",
            input={"file": "test.py"},
            status=PermissionStatus.APPROVED,
            resolved_by=PermissionResolver.LEADER,
            created_at=1000,
        )

        data = request.to_dict()

        assert data["id"] == "perm-123"
        assert data["toolName"] == "Edit"
        assert data["status"] == "approved"
        assert data["resolvedBy"] == "leader"

    def test_dict_to_permission_request(self):
        """Should deserialize from dict correctly."""
        data = {
            "id": "perm-456",
            "workerId": "worker@test",
            "workerName": "worker",
            "teamName": "test-team",
            "toolName": "Write",
            "toolUseId": "tool_789",
            "description": "Write file",
            "input": {"path": "out.txt"},
            "status": "rejected",
            "resolvedBy": "worker",
            "feedback": "Not allowed",
            "createdAt": 2000,
        }

        request = dict_to_permission_request(data)

        assert request.id == "perm-456"
        assert request.tool_name == "Write"
        assert request.status == PermissionStatus.REJECTED
        assert request.resolved_by == PermissionResolver.WORKER
        assert request.feedback == "Not allowed"


class TestPermissionFileSync:
    """Test permission file storage."""

    @pytest.fixture
    def temp_team_dir(self, tmp_path):
        """Create a temporary team directory."""
        team_dir = tmp_path / "teams" / "test-team"
        team_dir.mkdir(parents=True)

        # Patch get_team_dir temporarily
        import claude_code_py.utils.swarm.permission_sync as ps
        original_get_team_dir = ps.get_team_dir

        def mock_get_team_dir(team_name):
            return str(team_dir)

        ps.get_team_dir = mock_get_team_dir
        yield team_dir

        ps.get_team_dir = original_get_team_dir

    @pytest.mark.asyncio
    async def test_write_and_read_pending(self, temp_team_dir):
        """Should write and read pending request."""
        request = create_swarm_permission_request(
            tool_name="Bash",
            tool_use_id="tool_001",
            input={"command": "echo test"},
            description="Echo test",
            team_name="test-team",
            worker_id="worker@test-team",
            worker_name="worker",
        )

        await write_permission_request(request)

        pending = await read_pending_permissions("test-team")

        assert len(pending) == 1
        assert pending[0].tool_name == "Bash"
        assert pending[0].id == request.id

    @pytest.mark.asyncio
    async def test_resolve_permission(self, temp_team_dir):
        """Should resolve a pending request."""
        request = create_swarm_permission_request(
            tool_name="Read",
            tool_use_id="tool_002",
            input={"file_path": "/etc/passwd"},
            description="Read passwd",
            team_name="test-team",
            worker_id="worker@test-team",
            worker_name="worker",
        )

        await write_permission_request(request)

        resolution = PermissionResolution(
            decision="approved",
            resolved_by="leader",
        )

        success = await resolve_permission(request.id, resolution, "test-team")
        assert success

        # Should no longer be pending
        pending = await read_pending_permissions("test-team")
        assert len(pending) == 0

        # Should be in resolved
        resolved = await read_resolved_permission(request.id, "test-team")
        assert resolved is not None
        assert resolved.status == PermissionStatus.APPROVED

    @pytest.mark.asyncio
    async def test_delete_resolved(self, temp_team_dir):
        """Should delete resolved permission."""
        request = create_swarm_permission_request(
            tool_name="Edit",
            tool_use_id="tool_003",
            input={"file_path": "test.py"},
            description="Edit test",
            team_name="test-team",
            worker_id="worker@test-team",
            worker_name="worker",
        )

        await write_permission_request(request)

        resolution = PermissionResolution(
            decision="rejected",
            resolved_by="worker",
            feedback="Denied",
        )

        await resolve_permission(request.id, resolution, "test-team")

        # Delete it
        success = await delete_resolved_permission(request.id, "test-team")
        assert success

        # Should be gone
        resolved = await read_resolved_permission(request.id, "test-team")
        assert resolved is None


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_swarm_worker_false_no_context(self):
        """Should return False when no context."""
        # Without setting context, should return False
        result = is_swarm_worker()
        assert result is False

    def test_is_team_leader_no_context(self):
        """Should return False when no team context."""
        result = is_team_leader()
        # Without team context, returns False (not in a team)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])