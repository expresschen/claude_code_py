"""Tests for teammate_mailbox.py"""
import json
import pytest

from claude_code_py.utils.teammate_mailbox import (
    TeammateMessage,
    IdleNotificationMessage,
    PermissionRequestMessage,
    PermissionResponseMessage,
    ShutdownRequestMessage,
    ShutdownApprovedMessage,
    ShutdownRejectedMessage,
    TaskAssignmentMessage,
    create_idle_notification,
    is_idle_notification,
    is_permission_request,
    is_permission_response,
    is_shutdown_request,
    is_shutdown_approved,
    is_shutdown_rejected,
    is_task_assignment,
    is_structured_protocol_message,
    format_teammate_messages,
)


class TestMessageTypes:
    """Test message dataclasses."""

    def test_teammate_message(self):
        """TeammateMessage should work correctly."""
        msg = TeammateMessage(
            from_agent="worker",
            text="Hello leader",
            timestamp="2026-04-27T10:00:00",
            read=False,
            color="blue",
            summary="greeting",
        )

        assert msg.from_agent == "worker"
        assert msg.text == "Hello leader"
        assert msg.read is False

    def test_idle_notification_message(self):
        """IdleNotificationMessage should work correctly."""
        msg = IdleNotificationMessage(
            type="idle_notification",
            from_agent="researcher",
            timestamp="2026-04-27T10:00:00",
            idle_reason="available",
            summary="Done searching",
        )

        assert msg.type == "idle_notification"
        assert msg.idle_reason == "available"

    def test_permission_request_message(self):
        """PermissionRequestMessage should work correctly."""
        msg = PermissionRequestMessage(
            type="permission_request",
            request_id="perm-123",
            agent_id="worker@test",
            tool_name="Bash",
            tool_use_id="tool_001",
            description="Run command",
            input={"command": "ls"},
        )

        assert msg.request_id == "perm-123"
        assert msg.tool_name == "Bash"

    def test_shutdown_request_message(self):
        """ShutdownRequestMessage should work correctly."""
        msg = ShutdownRequestMessage(
            type="shutdown_request",
            request_id="shutdown-001",
            from_agent="team-lead",
            reason="Team completed",
            timestamp="2026-04-27T10:00:00",
        )

        assert msg.type == "shutdown_request"
        assert msg.reason == "Team completed"


class TestMessageHelpers:
    """Test message helper functions."""

    def test_create_idle_notification(self):
        """Should create idle notification."""
        msg = create_idle_notification(
            agent_id="researcher@test",
            options={"idle_reason": "available", "summary": "Done"},
        )

        assert msg.type == "idle_notification"
        assert msg.from_agent == "researcher@test"
        assert msg.idle_reason == "available"

    def test_is_idle_notification(self):
        """Should detect idle notification."""
        text = json.dumps({
            "type": "idle_notification",
            "from": "worker",
            "timestamp": "2026-04-27T10:00:00",
            "idle_reason": "available",
        })

        msg = is_idle_notification(text)
        assert msg is not None
        assert msg.from_agent == "worker"

    def test_is_idle_notification_invalid(self):
        """Should return None for invalid message."""
        msg = is_idle_notification("not json")
        assert msg is None

        msg = is_idle_notification(json.dumps({"type": "other"}))
        assert msg is None

    def test_is_permission_request(self):
        """Should detect permission request."""
        text = json.dumps({
            "type": "permission_request",
            "request_id": "perm-001",
            "agent_id": "worker",
            "tool_name": "Edit",
            "tool_use_id": "tool_123",
            "description": "Edit file",
            "input": {"file": "test.py"},
        })

        msg = is_permission_request(text)
        assert msg is not None
        assert msg.tool_name == "Edit"

    def test_is_permission_response(self):
        """Should detect permission response."""
        text = json.dumps({
            "type": "permission_response",
            "request_id": "perm-001",
            "subtype": "success",
            "response": {"updated_input": {}},
        })

        msg = is_permission_response(text)
        assert msg is not None
        assert msg.subtype == "success"

    def test_is_shutdown_request(self):
        """Should detect shutdown request."""
        text = json.dumps({
            "type": "shutdown_request",
            "requestId": "shutdown-001",
            "from": "team-lead",
            "reason": "Done",
            "timestamp": "2026-04-27T10:00:00",
        })

        msg = is_shutdown_request(text)
        assert msg is not None
        assert msg.from_agent == "team-lead"

    def test_is_shutdown_approved(self):
        """Should detect shutdown approved."""
        text = json.dumps({
            "type": "shutdown_approved",
            "requestId": "shutdown-001",
            "from": "worker",
            "timestamp": "2026-04-27T10:00:00",
        })

        msg = is_shutdown_approved(text)
        assert msg is not None

    def test_is_shutdown_rejected(self):
        """Should detect shutdown rejected."""
        text = json.dumps({
            "type": "shutdown_rejected",
            "requestId": "shutdown-001",
            "from": "worker",
            "reason": "Still working",
            "timestamp": "2026-04-27T10:00:00",
        })

        msg = is_shutdown_rejected(text)
        assert msg is not None
        assert msg.reason == "Still working"

    def test_is_task_assignment(self):
        """Should detect task assignment."""
        text = json.dumps({
            "type": "task_assignment",
            "taskId": "task-001",
            "subject": "Fix bug",
            "description": "Fix the login bug",
            "assignedBy": "team-lead",
            "timestamp": "2026-04-27T10:00:00",
        })

        msg = is_task_assignment(text)
        assert msg is not None
        assert msg.task_id == "task-001"

    def test_is_structured_protocol_message(self):
        """Should detect structured messages."""
        for msg_type in [
            "permission_request",
            "permission_response",
            "idle_notification",
            "shutdown_request",
            "shutdown_approved",
            "shutdown_rejected",
            "plan_approval_request",
            "plan_approval_response",
            "task_assignment",
        ]:
            text = json.dumps({"type": msg_type})
            assert is_structured_protocol_message(text) is True

        # Non-structured
        assert is_structured_protocol_message("plain text") is False
        assert is_structured_protocol_message(json.dumps({"other": "field"})) is False


class TestMessageFormatting:
    """Test message formatting."""

    def test_format_teammate_messages(self):
        """Should format messages as XML."""
        messages = [
            TeammateMessage(
                from_agent="worker",
                text="Hello",
                timestamp="2026-04-27T10:00:00",
                color="blue",
            ),
            TeammateMessage(
                from_agent="researcher",
                text="Found results",
                timestamp="2026-04-27T10:01:00",
                color="green",
                summary="search done",
            ),
        ]

        result = format_teammate_messages(messages)

        assert '<teammate_message teammate_id="worker" color="blue">' in result
        assert "Hello" in result
        assert '<teammate_message teammate_id="researcher"' in result
        assert "Found results" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])