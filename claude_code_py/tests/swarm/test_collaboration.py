"""Tests for team collaboration."""

import pytest
import tempfile
import os
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from claude_code_py.utils.teammate_mailbox import (
    write_to_mailbox,
    read_mailbox,
    read_unread_messages,
    mark_messages_as_read,
    clear_mailbox,
    TeammateMessage,
    sanitize_component,
    get_inbox_path,
    get_teams_dir,
)
from claude_code_py.utils.team.team_file import (
    write_team_file,
    read_team_file,
    TeamFile,
    TeamMember,
    TeamAllowedPath,
    BackendType,
    format_agent_id,
    sanitize_team_name,
    sanitize_agent_name,
    add_member_to_team,
    remove_member_by_agent_id,
    get_team_file_path,
    ensure_team_dir,
)
from claude_code_py.utils.swarm.constants import TEAM_LEAD_NAME


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
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def use_temp_config(temp_config_home):
    """Use temp config home for all tests."""
    yield temp_config_home


@pytest.mark.asyncio
class TestMailboxOperations:
    """Tests for mailbox operations."""

    async def test_mailbox_write_read(self):
        """Test mailbox write and read."""
        team_name = "test-team-unique-1"
        inbox_name = "lead-unique-1"

        await write_to_mailbox(inbox_name, TeammateMessage(
            from_agent="worker",
            text="Hello",
            timestamp="2024-01-01T00:00:00",
            read=False,
        ), team_name)

        messages = await read_mailbox(inbox_name, team_name)
        assert len(messages) >= 1
        assert messages[-1].from_agent == "worker"
        assert messages[-1].text == "Hello"

    async def test_mailbox_multiple_messages(self):
        """Test writing multiple messages."""
        team_name = "multi-msg-team-unique"
        inbox_name = "recipient-unique"

        for i in range(3):
            await write_to_mailbox(inbox_name, TeammateMessage(
                from_agent=f"sender{i}",
                text=f"Message {i}",
                timestamp=f"2024-01-01T00:0{i}:00",
                read=False,
            ), team_name)

        messages = await read_mailbox(inbox_name, team_name)
        assert len(messages) == 3
        assert messages[0].from_agent == "sender0"
        assert messages[2].text == "Message 2"

    async def test_read_unread_messages(self):
        """Test reading only unread messages."""
        team_name = "unread-team-unique"
        inbox_name = "inbox-unique"

        # Write messages
        await write_to_mailbox(inbox_name, TeammateMessage(
            from_agent="a",
            text="First",
            timestamp="2024-01-01T00:00:00",
            read=False,
        ), team_name)

        await write_to_mailbox(inbox_name, TeammateMessage(
            from_agent="b",
            text="Second",
            timestamp="2024-01-01T00:01:00",
            read=True,
        ), team_name)

        unread = await read_unread_messages(inbox_name, team_name)
        assert len(unread) == 1
        assert unread[0].from_agent == "a"
        assert unread[0].text == "First"

    async def test_mark_messages_as_read(self):
        """Test marking messages as read."""
        team_name = "mark-read-team-unique"
        inbox_name = "target-unique"

        await write_to_mailbox(inbox_name, TeammateMessage(
            from_agent="sender",
            text="Test",
            timestamp="2024-01-01T00:00:00",
            read=False,
        ), team_name)

        await mark_messages_as_read(inbox_name, team_name)

        messages = await read_mailbox(inbox_name, team_name)
        assert len(messages) == 1
        assert messages[0].read == True

    async def test_clear_mailbox(self):
        """Test clearing mailbox."""
        team_name = "clear-team-unique"
        inbox_name = "clear_target_unique"

        await write_to_mailbox(inbox_name, TeammateMessage(
            from_agent="sender",
            text="Test",
            timestamp="2024-01-01T00:00:00",
            read=False,
        ), team_name)

        await clear_mailbox(inbox_name, team_name)

        messages = await read_mailbox(inbox_name, team_name)
        assert len(messages) == 0

    async def test_mailbox_nonexistent(self):
        """Test reading from nonexistent mailbox."""
        messages = await read_mailbox("nonexistent_agent_unique", "nonexistent_team_unique")
        assert messages == []


class TestMailboxPathHelpers:
    """Tests for mailbox path helpers."""

    def test_sanitize_component(self):
        """Test sanitize_component function."""
        assert sanitize_component("normal") == "normal"
        assert sanitize_component("with spaces") == "with-spaces"
        assert sanitize_component("special@chars!") == "special-chars-"
        assert sanitize_component("test-team") == "test-team"

    def test_get_inbox_path(self):
        """Test get_inbox_path function."""
        path = get_inbox_path("agent", "team")
        assert path.endswith("agent.json")
        assert "team" in path
        assert "inboxes" in path

    def test_get_teams_dir(self):
        """Test get_teams_dir returns valid path."""
        teams_dir = get_teams_dir()
        assert teams_dir.name == "teams"
        assert ".claude" in str(teams_dir)


class TestTeamFileOperations:
    """Tests for team file CRUD."""

    def test_team_file_write_read(self):
        """Test team file write and read."""
        team_name = "test-team-file"

        team_file = TeamFile(
            name=team_name,
            created_at=1234567890,
            lead_agent_id=f"{TEAM_LEAD_NAME}@{team_name}",
            members=[
                TeamMember(
                    agent_id=f"{TEAM_LEAD_NAME}@{team_name}",
                    name=TEAM_LEAD_NAME,
                    backend_type=BackendType.IN_PROCESS,
                    joined_at=1234567890,
                )
            ],
        )

        write_team_file(team_name, team_file)
        read = read_team_file(team_name)

        assert read is not None
        assert read.name == team_name
        assert len(read.members) == 1
        assert read.members[0].name == TEAM_LEAD_NAME

    def test_team_file_with_multiple_members(self):
        """Test team file with multiple members."""
        team_name = "multi-member-team"

        team_file = TeamFile(
            name=team_name,
            created_at=1234567890,
            lead_agent_id=f"{TEAM_LEAD_NAME}@{team_name}",
            members=[
                TeamMember(
                    agent_id=f"{TEAM_LEAD_NAME}@{team_name}",
                    name=TEAM_LEAD_NAME,
                    backend_type=BackendType.IN_PROCESS,
                    joined_at=1234567890,
                ),
                TeamMember(
                    agent_id="worker@multi-member-team",
                    name="worker",
                    backend_type=BackendType.IN_PROCESS,
                    joined_at=1234567895,
                    color="blue",
                ),
            ],
        )

        write_team_file(team_name, team_file)
        read = read_team_file(team_name)

        assert read is not None
        assert len(read.members) == 2
        assert read.members[1].color == "blue"

    def test_team_file_nonexistent(self):
        """Test reading nonexistent team file."""
        read = read_team_file("nonexistent-team")
        assert read is None

    def test_add_member_to_team(self):
        """Test adding member to team."""
        team_name = "add-member-team"

        # Create initial team
        team_file = TeamFile(
            name=team_name,
            created_at=1234567890,
            lead_agent_id=f"{TEAM_LEAD_NAME}@{team_name}",
            members=[
                TeamMember(
                    agent_id=f"{TEAM_LEAD_NAME}@{team_name}",
                    name=TEAM_LEAD_NAME,
                    backend_type=BackendType.IN_PROCESS,
                    joined_at=1234567890,
                )
            ],
        )
        write_team_file(team_name, team_file)

        # Add member
        new_member = TeamMember(
            agent_id="researcher@add-member-team",
            name="researcher",
            backend_type=BackendType.IN_PROCESS,
            joined_at=1234567895,
        )
        result = add_member_to_team(team_name, new_member)
        assert result == True

        read = read_team_file(team_name)
        assert len(read.members) == 2

    def test_remove_member_by_agent_id(self):
        """Test removing member by agent ID."""
        team_name = "remove-member-team"

        # Create team with members
        team_file = TeamFile(
            name=team_name,
            created_at=1234567890,
            lead_agent_id=f"{TEAM_LEAD_NAME}@{team_name}",
            members=[
                TeamMember(
                    agent_id=f"{TEAM_LEAD_NAME}@{team_name}",
                    name=TEAM_LEAD_NAME,
                    backend_type=BackendType.IN_PROCESS,
                    joined_at=1234567890,
                ),
                TeamMember(
                    agent_id="to_remove@remove-member-team",
                    name="to_remove",
                    backend_type=BackendType.IN_PROCESS,
                    joined_at=1234567895,
                ),
            ],
        )
        write_team_file(team_name, team_file)

        # Remove member
        result = remove_member_by_agent_id(team_name, "to_remove@remove-member-team")
        assert result == True

        read = read_team_file(team_name)
        assert len(read.members) == 1
        assert read.members[0].name == TEAM_LEAD_NAME

    def test_remove_nonexistent_member(self):
        """Test removing nonexistent member."""
        team_name = "remove-nonexistent-team"

        # Create team
        team_file = TeamFile(
            name=team_name,
            created_at=1234567890,
            lead_agent_id=f"{TEAM_LEAD_NAME}@{team_name}",
            members=[
                TeamMember(
                    agent_id=f"{TEAM_LEAD_NAME}@{team_name}",
                    name=TEAM_LEAD_NAME,
                    backend_type=BackendType.IN_PROCESS,
                    joined_at=1234567890,
                )
            ],
        )
        write_team_file(team_name, team_file)

        # Try to remove nonexistent member
        result = remove_member_by_agent_id(team_name, "nonexistent@team")
        assert result == False


class TestTeamMemberDataclass:
    """Tests for TeamMember dataclass."""

    def test_team_member_creation(self):
        """Test TeamMember creation."""
        member = TeamMember(
            agent_id="test@team",
            name="test",
            backend_type=BackendType.IN_PROCESS,
            joined_at=1234567890,
        )
        assert member.agent_id == "test@team"
        assert member.name == "test"
        assert member.backend_type == BackendType.IN_PROCESS
        assert member.is_active == True
        assert member.mode == "default"

    def test_team_member_to_dict(self):
        """Test TeamMember to_dict serialization."""
        member = TeamMember(
            agent_id="worker@team",
            name="worker",
            backend_type=BackendType.IN_PROCESS,
            joined_at=1234567890,
            color="blue",
            cwd="/home/user/project",
        )
        data = member.to_dict()
        assert data["agentId"] == "worker@team"
        assert data["name"] == "worker"
        assert data["backendType"] == "in-process"
        assert data["color"] == "blue"
        assert data["cwd"] == "/home/user/project"


class TestTeamFileDataclass:
    """Tests for TeamFile dataclass."""

    def test_team_file_creation(self):
        """Test TeamFile creation."""
        team_file = TeamFile(
            name="my-team",
            created_at=1234567890,
            lead_agent_id="lead@my-team",
        )
        assert team_file.name == "my-team"
        assert team_file.lead_agent_id == "lead@my-team"
        assert team_file.members == []
        assert team_file.hidden_pane_ids == []

    def test_team_file_to_dict(self):
        """Test TeamFile to_dict serialization."""
        team_file = TeamFile(
            name="test-team",
            created_at=1234567890,
            lead_agent_id="lead@test-team",
            members=[
                TeamMember(
                    agent_id="lead@test-team",
                    name="lead",
                    backend_type=BackendType.IN_PROCESS,
                    joined_at=1234567890,
                )
            ],
        )
        data = team_file.to_dict()
        assert data["name"] == "test-team"
        assert data["leadAgentId"] == "lead@test-team"
        assert len(data["members"]) == 1


class TestFormatAgentId:
    """Tests for format_agent_id function."""

    def test_format_agent_id_basic(self):
        """Test basic agent ID formatting."""
        agent_id = format_agent_id("worker", "test-team")
        assert agent_id == "worker@test-team"

    def test_format_agent_id_with_special_chars(self):
        """Test agent ID with special characters."""
        # sanitize_agent_name replaces @ with -
        agent_id = format_agent_id("test@agent", "my-team")
        assert agent_id == "test-agent@my-team"

    def test_sanitize_team_name(self):
        """Test sanitize_team_name function."""
        assert sanitize_team_name("My Team") == "my-team"
        assert sanitize_team_name("TestTeam") == "testteam"
        assert sanitize_team_name("test_team") == "test-team"

    def test_sanitize_agent_name(self):
        """Test sanitize_agent_name function."""
        assert sanitize_agent_name("test@agent") == "test-agent"
        assert sanitize_agent_name("normal") == "normal"


class TestBackendType:
    """Tests for BackendType enum."""

    def test_backend_types(self):
        """Test BackendType enum values."""
        assert BackendType.TMUX.value == "tmux"
        assert BackendType.ITERM2.value == "iterm2"
        assert BackendType.IN_PROCESS.value == "in-process"

    def test_backend_type_comparison(self):
        """Test BackendType comparison."""
        assert BackendType.IN_PROCESS == BackendType.IN_PROCESS
        assert BackendType.TMUX != BackendType.IN_PROCESS


class TestTeamAllowedPath:
    """Tests for TeamAllowedPath dataclass."""

    def test_team_allowed_path_creation(self):
        """Test TeamAllowedPath creation."""
        path = TeamAllowedPath(
            path="/home/user/project",
            tool_name="Edit",
            added_by="lead",
            added_at=1234567890,
        )
        assert path.path == "/home/user/project"
        assert path.tool_name == "Edit"
        assert path.added_by == "lead"


class TestPathHelpers:
    """Tests for path helper functions."""

    def test_get_team_file_path(self):
        """Test get_team_file_path function."""
        path = get_team_file_path("my-team")
        assert path.name == "config.json"
        assert "my-team" in str(path)
        assert "teams" in str(path)

    def test_ensure_team_dir(self):
        """Test ensure_team_dir creates directory."""
        team_name = "new-team"
        dir_path = ensure_team_dir(team_name)
        assert dir_path.exists()
        assert dir_path.name == "new-team"