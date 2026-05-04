"""Tests for Task V2 file storage system.

Tests:
- File locking mechanism
- High water mark to prevent ID reuse
- Task CRUD operations
- Atomic task claiming
- Agent status tracking
- Teammate task unassignment
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest import TestCase, main
from concurrent.futures import ThreadPoolExecutor, as_completed

from claude_code_py.utils.task.file_storage import (
    TaskStatus,
    Task,
    FileLock,
    ClaimTaskResult,
    AgentStatus,
    UnassignTasksResult,
    sanitize_path_component,
    get_tasks_dir,
    get_task_path,
    ensure_tasks_dir,
    read_high_water_mark,
    write_high_water_mark,
    find_highest_task_id,
    find_highest_task_id_from_files,
    create_task,
    get_task,
    update_task,
    delete_task,
    list_tasks,
    reset_task_list,
    block_task,
    claim_task,
    get_agent_statuses,
    unassign_teammate_tasks,
    task_to_dict,
    dict_to_task,
    set_leader_team_name,
    clear_leader_team_name,
    notify_tasks_updated,
    on_tasks_updated,
)


class TestFileLock(TestCase):
    """Test cross-process file locking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.lock_path = Path(self.temp_dir) / ".lock"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_lock_acquire_release(self):
        """Test basic lock acquire and release."""
        lock = FileLock(self.lock_path)
        self.assertTrue(lock.acquire())
        lock.release()

    def test_lock_context_manager(self):
        """Test lock as context manager."""
        with FileLock(self.lock_path):
            # Lock is held
            pass
        # Lock is released

    def test_lock_timeout(self):
        """Test lock timeout when held by another thread."""
        lock1 = FileLock(self.lock_path, timeout_ms=500)
        lock2 = FileLock(self.lock_path, timeout_ms=500, retries=5)

        lock1.acquire()

        # Thread 2 should timeout
        result = lock2.acquire()
        self.assertFalse(result)

        lock1.release()


class TestHighWaterMark(TestCase):
    """Test high water mark to prevent ID reuse."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task_list_id = "test_tasks"
        # Override tasks dir
        os.environ["CLAUDE_CONFIG_HOME"] = self.temp_dir

    def tearDown(self):
        import shutil
        del os.environ["CLAUDE_CONFIG_HOME"]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_write_high_water_mark(self):
        """Test reading and writing high water mark."""
        ensure_tasks_dir(self.task_list_id)

        write_high_water_mark(self.task_list_id, 5)
        self.assertEqual(read_high_water_mark(self.task_list_id), 5)

        write_high_water_mark(self.task_list_id, 10)
        self.assertEqual(read_high_water_mark(self.task_list_id), 10)

    def test_high_water_mark_default(self):
        """Test default high water mark is 0."""
        self.assertEqual(read_high_water_mark(self.task_list_id), 0)

    def test_find_highest_task_id(self):
        """Test finding highest task ID considers both files and mark."""
        ensure_tasks_dir(self.task_list_id)

        # Create tasks 1, 2, 3
        for i in [1, 2, 3]:
            task = Task(id=str(i), subject=f"Task {i}", description="", status=TaskStatus.PENDING)
            path = get_task_path(self.task_list_id, str(i))
            path.write_text(json.dumps(task_to_dict(task)))

        # Set high water mark to 5
        write_high_water_mark(self.task_list_id, 5)

        # Highest should be 5 (from mark)
        self.assertEqual(find_highest_task_id(self.task_list_id), 5)

    def test_delete_updates_high_water_mark(self):
        """Test deleting a task updates high water mark."""
        ensure_tasks_dir(self.task_list_id)

        # Create task
        task_id = create_task(self.task_list_id, "Test", "Description")

        # Delete task
        delete_task(self.task_list_id, task_id)

        # High water mark should be updated
        self.assertEqual(read_high_water_mark(self.task_list_id), int(task_id))

        # Next task ID should be higher
        next_id = create_task(self.task_list_id, "Next", "Description")
        self.assertEqual(int(next_id), int(task_id) + 1)


class TestTaskCRUD(TestCase):
    """Test Task CRUD operations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task_list_id = "test_tasks"
        os.environ["CLAUDE_CONFIG_HOME"] = self.temp_dir

    def tearDown(self):
        import shutil
        del os.environ["CLAUDE_CONFIG_HOME"]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_task(self):
        """Test creating a task."""
        task_id = create_task(
            self.task_list_id,
            subject="Test Task",
            description="Test description",
            activeForm="Testing",
        )

        self.assertIsNotNone(task_id)
        self.assertEqual(task_id, "1")  # First task

        task = get_task(self.task_list_id, task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.subject, "Test Task")
        self.assertEqual(task.status, TaskStatus.PENDING)

    def test_update_task(self):
        """Test updating a task."""
        task_id = create_task(self.task_list_id, "Original", "Desc")

        updated = update_task(
            self.task_list_id,
            task_id,
            subject="Updated",
            status=TaskStatus.IN_PROGRESS,
        )

        self.assertIsNotNone(updated)
        self.assertEqual(updated.subject, "Updated")
        self.assertEqual(updated.status, TaskStatus.IN_PROGRESS)

    def test_delete_task(self):
        """Test deleting a task."""
        task_id = create_task(self.task_list_id, "To Delete", "Desc")

        result = delete_task(self.task_list_id, task_id)
        self.assertTrue(result)

        task = get_task(self.task_list_id, task_id)
        self.assertIsNone(task)

    def test_list_tasks(self):
        """Test listing all tasks."""
        create_task(self.task_list_id, "Task 1", "Desc")
        create_task(self.task_list_id, "Task 2", "Desc")
        create_task(self.task_list_id, "Task 3", "Desc")

        tasks = list_tasks(self.task_list_id)
        self.assertEqual(len(tasks), 3)

    def test_reset_task_list(self):
        """Test resetting a task list."""
        create_task(self.task_list_id, "Task 1", "Desc")
        create_task(self.task_list_id, "Task 2", "Desc")

        reset_task_list(self.task_list_id)

        tasks = list_tasks(self.task_list_id)
        self.assertEqual(len(tasks), 0)

        # High water mark should prevent ID reuse
        next_id = create_task(self.task_list_id, "New", "Desc")
        self.assertEqual(int(next_id), 3)  # After 1, 2


class TestTaskBlocking(TestCase):
    """Test task blocking relationships."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task_list_id = "test_tasks"
        os.environ["CLAUDE_CONFIG_HOME"] = self.temp_dir

    def tearDown(self):
        import shutil
        del os.environ["CLAUDE_CONFIG_HOME"]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_block_task(self):
        """Test setting up blocking relationship."""
        task_a = create_task(self.task_list_id, "Task A", "Blocks B")
        task_b = create_task(self.task_list_id, "Task B", "Blocked by A")

        result = block_task(self.task_list_id, task_a, task_b)
        self.assertTrue(result)

        a = get_task(self.task_list_id, task_a)
        b = get_task(self.task_list_id, task_b)

        self.assertIn(task_b, a.blocks)
        self.assertIn(task_a, b.blockedBy)


class TestClaimTask(TestCase):
    """Test atomic task claiming."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task_list_id = "test_tasks"
        os.environ["CLAUDE_CONFIG_HOME"] = self.temp_dir

    def tearDown(self):
        import shutil
        del os.environ["CLAUDE_CONFIG_HOME"]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_claim_available_task(self):
        """Test claiming an available task."""
        task_id = create_task(self.task_list_id, "Available", "Desc")

        result = claim_task(self.task_list_id, task_id, "agent-1")

        self.assertTrue(result.success)
        self.assertIsNotNone(result.task)
        self.assertEqual(result.task.owner, "agent-1")

    def test_claim_already_claimed(self):
        """Test claiming an already claimed task."""
        task_id = create_task(self.task_list_id, "Claimed", "Desc")

        # First claim succeeds
        result1 = claim_task(self.task_list_id, task_id, "agent-1")
        self.assertTrue(result1.success)

        # Second claim fails
        result2 = claim_task(self.task_list_id, task_id, "agent-2")
        self.assertFalse(result2.success)
        self.assertEqual(result2.reason, "already_claimed")

    def test_claim_blocked_task(self):
        """Test claiming a blocked task."""
        blocker_id = create_task(self.task_list_id, "Blocker", "Desc")
        blocked_id = create_task(self.task_list_id, "Blocked", "Desc")
        block_task(self.task_list_id, blocker_id, blocked_id)

        result = claim_task(self.task_list_id, blocked_id, "agent-1")

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "blocked")
        self.assertIn(blocker_id, result.blocked_by_tasks)

    def test_claim_with_busy_check(self):
        """Test claiming with agent busy check."""
        task1_id = create_task(self.task_list_id, "Task 1", "Desc")
        task2_id = create_task(self.task_list_id, "Task 2", "Desc")

        # Claim task 1
        result1 = claim_task(self.task_list_id, task1_id, "agent-1")
        self.assertTrue(result1.success)

        # Update task 1 to in_progress (so agent is "busy")
        update_task(self.task_list_id, task1_id, status=TaskStatus.IN_PROGRESS)

        # Try to claim task 2 with busy check
        result2 = claim_task(self.task_list_id, task2_id, "agent-1", check_agent_busy=True)

        self.assertFalse(result2.success)
        self.assertEqual(result2.reason, "agent_busy")
        self.assertIn(task1_id, result2.busy_with_tasks)


class TestAgentStatuses(TestCase):
    """Test agent status tracking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["CLAUDE_CONFIG_HOME"] = self.temp_dir

    def tearDown(self):
        import shutil
        del os.environ["CLAUDE_CONFIG_HOME"]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_agent_statuses(self):
        """Test getting agent statuses based on task ownership."""
        # Create a mock team config
        team_name = "test-team"
        teams_dir = Path(self.temp_dir) / "teams" / sanitize_path_component(team_name)
        teams_dir.mkdir(parents=True, exist_ok=True)

        team_config = {
            "leadAgentId": "lead@test-team",
            "members": [
                {"agentId": "agent-1@test-team", "name": "agent-1"},
                {"agentId": "agent-2@test-team", "name": "agent-2"},
            ]
        }
        (teams_dir / "config.json").write_text(json.dumps(team_config))

        # Create tasks for the team
        task_list_id = sanitize_path_component(team_name)
        task1_id = create_task(task_list_id, "Task 1", "Desc", owner="agent-1@test-team")
        task2_id = create_task(task_list_id, "Task 2", "Desc", owner="agent-1@test-team")
        create_task(task_list_id, "Task 3", "Desc")  # No owner

        # Get statuses
        statuses = get_agent_statuses(team_name)

        self.assertIsNotNone(statuses)
        self.assertEqual(len(statuses), 2)

        # Agent 1 should be busy with 2 tasks
        agent1_status = next(s for s in statuses if s.name == "agent-1")
        self.assertEqual(agent1_status.status, "busy")
        self.assertEqual(len(agent1_status.current_tasks), 2)

        # Agent 2 should be idle
        agent2_status = next(s for s in statuses if s.name == "agent-2")
        self.assertEqual(agent2_status.status, "idle")


class TestUnassignTeammateTasks(TestCase):
    """Test unassigning tasks from a teammate."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["CLAUDE_CONFIG_HOME"] = self.temp_dir

    def tearDown(self):
        import shutil
        del os.environ["CLAUDE_CONFIG_HOME"]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unassign_teammate_tasks(self):
        """Test unassigning tasks when teammate shuts down."""
        team_name = "test-team"
        task_list_id = sanitize_path_component(team_name)

        # Create tasks owned by teammate
        task1_id = create_task(task_list_id, "Task 1", "Desc", owner="teammate-1")
        task2_id = create_task(task_list_id, "Task 2", "Desc", owner="teammate-1")
        task3_id = create_task(task_list_id, "Task 3", "Desc", owner="teammate-1", status=TaskStatus.COMPLETED)
        create_task(task_list_id, "Task 4", "Desc", owner="other-agent")

        # Update tasks to in_progress
        update_task(task_list_id, task1_id, status=TaskStatus.IN_PROGRESS)
        update_task(task_list_id, task2_id, status=TaskStatus.IN_PROGRESS)

        # Unassign teammate tasks
        result = unassign_teammate_tasks(
            team_name,
            "teammate-1",
            "teammate-1",
            reason="shutdown",
        )

        self.assertEqual(len(result.unassigned_tasks), 2)  # Only unresolved tasks
        self.assertIn("shutdown", result.notification_message)

        # Tasks should be reset to pending with no owner
        task1 = get_task(task_list_id, task1_id)
        task2 = get_task(task_list_id, task2_id)

        self.assertIsNone(task1.owner)
        self.assertEqual(task1.status, TaskStatus.PENDING)
        self.assertIsNone(task2.owner)
        self.assertEqual(task2.status, TaskStatus.PENDING)

        # Completed task should remain unchanged
        task3 = get_task(task_list_id, task3_id)
        self.assertEqual(task3.status, TaskStatus.COMPLETED)

        # Other agent's task should remain unchanged
        tasks = list_tasks(task_list_id)
        other_task = next(t for t in tasks if t.owner == "other-agent")
        self.assertIsNotNone(other_task)


class TestConcurrency(TestCase):
    """Test concurrent task operations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task_list_id = "concurrent_test"
        os.environ["CLAUDE_CONFIG_HOME"] = self.temp_dir

    def tearDown(self):
        import shutil
        del os.environ["CLAUDE_CONFIG_HOME"]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_task_creation(self):
        """Test creating tasks concurrently doesn't cause ID conflicts."""
        num_tasks = 10
        num_threads = 5

        created_ids = []

        def create_and_record(i):
            task_id = create_task(
                self.task_list_id,
                f"Concurrent Task {i}",
                "Description",
            )
            return task_id

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_and_record, i) for i in range(num_tasks)]
            for future in as_completed(futures):
                created_ids.append(future.result())

        # All IDs should be unique
        self.assertEqual(len(created_ids), num_tasks)
        self.assertEqual(len(set(created_ids)), num_tasks)

        # IDs should be sequential integers
        int_ids = sorted(int(id) for id in created_ids)
        self.assertEqual(int_ids, list(range(1, num_tasks + 1)))

    def test_concurrent_claim(self):
        """Test concurrent claiming of same task."""
        task_id = create_task(self.task_list_id, "Race Task", "Desc")

        results = []

        def claim_task_thread(agent_id):
            result = claim_task(self.task_list_id, task_id, agent_id)
            return (agent_id, result.success)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(claim_task_thread, f"agent-{i}") for i in range(5)]
            for future in as_completed(futures):
                results.append(future.result())

        # Only one should succeed
        successful_claims = [r for r in results if r[1]]
        self.assertEqual(len(successful_claims), 1)


class TestUpdateNotifications(TestCase):
    """Test task update notifications."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task_list_id = "test_tasks"
        os.environ["CLAUDE_CONFIG_HOME"] = self.temp_dir
        self.notifications = []

    def tearDown(self):
        import shutil
        del os.environ["CLAUDE_CONFIG_HOME"]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_on_tasks_updated(self):
        """Test notification callback is called on updates."""
        def callback():
            self.notifications.append(time.time())

        unsubscribe = on_tasks_updated(callback)

        create_task(self.task_list_id, "Task 1", "Desc")
        self.assertEqual(len(self.notifications), 1)

        unsubscribe()

        create_task(self.task_list_id, "Task 2", "Desc")
        # Should not receive notification after unsubscribe
        self.assertEqual(len(self.notifications), 1)


if __name__ == "__main__":
    main()