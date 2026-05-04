"""Async helpers for reliable task execution.

Solves the "create_task doesn't immediately execute" problem by providing
utilities that ensure background tasks start running promptly.

Key concept: Background Event Loop
- A dedicated thread with its own continuously-running event loop
- Tasks submitted via run_coroutine_threadsafe() execute immediately
- Independent of main REPL's event loop (which pauses while waiting for input)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Coroutine, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Background Event Loop - For non-blocking task execution
# =============================================================================

# Global background event loop (singleton)
_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BACKGROUND_THREAD: Optional[threading.Thread] = None
_BACKGROUND_LOOP_LOCK = threading.Lock()


def get_background_loop() -> asyncio.AbstractEventLoop:
    """Get or create the global background event loop.

    This provides a continuously-running event loop in a dedicated thread,
    allowing tasks to execute immediately without blocking on user input.

    Returns:
        The background event loop (always running)
    """
    global _BACKGROUND_LOOP, _BACKGROUND_THREAD

    with _BACKGROUND_LOOP_LOCK:
        if _BACKGROUND_LOOP is None or not _BACKGROUND_LOOP.is_running():
            # Create new background loop
            _BACKGROUND_LOOP = asyncio.new_event_loop()

            def run_loop():
                """Run the background event loop forever."""
                asyncio.set_event_loop(_BACKGROUND_LOOP)
                _BACKGROUND_LOOP.run_forever()

            _BACKGROUND_THREAD = threading.Thread(
                target=run_loop,
                name="background-event-loop",
                daemon=True,
            )
            _BACKGROUND_THREAD.start()

            # Wait for loop to start running
            while not _BACKGROUND_LOOP.is_running():
                pass  # Spin briefly (loop starts quickly)

            logger.debug("Background event loop started")

        return _BACKGROUND_LOOP


def stop_background_loop() -> None:
    """Stop the background event loop (for cleanup)."""
    global _BACKGROUND_LOOP, _BACKGROUND_THREAD

    with _BACKGROUND_LOOP_LOCK:
        if _BACKGROUND_LOOP and _BACKGROUND_LOOP.is_running():
            _BACKGROUND_LOOP.call_soon_threadsafe(_BACKGROUND_LOOP.stop)
            if _BACKGROUND_THREAD:
                _BACKGROUND_THREAD.join(timeout=2.0)
            _BACKGROUND_LOOP = None
            _BACKGROUND_THREAD = None
            logger.debug("Background event loop stopped")


async def create_task_with_yield(
    coro: Coroutine[Any, Any, Any],
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> asyncio.Task:
    """Create asyncio task and yield to ensure it starts.

    This solves the "task not immediately executing" problem by yielding
    control after create_task, giving the event loop a chance to schedule
    the task before returning.

    Args:
        coro: Coroutine to run as background task
        loop: Optional event loop (uses get_running_loop if None)

    Returns:
        asyncio.Task that has been given chance to start

    Example:
        # Old way (task may not start immediately)
        task = asyncio.create_task(my_coro())

        # New way (task starts before return)
        task = await create_task_with_yield(my_coro())
    """
    if loop is None:
        loop = asyncio.get_running_loop()

    task = loop.create_task(coro)

    # Yield control to let the event loop schedule the task
    await asyncio.sleep(0)

    return task


def create_task_in_thread(
    coro: Coroutine[Any, Any, Any],
    name: Optional[str] = None,
) -> threading.Thread:
    """Run coroutine in a dedicated thread with its own event loop.

    This ensures the coroutine starts immediately, independent of the
    main event loop's state. Used when:
    - We can't yield (non-async context)
    - We need guaranteed immediate execution
    - The task should run independently

    Args:
        coro: Coroutine to run
        name: Optional thread name (for debugging)

    Returns:
        Thread object (daemon=True, auto-exits when main thread exits)

    Example:
        thread = create_task_in_thread(my_coro(), name="worker-1")
        # Thread is already running
    """
    def run_in_thread():
        thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_loop)
        try:
            thread_loop.run_until_complete(coro)
        except Exception:
            pass  # Errors should be handled inside coroutine
        finally:
            thread_loop.close()

    thread_name = name or f"async-helper-{id(coro)}"
    thread = threading.Thread(
        target=run_in_thread,
        name=thread_name,
        daemon=True,
    )
    thread.start()

    return thread


async def ensure_task_running(
    task: asyncio.Task,
    max_wait_ms: int = 100,
) -> bool:
    """Wait for a task to start running (transition from pending).

    Useful when you need to confirm a task has begun execution before
    proceeding. Tasks start in "pending" state and transition to "running"
    when the event loop picks them up.

    Args:
        task: asyncio.Task to check
        max_wait_ms: Maximum milliseconds to wait

    Returns:
        True if task started running, False if still pending or done
    """
    # Check if already running or done
    if task.done():
        return False

    # Wait a bit to give task chance to start
    # Note: asyncio.Task has no "running" state exposed, so we just
    # yield and assume it started if not done
    await asyncio.sleep(max_wait_ms / 1000.0)

    return not task.done()


class BackgroundTaskManager:
    """Manager for reliable background task execution.

    Provides a dedicated event loop running in a background thread,
    ensuring tasks can be submitted and will execute immediately
    regardless of the main loop's state.

    Example:
        manager = BackgroundTaskManager()
        manager.start()

        # Submit task (runs immediately)
        future = manager.submit(my_coro())

        # Later, check result
        if future.done():
            result = future.result()

        manager.stop()
    """

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the background event loop thread."""
        if self._running:
            return

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(
            target=run_loop,
            name="background-task-manager",
            daemon=True,
        )
        self._thread.start()

        # Wait for loop to be created
        while self._loop is None:
            pass  # Spin briefly (loop created quickly)

        self._running = True

    def stop(self) -> None:
        """Stop the background event loop."""
        if not self._running or self._loop is None:
            return

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._running = False

    def submit(
        self,
        coro: Coroutine[Any, Any, Any],
    ) -> asyncio.Future:
        """Submit a coroutine to run in the background loop.

        Args:
            coro: Coroutine to execute

        Returns:
            Future that can be used to check/result

        Raises:
            RuntimeError: If manager not started
        """
        if not self._running or self._loop is None:
            raise RuntimeError("BackgroundTaskManager not started")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future

    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running


__all__ = [
    "get_background_loop",
    "stop_background_loop",
    "create_task_with_yield",
    "create_task_in_thread",
    "ensure_task_running",
    "BackgroundTaskManager",
]