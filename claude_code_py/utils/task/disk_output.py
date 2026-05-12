"""Task output file path utilities and writer.

Provides utilities for writing task outputs to disk files,
used by background shell tasks, remote agents, etc.
"""

from __future__ import annotations

import os
import json
import threading
from pathlib import Path
from typing import Any, Optional, Callable, TextIO


def get_task_output_path(task_id: str) -> str:
    """Get the output file path for a task.

    Args:
        task_id: Task ID

    Returns:
        Path to task output file
    """
    base_dir = Path(os.environ.get("CLAUDE_CODE_TASK_DIR", "/tmp/claude-code-tasks"))
    base_dir.mkdir(parents=True, exist_ok=True)

    return str(base_dir / f"{task_id}.log")


class TaskOutputWriter:
    """Writer for task output files.

    Handles:
    - Streaming writes to disk
    - Line-by-line output
    - JSON structured output
    - Thread-safe writing
    """

    def __init__(
        self,
        task_id: str,
        format: str = "text",  # "text" or "json"
        on_write: Optional[Callable[[str], None]] = None,
    ):
        """Initialize output writer.

        Args:
            task_id: Task ID
            format: Output format ("text" or "json")
            on_write: Optional callback for each write
        """
        self.task_id = task_id
        self.format = format
        self.on_write = on_write
        self.path = Path(get_task_output_path(task_id))
        self._file: Optional[TextIO] = None
        self._lock = threading.Lock()
        self._offset = 0

    def open(self) -> None:
        """Open the output file for writing."""
        with self._lock:
            if self._file is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        """Close the output file."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def write(self, content: str) -> int:
        """Write content to the output file.

        Args:
            content: Content to write

        Returns:
            Number of bytes written
        """
        with self._lock:
            if self._file is None:
                self.open()

            if self.format == "json":
                # Write as JSON line
                line = json.dumps({
                    "timestamp": _get_timestamp(),
                    "content": content,
                })
                self._file.write(line + "\n")
            else:
                self._file.write(content)

            self._file.flush()
            written = len(content)
            self._offset += written

            if self.on_write:
                self.on_write(content)

            return written

    def write_line(self, line: str) -> int:
        """Write a line to the output file.

        Args:
            line: Line to write (without newline)

        Returns:
            Number of bytes written
        """
        return self.write(line + "\n")

    def get_offset(self) -> int:
        """Get current write offset.

        Returns:
            Current offset in bytes
        """
        return self._offset

    def read_all(self) -> str:
        """Read all content from the output file.

        Returns:
            All content as string
        """
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def read_from(self, offset: int) -> str:
        """Read content from offset to end.

        Args:
            offset: Starting offset

        Returns:
            Content from offset
        """
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                f.seek(offset)
                return f.read()
        except FileNotFoundError:
            return ""

    def __enter__(self) -> "TaskOutputWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def _get_timestamp() -> str:
    """Get ISO format timestamp."""
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"


__all__ = [
    "get_task_output_path",
    "TaskOutputWriter",
]