"""EscapeInput: terminal input with Escape/Ctrl-C detection during execution.

Two distinct phases:
  1. Waiting for user input  → cooked mode (readline, echo, line discipline)
  2. Claude executing        → raw mode  (byte-by-byte, catch Escape / Ctrl-C)

Never overlap these two modes.
"""

from __future__ import annotations

import asyncio
import sys
import termios
import tty
from typing import Callable, Optional
import select
from rich.console import Console
import os

class EscapeInput:
    def __init__(self, console: Optional[Console] = None) -> None:
        import readline  # activates readline editing for input(), side-effect import
        self.console = console or Console()

        # Raw-mode listener state (execution phase only)
        self._listener_active: bool = False
        self._listener_fd: Optional[int] = None
        self._saved_settings: Optional[list] = None

        self._on_escape: Optional[Callable] = None
        self._on_ctrl_c: Optional[Callable] = None

        # prompt_toolkit session – set later by REPL.start() if needed
        self._session = None

    # ------------------------------------------------------------------
    # Phase 1: get user input (cooked mode, readline works normally)
    # ------------------------------------------------------------------

    async def input_async(
        self,
        prompt: str = "> ",
        on_escape: Optional[Callable] = None,
        on_ctrl_c: Optional[Callable] = None,
        on_shift_up: Optional[Callable] = None,
        on_shift_down: Optional[Callable] = None,
    ) -> tuple[str, bool]:
        """Read a line from the user in cooked mode.

        Returns (text, was_aborted).  was_aborted is True only on EOF.
        Escape / Ctrl-C during *input* are handled by the OS / readline;
        they are NOT intercepted here (that would require raw mode which
        breaks readline).
        """
        # Guarantee cooked mode while we are waiting for user input.
        # If a previous execution listener left the terminal in raw mode
        # (e.g. due to an exception), fix it now.
        self._ensure_cooked()

        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(None, input, prompt)
            return text, False
        except EOFError:
            return "", True
        except KeyboardInterrupt:
            # Ctrl-C during input: surface to REPL loop as abort
            return "", True

    # ------------------------------------------------------------------
    # Phase 2: execution listener (raw mode, Escape / Ctrl-C detection)
    # ------------------------------------------------------------------

    def start_execution_listener(
        self,
        on_escape: Optional[Callable] = None,
        on_ctrl_c: Optional[Callable] = None,
    ) -> None:
        """Switch terminal to raw mode and start listening for special keys.

        Call this AFTER input_async returns and BEFORE blocking on Claude.
        """
        if not sys.stdin.isatty():
            return
        if self._listener_active:
            return  # already running

        self._on_escape = on_escape
        self._on_ctrl_c = on_ctrl_c

        fd = sys.stdin.fileno()
        self._listener_fd = fd
        self._saved_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        # setraw() disables output post-processing (OPOST), which prevents
        # \n → \r\n translation. Re-enable it so text output isn't garbled
        # with stair-stepped indentation during execution.
        self._enable_output_processing(fd)

        loop = asyncio.get_event_loop()
        loop.add_reader(fd, self._on_stdin_ready)
        self._listener_active = True

    def stop_execution_listener(self) -> None:
        """Remove raw-mode listener and restore cooked mode.

        Safe to call even if listener was never started.
        """
        if not self._listener_active:
            return
        self._listener_active = False

        loop = asyncio.get_event_loop()
        try:
            loop.remove_reader(self._listener_fd)
        except Exception:
            pass

        self._restore_saved_settings()
        self._listener_fd = None
        self._saved_settings = None

    # ------------------------------------------------------------------
    # suspend / restore (used by async_input in main.py for permission
    # prompts that fire *during* execution)
    # ------------------------------------------------------------------

    def suspend_listener(self) -> None:
        """Temporarily pause raw-mode listener so a blocking input() can run."""
        if self._listener_active:
            loop = asyncio.get_event_loop()
            try:
                loop.remove_reader(self._listener_fd)
            except Exception:
                pass
        # Restore cooked so the nested input() works with readline.
        # Always attempt restoration even if listener_active is False —
        # the terminal may still be raw due to an exception or edge case.
        self._restore_saved_settings()

    # Aliases used by async_input in main.py
    # pause_listener = suspend_listener

    def restore_listener(self) -> None:
        """Resume raw-mode listener after nested input() returns."""
        if not self._listener_active:
            return
        if self._listener_fd is None:
            return
        # Re-acquire fresh settings before going raw again.
        self._saved_settings = termios.tcgetattr(self._listener_fd)
        tty.setraw(self._listener_fd)
        self._enable_output_processing(self._listener_fd)
        loop = asyncio.get_event_loop()
        loop.add_reader(self._listener_fd, self._on_stdin_ready)

    # Alias used by async_input in main.py
    # resume_listener = restore_listener

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enable_output_processing(fd: int) -> None:
        """Re-enable OPOST after setraw() so \\n is translated to \\r\\n."""
        try:
            iflag, oflag, cflag, lflag, ispeed, ospeed, cc = termios.tcgetattr(fd)
            oflag |= termios.OPOST
            termios.tcsetattr(fd, termios.TCSADRAIN, [iflag, oflag, cflag, lflag, ispeed, ospeed, cc])
        except Exception:
            pass

    def _restore_saved_settings(self) -> None:
        if self._listener_fd is not None and self._saved_settings is not None:
            try:
                termios.tcsetattr(
                    self._listener_fd, termios.TCSADRAIN, self._saved_settings
                )
            except Exception:
                pass

    def _ensure_cooked(self) -> None:
        """If the terminal is somehow still in raw mode, fix it."""
        # Always attempt to restore cooked settings, even if listener_active
        # is False. The terminal may be raw due to an exception or edge case.
        self._restore_saved_settings()
        if self._listener_active:
            self.stop_execution_listener()



    def _on_stdin_ready(self) -> None:
        try:
            ch = os.read(self._listener_fd, 1)  # 直接读 fd，不经过 buffer 层
        except Exception:
            return

        if not ch:
            return

        if ch == b'\x1b':
            # 非阻塞排空后续 CSI 字节
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if not r:
                    break
                try:
                    os.read(self._listener_fd, 1)  # 同样用 os.read
                except Exception:
                    break
            if self._on_escape:
                self._on_escape()

        elif ch in (b'\x03', b'\x1a'):
            if self._on_ctrl_c:
                self._on_ctrl_c()