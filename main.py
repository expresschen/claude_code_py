"""Claude Code Python CLI Entry Point.

This provides the main entry point for running Claude Code in Python.
Supports both interactive REPL mode and SDK/headless mode.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Configure logging to file only (not console) to avoid interfering with REPL
# This prevents logs from covering the input prompt
log_dir = Path.home() / ".claude" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / "claude_code.log",
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Allow direct execution: python main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Any, Optional, Callable
from dataclasses import dataclass, replace


# =============================================================================
# Async Input Helper
# =============================================================================

# Global reference to escape input handler (set by REPL class)
_escape_input_handler: Optional[Any] = None
# Global reference to status display (set by REPL class, used by async_input)
_status_display_ref: Optional[Any] = None


async def async_input(prompt: str = "") -> str:
    global _escape_input_handler, _status_display_ref
    handler = _escape_input_handler
    status_display = _status_display_ref

    if handler is not None:
        handler.suspend_listener()

    # Pause status display so Rich Live doesn't overwrite the input() prompt
    if status_display is not None:
        try:
            status_display.pause()
        except Exception:
            pass

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, input, prompt + "\n> ")
        return result

    finally:
        if status_display is not None:
            try:
                status_display.resume()
            except Exception:
                pass
        if handler is not None:
            handler.restore_listener()


async def interruptible_input(
    prompt: str,
    interrupt_event: asyncio.Event,
    check_interval: float = 0.1,
) -> tuple[str, bool]:
    """Async input that can be interrupted by an event.

    Args:
        prompt: Prompt string to display
        interrupt_event: Event that signals interruption
        check_interval: How often to check for interruption (seconds)

    Returns:
        Tuple of (user_input, was_interrupted)
        - If interrupted: ("", True)
        - If completed: (user_input, False)
    """
    loop = asyncio.get_event_loop()

    # Start input in executor
    input_future = loop.run_in_executor(None, input, prompt)

    # Poll for completion or interruption
    while not input_future.done():
        if interrupt_event.is_set():
            # Interrupt detected - need to cancel input
            # Note: input() is blocking in executor, we can't truly cancel it
            # But we can return immediately and let the caller handle it
            # The input thread will continue but its result will be ignored
            return "", True

        await asyncio.sleep(check_interval)

    # Input completed normally
    try:
        result = input_future.result()
        return result, False
    except Exception:
        return "", False


from claude_code_py import (
    QueryEngine,
    Store,
)
from claude_code_py.engine import QuerySource, PromptInputMode
from claude_code_py.tools import get_all_base_tools
from claude_code_py.core_types.permissions import PermissionMode, PermissionResult
from claude_code_py.tool.context import ToolUseContext, CanUseToolFn
from claude_code_py.tool.base import Tool
from claude_code_py.engine.query_engine import QueryEngineConfig
from claude_code_py.utils.api_config import get_api_config
from claude_code_py.state.app_state import get_default_app_state
from claude_code_py.utils.managed_env import setup_environment_from_settings

# Rich console for output
from rich.console import Console
from rich.text import Text


# =============================================================================
# Configuration
# =============================================================================


def get_default_system_prompt() -> str:
    """Get the default system prompt."""
    return """You are Claude Code, Anthropic's official CLI for Claude.
You are an interactive agent that helps users with software engineering tasks.
Use the instructions below and the tools available to assist the user.

# System
- All text you output outside of tool use is displayed to the user
- Tools are executed in a user-selected permission mode
- When you attempt to call a tool that is not automatically allowed, the user will be prompted
- Tool results may include data from external sources

# Doing tasks
- The user will primarily request you to perform software engineering tasks
- When given an unclear or generic instruction, consider it in the context of software engineering
- You are highly capable and often allow users to complete ambitious tasks
- In general, do not propose changes to code you haven't read

# Environment
- Primary working directory: {cwd}
- Platform: {platform}
"""


def get_api_key() -> Optional[str]:
    """Get the API key/auth token from configuration.

    Returns:
        API key or auth token, or None
    """
    config = get_api_config()
    return config.get_auth_token()


# =============================================================================
# Permission Handling
# =============================================================================

# Global denial tracking state
_denial_tracking_state = None


def get_denial_tracking_state():
    """Get or create the global denial tracking state."""
    global _denial_tracking_state
    if _denial_tracking_state is None:
        from claude_code_py.utils.permissions.denial_tracking import create_denial_tracking_state
        _denial_tracking_state = create_denial_tracking_state()
    return _denial_tracking_state


async def default_can_use_tool(
    tool: Tool,
    input: Any,
    context: ToolUseContext,
    assistant_message: Any,
    tool_use_id: Optional[str] = None,
    force_decision: Optional[str] = None,
) -> PermissionResult:
    """Default permission handler for tool use.

    This implements the full permission checking flow:
    1. Check bypass mode
    2. Check deny rules
    3. Check ask rules
    4. Check allow rules
    5. Auto mode: run classifier
    6. Default mode: check danger + prompt user
    7. Plan mode: reject non-read operations

    Args:
        tool: Tool to check
        input: Tool input
        context: Tool use context
        assistant_message: Parent message
        tool_use_id: Tool use ID
        force_decision: Forced decision

    Returns:
        Permission result
    """
    from claude_code_py.utils.permissions import (
        get_deny_rule_for_tool,
        get_ask_rule_for_tool,
        get_allow_rule_for_tool,
        check_tool_input_dangerous,
        is_critical_danger,
        should_block_without_confirmation,
        requires_explicit_user_confirmation,
        is_auto_mode_allowlisted_tool,
        classify_action,
        classify_and_decide,
        TwoStageMode,
        ClassifierResult,
        get_denial_tracking_state,
        record_denial,
        record_success,
        should_fallback_to_prompting,
        get_denial_history_message,
        update_denial_tracking_state,
        is_iron_gate_closed,
        check_accept_edits_fast_path,
    )

    # Get permission context
    perm_context = getattr(context.options, "tool_permission_context", None)

    # Determine mode
    mode = getattr(perm_context, "mode", "default") if perm_context else "default"

    # Override mode from environment
    if os.environ.get("CLAUDE_CODE_ACCEPT_ALL", "").lower() == "true":
        mode = "accept-all"
    elif os.environ.get("CLAUDE_CODE_AUTO_MODE", "").lower() == "true":
        mode = "auto"

    # Force decision override
    if force_decision == "allow":
        return PermissionResult.allow(updated_input=input)
    if force_decision == "deny":
        return PermissionResult.deny(reason="Force denied")

    # =========================================================================
    # 1. Bypass mode - allow everything
    # =========================================================================
    if mode == "accept-all" or mode == "bypass":
        return PermissionResult.allow(updated_input=input)

    # =========================================================================
    # 2. Check deny rules - reject immediately
    # =========================================================================
    deny_rule = get_deny_rule_for_tool(perm_context, tool)
    if deny_rule:
        reason = f"Blocked by deny rule from {deny_rule.source.value}"
        return PermissionResult.deny(reason=reason)

    # =========================================================================
    # 3. Check ask rules - force user prompt
    # =========================================================================
    ask_rule = get_ask_rule_for_tool(perm_context, tool)
    if ask_rule:
        # async_input now safely handles raw-mode teardown via pause_listener
        response = await async_input(f"\n⚠️  Tool '{tool.name}' requires permission. Allow? [y/N/a=always] ")
        response = response.strip().lower()

        if response in ("y", "yes"):
            return PermissionResult.allow(updated_input=input)
        if response in ("a", "always"):
            # TODO: Add to always_allow rules
            return PermissionResult.allow(updated_input=input)

        return PermissionResult.deny(reason="User denied")

    # =========================================================================
    # 4. Check allow rules - allow immediately
    # =========================================================================
    allow_rule = get_allow_rule_for_tool(perm_context, tool, input)
    if allow_rule:
        return PermissionResult.allow(updated_input=input)

    # =========================================================================
    # 5. Check if tool is read-only (always safe)
    # =========================================================================
    if hasattr(tool, "is_read_only") and tool.is_read_only(input):
        return PermissionResult.allow(updated_input=input)

    # =========================================================================
    # 6. Auto mode - use LLM classifier
    # =========================================================================
    if mode == "auto":
        # Check if tool is in safe allowlist (skip classifier entirely)
        if is_auto_mode_allowlisted_tool(tool.name):
            return PermissionResult.allow(updated_input=input)

        # Get messages and tools for classifier
        messages = getattr(context, "messages", [])
        tools = getattr(context.options, "tools", [])

        # Check acceptEdits fast path (skip classifier for safe edits)
        fast_path_allowed = await check_accept_edits_fast_path(tool, input, context)
        if fast_path_allowed is True:
            # Record success and return allow
            state = get_denial_tracking_state()
            update_denial_tracking_state(record_success(state, tool.name, input))
            return PermissionResult.allow(updated_input=input)

        # Run classifier with iron gate and denial tracking
        model = getattr(context.options, "main_loop_model", "claude-sonnet-4-6")

        try:
            classifier_result = await classify_and_decide(
                tool_name=tool.name,
                tool_input=input,
                messages=messages,
                tools=tools,
                context=perm_context,
                model=model,
            )

            if not classifier_result.should_block:
                # Allowed by classifier
                return PermissionResult.allow(updated_input=input)

            # Blocked by classifier
            reason = classifier_result.reason or "Blocked by classifier"

            # Check for transcript too long - fallback to prompting
            if classifier_result.transcript_too_long:
                response = await async_input(f"\n⚠️  Classifier: Transcript exceeded context window\n   Please review: {reason}\n   Allow? [y/N] ")
                response = response.strip().lower()
                if response in ("y", "yes"):
                    return PermissionResult.allow(updated_input=input)
                return PermissionResult.deny(reason=reason)

            # Check if classifier unavailable - apply iron gate
            if classifier_result.unavailable:
                if is_iron_gate_closed():
                    # Fail closed - deny
                    response = await async_input(f"\n⚠️  Classifier unavailable, blocking for safety (iron gate closed)\n   Reason: {reason}\n   Override? [y/N] ")
                    response = response.strip().lower()
                    if response in ("y", "yes"):
                        return PermissionResult.allow(updated_input=input)
                    return PermissionResult.deny(reason=reason)
                else:
                    # Fail open - allow with warning
                    print(f"\n⚠️  Classifier unavailable, allowing (iron gate open)")
                    return PermissionResult.allow(updated_input=input)

            # Check if we should fallback to prompting (denial limits)
            state = get_denial_tracking_state()
            if should_fallback_to_prompting(state):
                history_msg = get_denial_history_message(state, tool.name, input)

                prompt_lines = [f"\n⚠️  Classifier blocked: {reason}"]
                if history_msg:
                    prompt_lines.append(f"   {history_msg}")
                prompt_lines.append("   Override? [y/N] ")
                response = await async_input("\n".join(prompt_lines))
                response = response.strip().lower()
                if response in ("y", "yes"):
                    # Reset denials on override
                    from claude_code_py.utils.permissions import reset_denial_tracking_state
                    reset_denial_tracking_state()
                    return PermissionResult.allow(updated_input=input)

            return PermissionResult.deny(reason=reason)

        except Exception as e:
            # Classifier error - prompt user for safety
            response = await async_input(f"\n⚠️  Classifier error: {e}\n   Allow '{tool.name}' manually? [y/N] ")
            response = response.strip().lower()
            if response in ("y", "yes"):
                return PermissionResult.allow(updated_input=input)

            return PermissionResult.deny(reason=f"Classifier error: {e}")

    # =========================================================================
    # 7. Plan mode - reject non-read operations
    # =========================================================================
    if mode == "plan":
        # In plan mode, only allow read-only operations
        if hasattr(tool, "is_read_only") and tool.is_read_only(input):
            return PermissionResult.allow(updated_input=input)

        # Check for EnterPlanMode/ExitPlanMode - allowed in plan mode
        if tool.name in ("EnterPlanMode", "ExitPlanMode", "AskUserQuestion"):
            return PermissionResult.allow(updated_input=input)

        return PermissionResult.deny(
            reason="Plan mode only allows read operations and plan tools"
        )

    # =========================================================================
    # 8. Default mode - check danger + prompt user
    # =========================================================================

    # Check for dangerous patterns
    danger_result = check_tool_input_dangerous(tool.name, input)

    if danger_result.is_dangerous:
        # Critical/high danger - always prompt
        if should_block_without_confirmation(danger_result):
            response = await async_input(
                f"\n⚠️  DANGER: {danger_result.reason}\n"
                f"   Severity: {danger_result.severity}\n"
                f"   This operation could cause irreversible damage.\n"
                f"   Proceed anyway? [y/N] "
            )
            response = response.strip().lower()
            if response in ("y", "yes"):
                return PermissionResult.allow(updated_input=input)

            return PermissionResult.deny(reason=danger_result.reason)

        # Medium danger - warn but allow
        if requires_explicit_user_confirmation(danger_result):
            response = await async_input(f"\n⚠️  Warning: {danger_result.reason}\n   Proceed? [Y/n] ")
            response = response.strip().lower()
            if response in ("n", "no"):
                return PermissionResult.deny(reason=danger_result.reason)

    # Check if tool is destructive (legacy check)
    if hasattr(tool, "is_destructive") and tool.is_destructive(input):
        response = await async_input(f"\n⚠️  Tool '{tool.name}' is destructive. Allow? [y/N] ")
        response = response.strip().lower()
        if response in ("y", "yes"):
            return PermissionResult.allow(updated_input=input)
        return PermissionResult.deny(reason="User denied destructive operation")

    # Default: allow with tracking
    record_success(get_denial_tracking_state(), tool.name, input)
    return PermissionResult.allow(updated_input=input)


# =============================================================================
# REPL Mode
# =============================================================================


class REPL:
    """Interactive REPL for Claude Code."""

    def __init__(
        self,
        cwd: Optional[str] = None,
        permission_mode: PermissionMode = "default",
        verbose: bool = False,
        use_rich_display: bool = True,
    ):
        """Initialize the REPL.

        Args:
            cwd: Working directory
            permission_mode: Permission mode
            verbose: Verbose output
            use_rich_display: Use Rich Live for teammate status display
        """
        self.cwd = cwd or str(Path.cwd())
        self.permission_mode = permission_mode
        self.verbose = verbose

        # Rich console for output
        self._console = Console()

        # Initialize state store with correct cwd
        self._store = Store(get_default_app_state(cwd=self.cwd))

        # Initialize tools
        self._tools = get_all_base_tools()

        # Initialize query engine
        self._engine: Optional[QueryEngine] = None

        # Loading state (for InboxPoller idle/busy check)
        self._is_loading: bool = False

        # Waiting for input state (prevents message submission during blocking stdin)
        self._waiting_for_input: bool = False

        # Interrupt event for permission/question handling
        self._interrupt_event: asyncio.Event = asyncio.Event()

        # Teammate status display (Rich Live)
        self._use_rich_display = use_rich_display
        self._status_display: Optional[Any] = None

        # InboxPoller for teammate messages
        self._inbox_poller: Optional[Any] = None

        # Abort controller for Escape key handling
        from claude_code_py.utils.abort_controller import AbortController
        self._abort_controller: AbortController = AbortController()

        # Escape-aware input handler
        from claude_code_py.ui.escape_input import EscapeInput
        self._escape_input = EscapeInput(self._console)

        # Set global reference for async_input to pause/resume listener
        # and to reuse the shared PromptSession for permission prompts.
        global _escape_input_handler
        _escape_input_handler = self._escape_input

    async def start(self) -> None:
        """Start the REPL."""
        self._console.print("=" * 60)
        self._console.print("Claude Code Python - Interactive Mode")
        self._console.print("=" * 60)
        self._console.print(f"Working directory: {self.cwd}")
        self._console.print(f"Tools available: {len(self._tools)}")
        self._console.print(f"Permission mode: {self.permission_mode}")
        self._console.print("=" * 60)
        self._console.print("Type your message and press Enter. Ctrl+C to exit.")
        self._console.print("=" * 60)
        self._console.print()

        # Pre-initialize PromptSession for permission prompts
        # This ensures _session is available before any tool calls
        # if self._escape_input._session is None:
        #     from prompt_toolkit import PromptSession
        #     from prompt_toolkit.history import InMemoryHistory
        #     self._escape_input._session = PromptSession(history=InMemoryHistory())

        # Initialize query engine
        self._init_engine()

        # Register permission bridge for in-process teammates
        self._register_permission_bridge()

        # Start teammate status display (Rich Live)
        if self._use_rich_display:
            self._start_status_display()
            # Set global reference for async_input to pause/resume the display
            global _status_display_ref
            _status_display_ref = self._status_display

        # Register message submission callback for InboxPoller
        self._register_inbox_callback()

        # Run REPL loop
        while True:
            try:
                # Clear interrupt event first, then drain pending items.
                # Order matters: clearing after drain creates a race window
                # where InboxPoller sets the interrupt between drain and clear,
                # losing the signal and delaying handling until next user input.
                self._interrupt_event.clear()

                # Mark as waiting for input BEFORE any async work (prevents
                # InboxPoller from delivering messages and switching terminal
                # to raw mode while we're about to read from stdin).
                self._waiting_for_input = True
                self._store.set_state(lambda prev: replace(prev, is_waiting_for_input=True))

                # Check for pending questions from teammates BEFORE waiting for input
                await self._handle_pending_questions()

                # Pause status display while waiting for input
                if self._status_display:
                    self._status_display.pause()

                # Check if viewing an agent - change prompt accordingly
                app_state = self._store.get_state()
                viewing_task_id = app_state.viewing_agent_task_id
                if viewing_task_id and viewing_task_id in app_state.tasks:
                    task = app_state.tasks[viewing_task_id]
                    agent_name = getattr(task, 'identity', None)
                    if agent_name:
                        agent_name = agent_name.agent_name
                    else:
                        agent_name = "agent"
                    prompt = f"\n@{agent_name}> "
                else:
                    prompt = "\n> "

                # Get user input with Escape key detection (async version)
                user_input, was_aborted = await self._escape_input.input_async(
                    prompt,
                    on_escape=self._handle_escape,
                    on_ctrl_c=self._handle_ctrl_c,
                    on_shift_up=lambda: self._handle_shift_navigation(-1),
                    on_shift_down=lambda: self._handle_shift_navigation(1),
                )

                # No longer waiting for input — this creates a deliberate
                # window where both is_loading and is_waiting_for_input are
                # False, allowing InboxPoller to deliver pending messages.
                # _process_incoming_message no longer touches terminal mode,
                # so delivery during this gap is safe.
                self._waiting_for_input = False
                self._store.set_state(lambda prev: replace(prev, is_waiting_for_input=False))

                # Resume status display after input received
                if self._status_display:
                    self._status_display.resume()

                # If interrupted or aborted, handle pending requests and continue loop
                if was_aborted:
                    # Clear the interrupt event
                    self._interrupt_event.clear()
                    # Note: Fresh abort controller will be created before next query
                    # Pending questions will be drained at the top of
                    # the loop — no need to handle them here too.
                    continue  # Restart loop to wait for input again

                if not user_input:
                    continue

                user_input = user_input.strip()

                # Handle special commands
                if user_input.startswith("/"):
                    handled = await self._handle_slash_command(user_input)
                    if handled:
                        continue

                # Process message
                await self._process_message(user_input)

            except KeyboardInterrupt:
                self._console.print("\n\nExiting...")
                break
            except EOFError:
                self._console.print("\n\nExiting...")
                break
            except Exception as e:
                self._console.print(f"\n❌ Error: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()

        # Stop teammate status display
        self._stop_status_display()

        # Stop InboxPoller
        self._stop_inbox_poller()

    def _start_status_display(self) -> None:
        """Start the Rich Live teammate status display."""
        from claude_code_py.utils.rich_status_display import create_status_display

        self._status_display = create_status_display(
            self._store.get_state,
            self._store.set_state,
            console=self._console,  # Share REPL's console to avoid conflicts
        )
        self._status_display.start()

    def _stop_status_display(self) -> None:
        """Stop the Rich Live teammate status display."""
        if self._status_display:
            self._status_display.stop()
            self._status_display = None

    async def _handle_pending_questions(self) -> None:
        """Check and handle pending questions from teammates.

        This is called before each REPL input cycle. If there are pending
        questions from teammates (AskUserQuestion requests), display them
        to the user and send answers back to the teammates.
        """
        app_state = self._store.get_state()
        pending_questions = app_state.pending_questions or []

        if not pending_questions:
            return

        # Get the first pending question
        question = pending_questions[0]
        request_id = question.get("request_id")
        from_agent = question.get("from_agent")
        team_name = question.get("team_name")
        questions = question.get("questions", [])

        # Display questions to user
        self._console.print("\n" + "=" * 60)
        self._console.print(f"[yellow] teammate '{from_agent}' needs your input:[/yellow]")
        self._console.print("=" * 60)

        answers = {}
        for i, q in enumerate(questions, 1):
            header = q.get("header", f"Question {i}")
            question_text = q.get("question", "")
            options = q.get("options", [])
            multi_select = q.get("multiSelect", False)

            self._console.print(f"\n[cyan]{header}:[/cyan] {question_text}")
            for j, opt in enumerate(options, 1):
                label = opt.get("label", f"Option {j}")
                desc = opt.get("description", "")
                self._console.print(f"  {j}. {label}")
                if desc:
                    self._console.print(f"     [dim]{desc}[/dim]")

            # Get user answer for this question (async to avoid blocking event loop)
            if multi_select:
                prompt = "  Select multiple (comma-separated numbers): "
            else:
                prompt = "  Your answer (number or text): "

            answer_input = await async_input(prompt)
            answer_input = answer_input.strip()

            # Parse answer
            if answer_input.isdigit():
                # User entered a number - map to option label
                idx = int(answer_input) - 1
                if 0 <= idx < len(options):
                    answers[question_text] = options[idx].get("label", answer_input)
                else:
                    answers[question_text] = answer_input
            else:
                # User entered text directly
                answers[question_text] = answer_input

        # Send answers back to teammate via mailbox
        import json
        from datetime import datetime
        from claude_code_py.utils.teammate_mailbox import (
            write_to_mailbox,
            create_question_response_message,
            TeammateMessage,
            TEAM_LEAD_NAME,
        )

        response = create_question_response_message(
            request_id=request_id,
            answers=answers,
        )

        await write_to_mailbox(
            from_agent,
            TeammateMessage(
                from_agent=TEAM_LEAD_NAME,
                text=json.dumps({
                    "type": response.type,
                    "request_id": response.request_id,
                    "subtype": response.subtype,
                    "answers": response.answers,
                    "error": response.error,
                }),
                timestamp=datetime.now().isoformat(),
            ),
            team_name,
        )

        self._console.print(f"\n[green]✓ Answers sent to {from_agent}[/green]")

        # Remove the handled question from pending queue
        self._store.set_state(lambda prev: replace(
            prev,
            pending_questions=[q for q in prev.pending_questions if q.get("id") != question.get("id")],
        ))

    async def _handle_pending_permissions(self) -> None:
        """Check and handle ALL pending permission requests from teammates.

        This is called before each REPL input cycle AND during main_loop
        iterations. Processes all pending requests in FIFO order.
        """
        import json
        from datetime import datetime
        from claude_code_py.utils.swarm.permission_sync import (
            send_permission_response_via_mailbox,
        )

        while True:
            app_state = self._store.get_state()
            pending_permissions = app_state.pending_permissions or []

            if not pending_permissions:
                break

            # Get the first pending permission request
            perm = pending_permissions[0]
            request_id = perm.get("request_id")
            from_agent = perm.get("from_agent")
            team_name = perm.get("team_name")
            tool_name = perm.get("tool_name")
            description = perm.get("description", "")
            input_data = perm.get("input", {})

            # Display permission request to user
            self._console.print("\n" + "=" * 60)
            self._console.print(f"[yellow] teammate '{from_agent}' requests permission:[/yellow]")
            self._console.print("=" * 60)
            self._console.print(f"\n[cyan]Tool:[/cyan] {tool_name}")
            if description:
                self._console.print(f"[cyan]Description:[/cyan] {description}")
            if input_data:
                for key, value in input_data.items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    self._console.print(f"[cyan]{key}:[/cyan] {value}")

            self._console.print(f"\n[bold]Allow this tool call?[/bold] [y/N/a=always] ")
            response = await async_input()
            response = response.strip().lower()

            approved = response in ("y", "yes", "a", "always")

            # Bridge fast path: invoke callbacks directly (in-process workers)
            callbacks = perm.get("_bridge_callbacks")
            if callbacks and approved:
                updated_input = input_data
                callbacks["on_allow"](updated_input=updated_input)
            elif callbacks and not approved:
                callbacks["on_reject"](feedback="User denied permission")
            else:
                # Mailbox fallback: send response via teammate mailbox
                await send_permission_response_via_mailbox(
                    request_id=request_id,
                    team_name=team_name,
                    recipient_name=from_agent,
                    approved=approved,
                    error=None if approved else "User denied permission",
                )

            if approved:
                self._console.print(f"[green]✓ Permission approved for {from_agent}[/green]")
            else:
                self._console.print(f"[red]✗ Permission denied for {from_agent}[/red]")

            # Remove the handled permission from pending queue
            self._store.set_state(lambda prev: replace(
                prev,
                pending_permissions=[p for p in prev.pending_permissions if p.get("id") != perm.get("id")],
            ))

    async def _show_permission_dialog(self) -> None:
        """Callback for query main_loop to show permission dialogs.

        This is registered as ToolUseContext.show_permission_dialog
        and called from _check_worker_permission_requests during
        message processing.
        """
      
        await self._handle_pending_permissions()

    def _stop_inbox_poller(self) -> None:
        """Stop the InboxPoller for teammate messages."""
        poller = self._inbox_poller
        if poller:
            try:
                # InboxPoller.stop() is async, need to call it properly
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule stop on the event loop (capture poller in lambda)
                    loop.call_soon_threadsafe(
                        lambda p=poller: asyncio.create_task(p.stop())
                    )
                else:
                    loop.run_until_complete(poller.stop())
            except Exception:
                pass
            self._inbox_poller = None

    def _handle_shift_navigation(self, delta: int) -> None:
        """Handle Shift+Up/Down for teammate navigation.

        Args:
            delta: +1 for down, -1 for up
        """
        print("[DEBUG REPL] _handle_shift_navigation called, delta=%d" % delta)
        from claude_code_py.ui import step_teammate_selection
        app_state = self._store.get_state()
        step_teammate_selection(delta, self._store.set_state, app_state)
        print("[DEBUG REPL] step_teammate_selection completed")

    def _handle_ctrl_c(self) -> None:
        """Handle Ctrl+C / Ctrl+Z - exit program."""
        self._console.print("\n\nExiting...")
        # Stop status display and inbox poller
        self._stop_status_display()
        self._stop_inbox_poller()
        os._exit(0) 
        # import sys
        # sys.exit(0)

    def _handle_escape(self) -> bool:
        """Handle Escape key press.

        Unified handler for Escape key in both input and query execution phases.

        Two-level abort pattern:
        - If viewing a running teammate: abort current work (not whole teammate)
        - If viewing a non-running teammate: exit view
        - If not viewing any agent: abort leader's current query

        Returns:
            True if handled (abort or exit view), False if no action taken
        """
        app_state = self._store.get_state()
        view_mode = app_state.view_selection_mode

        if view_mode == "viewing-agent":
            # Viewing a specific agent - check if running
            viewing_task_id = app_state.viewing_agent_task_id
            if viewing_task_id and viewing_task_id in app_state.tasks:
                task = app_state.tasks[viewing_task_id]
                # Check if task is running and has current_work_abort_controller
                if hasattr(task, 'status') and task.status == "running":
                    if hasattr(task, 'current_work_abort_controller') and task.current_work_abort_controller:
                        # Abort current work only (not the whole teammate)
                        task.current_work_abort_controller.abort()
                        self._console.print("\n[yellow]Aborted current work[/yellow]")
                        return True
            # Task not running or no work controller - exit view
            from claude_code_py.ui import exit_teammate_view
            self._store.set_state(exit_teammate_view)
            self._console.print("\n[yellow]Exited agent view[/yellow]")
            return True

        elif view_mode == "selecting-agent":
            # In selection mode - exit selection
            from claude_code_py.ui import exit_teammate_view
            self._store.set_state(exit_teammate_view)
            self._console.print("\n[yellow]Exited agent selection[/yellow]")
            return True

        else:
            # Not viewing any agent - abort leader's current query
            if self._abort_controller and not self._abort_controller.signal.aborted:
                self._abort_controller.abort()
                self._console.print("\n[yellow]Aborted current operation[/yellow]")
                return True

        return False

    def _init_engine(self) -> None:
        """Initialize the QueryEngine."""
        from claude_code_py.tool.context import ThinkingConfig

        api_config = get_api_config()

        # Configure thinking mode
        # Options:
        #   - type="adaptive" - Adaptive thinking (no budget limit, for Claude 4.6+)
        #   - type="enabled", budget_tokens=N - Fixed budget thinking
        #   - type="disabled" - Disable thinking
        thinking_config = ThinkingConfig(type="enabled", budget_tokens=10000)

        config = QueryEngineConfig(
            cwd=self.cwd,
            tools=self._tools,
            commands=[],
            mcp_clients=[],
            agents=[],
            can_use_tool=default_can_use_tool,
            get_app_state=self._store.get_state,
            set_app_state=self._store.set_state,
            verbose=self.verbose,
            fallback_model=api_config.model,
            show_permission_dialog=self._show_permission_dialog,
            thinking_config=thinking_config,
            abort_controller=self._abort_controller,  # Share abort controller with REPL
        )

        self._engine = QueryEngine(config)

    def _register_permission_bridge(self) -> None:
        """Register permission bridge for in-process teammates.

        This allows teammates to use the leader's pending_permissions queue
        for permission handling via Bridge path instead of mailbox fallback.
        """
        from claude_code_py.utils.swarm.permission_bridge import (
            register_leader_permission_queue,
            register_leader_permission_context_setter,
        )
        from claude_code_py.utils.swarm.constants import is_agent_teams_enabled

        if not is_agent_teams_enabled():
            return

        # Create permission queue setter that adds to pending_permissions
        def permission_queue_setter(
            updater: Callable[[list], list]
        ) -> None:
            """Update the pending_permissions queue."""
            current_state = self._store.get_state()
            current_queue = current_state.pending_permissions or []

            # Apply the updater function (e.g., append new item)
            new_queue = updater(current_queue)

            # Update AppState
            self._store.set_state(lambda prev: replace(
                prev,
                pending_permissions=new_queue,
            ))

        # Create permission context setter
        def permission_context_setter(
            context: Any,
            options: Optional[Dict] = None,
        ) -> None:
            """Update the tool_permission_context."""
            self._store.set_state(lambda prev: replace(
                prev,
                tool_permission_context=context,
            ))

        # Register the setters
        register_leader_permission_queue(permission_queue_setter)
        register_leader_permission_context_setter(permission_context_setter)
        # print("[DEBUG REPL] Permission bridge registered")

    def _register_inbox_callback(self) -> None:
        """Register the message submission callback and start InboxPoller.

        This allows InboxPoller to submit teammate messages to the LLM.
        """
        from claude_code_py.utils.swarm.constants import is_agent_teams_enabled
        from dataclasses import replace

        def handle_incoming_prompt(formatted_content: str) -> bool:
            """Handle incoming teammate message.

            Args:
                formatted_content: XML-wrapped message content

            Returns:
                True if submission succeeded, False if rejected
            """
            # Only reject if busy processing (LLM running).
            # We intentionally do NOT check _waiting_for_input here because
            # _process_incoming_message no longer switches terminal mode,
            # so it is safe to start even while input() is blocking.
            if self._is_loading:
                return False

            # Set loading state
            self._is_loading = True
            self._store.set_state(lambda prev: replace(prev, is_loading=True))

            # Start async processing (non-blocking)
            asyncio.create_task(self._process_incoming_message(formatted_content))

            return True

        # Start InboxPoller when agent teams is enabled
        # Pass callback directly instead of using global registry
        # This mirrors TypeScript's useInboxPoller hook in REPL.tsx
        # print(f"[DEBUG REPL] is_agent_teams_enabled = {is_agent_teams_enabled()}")
        if is_agent_teams_enabled():
            print("[DEBUG REPL] Starting InboxPoller...")
            self._start_inbox_poller(handle_incoming_prompt)

    def _start_inbox_poller(self, submit_message_fn: Callable[[str], bool]) -> None:
        """Start the InboxPoller for the leader session.

        This is called when agent teams feature is enabled.
        The poller monitors the leader's mailbox for messages from teammates.

        Args:
            submit_message_fn: Callback to submit message to QueryEngine
        """
        from claude_code_py.utils.swarm.inbox_poller import create_inbox_poller

        print("[DEBUG REPL._start_inbox_poller] Creating InboxPoller...")

        # Get team_name from AppState.teamContext if available
        app_state = self._store.get_state()
        team_context = app_state.team_context
        team_name = team_context.get("teamName") if team_context else None

        print(f"[DEBUG REPL._start_inbox_poller] team_name from AppState = {team_name}")

        # If no team yet, we still create the poller - it will get team_name
        # from AppState when TeamCreate is called
        if not team_name:
            # Use a placeholder - the poller will read from AppState each cycle
            team_name = "pending"
            print("[DEBUG REPL._start_inbox_poller] No team yet, using placeholder 'pending'")

        # Create interrupt callback
        def interrupt_callback() -> None:
            """Trigger interrupt to break out of input wait."""
            self._interrupt_event.set()

        poller = create_inbox_poller(
            team_name,
            self._store.get_state,
            self._store.set_state,
            submit_message_fn=submit_message_fn,
            interrupt_fn=interrupt_callback,
            show_permission_dialog=self._show_permission_dialog,
        )
        poller.start()

        print("[DEBUG REPL._start_inbox_poller] InboxPoller started and running")

        # Store poller reference
        self._inbox_poller = poller

    async def _process_incoming_message(self, formatted_content: str) -> None:
        """Process an incoming teammate message asynchronously.

        This is called by the InboxPoller callback.
        Does NOT switch terminal to raw mode — this runs in the background
        and must not interfere with the REPL's terminal state (e.g., input()
        echo during input_async, or the outer _process_message's listener).

        Args:
            formatted_content: XML-wrapped message content
        """
        if not self._engine:
            return

        try:
            # Process message through query engine
            response_text = []
            async for event in self._engine.submit_message(formatted_content):
                # Check if aborted
                if self._abort_controller.signal.aborted:
                    self._console.print("\n[yellow]Query aborted by user[/yellow]")
                    break

                # Process events
                if hasattr(event, "type"):
                    if event.type == "assistant":
                        content = event.message.get("content", [])
                        for i, block in enumerate(content):
                            if isinstance(block, dict):
                                block_type = block.get("type", "unknown")
                                if block_type == "text":
                                    text = block.get("text", "")
                                    response_text.append(text)
                                    self._console.print(text, end="")
                                elif block_type == "thinking":
                                    # Thinking blocks are internal
                                    pass
                                elif block_type == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    self._console.print(f"\n[dim]⟳ Using tool: {tool_name}[/dim]")
                    elif event.type == "progress":
                        progress_content = getattr(event, "content", "")
                        if progress_content:
                            self._console.print(f"\n[dim]⏳ {progress_content}[/dim]")

            self._console.print()

        except Exception as e:
            self._console.print(f"\n[red]❌ Error: {e}[/red]")
            if self.verbose:
                import traceback
                traceback.print_exc()

        finally:
            # Reset loading state
            self._is_loading = False
            self._store.set_state(lambda prev: replace(prev, is_loading=False))

            # Trigger immediate delivery of any pending messages (matches TS useEffect)
            if self._inbox_poller:
                self._inbox_poller.deliver_pending_now()

    async def _process_message(self, user_input: str) -> None:
        """Process a user message.

        Args:
            user_input: User input string
        """
        from dataclasses import replace
        from claude_code_py.ui.teammate_view import get_viewed_teammate_task
        from claude_code_py.task.in_process_teammate import is_in_process_teammate_task

        if not self._engine:
            self._console.print("[red]❌ Engine not initialized[/red]")
            return

        # Check if viewing an agent - route message to that agent
        viewed_task = get_viewed_teammate_task(self._store.get_state())
        if viewed_task:
            # Inject message to the viewed teammate
            self._console.print(f"\n[cyan]Sending message to @{viewed_task.identity.agent_name}...[/cyan]")

            # Add message to teammate's pending queue
            from claude_code_py.core_types.message import UserMessage
            new_msg = UserMessage(content=user_input)

            # Update task with new message
            def add_message(prev):
                task = prev.tasks.get(viewed_task.id)
                if task and is_in_process_teammate_task(task):
                    new_pending = list(task.pending_user_messages) + [user_input]
                    new_messages = list(task.messages) + [{"role": "user", "content": user_input}]
                    new_tasks = dict(prev.tasks)
                    new_tasks[viewed_task.id] = replace(
                        task,
                        pending_user_messages=new_pending,
                        messages=new_messages,
                    )
                    return replace(prev, tasks=new_tasks)
                return prev

            self._store.set_state(add_message)

            # Poll for agent response and display new messages
            await self._poll_agent_response(viewed_task)
            return

        # Normal processing - leader handles the message
        # Create fresh abort controller for this query (matches TypeScript pattern)
        from claude_code_py.utils.abort_controller import AbortController
        self._abort_controller = AbortController()
        self._engine.config.abort_controller = self._abort_controller

        # Set loading state
        self._is_loading = True
        self._store.set_state(lambda prev: replace(prev, is_loading=True))

        self._console.print("\n[cyan]Claude is thinking...[/cyan] [dim](Press Escape to interrupt)[/dim]")

        # Start keyboard listener for execution phase
        self._escape_input.start_execution_listener(
            on_escape=self._handle_escape,
            on_ctrl_c=self._handle_ctrl_c,
        )

        try:
            # Stream response
            response_text = []

            async for event in self._engine.submit_message(user_input):
                # Check if aborted
                if self._abort_controller.signal.aborted:
                    self._console.print("\n[yellow]Query aborted by user[/yellow]")
                    break

                # Process events
                if hasattr(event, "type"):
                    if event.type == "assistant":
                        # Extract text content
                        content = event.message.get("content", [])
                        for i, block in enumerate(content):
                            if isinstance(block, dict):
                                block_type = block.get("type", "unknown")
                                if block_type == "text":
                                    text = block.get("text", "")
                                    response_text.append(text)
                                    # Print text directly (no extra formatting)
                                    self._console.print(text, end="")
                                elif block_type == "thinking":
                                    # Thinking blocks are internal, not shown to user
                                    pass
                                elif block_type == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    self._console.print(f"\n[dim]⟳ Using tool: {tool_name}[/dim]")
                    elif event.type == "progress":
                        progress_content = getattr(event, "content", "")
                        if progress_content:
                            self._console.print(f"\n[dim]⏳ {progress_content}[/dim]")
                    elif event.type == "system" and hasattr(event, "subtype") and event.subtype == "error":
                        self._console.print(f"\n[red]❌ {event.content}[/red]")

            self._console.print()  # New line after response

        except Exception as e:
            self._console.print(f"\n[red]❌ Error processing message: {e}[/red]")
            if self.verbose:
                import traceback
                traceback.print_exc()

        finally:
            # Stop keyboard listener
            self._escape_input.stop_execution_listener()
            # Reset loading state
            self._is_loading = False
            self._store.set_state(lambda prev: replace(prev, is_loading=False))

            # Trigger immediate delivery of any pending messages (matches TS useEffect)
            if self._inbox_poller:
                self._inbox_poller.deliver_pending_now()

    async def _handle_slash_command(self, command: str) -> bool:
        """Handle slash commands.

        Args:
            command: Slash command string

        Returns:
            True if handled, False otherwise
        """
        cmd = command.lower().strip()

        if cmd == "/help":
            self._console.print("""
[cyan]Available commands:[/cyan]
  /help       - Show this help
  /tools      - List available tools
  /clear      - Clear conversation history
  /mode       - Show/change permission mode
  /exit       - Exit the REPL

[cyan]Agent view commands:[/cyan]
  /agents     - Show list of running agents
  /view <name> - View an agent's status and messages
  /exit-view  - Return to leader view
  /next       - Select next agent
  /prev       - Select previous agent
""")
            return True

        elif cmd == "/tools":
            self._console.print("\n[cyan]Available tools:[/cyan]")
            for tool in self._tools:
                self._console.print(f"  - {tool.name}")
            return True

        elif cmd == "/clear":
            if self._engine:
                self._engine._messages = []
            self._console.print("[green]✅ Conversation cleared[/green]")
            return True

        elif cmd == "/mode":
            self._console.print(f"\nCurrent permission mode: [cyan]{self.permission_mode}[/cyan]")
            self._console.print("Available modes: default, accept-all, plan")
            return True

        elif cmd == "/exit":
            self._console.print("[yellow]Exiting...[/yellow]")
            sys.exit(0)

        # Agent view commands
        elif cmd == "/agents":
            return await self._handle_agents_command()

        elif cmd.startswith("/view "):
            agent_name = command[6:].strip()
            return await self._handle_view_agent_command(agent_name)

        elif cmd == "/exit-view":
            return self._handle_exit_view_command()

        elif cmd == "/next":
            return self._handle_next_agent_command()

        elif cmd == "/prev":
            return self._handle_prev_agent_command()

        return False

    async def _handle_agents_command(self) -> bool:
        """Handle /agents command to show agent list."""
        from claude_code_py.ui.teammate_view import (
            TeammateViewUI,
            get_running_teammates_sorted,
        )
        from dataclasses import replace

        ui = TeammateViewUI(self._console)
        app_state = self._store.get_state()
        teammates = get_running_teammates_sorted(app_state)

        if not teammates:
            self._console.print("\n[yellow]No running agents[/yellow]")
            return True

        # Set expanded view state
        self._store.set_state(lambda prev: replace(
            prev,
            expanded_view="teammates",
            view_selection_mode="selecting-agent",
            selected_agent_index=-1,
        ))

        self._console.print("\n")
        self._console.print(ui.render_agent_list(app_state, teammates))
        self._console.print("\n[dim]Commands: /view <name> to view, /next or /prev to navigate[/dim]")
        return True

    async def _handle_view_agent_command(self, agent_name: str) -> bool:
        """Handle /view <agent> command to view an agent's status."""
        from claude_code_py.ui.teammate_view import (
            TeammateViewUI,
            get_running_teammates_sorted,
            enter_teammate_view,
        )

        if not agent_name:
            self._console.print("[red]Usage: /view <agent_name>[/red]")
            return True

        app_state = self._store.get_state()
        teammates = get_running_teammates_sorted(app_state)

        # Find agent by name
        for task in teammates:
            if task.identity.agent_name == agent_name:
                enter_teammate_view(task.id, self._store.set_state)

                # Re-read state to get bootstrapped messages
                updated_state = self._store.get_state()
                updated_task = updated_state.tasks.get(task.id)

                # Render the view with bootstrapped messages
                ui = TeammateViewUI(self._console)
                self._console.print("\n")
                if updated_task:
                    self._console.print(ui.render_full_view(updated_task))
                else:
                    self._console.print(ui.render_full_view(task))
                self._console.print("\n[dim]Type your message to send, /exit-view to return to leader, Esc to go back[/dim]")
                return True

        self._console.print(f"[red]Agent '{agent_name}' not found[/red]")
        self._console.print(f"[dim]Available agents: {', '.join(t.identity.agent_name for t in teammates)}[/dim]")
        return True

    async def _poll_agent_response(self, viewed_task) -> None:
        """Poll the viewed agent for new response messages.

        After injecting a user message to a viewed teammate, poll task.messages
        for new assistant responses. Display new messages as they arrive.
        Break when the teammate goes idle (response complete) or on Escape.

        Args:
            viewed_task: The InProcessTeammateTaskState being viewed
        """
        import asyncio
        from claude_code_py.ui.teammate_view import TeammateViewUI

        ui = TeammateViewUI(self._console)
        seen_count = len(viewed_task.messages)
        poll_interval = 0.3  # seconds
        max_polls = 300  # ~90 seconds timeout
        poll_count = 0

        self._console.print("[dim]Waiting for response... (Esc to stop waiting)[/dim]")

        while poll_count < max_polls:
            await asyncio.sleep(poll_interval)
            poll_count += 1

            # Re-read task state (AppState may have been updated by runner)
            state = self._store.get_state()
            task = state.tasks.get(viewed_task.id)
            if not task:
                break

            # Check for new messages
            current_msgs = getattr(task, 'messages', []) or []
            new_msgs = current_msgs[seen_count:]
            if new_msgs:
                for msg in new_msgs:
                    # Handle both Pydantic Message objects and raw dicts
                    if hasattr(msg, "type") and hasattr(msg, "message"):
                        role = msg.message.get("role", msg.type)
                        content = msg.message.get("content", "")
                    elif isinstance(msg, dict):
                        # Two shapes: serialized Pydantic ({message: {role, content}})
                        # or flat injected user msgs ({role, content})
                        inner = msg.get("message")
                        if isinstance(inner, dict) and "role" in inner:
                            role = inner.get("role", msg.get("type", "unknown"))
                            content = inner.get("content", "")
                        else:
                            role = msg.get("role") or msg.get("type", "unknown")
                            content = msg.get("content", "")
                    else:
                        continue
                    if isinstance(content, list):
                        # Handle content blocks (tool_use, tool_result, etc.)
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif block.get("type") == "tool_use":
                                    text_parts.append(f"[tool: {block.get('name', '?')}]")
                                elif block.get("type") == "tool_result":
                                    text_parts.append("[tool result]")
                            else:
                                text_parts.append(str(block))
                        content = " ".join(text_parts)
                    if role == "assistant":
                        self._console.print(f"\n[bold green]@{task.identity.agent_name}:[/bold green] {content}")
                    elif role == "user":
                        self._console.print(f"\n[bold blue]You:[/bold blue] {content}")
                seen_count = len(current_msgs)

            # Check if teammate is idle (response complete)
            is_idle = getattr(task, 'is_idle', False)
            status = getattr(task, 'status', 'running')
            if is_idle or status in ('completed', 'killed', 'failed', 'error'):
                break

            # Check for Escape press (non-blocking poll)
            if self._interrupt_event.is_set():
                self._interrupt_event.clear()
                self._console.print("\n[dim]Stopped waiting[/dim]")
                break

        self._console.print()

    def _handle_exit_view_command(self) -> bool:
        """Handle /exit-view command to return to leader view."""
        from claude_code_py.ui.teammate_view import exit_teammate_view

        app_state = self._store.get_state()
        if not app_state.viewing_agent_task_id:
            self._console.print("[yellow]Not viewing any agent[/yellow]")
            return True

        exit_teammate_view(self._store.set_state)
        self._console.print("[green]Returned to leader view[/green]")
        return True

    def _handle_next_agent_command(self) -> bool:
        """Handle /next command to select next agent."""
        from claude_code_py.ui.teammate_view import (
            step_teammate_selection,
            get_running_teammates_sorted,
        )
        from dataclasses import replace

        app_state = self._store.get_state()
        teammates = get_running_teammates_sorted(app_state)

        if not teammates:
            self._console.print("[yellow]No running agents[/yellow]")
            return True

        step_teammate_selection(1, self._store.set_state, app_state)

        # Show current selection
        new_state = self._store.get_state()
        idx = new_state.selected_agent_index

        if idx == -1:
            self._console.print("[cyan]Selected: leader[/cyan]")
        elif idx < len(teammates):
            task = teammates[idx]
            self._console.print(f"[cyan]Selected: @{task.identity.agent_name}[/cyan]")
        else:
            self._console.print("[dim]Selected: hide[/dim]")

        return True

    def _handle_prev_agent_command(self) -> bool:
        """Handle /prev command to select previous agent."""
        from claude_code_py.ui.teammate_view import (
            step_teammate_selection,
            get_running_teammates_sorted,
        )

        app_state = self._store.get_state()
        teammates = get_running_teammates_sorted(app_state)

        if not teammates:
            self._console.print("[yellow]No running agents[/yellow]")
            return True

        step_teammate_selection(-1, self._store.set_state, app_state)

        # Show current selection
        new_state = self._store.get_state()
        idx = new_state.selected_agent_index

        if idx == -1:
            self._console.print("[cyan]Selected: leader[/cyan]")
        elif idx < len(teammates):
            task = teammates[idx]
            self._console.print(f"[cyan]Selected: @{task.identity.agent_name}[/cyan]")
        else:
            self._console.print("[dim]Selected: hide[/dim]")

        return True


# =============================================================================
# SDK Mode
# =============================================================================


async def run_sdk_mode(
    prompt: str,
    cwd: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Run in SDK/headless mode.

    Args:
        prompt: User prompt
        cwd: Working directory
        output_file: Output file path

    Returns:
        Response text
    """
    cwd = cwd or str(Path.cwd())

    # Initialize store with correct cwd
    store = Store(get_default_app_state(cwd=cwd))

    # Initialize tools
    tools = get_all_base_tools()

    # Get model from api config
    api_config = get_api_config()

    # Configure thinking mode
    from claude_code_py.tool.context import ThinkingConfig
    thinking_config = ThinkingConfig(type="enabled", budget_tokens=10000)

    # Initialize engine
    config = QueryEngineConfig(
        cwd=cwd,
        tools=tools,
        commands=[],
        mcp_clients=[],
        agents=[],
        can_use_tool=default_can_use_tool,
        get_app_state=store.get_state,
        set_app_state=store.set_state,
        verbose=False,
        fallback_model=api_config.model,
        thinking_config=thinking_config,
    )

    engine = QueryEngine(config)

    # Process message
    response_parts = []
    async for event in engine.submit_message(prompt):
        if hasattr(event, "type"):
            if event.type == "assistant":
                content = event.message.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        response_parts.append(block.get("text", ""))

    response = "".join(response_parts)

    # Write to file if specified
    if output_file:
        Path(output_file).write_text(response)

    return response


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Claude Code Python - AI-powered coding assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  claude_code_py                    # Interactive REPL mode
  claude_code_py "Fix the bug"      # SDK mode with prompt
  claude_code_py -p "Explain code"  # SDK mode with print flag
  claude_code_py --cwd /path/to/dir # Specify working directory
"""
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="User prompt (if provided, runs in SDK mode)",
    )
    parser.add_argument(
        "-p", "--print",
        action="store_true",
        help="Run in print/headless mode",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default="/Users/chentianzeng/Documents/code_project/test_agent_tema",
        help="Working directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for SDK mode",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["default", "accept-all", "plan"],
        default="default",
        help="Permission mode",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--accept-all",
        action="store_true",
        help="Accept all tool calls without prompting",
    )

    args = parser.parse_args()

    # Determine working directory
    cwd = args.cwd or str(Path.cwd())

    # Apply environment variables from settings.json
    # Use trust_established=True for SDK/headless mode
    trust_established = bool(args.prompt or args.print)
    setup_environment_from_settings(cwd=cwd, trust_established=trust_established)

    # Check for API configuration
    config = get_api_config()
    if not config.is_valid():
        print("⚠️  Warning: No API authentication configured")
        print("   Set one of:")
        print("     export ANTHROPIC_API_KEY=your-key")
        print("     export ANTHROPIC_AUTH_TOKEN=your-token")
        if config.base_url:
            print(f"   Base URL: {config.base_url}")
        print()

    # Handle accept-all
    if args.accept_all:
        os.environ["CLAUDE_CODE_ACCEPT_ALL"] = "true"
        args.mode = "accept-all"

    # Run in appropriate mode
    if args.prompt or args.print:
        # SDK mode
        if not args.prompt:
            print("❌ Error: --print mode requires a prompt")
            sys.exit(1)

        response = asyncio.run(run_sdk_mode(
            prompt=args.prompt,
            cwd=cwd,
            output_file=args.output,
        ))
        print(response)
    else:
        # REPL mode
        repl = REPL(
            cwd=cwd,
            permission_mode=args.mode,
            verbose=args.verbose,
        )
        asyncio.run(repl.start())


if __name__ == "__main__":
    main()