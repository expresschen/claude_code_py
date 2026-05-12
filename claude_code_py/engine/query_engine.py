"""Query Engine - manages query lifecycle and session state.

This is a Python implementation of QueryEngine.ts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional, Union

from claude_code_py.state.app_state import AppState, SetAppState
from claude_code_py.tool.context import (
    CanUseToolFn,
    Command,
    MCPServerConnection,
    AgentDefinition,
    ThinkingConfig,
    ToolUseContext,
    ToolUseContextOptions,
)
from claude_code_py.tool.base import Tool
from claude_code_py.core_types.message import Message
from claude_code_py.utils.abort_controller import AbortController, create_abort_controller
from claude_code_py.storage.session import SessionStorage


@dataclass
class QueryEngineConfig:
    """Configuration for QueryEngine."""

    # Working directory
    cwd: str

    # Tools
    tools: list[Tool]

    # Commands
    commands: list[Command]

    # MCP
    mcp_clients: list[MCPServerConnection]

    # Agents
    agents: list[AgentDefinition]

    # Permissions
    can_use_tool: CanUseToolFn

    # State
    get_app_state: Callable[[], AppState]
    set_app_state: SetAppState

    # Messages
    initial_messages: Optional[list[Message]] = None

    # File cache
    read_file_cache: dict[str, Any] = field(default_factory=dict)

    # Prompts
    custom_system_prompt: Optional[str] = None
    append_system_prompt: Optional[str] = None

    # Model
    user_specified_model: Optional[str] = None
    fallback_model: Optional[str] = None

    # Thinking
    thinking_config: Optional[ThinkingConfig] = None

    # Limits
    max_turns: Optional[int] = None
    max_budget_usd: Optional[float] = None
    task_budget: Optional[dict[str, int]] = None

    # Schema
    json_schema: Optional[dict[str, Any]] = None

    show_permission_dialog: Optional[Callable[[], Any]] = None

    # Options
    verbose: bool = False
    replay_user_messages: bool = False
    include_partial_messages: bool = False

    # Session persistence
    persist_session: bool = True
    session_id: Optional[str] = None

    # Control
    abort_controller: Optional[AbortController] = None

    # Callbacks
    handle_elicitation: Optional[Callable] = None
    set_sdk_status: Optional[Callable] = None


class QueryEngine:
    """QueryEngine owns the query lifecycle and session state.

    One QueryEngine per conversation. Each submit_message() call starts
    a new turn within the same conversation.

    This extracts the core logic from the REPL into a standalone class
    that can be used by both headless/SDK mode and interactive mode.
    """

    def __init__(self, config: QueryEngineConfig):
        """Initialize the QueryEngine.

        Args:
            config: Engine configuration
        """
        self.config = config
        self._messages: list[Message] = config.initial_messages or []
        # Note: We don't store _abort_controller - we get it from config each time
        # This allows the caller to update the controller between queries
        self._permission_denials: list[dict[str, Any]] = []
        self._total_usage: dict[str, int] = {}
        self._has_handled_orphaned_permission = False
        self._read_file_state = config.read_file_cache.copy()
        self._discovered_skill_names: set[str] = set()
        self._loaded_nested_memory_paths: set[str] = set()

        # Session storage for persistence
        self._session_storage: Optional[SessionStorage] = None
        if config.persist_session:
            self._session_storage = SessionStorage(config.session_id, config.cwd)

    @property
    def _abort_controller(self) -> AbortController:
        """Get the abort controller from config, creating one if needed."""
        return self.config.abort_controller or create_abort_controller()

    async def submit_message(
        self,
        prompt: Union[str, list[dict[str, Any]]],
        options: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[Any, None]:
        """Submit a message and process the query loop.

        Args:
            prompt: User prompt (string or content blocks)
            options: Optional message options (uuid, is_meta)

        Yields:
            SDKMessage events from processing
        """
        from .query import query, QueryParams

        options = options or {}
        uuid_str = options.get("uuid") or str(uuid.uuid4())
        is_meta = options.get("is_meta", False)

        # Create tool use context
        context = self._create_tool_use_context()

        # Process user input
        messages_from_input = await self._process_user_input(
            prompt=prompt,
            uuid=uuid_str,
            is_meta=is_meta,
            context=context,
        )

        # Add messages to history
        self._messages.extend(messages_from_input)

        # Record user messages to session storage (before query loop)
        # This ensures messages are persisted even if process is killed
        if self._session_storage:
            await self._record_transcript(messages_from_input)

        # Build query params
        params = QueryParams(
            messages=self._messages.copy(),
            system_prompt=await self._build_system_prompt(),
            user_context={},
            system_context={},
            can_use_tool=self._wrap_can_use_tool(),
            tool_use_context=context,
            query_source="sdk",
            fallback_model=self.config.user_specified_model or self.config.fallback_model,
        )

        # Run query loop
        async for event in query(params):
            # Track messages
            if hasattr(event, "type"):
                if event.type in ("assistant", "user", "system"):
                    self._messages.append(event)
                    # Record to session storage
                    if self._session_storage:
                        await self._record_transcript([event])

            yield event

    async def _record_transcript(self, messages: list[Message]) -> None:
        """Record messages to session storage.

        Args:
            messages: Messages to record
        """
        if not self._session_storage:
            return

        for msg in messages:
            try:
                self._session_storage.append_message(msg)
            except Exception:
                # Don't fail the query if session storage fails
                pass

    def flush_session(self) -> None:
        """Flush session storage to disk."""
        if self._session_storage:
            self._session_storage._save_meta()

    def get_session_storage(self) -> Optional[SessionStorage]:
        """Get the session storage instance.

        Returns:
            SessionStorage or None if persistence disabled
        """
        return self._session_storage

    def interrupt(self) -> None:
        """Interrupt the current query."""
        self._abort_controller.abort()

    def get_messages(self) -> list[Message]:
        """Get all messages in the conversation.

        Returns:
            Copy of message list
        """
        return self._messages.copy()

    def get_read_file_state(self) -> dict[str, Any]:
        """Get the file state cache.

        Returns:
            File state cache
        """
        return self._read_file_state.copy()

    def get_session_id(self) -> str:
        """Get the session ID.

        Returns:
            Session ID string
        """
        if self._session_storage:
            return self._session_storage.session_id
        return self.config.session_id or str(uuid.uuid4())

    def set_model(self, model: str) -> None:
        """Set the model for subsequent queries.

        Args:
            model: Model identifier
        """
        self.config.user_specified_model = model

    def _create_tool_use_context(self) -> ToolUseContext:
        """Create a ToolUseContext for this engine.

        Returns:
            ToolUseContext instance
        """
        from claude_code_py.core_types.permissions import ToolPermissionContext

        # Get cwd from app state if available, otherwise use config
        cwd = self.config.cwd
        app_state = self.config.get_app_state()
        if app_state and hasattr(app_state, "cwd"):
            cwd = app_state.cwd

        # Create permission context with cwd
        perm_context = ToolPermissionContext(cwd=cwd)

        options = ToolUseContextOptions(
            cwd=cwd,
            tool_permission_context=perm_context,
            commands=self.config.commands,
            tools=self.config.tools,
            mcp_clients=self.config.mcp_clients,
            main_loop_model=self.config.user_specified_model or "claude-sonnet-4-6",
            thinking_config=self.config.thinking_config or ThinkingConfig(),
            debug=self.config.verbose,
            verbose=self.config.verbose,
            max_budget_usd=self.config.max_budget_usd,
            custom_system_prompt=self.config.custom_system_prompt,
            append_system_prompt=self.config.append_system_prompt,
        )

        return ToolUseContext(
            options=options,
            abort_controller=self._abort_controller,
            messages=self._messages,
            get_app_state=self.config.get_app_state,
            set_app_state=self.config.set_app_state,
            show_permission_dialog=self.config.show_permission_dialog,
        )

    def _wrap_can_use_tool(self) -> CanUseToolFn:
        """Wrap can_use_tool to track permission denials.

        Returns:
            Wrapped can_use_tool function
        """
        original = self.config.can_use_tool

        async def wrapped(
            tool: Tool,
            input: Any,
            context: ToolUseContext,
            assistant_message: Any,
            tool_use_id: Optional[str],
            force_decision: Optional[str] = None,
        ) -> Any:
            result = await original(
                tool, input, context, assistant_message, tool_use_id, force_decision
            )

            # Track denials
            if result.behavior != "allow":
                self._permission_denials.append({
                    "tool_name": tool.name,
                    "tool_use_id": tool_use_id,
                    "tool_input": input,
                })

            return result

        return wrapped

    async def _build_system_prompt(self) -> str:
        """Build the system prompt.

        Returns:
            System prompt string
        """
        from claude_code_py.constants.prompts import get_system_prompt

        return await get_system_prompt(
            tools=self.config.tools,
            model=self.config.user_specified_model or self.config.fallback_model or "claude-sonnet-4-6",
            additional_working_dirs=None,
            custom_system_prompt=self.config.custom_system_prompt,
            append_system_prompt=self.config.append_system_prompt,
            cwd=self.config.cwd,
        )

    async def _process_user_input(
        self,
        prompt: Union[str, list[dict[str, Any]]],
        uuid: str,
        is_meta: bool,
        context: ToolUseContext,
    ) -> list[Message]:
        """Process user input into messages.

        This uses the full process_user_input pipeline including:
        - Memory recall
        - Slash command routing
        - Attachment handling
        - Hook execution

        Args:
            prompt: User prompt
            uuid: Message UUID
            is_meta: Whether this is a meta message
            context: Tool use context

        Returns:
            List of messages from the input
        """
        from .process_input import (
            process_user_input,
            ProcessUserInputOptions,
            PromptInputMode,
            QuerySource,
        )

        # Build options
        options = ProcessUserInputOptions(
            input=prompt,
            mode=PromptInputMode.PROMPT,
            uuid=uuid,
            is_meta=is_meta,
            query_source=QuerySource.SDK,
            messages=self._messages.copy(),
        )

        # Process input
        result = await process_user_input(
            options=options,
            context=context,
            can_use_tool=self.config.can_use_tool,
        )

        return result.messages


# Empty usage constant
EMPTY_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def accumulate_usage(total: dict[str, int], current: dict[str, int]) -> dict[str, int]:
    """Accumulate usage from current into total.

    Args:
        total: Total usage so far
        current: Current message usage

    Returns:
        Updated total usage
    """
    return {
        "input_tokens": total.get("input_tokens", 0) + current.get("input_tokens", 0),
        "output_tokens": total.get("output_tokens", 0) + current.get("output_tokens", 0),
        "cache_creation_input_tokens": total.get("cache_creation_input_tokens", 0)
        + current.get("cache_creation_input_tokens", 0),
        "cache_read_input_tokens": total.get("cache_read_input_tokens", 0)
        + current.get("cache_read_input_tokens", 0),
    }


def update_usage(current: dict[str, int], update: dict[str, int]) -> dict[str, int]:
    """Update usage with new values.

    Args:
        current: Current usage
        update: Update values

    Returns:
        Updated usage
    """
    result = current.copy()
    for key, value in update.items():
        if key in result:
            result[key] += value
        else:
            result[key] = value
    return result