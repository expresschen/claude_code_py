"""Process user input into messages.

This implements the user input processing pipeline from processUserInput.ts,
including memory recall, slash command routing, and attachment handling.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

from claude_code_py.core_types.message import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ProgressMessage,
    AttachmentMessage,
    Message,
)
from claude_code_py.core_types.permissions import PermissionMode
from claude_code_py.memory import (
    is_auto_memory_enabled,
    get_auto_mem_path,
    build_memory_prompt,
    ensure_memory_dir_exists,
    AgentMemoryScope,
    get_agent_memory_dir,
    SessionMemory,
    collect_surfaced_memories,
    extract_recent_tools,
)
from claude_code_py.memory.find_relevant import (
    find_relevant_memories,
    get_memory_files_to_attachments,
)


# =============================================================================
# Input Mode Types
# =============================================================================


class PromptInputMode(str, Enum):
    """Mode for prompt input."""

    PROMPT = "prompt"  # Regular interactive prompt
    BASH = "bash"  # Bash command mode (! prefix)
    PRINT = "print"  # Print/headless mode


class QuerySource(str, Enum):
    """Source of the query."""

    CLI = "cli"
    SDK = "sdk"
    BRIDGE = "bridge"
    HOOK = "hook"
    SCHEDULED_TASK = "scheduled_task"


# =============================================================================
# Process Result Types
# =============================================================================


@dataclass
class ProcessUserInputResult:
    """Result of processing user input."""

    messages: list[Message] = field(default_factory=list)
    should_query: bool = True
    allowed_tools: Optional[list[str]] = None
    model: Optional[str] = None
    effort: Optional[str] = None
    result_text: Optional[str] = None
    next_input: Optional[str] = None
    submit_next_input: bool = False


# =============================================================================
# Hook Result Types
# =============================================================================


@dataclass
class HookResult:
    """Result from a UserPromptSubmit hook."""

    message: Optional[Message] = None
    blocking_error: Optional[str] = None
    prevent_continuation: bool = False
    stop_reason: Optional[str] = None
    additional_contexts: Optional[list[str]] = None


# =============================================================================
# Context Types
# =============================================================================


@dataclass
class IDESelection:
    """IDE selection context."""

    file_path: Optional[str] = None
    content: Optional[str] = None
    selection_range: Optional[dict[str, int]] = None


@dataclass
class PastedContent:
    """Pasted content from user."""

    id: int
    content: str
    media_type: Optional[str] = None
    source_path: Optional[str] = None
    dimensions: Optional[dict[str, int]] = None


@dataclass
class ProcessUserInputOptions:
    """Options for process_user_input."""

    input: Union[str, list[dict[str, Any]]]
    pre_expansion_input: Optional[str] = None
    mode: PromptInputMode = PromptInputMode.PROMPT
    pasted_contents: Optional[dict[int, PastedContent]] = None
    ide_selection: Optional[IDESelection] = None
    messages: Optional[list[Message]] = None
    uuid: Optional[str] = None
    is_already_processing: bool = False
    query_source: Optional[QuerySource] = None
    skip_slash_commands: bool = False
    bridge_origin: bool = False
    is_meta: bool = False
    skip_attachments: bool = False


# =============================================================================
# Main Processing Function
# =============================================================================


async def process_user_input(
    options: ProcessUserInputOptions,
    context: Any,
    can_use_tool: Optional[Callable] = None,
    set_user_input_on_processing: Optional[Callable[[str], None]] = None,
) -> ProcessUserInputResult:
    """Process user input into messages.

    This is the main entry point for processing user input. It:
    1. Shows processing indicator
    2. Processes the input base (routing to appropriate handler)
    3. Executes UserPromptSubmit hooks
    4. Returns messages and query flags

    Args:
        options: Input processing options
        context: Tool use context
        can_use_tool: Permission checking function
        set_user_input_on_processing: Callback to show processing indicator

    Returns:
        ProcessUserInputResult with messages and flags
    """
    input_str = (
        options.input if isinstance(options.input, str) else None
    )

    # Show processing indicator for interactive mode (skip for meta messages)
    if (
        options.mode == PromptInputMode.PROMPT
        and input_str is not None
        and not options.is_meta
        and set_user_input_on_processing
    ):
        set_user_input_on_processing(input_str)

    # Get app state
    app_state = context.get_app_state() if hasattr(context, "get_app_state") else None
    permission_mode = (
        app_state.tool_permission_context.mode
        if app_state and hasattr(app_state, "tool_permission_context")
        else PermissionMode.INTERACTIVE
    )

    # Process input base
    result = await process_user_input_base(
        options,
        context,
        can_use_tool,
        permission_mode,
    )

    if not result.should_query:
        return result

    # Execute UserPromptSubmit hooks (if any)
    hook_results = await execute_user_prompt_submit_hooks(
        input_str or "",
        context,
    )

    for hook_result in hook_results:
        # Handle blocking error - erase original input
        if hook_result.blocking_error:
            return ProcessUserInputResult(
                messages=[
                    create_system_message(
                        f"{hook_result.blocking_error}\n\nOriginal prompt: {input_str}",
                        "warning",
                    ),
                ],
                should_query=False,
                allowed_tools=result.allowed_tools,
            )

        # Handle prevent continuation - keep original prompt
        if hook_result.prevent_continuation:
            stop_msg = (
                f"Operation stopped by hook: {hook_result.stop_reason}"
                if hook_result.stop_reason
                else "Operation stopped by hook"
            )
            result.messages.append(create_user_message(stop_msg))
            result.should_query = False
            return result

        # Collect additional contexts
        if hook_result.additional_contexts:
            result.messages.append(
                create_attachment_message(
                    type="hook_additional_context",
                    content=hook_result.additional_contexts,
                    hook_name="UserPromptSubmit",
                    tool_use_id=f"hook-{str(uuid.uuid4())}",
                    hook_event="UserPromptSubmit",
                )
            )

        # Add hook message
        if hook_result.message:
            result.messages.append(hook_result.message)

    return result


# =============================================================================
# Base Processing Function
# =============================================================================


async def process_user_input_base(
    options: ProcessUserInputOptions,
    context: Any,
    can_use_tool: Optional[Callable] = None,
    permission_mode: Optional[PermissionMode] = None,
) -> ProcessUserInputResult:
    """Process user input base logic.

    This handles:
    - Image processing and resizing
    - Pasted content handling
    - Bridge-safe command checking
    - Slash command routing
    - Bash command routing
    - Regular prompt processing
    - Memory recall integration

    Args:
        options: Input processing options
        context: Tool use context
        can_use_tool: Permission checking function
        permission_mode: Permission mode

    Returns:
        ProcessUserInputResult
    """
    input_str: Optional[str] = None
    preceding_blocks: list[dict[str, Any]] = []
    image_metadata_texts: list[str] = []

    # Normalize input
    normalized_input: Union[str, list[dict[str, Any]]] = options.input

    if isinstance(options.input, str):
        input_str = options.input
    elif options.input:
        # Process array input (handle images)
        processed_blocks: list[dict[str, Any]] = []
        for block in options.input:
            if block.get("type") == "image":
                # Image processing placeholder (would resize/downsample)
                processed_blocks.append(block)
            else:
                processed_blocks.append(block)

        normalized_input = processed_blocks

        # Extract string from last text block
        last_block = processed_blocks[-1] if processed_blocks else None
        if last_block and last_block.get("type") == "text":
            input_str = last_block.get("text", "")
            preceding_blocks = processed_blocks[:-1]
        else:
            preceding_blocks = processed_blocks

    if input_str is None and options.mode != PromptInputMode.PROMPT:
        raise ValueError(f"Mode: {options.mode} requires a string input.")

    # Process pasted images
    image_content_blocks: list[dict[str, Any]] = []
    if options.pasted_contents:
        for pasted in options.pasted_contents.values():
            if is_valid_image_paste(pasted):
                # Create image block (placeholder for actual resize)
                image_block = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": pasted.media_type or "image/png",
                        "data": pasted.content,
                    },
                }
                image_content_blocks.append(image_block)

    # Bridge-safe slash command override
    effective_skip_slash = options.skip_slash_commands
    if (
        options.bridge_origin
        and input_str
        and input_str.startswith("/")
    ):
        # Check if command is bridge-safe
        cmd_name = parse_slash_command(input_str)
        if cmd_name and is_bridge_safe_command(cmd_name):
            effective_skip_slash = False
        elif cmd_name:
            # Known but unsafe command
            return ProcessUserInputResult(
                messages=[
                    create_user_message(input_str, options.uuid),
                    create_command_input_message(
                        f"<local-command-stdout>{cmd_name} isn't available over Remote Control.</local-command-stdout>"
                    ),
                ],
                should_query=False,
                result_text=f"{cmd_name} isn't available over Remote Control.",
            )

    # Determine if we should extract attachments
    should_extract_attachments = (
        not options.skip_attachments
        and input_str is not None
        and (
            options.mode != PromptInputMode.PROMPT
            or effective_skip_slash
            or not input_str.startswith("/")
        )
    )

    # Load attachment messages
    attachment_messages: list[AttachmentMessage] = []
    if should_extract_attachments:
        attachment_messages = await get_attachment_messages(
            input_str,
            context,
            options.ide_selection,
            options.messages,
            options.query_source,
        )

    # Memory recall - find relevant memories
    memory_attachments: list[AttachmentMessage] = []
    if input_str and should_extract_attachments:
        memory_attachments = await load_relevant_memories(
            input_str,
            context,
        )
        attachment_messages.extend(memory_attachments)

    # Bash commands (! prefix)
    if input_str and options.mode == PromptInputMode.BASH:
        return await process_bash_command(
            input_str,
            preceding_blocks,
            attachment_messages,
            context,
        )

    # Slash commands
    if (
        input_str
        and not effective_skip_slash
        and input_str.startswith("/")
    ):
        return await process_slash_command(
            input_str,
            preceding_blocks,
            image_content_blocks,
            attachment_messages,
            context,
            options.uuid,
            options.is_already_processing,
            can_use_tool,
        )

    # Log agent mentions for analytics
    if input_str and options.mode == PromptInputMode.PROMPT:
        agent_mention = find_agent_mention(attachment_messages)
        if agent_mention:
            # Would log analytics event here
            pass

    # Regular user prompt
    return process_text_prompt(
        normalized_input,
        image_content_blocks,
        attachment_messages,
        options.uuid,
        permission_mode,
        options.is_meta,
    )


# =============================================================================
# Memory Recall
# =============================================================================


async def load_relevant_memories(
    input_str: str,
    context: Any,
    loaded_paths: Optional[set[str]] = None,
) -> list[AttachmentMessage]:
    """Load relevant memories for the input.

    This implements the memory recall logic from TypeScript:
    1. Check if auto memory is enabled
    2. Extract already surfaced paths from messages (deduplication)
    3. Extract recent tools from messages (noise filtering)
    4. Find relevant memory files using model/keyword selection
    5. Convert to attachment messages

    Args:
        input_str: User input string
        context: Tool use context (contains messages list)
        loaded_paths: Already loaded memory paths (external dedup)

    Returns:
        List of attachment messages with memory content
    """
    loaded_paths = loaded_paths or set()

    # Check if auto memory is enabled
    if not is_auto_memory_enabled():
        return []

    # Get memory directory (use cwd from context)
    cwd = context.get_cwd() if hasattr(context, "get_cwd") else None
    memory_dir = get_auto_mem_path(cwd)

    # Ensure directory exists
    await ensure_memory_dir_exists(memory_dir)

    # Extract already surfaced memory paths from messages
    # This prevents re-injecting the same memory files across turns
    messages = getattr(context, "messages", [])
    surfaced_info = collect_surfaced_memories(messages)

    # Combine loaded_paths (external) with surfaced_paths (from messages)
    # The selector spends its 5-slot budget on fresh candidates
    already_surfaced = loaded_paths | surfaced_info.paths

    # Extract recent tools from messages
    # These are tools actively being used - exclude their docs to avoid noise
    recent_tools = extract_recent_tools(messages)

    # Find relevant memories
    relevant_memories = await find_relevant_memories(
        memory_dir,
        input_str,
        already_surfaced,
        max_results=5,
        recent_tools=recent_tools,
    )

    # Convert to attachments
    attachments = get_memory_files_to_attachments(
        relevant_memories,
        loaded_paths,  # Update this set for caller tracking
    )

    # Convert to AttachmentMessage format
    result: list[AttachmentMessage] = []
    for att in attachments:
        if att.get("type") == "nested_memory":
            result.append(
                create_attachment_message(
                    type="nested_memory",
                    content=att.get("content", ""),
                    name=att.get("name", ""),
                    description=att.get("description", ""),
                    path=att.get("path", ""),
                )
            )

    return result


async def load_agent_memory_for_agent(
    agent_type: str,
    scope: AgentMemoryScope,
    context: Any,
    cwd: Optional[str] = None,
) -> Optional[str]:
    """Load agent memory prompt for a specific agent.

    Args:
        agent_type: Agent type identifier
        scope: Memory scope
        context: Tool use context
        cwd: Working directory

    Returns:
        Agent memory prompt or None
    """
    from claude_code_py.memory.agent_memory import load_agent_memory_prompt

    return await load_agent_memory_prompt(agent_type, scope, cwd)


def get_session_memory_content(context: Any) -> Optional[str]:
    """Get session memory content.

    Args:
        context: Tool use context

    Returns:
        Session memory content or None
    """
    session_memory = SessionMemory()

    if not session_memory.exists():
        return None

    return session_memory.read()


# =============================================================================
# Attachment Processing
# =============================================================================


async def get_attachment_messages(
    input_str: Optional[str],
    context: Any,
    ide_selection: Optional[IDESelection] = None,
    messages: Optional[list[Message]] = None,
    query_source: Optional[QuerySource] = None,
) -> list[AttachmentMessage]:
    """Get attachment messages for the input.

    This extracts:
    - Agent mentions (@agent-<name>)
    - IDE selection context
    - File attachments

    Args:
        input_str: User input string
        context: Tool use context
        ide_selection: IDE selection context
        messages: Existing messages
        query_source: Query source

    Returns:
        List of attachment messages
    """
    attachments: list[AttachmentMessage] = []

    if not input_str:
        return attachments

    # Parse agent mentions
    agent_mentions = parse_agent_mentions(input_str)
    for agent_type in agent_mentions:
        attachments.append(
            create_attachment_message(
                type="agent_mention",
                agent_type=agent_type,
                content=input_str,
            )
        )

    # Add IDE selection if present
    if ide_selection and ide_selection.content:
        attachments.append(
            create_attachment_message(
                type="ide_selection",
                file_path=ide_selection.file_path,
                content=ide_selection.content,
                selection_range=ide_selection.selection_range,
            )
        )

    return attachments


def parse_agent_mentions(input_str: str) -> list[str]:
    """Parse agent mentions from input.

    Args:
        input_str: Input string

    Returns:
        List of agent types mentioned
    """
    import re

    # Match @agent-<name> patterns
    pattern = r"@agent-([\w-]+)"
    matches = re.findall(pattern, input_str)

    return list(set(matches))  # Unique agent types


def find_agent_mention(
    attachments: list[AttachmentMessage],
) -> Optional[AttachmentMessage]:
    """Find agent mention attachment.

    Args:
        attachments: List of attachments

    Returns:
        Agent mention attachment or None
    """
    for att in attachments:
        if att.attachment.get("type") == "agent_mention":
            return att
    return None


# =============================================================================
# Command Processing
# =============================================================================


async def process_slash_command(
    command_str: str,
    preceding_blocks: list[dict[str, Any]],
    image_blocks: list[dict[str, Any]],
    attachment_messages: list[AttachmentMessage],
    context: Any,
    uuid: Optional[str] = None,
    is_already_processing: bool = False,
    can_use_tool: Optional[Callable] = None,
) -> ProcessUserInputResult:
    """Process a slash command.

    Args:
        command_str: Command string (starts with /)
        preceding_blocks: Preceding content blocks
        image_blocks: Image content blocks
        attachment_messages: Attachment messages
        context: Tool use context
        uuid: Message UUID
        is_already_processing: Whether already processing
        can_use_tool: Permission checker

    Returns:
        ProcessUserInputResult
    """
    # Parse command
    parsed = parse_slash_command(command_str)
    if not parsed:
        # Invalid command format - treat as regular text
        return ProcessUserInputResult(
            messages=[create_user_message(command_str, uuid)],
            should_query=True,
        )

    command_name = parsed.get("command_name", "")
    args = parsed.get("args", "")

    # Find command in context
    commands = context.options.commands if hasattr(context, "options") else []

    command = find_command(command_name, commands)
    if not command:
        # Unknown command
        return ProcessUserInputResult(
            messages=[
                create_user_message(command_str, uuid),
                create_system_message(
                    f"Unknown command: {command_name}",
                    "warning",
                ),
            ],
            should_query=False,
            result_text=f"Unknown command: {command_name}",
        )

    # Execute command (placeholder - would call actual command handler)
    # For now, return a placeholder result
    return ProcessUserInputResult(
        messages=[create_user_message(command_str, uuid)],
        should_query=False,
        result_text=f"Command {command_name} executed (placeholder)",
    )


async def process_bash_command(
    command_str: str,
    preceding_blocks: list[dict[str, Any]],
    attachment_messages: list[AttachmentMessage],
    context: Any,
) -> ProcessUserInputResult:
    """Process a bash command (! prefix).

    Args:
        command_str: Bash command string
        preceding_blocks: Preceding content blocks
        attachment_messages: Attachment messages
        context: Tool use context

    Returns:
        ProcessUserInputResult
    """
    # Remove ! prefix
    actual_command = command_str[1:] if command_str.startswith("!") else command_str

    # Create bash command attachment
    messages: list[Message] = [
        create_user_message(command_str),
        create_attachment_message(
            type="bash_command",
            content=actual_command,
        ),
    ]

    # Placeholder - would execute via BashTool
    return ProcessUserInputResult(
        messages=messages,
        should_query=False,
        result_text=f"Bash command: {actual_command} (placeholder)",
    )


# =============================================================================
# Text Prompt Processing
# =============================================================================


def process_text_prompt(
    input: Union[str, list[dict[str, Any]]],
    image_blocks: list[dict[str, Any]],
    attachment_messages: list[AttachmentMessage],
    uuid: Optional[str] = None,
    permission_mode: Optional[PermissionMode] = None,
    is_meta: bool = False,
) -> ProcessUserInputResult:
    """Process a regular text prompt.

    Args:
        input: User input
        image_blocks: Image blocks
        attachment_messages: Attachment messages
        uuid: Message UUID
        permission_mode: Permission mode
        is_meta: Whether this is a meta message

    Returns:
        ProcessUserInputResult
    """
    messages: list[Message] = []

    # Create user message
    if isinstance(input, str):
        content = input
    else:
        content = input

    user_msg = create_user_message(
        content,
        uuid,
        is_meta=is_meta,
    )
    messages.append(user_msg)

    # Add attachments
    messages.extend(attachment_messages)

    return ProcessUserInputResult(
        messages=messages,
        should_query=True,
    )


# =============================================================================
# Hook Execution
# =============================================================================


async def execute_user_prompt_submit_hooks(
    input_str: str,
    context: Any,
) -> list[HookResult]:
    """Execute UserPromptSubmit hooks.

    Args:
        input_str: User input string
        context: Tool use context

    Returns:
        List of hook results
    """
    # Placeholder - would load and execute hooks from settings
    # For now, return empty list
    return []


# =============================================================================
# Command Helpers
# =============================================================================


def parse_slash_command(input_str: str) -> Optional[dict[str, Any]]:
    """Parse a slash command.

    Args:
        input_str: Input string

    Returns:
        Parsed command dict or None
    """
    import re

    if not input_str.startswith("/"):
        return None

    # Match /command [args] format
    match = re.match(r"^/([\w-]+)(?:\s+(.*))?$", input_str)
    if not match:
        return None

    return {
        "command_name": match.group(1),
        "args": match.group(2) or "",
    }


def find_command(name: str, commands: list[Any]) -> Optional[Any]:
    """Find a command by name.

    Args:
        name: Command name
        commands: List of commands

    Returns:
        Command or None
    """
    for cmd in commands:
        cmd_name = getattr(cmd, "name", None) or cmd.get("name", "")
        if cmd_name == name:
            return cmd

        # Check aliases
        aliases = getattr(cmd, "aliases", None) or cmd.get("aliases", [])
        if name in aliases:
            return cmd

    return None


def is_bridge_safe_command(command_name: str) -> bool:
    """Check if a command is safe to execute over bridge.

    Args:
        command_name: Command name

    Returns:
        True if bridge-safe
    """
    # Commands that are safe to execute remotely
    safe_commands = {
        "help",
        "model",
        "clear",
        "compact",
        "config",
        "doctor",
        "status",
        "init",
        "mcp",
    }

    return command_name.lower() in safe_commands


# =============================================================================
# Image Helpers
# =============================================================================


def is_valid_image_paste(pasted: PastedContent) -> bool:
    """Check if pasted content is a valid image.

    Args:
        pasted: Pasted content

    Returns:
        True if valid image paste
    """
    if not pasted.media_type:
        return False

    valid_types = {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }

    return pasted.media_type in valid_types


# =============================================================================
# Message Factories
# =============================================================================


def create_user_message(
    content: Union[str, list[dict[str, Any]]],
    uuid_str: Optional[str] = None,
    is_meta: bool = False,
) -> UserMessage:
    """Create a user message.

    Args:
        content: Message content
        uuid_str: Message UUID
        is_meta: Whether meta message

    Returns:
        UserMessage
    """
    return UserMessage(
        uuid=uuid_str or str(uuid.uuid4()),
        message={"role": "user", "content": content},
        is_meta=is_meta,
    )


def create_system_message(
    content: str,
    subtype: Optional[str] = None,
) -> SystemMessage:
    """Create a system message.

    Args:
        content: Message content
        subtype: System message subtype

    Returns:
        SystemMessage
    """
    return SystemMessage(
        uuid=str(uuid.uuid4()),
        subtype=subtype,
        content=content,
    )


def create_attachment_message(
    type: str,
    content: Any = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    path: Optional[str] = None,
    file_path: Optional[str] = None,
    agent_type: Optional[str] = None,
    selection_range: Optional[dict[str, int]] = None,
    hook_name: Optional[str] = None,
    tool_use_id: Optional[str] = None,
    hook_event: Optional[str] = None,
) -> AttachmentMessage:
    """Create an attachment message.

    Args:
        type: Attachment type
        content: Attachment content
        name: Memory name (for nested_memory)
        description: Memory description
        path: Memory path
        file_path: File path (for ide_selection)
        agent_type: Agent type (for agent_mention)
        selection_range: Selection range
        hook_name: Hook name
        tool_use_id: Tool use ID
        hook_event: Hook event

    Returns:
        AttachmentMessage
    """
    attachment: dict[str, Any] = {"type": type}

    if content is not None:
        attachment["content"] = content
    if name is not None:
        attachment["name"] = name
    if description is not None:
        attachment["description"] = description
    if path is not None:
        attachment["path"] = path
    if file_path is not None:
        attachment["file_path"] = file_path
    if agent_type is not None:
        attachment["agent_type"] = agent_type
    if selection_range is not None:
        attachment["selection_range"] = selection_range
    if hook_name is not None:
        attachment["hook_name"] = hook_name
    if tool_use_id is not None:
        attachment["tool_use_id"] = tool_use_id
    if hook_event is not None:
        attachment["hook_event"] = hook_event

    return AttachmentMessage(
        uuid=str(uuid.uuid4()),
        attachment=attachment,
    )


def create_command_input_message(content: str) -> SystemMessage:
    """Create a command input message.

    Args:
        content: Message content

    Returns:
        SystemMessage with local_command subtype
    """
    return SystemMessage(
        uuid=str(uuid.uuid4()),
        subtype="local_command",
        content=content,
    )


# =============================================================================
# Memory Prompt Integration
# =============================================================================


async def build_memory_prompt_for_input(
    context: Any,
) -> Optional[str]:
    """Build the memory prompt for user input processing.

    This loads:
    - Auto memory (project-level memory)
    - Session memory (current session summary)

    Args:
        context: Tool use context

    Returns:
        Memory prompt string or None
    """
    if not is_auto_memory_enabled():
        return None

    # Get memory directory (use cwd from context)
    cwd = context.get_cwd() if hasattr(context, "get_cwd") else None
    memory_dir = get_auto_mem_path(cwd)
    await ensure_memory_dir_exists(memory_dir)

    # Build auto memory prompt
    auto_prompt = build_memory_prompt(
        display_name="auto memory",
        memory_dir=memory_dir,
    )

    # Add session memory if exists
    session_content = get_session_memory_content(context)
    if session_content:
        auto_prompt += f"\n\n## Session Memory\n\n{session_content}"

    return auto_prompt