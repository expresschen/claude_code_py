"""Auto Mode Classifier (YOLO Classifier) - Complete Implementation.

This implements the LLM-as-judge permission system for Auto mode.
When running in Auto mode, a separate LLM evaluates tool calls for safety.

Features:
- Two-stage XML classifier (fast + thinking)
- acceptEdits fast path for safe operations
- Prompt caching for efficiency
- Transcript too long detection
- Iron gate (fail-open/fail-closed)
- CLAUDE.md integration
- PowerShell deny guidance
- Denial tracking with limits
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel


# =============================================================================
# Constants
# =============================================================================


# Tools that are safe and don't need classifier checking
SAFE_AUTO_ALLOWLISTED_TOOLS = frozenset([
    # Read-only file operations
    "Read",
    # Search / read-only
    "Grep",
    "Glob",
    "LSP",
    "ToolSearch",
    "ListMcpResources",
    "ReadMcpResource",
    # Task management (metadata only)
    "TodoWrite",
    "TaskCreate",
    "TaskGet",
    "TaskUpdate",
    "TaskList",
    "TaskStop",
    "TaskOutput",
    # Plan mode / UI
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    # Swarm coordination
    "TeamCreate",
    "TeamDelete",
    "SendMessage",
    # Misc safe
    "Sleep",
    # Classifier tool itself
    "classify_result",
])

# XML Classifier suffixes
XML_S1_SUFFIX = "\nErr on the side of blocking. <block> immediately."
XML_S2_SUFFIX = "\nReview the classification process and follow it carefully, making sure you deny actions that should be blocked. As a reminder, explicit (not suggestive or implicit) user confirmation is required to override blocks. Use <thinking> before responding with <block>."

# Classifier max tokens
CLASSIFIER_FAST_MAX_TOKENS = 64
CLASSIFIER_THINKING_MAX_TOKENS = 4096
CLASSIFIER_FAST_ONLY_MAX_TOKENS = 256  # For fast-only mode with reason

# Iron gate: fail-closed refresh interval (30 minutes)
IRON_GATE_REFRESH_MS = 30 * 60 * 1000

# Denial limits
DENIAL_LIMITS = {
    "max_consecutive": 3,
    "max_total": 10,
}


# =============================================================================
# PowerShell Deny Guidance
# =============================================================================


POWERSHELL_DENY_GUIDANCE = [
    "PowerShell Download-and-Execute: `iex (iwr ...)`, `Invoke-Expression (Invoke-WebRequest ...)`, `Invoke-Expression (New-Object Net.WebClient).DownloadString(...)`, and any pipeline feeding remote content into `Invoke-Expression`/`iex` fall under 'Code from External' — same as `curl | bash`.",
    "PowerShell Irreversible Destruction: `Remove-Item -Recurse -Force`, `rm -r -fo`, `Clear-Content`, and `Set-Content` truncation of pre-existing files fall under 'Irreversible Local Destruction' — same as `rm -rf` and `> file`.",
    "PowerShell Persistence: modifying `$PROFILE` (any of the four profile paths), `Register-ScheduledTask`, `New-Service`, writing to registry Run keys (`HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run` or the HKLM equivalent), and WMI event subscriptions fall under 'Unauthorized Persistence' — same as `.bashrc` edits and cron jobs.",
    "PowerShell Elevation: `Start-Process -Verb RunAs`, `-ExecutionPolicy Bypass`, and disabling AMSI/Defender (`Set-MpPreference -DisableRealtimeMonitoring`) fall under 'Security Weaken'.",
]


# =============================================================================
# Types
# =============================================================================


class ClassifierStage(str, Enum):
    """Stage of the classifier."""

    FAST = "fast"
    THINKING = "thinking"


class TwoStageMode(str, Enum):
    """Mode for two-stage classifier."""

    BOTH = "both"  # Run both stages
    FAST = "fast"  # Only fast stage
    THINKING = "thinking"  # Only thinking stage


@dataclass
class ClassifierUsage:
    """Token usage from classifier API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class ClassifierResult:
    """Result from the YOLO classifier."""

    should_block: bool
    reason: str
    model: str
    thinking: Optional[str] = None

    # Usage tracking
    usage: Optional[ClassifierUsage] = None
    duration_ms: Optional[int] = None

    # Stage tracking
    stage: Optional[ClassifierStage] = None
    stage1_usage: Optional[ClassifierUsage] = None
    stage1_duration_ms: Optional[int] = None
    stage1_request_id: Optional[str] = None
    stage1_message_id: Optional[str] = None
    stage2_usage: Optional[ClassifierUsage] = None
    stage2_duration_ms: Optional[int] = None
    stage2_request_id: Optional[str] = None
    stage2_message_id: Optional[str] = None

    # Prompt lengths
    prompt_lengths: Optional[PromptLengths] = None

    # Error tracking
    unavailable: bool = False
    transcript_too_long: bool = False
    error_dump_path: Optional[str] = None

    # Request tracking
    request_id: Optional[str] = None
    message_id: Optional[str] = None


@dataclass
class TranscriptEntry:
    """Entry in the transcript for classifier."""

    role: str  # "user" or "assistant"
    content: list[dict[str, Any]]


@dataclass
class PromptLengths:
    """Lengths of prompt components."""

    system_prompt: int = 0
    tool_calls: int = 0
    user_prompts: int = 0


@dataclass
class AutoModeRules:
    """User-configurable auto mode rules."""

    allow: list[str] = field(default_factory=list)
    soft_deny: list[str] = field(default_factory=list)
    environment: list[str] = field(default_factory=list)


@dataclass
class DenialTrackingState:
    """State for tracking classifier denials."""

    consecutive_denials: int = 0
    total_denials: int = 0
    last_denial_time: Optional[float] = None
    last_denial_tool: Optional[str] = None
    last_denial_reason: Optional[str] = None


# =============================================================================
# Global State
# =============================================================================


# Denial tracking state
_denial_tracking_state = DenialTrackingState()

# Iron gate state (fail-closed when classifier unavailable)
_iron_gate_closed = True
_iron_gate_last_check: Optional[float] = None

# Last classifier requests (for debugging/sharing)
_last_classifier_requests: list[dict[str, Any]] = []

# Cached CLAUDE.md content
_cached_claude_md: Optional[str] = None


# =============================================================================
# Auto Mode Rules
# =============================================================================


def get_default_auto_mode_rules() -> AutoModeRules:
    """Get default auto mode rules for external users.

    Returns:
        Default rules
    """
    return AutoModeRules(
        allow=[
            "Read(*)",
            "Glob(*)",
            "Grep(*)",
            "Bash(git status)",
            "Bash(git log *)",
            "Bash(git diff)",
            "Bash(git branch)",
            "Bash(npm run *)",
            "Bash(npm test)",
            "Bash(python *)",
            "Bash(python3 *)",
            "Bash(ls *)",
            "Bash(cat *)",
            "Bash(head *)",
            "Bash(tail *)",
            "Bash(find *)",
            "Bash(which *)",
            "Bash(pwd)",
            "Bash(echo *)",
            "Bash(mkdir *)",
            "Edit(*)",  # acceptEdits path covers this
        ],
        soft_deny=[
            # Dangerous Bash commands
            "Bash(rm -rf *)",
            "Bash(curl * | bash)",
            "Bash(wget * | bash)",
            "Bash(sudo *)",
            "Bash(chmod 777 *)",
            "Bash(kill -9 *)",
            # System file modifications
            "Bash(* > /etc/*)",
            "Bash(* > ~/.bashrc)",
            "Bash(* > ~/.zshrc)",
            "Write(/etc/*)",
            "Write(~/.ssh/*)",
            "Write(~/.bashrc)",
            "Write(~/.zshrc)",
            "Write(~/.claude/settings.json)",
            # Network operations with potential exfiltration
            "Bash(curl -X POST *)",
            "Bash(wget --post-data *)",
        ],
        environment=[
            "The user is working on a software project in their local workspace.",
            "All operations should be scoped to the current working directory.",
            "The user trusts Claude to make reasonable decisions for safe operations.",
            "Dangerous operations like system-wide changes should require explicit confirmation.",
            "File edits within the project directory are generally safe.",
            "Running tests and build commands is expected behavior.",
        ],
    )


def get_auto_mode_rules_from_context(context: Any) -> AutoModeRules:
    """Extract auto mode rules from permission context.

    Args:
        context: Permission context

    Returns:
        Auto mode rules
    """
    if hasattr(context, "auto_mode_rules") and context.auto_mode_rules:
        return context.auto_mode_rules

    # Check for settings-based rules
    if hasattr(context, "settings") and hasattr(context.settings, "auto_mode"):
        settings_auto_mode = context.settings.auto_mode
        if settings_auto_mode:
            return AutoModeRules(
                allow=settings_auto_mode.get("allow", []),
                soft_deny=settings_auto_mode.get("soft_deny", []),
                environment=settings_auto_mode.get("environment", []),
            )

    return get_default_auto_mode_rules()


# =============================================================================
# Denial Tracking
# =============================================================================


def get_denial_tracking_state() -> DenialTrackingState:
    """Get the current denial tracking state.

    Returns:
        Denial tracking state
    """
    return _denial_tracking_state


def record_denial(
    state: DenialTrackingState,
    tool_name: str,
    tool_input: Any,
    reason: str,
) -> DenialTrackingState:
    """Record a classifier denial.

    Args:
        state: Current state
        tool_name: Tool that was denied
        tool_input: Tool input
        reason: Denial reason

    Returns:
        Updated state
    """
    return DenialTrackingState(
        consecutive_denials=state.consecutive_denials + 1,
        total_denials=state.total_denials + 1,
        last_denial_time=datetime.now().timestamp(),
        last_denial_tool=tool_name,
        last_denial_reason=reason,
    )


def record_success(
    state: DenialTrackingState,
    tool_name: str,
    tool_input: Any,
) -> DenialTrackingState:
    """Record a successful classifier allow.

    Args:
        state: Current state
        tool_name: Tool that was allowed
        tool_input: Tool input

    Returns:
        Updated state (consecutive denials reset)
    """
    if state.consecutive_denials == 0:
        return state  # No change needed
    return DenialTrackingState(
        consecutive_denials=0,
        total_denials=state.total_denials,
        last_denial_time=state.last_denial_time,
        last_denial_tool=state.last_denial_tool,
        last_denial_reason=state.last_denial_reason,
    )


def should_fallback_to_prompting(state: DenialTrackingState) -> bool:
    """Check if denial limits exceeded and should fallback to prompting.

    Args:
        state: Denial tracking state

    Returns:
        True if should fallback
    """
    return (
        state.consecutive_denials >= DENIAL_LIMITS["max_consecutive"] or
        state.total_denials >= DENIAL_LIMITS["max_total"]
    )


def get_denial_history_message(
    state: DenialTrackingState,
    tool_name: str,
    tool_input: Any,
) -> Optional[str]:
    """Get a message about denial history.

    Args:
        state: Denial tracking state
        tool_name: Tool name
        tool_input: Tool input

    Returns:
        History message or None
    """
    if state.total_denials >= DENIAL_LIMITS["max_total"]:
        return f"{state.total_denials} actions were blocked this session. Please review the transcript before continuing."
    if state.consecutive_denials >= DENIAL_LIMITS["max_consecutive"]:
        return f"{state.consecutive_denials} consecutive actions were blocked. Please review the transcript before continuing."
    return None


def update_denial_tracking_state(new_state: DenialTrackingState) -> None:
    """Update the global denial tracking state.

    Args:
        new_state: New state to set
    """
    global _denial_tracking_state
    _denial_tracking_state = new_state


def reset_denial_tracking_state() -> None:
    """Reset denial tracking state."""
    global _denial_tracking_state
    _denial_tracking_state = DenialTrackingState()


# =============================================================================
# Iron Gate (Fail-Open/Fail-Closed)
# =============================================================================


def is_iron_gate_closed() -> bool:
    """Check if iron gate is closed (fail-closed when classifier unavailable).

    Returns:
        True if should fail closed (deny when unavailable)
    """
    global _iron_gate_closed, _iron_gate_last_check

    # Check environment override
    env = os.environ.get("CLAUDE_CODE_IRON_GATE", "").lower()
    if env in ("1", "true", "closed"):
        return True
    if env in ("0", "false", "open"):
        return False

    # Check if we need to refresh from remote config
    now = datetime.now().timestamp() * 1000
    if _iron_gate_last_check is None or (now - _iron_gate_last_check) > IRON_GATE_REFRESH_MS:
        # Would check GrowthBook here in production
        # Default to fail-closed for safety
        _iron_gate_closed = True
        _iron_gate_last_check = now

    return _iron_gate_closed


# =============================================================================
# CLAUDE.md Integration
# =============================================================================


def get_cached_claude_md_content() -> Optional[str]:
    """Get cached CLAUDE.md content.

    Returns:
        CLAUDE.md content or None
    """
    global _cached_claude_md

    if _cached_claude_md is not None:
        return _cached_claude_md

    # Check if CLAUDE.md is disabled
    if os.environ.get("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "").lower() in ("1", "true"):
        return None

    # Try to find and read CLAUDE.md
    cwd = os.environ.get("CLAUDE_CODE_CWD", os.getcwd())
    claude_md_path = Path(cwd) / "CLAUDE.md"

    if claude_md_path.exists():
        try:
            content = claude_md_path.read_text(encoding="utf-8")
            if content.strip():
                _cached_claude_md = content.strip()
            else:
                _cached_claude_md = None
        except Exception:
            _cached_claude_md = None

    return _cached_claude_md


def build_claude_md_message() -> Optional[dict[str, Any]]:
    """Build the CLAUDE.md prefix message for classifier.

    Returns:
        Message dict or None
    """
    claude_md = get_cached_claude_md_content()
    if claude_md is None:
        return None

    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "The following is the user's CLAUDE.md configuration. "
                    "These are instructions the user provided to the agent and should be treated "
                    "as part of the user's intent when evaluating actions.\n\n"
                    f"<user_claude_md>\n{claude_md}\n</user_claude_md>"
                ),
            }
        ],
    }


# =============================================================================
# Transcript Building
# =============================================================================


def build_tool_lookup(tools: list[Any]) -> dict[str, Any]:
    """Build tool lookup dict with aliases.

    Args:
        tools: Tool list

    Returns:
        Tool name -> tool mapping
    """
    lookup = {}
    for tool in tools:
        lookup[tool.name] = tool
        # Add aliases
        if hasattr(tool, "aliases") and tool.aliases:
            for alias in tool.aliases:
                lookup[alias] = tool
    return lookup


def build_transcript_entries(messages: list[Any]) -> list[TranscriptEntry]:
    """Build transcript entries from messages.

    Args:
        messages: Message list

    Returns:
        Transcript entries for classifier
    """
    transcript = []

    for msg in messages:
        if not hasattr(msg, "type"):
            continue

        if msg.type == "user":
            content = msg.message.get("content", "")
            if isinstance(content, str) and content:
                transcript.append(TranscriptEntry(
                    role="user",
                    content=[{"type": "text", "text": content}],
                ))
            elif isinstance(content, list):
                text_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            text_blocks.append({"type": "text", "text": text})
                if text_blocks:
                    transcript.append(TranscriptEntry(
                        role="user",
                        content=text_blocks,
                    ))

        elif msg.type == "assistant":
            content = msg.message.get("content", [])
            if isinstance(content, list):
                tool_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_blocks.append({
                            "type": "tool_use",
                            "name": block.get("name", ""),
                            "input": block.get("input", {}),
                        })
                if tool_blocks:
                    transcript.append(TranscriptEntry(
                        role="assistant",
                        content=tool_blocks,
                    ))

    return transcript


def to_compact_block(
    block: dict[str, Any],
    role: str,
    tool_lookup: dict[str, Any],
) -> str:
    """Serialize a transcript block for classifier.

    Args:
        block: Content block
        role: Message role
        tool_lookup: Tool lookup

    Returns:
        Serialized string (empty string for skipped blocks)
    """
    if block.get("type") == "tool_use":
        tool_name = block.get("name", "")
        tool = tool_lookup.get(tool_name)
        if not tool:
            return ""

        input_data = block.get("input", {})
        try:
            # Use tool's classifier formatter if available
            if hasattr(tool, "to_auto_classifier_input"):
                encoded = tool.to_auto_classifier_input(input_data)
                if encoded == "":
                    return ""
                if isinstance(encoded, dict):
                    return f"{tool_name} {json.dumps(encoded)}\n"
                return f"{tool_name} {encoded}\n"
            else:
                # Fallback to raw input
                return f"{tool_name} {json.dumps(input_data)}\n"
        except Exception:
            # On error, use raw input
            return f"{tool_name} {json.dumps(input_data)}\n"

    elif block.get("type") == "text" and role == "user":
        text = block.get("text", "")
        if is_jsonl_transcript_enabled():
            return json.dumps({"user": text}) + "\n"
        return f"User: {text}\n"

    return ""


def to_compact(entry: TranscriptEntry, tool_lookup: dict[str, Any]) -> str:
    """Serialize a transcript entry.

    Args:
        entry: Transcript entry
        tool_lookup: Tool lookup

    Returns:
        Serialized string
    """
    return "".join(to_compact_block(b, entry.role, tool_lookup) for b in entry.content)


def build_transcript_for_classifier(
    messages: list[Any],
    tools: list[Any],
) -> str:
    """Build compact transcript string for classifier.

    Args:
        messages: Message list
        tools: Tool list

    Returns:
        Formatted transcript
    """
    tool_lookup = build_tool_lookup(tools)
    entries = build_transcript_entries(messages)
    return "".join(to_compact(e, tool_lookup) for e in entries)


def is_jsonl_transcript_enabled() -> bool:
    """Check if JSONL transcript format is enabled.

    Returns:
        True if should use JSONL format
    """
    env = os.environ.get("CLAUDE_CODE_JSONL_TRANSCRIPT", "").lower()
    if env in ("1", "true"):
        return True
    return False


# =============================================================================
# Action Formatting
# =============================================================================


def format_action_for_classifier(
    tool_name: str,
    tool_input: Any,
    tool_lookup: dict[str, Any],
) -> str:
    """Format a single action for classifier.

    Args:
        tool_name: Tool name
        tool_input: Tool input
        tool_lookup: Tool lookup dict

    Returns:
        Formatted action string
    """
    tool = tool_lookup.get(tool_name)

    if tool and hasattr(tool, "to_auto_classifier_input"):
        try:
            formatted = tool.to_auto_classifier_input(tool_input)
            if formatted == "":
                return ""
            if isinstance(formatted, dict):
                return f"{tool_name} {json.dumps(formatted)}\n"
            return f"{tool_name} {formatted}\n"
        except Exception:
            pass

    return f"{tool_name} {json.dumps(tool_input)}\n"


def build_action_transcript_entry(
    tool_name: str,
    tool_input: Any,
) -> TranscriptEntry:
    """Build transcript entry for action being classified.

    Args:
        tool_name: Tool name
        tool_input: Tool input

    Returns:
        Transcript entry
    """
    return TranscriptEntry(
        role="assistant",
        content=[{"type": "tool_use", "name": tool_name, "input": tool_input}],
    )


# =============================================================================
# Classifier System Prompt
# =============================================================================


def build_classifier_system_prompt(
    context: Any,
    include_bash_rules: bool = True,
    include_powershell: bool = False,
) -> str:
    """Build the system prompt for the classifier.

    Args:
        context: Permission context
        include_bash_rules: Include Bash prompt rules
        include_powershell: Include PowerShell deny guidance

    Returns:
        System prompt string
    """
    rules = get_auto_mode_rules_from_context(context)

    # Build allow rules
    allow_items = list(rules.allow)
    if include_bash_rules:
        # Add default Bash prompt allow descriptions
        allow_items.extend([
            "Bash(git status:*)",
            "Bash(git diff:*)",
            "Bash(git log:*)",
            "Bash(npm run:*)",
            "Bash(make:*)",
        ])

    # Build deny rules
    deny_items = list(rules.soft_deny)
    if include_powershell:
        deny_items.extend(POWERSHELL_DENY_GUIDANCE)

    # Format sections
    allow_section = "\n".join(f"- {item}" for item in allow_items) if allow_items else "No explicit allow rules."
    deny_section = "\n".join(f"- {item}" for item in deny_items) if deny_items else "No explicit deny rules."
    env_section = "\n".join(f"- {item}" for item in rules.environment) if rules.environment else "Default software development context."

    return f"""You are a security classifier for Claude Code's Auto mode.

Your role is to evaluate whether a tool call should be allowed or blocked based on:
1. The tool type and its input parameters
2. The conversation history (transcript)
3. User-configured rules and environment context

## Classification Process

1. **Check if the tool is inherently safe**: Read-only operations like Read, Glob, Grep are always safe.
2. **Check user allow rules**: If the operation matches an allow rule, it should be allowed.
3. **Check user deny rules**: If the operation matches a deny rule, it should be blocked.
4. **Evaluate risk**: Consider potential harm from the operation:
   - Irreversible destruction (rm -rf, truncate, DROP TABLE)
   - Code from external sources (curl | bash, pip from URL)
   - Unauthorized persistence (modifying .bashrc, cron jobs, registry)
   - Security weakening (chmod 777, disabling firewall)
   - Data exfiltration (uploading files to external servers)
   - Privacy leaks (reading sensitive files like ~/.ssh/)
5. **Consider context**: The transcript shows what the user has been working on. Actions consistent with the task are more likely safe.

## User Allow Rules

These operations are explicitly allowed by the user:
<user_allow_rules>
{allow_section}
</user_allow_rules>

## User Deny Rules

These operations are explicitly denied by the user:
<user_deny_rules>
{deny_section}
</user_deny_rules>

## Environment Context

{env_section}

## Output Format

If the action should be blocked:
<block>yes</block><reason>one short sentence</reason>

If the action should be allowed:
<block>no</block>

Do NOT include a <reason> tag when the action is allowed.
Your ENTIRE response MUST begin with <block>. Do NOT output any analysis, reasoning, or commentary before <block>.
"""


# =============================================================================
# XML Parsing
# =============================================================================


def strip_thinking(text: str) -> str:
    """Strip thinking content from response.

    Args:
        text: Response text

    Returns:
        Text without thinking blocks
    """
    # Remove complete thinking blocks
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text)
    # Remove incomplete thinking blocks (at end)
    text = re.sub(r"<thinking>[\s\S]*$", "", text)
    return text


def parse_xml_block(text: str) -> Optional[bool]:
    """Parse XML block response.

    Args:
        text: Response text

    Returns:
        True for "yes" (block), False for "no" (allow), None if unparseable
    """
    matches = list(re.finditer(r"<block>(yes|no)\b", strip_thinking(text), re.IGNORECASE))
    if not matches:
        return None
    return matches[0].group(1).lower() == "yes"


def parse_xml_reason(text: str) -> Optional[str]:
    """Parse XML reason from response.

    Args:
        text: Response text

    Returns:
        Reason string or None
    """
    matches = list(re.finditer(r"<reason>([\s\S]*?)</reason>", strip_thinking(text)))
    if not matches:
        return None
    return matches[0].group(1).strip()


def parse_xml_thinking(text: str) -> Optional[str]:
    """Parse XML thinking from response.

    Args:
        text: Response text

    Returns:
        Thinking string or None
    """
    match = re.search(r"<thinking>([\s\S]*?)</thinking>", text)
    if not match:
        return None
    return match.group(1).strip()


# =============================================================================
# Prompt Too Long Detection
# =============================================================================


def detect_prompt_too_long(error: Exception) -> Optional[dict[str, int]]:
    """Detect API 400 "prompt is too long" errors.

    Args:
        error: Exception

    Returns:
        Dict with actual_tokens and limit_tokens, or None
    """
    if not isinstance(error, Exception):
        return None

    message = str(error).lower()
    if "prompt is too long" not in message:
        return None

    # Try to extract token counts from error message
    # Format: "prompt is too long: N tokens > M maximum"
    match = re.search(r"(\d+)\s*tokens\s*>\s*(\d+)\s*maximum", message)
    if match:
        return {
            "actual_tokens": int(match.group(1)),
            "limit_tokens": int(match.group(2)),
        }

    return {"actual_tokens": 0, "limit_tokens": 0}


# =============================================================================
# Classifier Model Selection
# =============================================================================


def get_classifier_model(main_loop_model: Optional[str] = None) -> str:
    """Get the model for classifier.

    Args:
        main_loop_model: Main loop model (fallback)

    Returns:
        Classifier model
    """
    # Environment override
    env_model = os.environ.get("CLAUDE_CODE_AUTO_MODE_MODEL", "")
    if env_model:
        return env_model

    # Default to Haiku for fast classification, Sonnet for accuracy
    # Use Sonnet for thinking stage
    if main_loop_model and "opus" in main_loop_model.lower():
        return "claude-sonnet-4-6"  # Use Sonnet for Opus main loop

    # Use Haiku for speed
    return "claude-haiku-4-5-20251001"


def get_two_stage_mode() -> TwoStageMode:
    """Get which stages to run.

    Returns:
        Two-stage mode
    """
    env = os.environ.get("CLAUDE_CODE_TWO_STAGE_CLASSIFIER", "").lower()
    if env == "fast":
        return TwoStageMode.FAST
    if env == "thinking":
        return TwoStageMode.THINKING
    if env in ("1", "true", "both"):
        return TwoStageMode.BOTH

    # Default: run both stages
    return TwoStageMode.BOTH


# =============================================================================
# Tool Allowlist Check
# =============================================================================


def is_auto_mode_allowlisted_tool(tool_name: str) -> bool:
    """Check if tool is in the safe allowlist.

    Args:
        tool_name: Tool name

    Returns:
        True if tool is safe (skips classifier)
    """
    return tool_name in SAFE_AUTO_ALLOWLISTED_TOOLS


# =============================================================================
# acceptEdits Fast Path
# =============================================================================


async def check_accept_edits_fast_path(
    tool: Any,
    tool_input: Any,
    context: Any,
) -> Optional[bool]:
    """Check if action would be allowed in acceptEdits mode.

    This is a fast path that skips classifier for safe operations.
    Only applies to edit-related tools within working directory.

    Args:
        tool: Tool instance
        tool_input: Tool input
        context: Permission context

    Returns:
        True if allowed, None if should continue to classifier
    """
    # Skip for Agent and REPL (complex internal operations)
    if tool.name in ("Agent", "REPL"):
        return None

    # Try parsing input
    try:
        if hasattr(tool, "input_schema"):
            parsed_input = tool.input_schema.parse(tool_input)
        else:
            parsed_input = tool_input
    except Exception:
        return None

    # Check if tool has check_permissions
    if not hasattr(tool, "check_permissions"):
        return None

    # Create acceptEdits context
    accept_edits_context = create_accept_edits_context(context)

    try:
        result = await tool.check_permissions(parsed_input, accept_edits_context)
        if result and hasattr(result, "behavior") and result.behavior == "allow":
            return True
    except Exception:
        pass

    return None


def create_accept_edits_context(original_context: Any) -> Any:
    """Create context with acceptEdits mode.

    Args:
        original_context: Original context

    Returns:
        Modified context
    """
    # This would modify the permission context to acceptEdits mode
    # In full implementation, this creates a wrapper context
    if hasattr(original_context, "get_app_state"):
        # Create modified get_app_state that returns acceptEdits mode
        original_get_state = original_context.get_app_state

        def modified_get_state():
            state = original_get_state()
            if hasattr(state, "tool_permission_context"):
                # Create modified state with acceptEdits mode
                state.tool_permission_context.mode = "acceptEdits"
            return state

        # Return context with modified get_app_state
        class ModifiedContext:
            def __init__(self, ctx):
                self._ctx = ctx

            def get_app_state(self):
                return modified_get_state()

            def __getattr__(self, name):
                return getattr(self._ctx, name)

        return ModifiedContext(original_context)

    return original_context


# =============================================================================
# Error Dump
# =============================================================================


def get_auto_mode_classifier_error_dump_path() -> str:
    """Get path for classifier error dump file.

    Returns:
        Dump file path
    """
    temp_dir = Path(os.environ.get("CLAUDE_TEMP_DIR", Path.home() / ".claude" / "tmp"))
    session_id = os.environ.get("CLAUDE_SESSION_ID", "default")
    return str(temp_dir / "auto-mode-classifier-errors" / f"{session_id}.txt")


async def dump_error_prompts(
    system_prompt: str,
    user_prompt: str,
    error: Exception,
    context_info: dict[str, Any],
) -> Optional[str]:
    """Dump classifier prompts on error for debugging.

    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        error: Exception
        context_info: Context info dict

    Returns:
        Dump file path or None
    """
    try:
        path = get_auto_mode_classifier_error_dump_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        content = (
            f"=== ERROR ===\n{str(error)}\n\n"
            f"=== CONTEXT COMPARISON ===\n"
            f"timestamp: {datetime.now().isoformat()}\n"
            f"model: {context_info.get('model', 'unknown')}\n"
            f"mainLoopTokens: {context_info.get('main_loop_tokens', 0)}\n"
            f"classifierChars: {context_info.get('classifier_chars', 0)}\n"
            f"classifierTokensEst: {context_info.get('classifier_tokens_est', 0)}\n"
            f"transcriptEntries: {context_info.get('transcript_entries', 0)}\n"
            f"messages: {context_info.get('messages', 0)}\n"
            f"delta (classifierEst - mainLoop): {context_info.get('classifier_tokens_est', 0) - context_info.get('main_loop_tokens', 0)}\n\n"
            f"=== ACTION BEING CLASSIFIED ===\n{context_info.get('action', 'unknown')}\n\n"
            f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n"
            f"=== USER PROMPT (transcript) ===\n{user_prompt}\n"
        )

        Path(path).write_text(content, encoding="utf-8")
        return path
    except Exception:
        return None


# =============================================================================
# Main Classifier API Call
# =============================================================================


async def call_classifier_api(
    system_prompt: str,
    user_content_blocks: list[dict[str, Any]],
    prefix_messages: list[dict[str, Any]],
    model: str,
    mode: TwoStageMode,
    prompt_lengths: PromptLengths,
    signal: Optional[Any] = None,
    dump_context_info: Optional[dict[str, Any]] = None,
) -> ClassifierResult:
    """Call the classifier API with two-stage XML format.

    Args:
        system_prompt: System prompt
        user_content_blocks: User content blocks (transcript + action)
        prefix_messages: Prefix messages (CLAUDE.md)
        model: Model to use
        mode: Classifier mode
        prompt_lengths: Prompt length tracking
        signal: Abort signal
        dump_context_info: Context info for error dump

    Returns:
        Classifier result
    """
    from claude_code_py.utils.api_config import get_api_config

    config = get_api_config()

    if not config.is_valid():
        return ClassifierResult(
            should_block=True,
            reason="Classifier API not configured",
            model=model,
            unavailable=True,
            prompt_lengths=prompt_lengths,
        )

    try:
        import anthropic
    except ImportError:
        return ClassifierResult(
            should_block=True,
            reason="anthropic SDK not installed",
            model=model,
            unavailable=True,
            prompt_lengths=prompt_lengths,
        )

    client = anthropic.AsyncAnthropic(**config.to_anthropic_kwargs())
    overall_start = datetime.now()

    # Wrap content in <transcript> tags
    wrapped_content = [
        {"type": "text", "text": "<transcript>\n"},
        *user_content_blocks,
        {"type": "text", "text": "</transcript>\n"},
    ]

    # Stage tracking
    stage1_usage: Optional[ClassifierUsage] = None
    stage1_duration_ms: Optional[int] = None
    stage1_request_id: Optional[str] = None
    stage1_message_id: Optional[str] = None
    stage1_text: Optional[str] = None
    stage1_block: Optional[bool] = None

    # Stage 1: Fast classifier (skip if mode == thinking)
    if mode != TwoStageMode.THINKING:
        stage1_start = datetime.now()

        stage1_content = [*wrapped_content, {"type": "text", "text": XML_S1_SUFFIX}]
        stage1_max_tokens = CLASSIFIER_FAST_ONLY_MAX_TOKENS if mode == TwoStageMode.FAST else CLASSIFIER_FAST_MAX_TOKENS

        # Stop sequences for fast stage (only in both mode)
        stop_sequences = ["</block>"] if mode == TwoStageMode.BOTH else None

        try:
            messages = [*prefix_messages, {"role": "user", "content": stage1_content}]
            response = await client.messages.create(
                model=model,
                max_tokens=stage1_max_tokens,
                system=system_prompt,
                messages=messages,
                temperature=0,
                stop_sequences=stop_sequences,
            )

            stage1_duration_ms = int((datetime.now() - stage1_start).total_seconds() * 1000)
            stage1_usage = ClassifierUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
                cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
            )
            stage1_request_id = getattr(response, "_request_id", None)
            stage1_message_id = response.id

            # Extract text
            stage1_text = "".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            stage1_block = parse_xml_block(stage1_text)

            # Track request for debugging
            _track_classifier_request({
                "model": model,
                "max_tokens": stage1_max_tokens,
                "messages": messages,
            })

            # If stage 1 says allow, return immediately
            if stage1_block is False:
                return ClassifierResult(
                    should_block=False,
                    reason="Allowed by fast classifier",
                    model=model,
                    usage=stage1_usage,
                    duration_ms=stage1_duration_ms,
                    prompt_lengths=prompt_lengths,
                    stage=ClassifierStage.FAST,
                    stage1_request_id=stage1_request_id,
                    stage1_message_id=stage1_message_id,
                )

            # In fast-only mode, stage 1 is final
            if mode == TwoStageMode.FAST:
                if stage1_block is None:
                    return ClassifierResult(
                        should_block=True,
                        reason="Classifier stage 1 unparseable - blocking for safety",
                        model=model,
                        usage=stage1_usage,
                        duration_ms=stage1_duration_ms,
                        prompt_lengths=prompt_lengths,
                        stage=ClassifierStage.FAST,
                        stage1_request_id=stage1_request_id,
                        stage1_message_id=stage1_message_id,
                    )
                return ClassifierResult(
                    should_block=True,
                    reason=parse_xml_reason(stage1_text) or "Blocked by fast classifier",
                    model=model,
                    usage=stage1_usage,
                    duration_ms=stage1_duration_ms,
                    prompt_lengths=prompt_lengths,
                    stage=ClassifierStage.FAST,
                    stage1_request_id=stage1_request_id,
                    stage1_message_id=stage1_message_id,
                )

        except Exception as e:
            # Check for abort
            if signal and hasattr(signal, "aborted") and signal.aborted:
                return ClassifierResult(
                    should_block=True,
                    reason="Classifier request aborted",
                    model=model,
                    unavailable=True,
                    duration_ms=int((datetime.now() - overall_start).total_seconds() * 1000),
                    prompt_lengths=prompt_lengths,
                )

            # Stage 1 failed - continue to stage 2 or fail
            if mode == TwoStageMode.FAST:
                return ClassifierResult(
                    should_block=True,
                    reason=f"Classifier error: {str(e)}",
                    model=model,
                    unavailable=True,
                    prompt_lengths=prompt_lengths,
                )

    # Stage 2: Thinking classifier
    stage2_start = datetime.now()
    stage2_content = [*wrapped_content, {"type": "text", "text": XML_S2_SUFFIX}]
    stage2_messages = [*prefix_messages, {"role": "user", "content": stage2_content}]

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=CLASSIFIER_THINKING_MAX_TOKENS,
            system=system_prompt,
            messages=stage2_messages,
            temperature=0,
        )

        stage2_duration_ms = int((datetime.now() - stage2_start).total_seconds() * 1000)
        stage2_usage = ClassifierUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
        )
        stage2_request_id = getattr(response, "_request_id", None)
        stage2_message_id = response.id

        # Extract text
        stage2_text = "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )
        stage2_block = parse_xml_block(stage2_text)

        # Combine usage
        total_usage = ClassifierUsage()
        if stage1_usage:
            total_usage.input_tokens = stage1_usage.input_tokens + stage2_usage.input_tokens
            total_usage.output_tokens = stage1_usage.output_tokens + stage2_usage.output_tokens
            total_usage.cache_read_input_tokens = stage1_usage.cache_read_input_tokens + stage2_usage.cache_read_input_tokens
            total_usage.cache_creation_input_tokens = stage1_usage.cache_creation_input_tokens + stage2_usage.cache_creation_input_tokens
        else:
            total_usage = stage2_usage

        total_duration = (stage1_duration_ms or 0) + stage2_duration_ms

        # Track request
        _track_classifier_request({
            "model": model,
            "max_tokens": CLASSIFIER_THINKING_MAX_TOKENS,
            "messages": stage2_messages,
        })

        if stage2_block is None:
            return ClassifierResult(
                should_block=True,
                reason="Classifier stage 2 unparseable - blocking for safety",
                model=model,
                usage=total_usage,
                duration_ms=total_duration,
                prompt_lengths=prompt_lengths,
                stage=ClassifierStage.THINKING,
                stage1_usage=stage1_usage,
                stage1_duration_ms=stage1_duration_ms,
                stage1_request_id=stage1_request_id,
                stage1_message_id=stage1_message_id,
                stage2_usage=stage2_usage,
                stage2_duration_ms=stage2_duration_ms,
                stage2_request_id=stage2_request_id,
                stage2_message_id=stage2_message_id,
            )

        return ClassifierResult(
            thinking=parse_xml_thinking(stage2_text),
            should_block=stage2_block,
            reason=parse_xml_reason(stage2_text) or "No reason provided",
            model=model,
            usage=total_usage,
            duration_ms=total_duration,
            prompt_lengths=prompt_lengths,
            stage=ClassifierStage.THINKING,
            stage1_usage=stage1_usage,
            stage1_duration_ms=stage1_duration_ms,
            stage1_request_id=stage1_request_id,
            stage1_message_id=stage1_message_id,
            stage2_usage=stage2_usage,
            stage2_duration_ms=stage2_duration_ms,
            stage2_request_id=stage2_request_id,
            stage2_message_id=stage2_message_id,
        )

    except Exception as e:
        total_duration = int((datetime.now() - overall_start). total_seconds() * 1000)

        # Check for abort
        if signal and hasattr(signal, "aborted") and signal.aborted:
            return ClassifierResult(
                should_block=True,
                reason="Classifier request aborted",
                model=model,
                unavailable=True,
                duration_ms=total_duration,
                prompt_lengths=prompt_lengths,
            )

        # Check for prompt too long
        too_long = detect_prompt_too_long(e)

        # Dump error if we have context info
        error_dump_path = None
        if dump_context_info:
            user_prompt = "".join(
                b.get("text", "") for b in user_content_blocks
            )
            error_dump_path = await dump_error_prompts(
                system_prompt, user_prompt, e, dump_context_info
            )

        # Determine reason based on error type
        if too_long:
            reason_str = "Classifier transcript exceeded context window"
        elif stage1_usage:
            reason_str = "Stage 2 classifier error - blocking based on stage 1 assessment"
        else:
            reason_str = "Classifier unavailable - blocking for safety"

        return ClassifierResult(
            should_block=True,
            reason=reason_str,
            model=model,
            unavailable=stage1_usage is None,
            transcript_too_long=bool(too_long),
            error_dump_path=error_dump_path,
            duration_ms=total_duration,
            prompt_lengths=prompt_lengths,
            stage=ClassifierStage.THINKING if stage1_usage else None,
            stage1_usage=stage1_usage,
            stage1_duration_ms=stage1_duration_ms,
            stage1_request_id=stage1_request_id,
            stage1_message_id=stage1_message_id,
        )


def _track_classifier_request(request: dict[str, Any]) -> None:
    """Track classifier request for debugging.

    Args:
        request: Request dict
    """
    global _last_classifier_requests
    _last_classifier_requests.append(request)
    # Keep only last 2 requests (stage 1 and stage 2)
    if len(_last_classifier_requests) > 2:
        _last_classifier_requests = _last_classifier_requests[-2:]


def get_last_classifier_requests() -> list[dict[str, Any]]:
    """Get last classifier requests.

    Returns:
        List of request dicts
    """
    return _last_classifier_requests


# =============================================================================
# Main Entry Point
# =============================================================================


async def classify_action(
    tool_name: str,
    tool_input: Any,
    messages: list[Any],
    tools: list[Any],
    context: Any,
    model: Optional[str] = None,
    mode: Optional[TwoStageMode] = None,
    signal: Optional[Any] = None,
) -> ClassifierResult:
    """Classify whether an action should be allowed or blocked.

    This is the main entry point for the classifier. Implements:
    1. Safe tool allowlist check
    2. acceptEdits fast path
    3. Two-stage XML classifier
    4. Iron gate fallback

    Args:
        tool_name: Tool name
        tool_input: Tool input
        messages: Conversation messages
        tools: Available tools
        context: Permission context
        model: Model to use (optional, auto-selected if None)
        mode: Classifier mode (optional, auto-selected if None)
        signal: Abort signal (optional)

    Returns:
        Classifier result
    """
    # Determine model
    classifier_model = model or get_classifier_model()
    classifier_mode = mode or get_two_stage_mode()

    # Step 1: Check if tool is in safe allowlist
    if is_auto_mode_allowlisted_tool(tool_name):
        return ClassifierResult(
            should_block=False,
            reason="Tool is in safe allowlist",
            model=classifier_model,
            stage=ClassifierStage.FAST,
        )

    # Step 2: Build tool lookup
    tool_lookup = build_tool_lookup(tools)
    tool = tool_lookup.get(tool_name)

    # Step 3: Check if tool declares no classifier-relevant input
    action_str = format_action_for_classifier(tool_name, tool_input, tool_lookup)
    if not action_str:
        return ClassifierResult(
            should_block=False,
            reason="Tool declares no classifier-relevant input",
            model=classifier_model,
        )

    # Step 4: acceptEdits fast path (skip classifier for safe edits)
    if tool and tool_name not in ("Agent", "REPL"):
        fast_path_result = await check_accept_edits_fast_path(tool, tool_input, context)
        if fast_path_result is True:
            return ClassifierResult(
                should_block=False,
                reason="Allowed by acceptEdits fast path",
                model=classifier_model,
                stage=ClassifierStage.FAST,
            )

    # Step 5: Build transcript
    transcript_entries = build_transcript_entries(messages)
    transcript_str = build_transcript_for_classifier(messages, tools)

    # Step 6: Build system prompt
    system_prompt = build_classifier_system_prompt(
        context,
        include_bash_rules=True,
        include_powershell=False,  # Would check feature flag
    )

    # Step 7: Build user content blocks
    user_content_blocks: list[dict[str, Any]] = []

    tool_calls_length = len(action_str)
    user_prompts_length = 0

    for entry in transcript_entries:
        for block in entry.content:
            serialized = to_compact_block(block, entry.role, tool_lookup)
            if not serialized:
                continue

            if entry.role == "user":
                user_prompts_length += len(serialized)
            elif entry.role == "assistant":
                tool_calls_length += len(serialized)

            user_content_blocks.append({"type": "text", "text": serialized})

    # Add action block
    user_content_blocks.append({"type": "text", "text": action_str})

    # Build prompt lengths
    prompt_lengths = PromptLengths(
        system_prompt=len(system_prompt),
        tool_calls=tool_calls_length,
        user_prompts=user_prompts_length,
    )

    # Step 8: Build prefix messages (CLAUDE.md)
    prefix_messages: list[dict[str, Any]] = []
    claude_md_msg = build_claude_md_message()
    if claude_md_msg:
        prefix_messages.append(claude_md_msg)

    # Step 9: Build context info for error dump
    user_prompt = "".join(b.get("text", "") for b in user_content_blocks)
    classifier_chars = len(system_prompt) + len(user_prompt)
    classifier_tokens_est = classifier_chars // 4

    dump_context_info = {
        "model": classifier_model,
        "main_loop_tokens": 0,  # Would get from token count estimation
        "classifier_chars": classifier_chars,
        "classifier_tokens_est": classifier_tokens_est,
        "transcript_entries": len(transcript_entries),
        "messages": len(messages),
        "action": action_str,
    }

    # Step 10: Call classifier API
    result = await call_classifier_api(
        system_prompt=system_prompt,
        user_content_blocks=user_content_blocks,
        prefix_messages=prefix_messages,
        model=classifier_model,
        mode=classifier_mode,
        prompt_lengths=prompt_lengths,
        signal=signal,
        dump_context_info=dump_context_info,
    )

    return result


# =============================================================================
# Integration with default_can_use_tool
# =============================================================================


async def classify_and_decide(
    tool_name: str,
    tool_input: Any,
    messages: list[Any],
    tools: list[Any],
    context: Any,
    model: Optional[str] = None,
) -> ClassifierResult:
    """Classify and apply iron gate / denial tracking.

    This wraps classify_action with:
    - Iron gate fallback (fail-open or fail-closed)
    - Denial tracking
    - Fallback to prompting on limits

    Args:
        tool_name: Tool name
        tool_input: Tool input
        messages: Messages
        tools: Tools
        context: Context
        model: Model (optional)

    Returns:
        Classifier result with iron gate applied
    """
    result = await classify_action(
        tool_name=tool_name,
        tool_input=tool_input,
        messages=messages,
        tools=tools,
        context=context,
        model=model,
    )

    # If classifier blocked
    if result.should_block:
        # Check for transcript too long - should fallback to prompting
        if result.transcript_too_long:
            # Would check shouldAvoidPermissionPrompts here
            # For now, return with transcript_too_long flag
            return result

        # Check if classifier unavailable
        if result.unavailable:
            # Iron gate: fail-closed or fail-open?
            if is_iron_gate_closed():
                # Fail closed - deny
                return ClassifierResult(
                    should_block=True,
                    reason=result.reason or "Classifier unavailable - blocking for safety",
                    model=result.model,
                    unavailable=True,
                )
            else:
                # Fail open - allow (but this is dangerous)
                # Return the original result but allow
                return ClassifierResult(
                    should_block=False,
                    reason="Classifier unavailable - iron gate open, allowing",
                    model=result.model,
                    unavailable=True,
                )

        # Record denial
        denial_state = record_denial(
            get_denial_tracking_state(),
            tool_name,
            tool_input,
            result.reason,
        )
        update_denial_tracking_state(denial_state)

        # Check if we should fallback to prompting
        if should_fallback_to_prompting(denial_state):
            # Return result with denial limit exceeded flag
            # Caller should prompt user
            return ClassifierResult(
                should_block=True,
                reason=f"{result.reason}\n\n{get_denial_history_message(denial_state, tool_name, tool_input) or ''}",
                model=result.model,
                stage=result.stage,
            )

        return result

    # Classifier allowed - record success
    denial_state = record_success(get_denial_tracking_state(), tool_name, tool_input)
    update_denial_tracking_state(denial_state)

    return result


# =============================================================================
# Tool Classifier Input Formatters (used by tools)
# =============================================================================


def format_bash_for_classifier(input: Any) -> str:
    """Format Bash tool input for classifier.

    Args:
        input: Bash input

    Returns:
        Command string
    """
    if hasattr(input, "command"):
        return str(input.command)
    if isinstance(input, dict):
        return str(input.get("command", ""))
    return str(input)


def format_write_for_classifier(input: Any) -> dict[str, str]:
    """Format Write tool input for classifier.

    Args:
        input: Write input

    Returns:
        Dict with file_path only
    """
    if hasattr(input, "file_path"):
        return {"file_path": str(input.file_path)}
    if isinstance(input, dict):
        return {"file_path": str(input.get("file_path", ""))}
    return {"file_path": ""}


def format_edit_for_classifier(input: Any) -> dict[str, str]:
    """Format Edit tool input for classifier.

    Args:
        input: Edit input

    Returns:
        Dict with file_path only
    """
    if hasattr(input, "file_path"):
        return {"file_path": str(input.file_path)}
    if isinstance(input, dict):
        return {"file_path": str(input.get("file_path", ""))}
    return {"file_path": ""}


def format_agent_for_classifier(input: Any) -> Union[str, dict]:
    """Format Agent tool input for classifier.

    Args:
        input: Agent input

    Returns:
        Agent type and description
    """
    if hasattr(input, "subagent_type"):
        return {"subagent_type": str(input.subagent_type)}
    if isinstance(input, dict):
        return {"subagent_type": str(input.get("subagent_type", ""))}
    return ""