"""Error formatting utilities for tool execution.

Ported from: src/utils/toolErrors.ts
"""

from __future__ import annotations

import errno
from typing import Optional, Union

from pydantic import ValidationError as PydanticValidationError

from claude_code_py.tool.result import ToolError, ShellError, TimeoutError
from claude_code_py.utils.abort_controller import AbortError

# Maximum error message length before truncation
MAX_ERROR_LENGTH = 10000
HALF_TRUNCATION_LENGTH = 5000

# Default message for user interrupts in tool results
INTERRUPT_MESSAGE_FOR_TOOL_USE = "Command interrupted by user"


def format_error(error: Union[Exception, object]) -> str:
    """Format an error for inclusion in tool_result messages.

    Handles special error types (AbortError, ShellError) and truncates
    messages exceeding MAX_ERROR_LENGTH by keeping first/last 5000 chars.

    Args:
        error: The error to format

    Returns:
        Formatted error string suitable for model context
    """
    if isinstance(error, AbortError):
        return str(error) or INTERRUPT_MESSAGE_FOR_TOOL_USE

    if not isinstance(error, Exception):
        return str(error)

    parts = _get_error_parts(error)
    full_message = (
        "\n".join(p for p in parts if p).strip()
        or "Command failed with no output"
    )

    if len(full_message) <= MAX_ERROR_LENGTH:
        return full_message

    # Truncate: keep first 5000 + last 5000 chars
    start = full_message[:HALF_TRUNCATION_LENGTH]
    end = full_message[-HALF_TRUNCATION_LENGTH:]
    truncated_count = len(full_message) - MAX_ERROR_LENGTH
    return f"{start}\n\n... [{truncated_count} characters truncated] ...\n\n{end}"


def _get_error_parts(error: Exception) -> list[str]:
    """Extract message parts from an error.

    ShellError gets special formatting (exit code, stderr, stdout).
    Other errors collect message + optional stderr/stdout duck-type fields.

    Args:
        error: The exception

    Returns:
        List of non-empty string parts
    """
    if isinstance(error, ShellError):
        parts: list[str] = [
            f"Exit code {error.exit_code}",
            INTERRUPT_MESSAGE_FOR_TOOL_USE if error.interrupted else "",
            error.stderr,
            error.stdout,
        ]
        return parts

    parts = [str(error)]
    if hasattr(error, "stderr") and isinstance(error.stderr, str):
        parts.append(error.stderr)
    if hasattr(error, "stdout") and isinstance(error.stdout, str):
        parts.append(error.stdout)
    return parts


def format_validation_error(
    tool_name: str,
    error: PydanticValidationError,
) -> str:
    """Format a Pydantic validation error into a human-readable message.

    Categorizes errors into:
    - Missing required parameters
    - Unexpected parameters
    - Type mismatches

    Args:
        tool_name: Name of the tool that failed validation
        error: Pydantic ValidationError

    Returns:
        Human-readable error message for model context
    """
    missing_params = []
    unexpected_params = []
    type_mismatch_params = []

    for err in error.errors():
        path_str = _format_validation_path(err.get("loc", []))

        if err.get("type") == "missing":
            missing_params.append(path_str)
        elif err.get("type") == "extra_forbidden" or err.get("type") == "unexpected":
            # Pydantic v2 uses "extra_forbidden"
            keys = err.get("input", {})
            if isinstance(keys, dict):
                unexpected_params.extend(keys.keys())
            else:
                unexpected_params.append(path_str)
        elif err.get("type") in ("int_type", "float_type", "bool_type",
                                  "string_type", "list_type", "dict_type",
                                  "model_type", "is_instance"):
            # Type mismatch
            expected = err.get("type", "unknown").replace("_type", "")
            msg = err.get("msg", "")
            # Try to extract "received" type from message
            received = "unknown"
            import re
            match = re.search(r"received (\w+)", msg)
            if match:
                received = match.group(1)
            elif "Input should be" in msg:
                # Pydantic v2 format: "Input should be a valid integer"
                expected = msg.split("valid")[-1].strip().split()[0] if "valid" in msg else expected

            type_mismatch_params.append({
                "param": path_str,
                "expected": expected,
                "received": received,
            })

    error_parts = []

    for param in missing_params:
        error_parts.append(f"The required parameter `{param}` is missing")

    for param in unexpected_params:
        error_parts.append(f"An unexpected parameter `{param}` was provided")

    for info in type_mismatch_params:
        error_parts.append(
            f"The parameter `{info['param']}` type is expected as "
            f"`{info['expected']}` but provided as `{info['received']}`"
        )

    if error_parts:
        count = len(error_parts)
        label = "issues" if count > 1 else "issue"
        return f"{tool_name} failed due to the following {label}:\n" + "\n".join(error_parts)

    # Fallback to raw error message
    return str(error)


def _format_validation_path(path: tuple) -> str:
    """Format a Pydantic validation error path into a readable string.

    Args:
        path: Tuple of path segments (strings and ints)

    Returns:
        Dot-separated path string like "args.command" or "items[0]"
    """
    if not path:
        return ""

    parts = []
    for i, segment in enumerate(path):
        if isinstance(segment, int):
            parts.append(f"{parts.pop() if parts else ''}[{segment}]")
        else:
            if i == 0:
                parts.append(str(segment))
            else:
                parts.append(f"{parts[-1]}.{segment}" if parts else str(segment))
                # Replace last element with combined path
                if len(parts) > 1:
                    parts[-2] = parts[-1]
                    parts.pop()

    # Simpler approach: just join with dots, brackets for ints
    result = ""
    for i, segment in enumerate(path):
        if isinstance(segment, int):
            result += f"[{segment}]"
        else:
            if i == 0:
                result = str(segment)
            else:
                result += f".{segment}"
    return result


def classify_tool_error(error: Union[Exception, object]) -> str:
    """Classify a tool error for telemetry-safe reporting.

    Maps errors to stable, telemetry-safe strings:
    - Filesystem errno codes (ENOENT, EACCES, etc.) -> "Error:{code}"
    - Named error types with stable .name -> use name
    - Fallback -> "Error" or "UnknownError"

    Args:
        error: The error to classify

    Returns:
        Telemetry-safe error classification string
    """
    if not isinstance(error, Exception):
        return "UnknownError"

    # Check for errno code (filesystem errors)
    errno_code = _get_errno_code(error)
    if errno_code:
        return f"Error:{errno_code}"

    # Use error class name if it's meaningful (>3 chars, not just "Error")
    name = type(error).__name__
    if name and name != "Error" and len(name) > 3:
        return name[:60]

    return "Error"


def _get_errno_code(error: Exception) -> Optional[str]:
    """Extract errno code from an exception if present.

    Args:
        error: Exception to check

    Returns:
        Errno code string (e.g., "ENOENT", "EACCES") or None
    """
    if hasattr(error, "errno") and isinstance(error.errno, int):
        try:
            return errno.errorcode.get(error.errno)
        except (AttributeError, KeyError):
            pass

    # Also check for OSError-style errors
    if isinstance(error, OSError) and error.errno:
        try:
            return errno.errorcode.get(error.errno)
        except (AttributeError, KeyError):
            pass

    return None


def short_error_stack(error: Union[Exception, object], max_frames: int = 5) -> str:
    """Truncate stack trace to top N frames for model context.

    Saves context tokens by keeping only the most relevant frames.

    Args:
        error: The error
        max_frames: Maximum number of frames to keep

    Returns:
        Truncated stack trace string
    """
    if not isinstance(error, Exception):
        return str(error)

    if not hasattr(error, "__traceback__") or error.__traceback__ is None:
        return str(error)

    import traceback
    tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
    if len(tb_lines) <= max_frames + 1:
        return "".join(tb_lines)

    # Keep header (error message line) + first max_frames frames
    header = tb_lines[0]
    frames = tb_lines[1:max_frames + 1]
    return header + "".join(frames) + f"... [{len(tb_lines) - 1 - max_frames} frames truncated]"