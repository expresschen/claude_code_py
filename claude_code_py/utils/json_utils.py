"""JSON utilities with better error handling."""

from __future__ import annotations

import json
from typing import Any, Optional, TypeVar

T = TypeVar("T")


class JSONParseError(Exception):
    """Error parsing JSON."""

    def __init__(self, message: str, source: Optional[str] = None):
        super().__init__(message)
        self.source = source


def json_parse(
    text: str,
    *,
    strict: bool = False,
) -> Any:
    """Parse JSON string with better error handling.

    Args:
        text: JSON string to parse
        strict: If True, raise on trailing content

    Returns:
        Parsed JSON value

    Raises:
        JSONParseError: If parsing fails
    """
    try:
        if strict:
            decoder = json.JSONDecoder()
            result, end = decoder.raw_decode(text)
            if end < len(text):
                raise JSONParseError(
                    f"Trailing content after JSON at position {end}",
                    source=text[: end + 50],
                )
            return result
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise JSONParseError(
            f"JSON parse error at position {e.pos}: {e.msg}",
            source=text[: e.pos + 50] if e.pos else text[:50],
        ) from e


def json_stringify(
    value: Any,
    *,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> str:
    """Serialize value to JSON string.

    Args:
        value: Value to serialize
        indent: Indentation level (None for compact)
        ensure_ascii: If True, escape non-ASCII characters
        sort_keys: If True, sort dictionary keys

    Returns:
        JSON string
    """
    return json.dumps(
        value,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        default=_json_default,
    )


def _json_default(obj: Any) -> Any:
    """Default JSON serializer for non-standard types."""
    if hasattr(obj, "model_dump"):
        # Pydantic model
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        # Dataclass or regular object
        return obj.__dict__
    if hasattr(obj, "to_dict"):
        # Custom to_dict method
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def try_json_parse(text: str) -> tuple[Any, Optional[Exception]]:
    """Try to parse JSON, returning (result, error) tuple.

    Args:
        text: JSON string to parse

    Returns:
        Tuple of (parsed_value or None, error or None)
    """
    try:
        return json_parse(text), None
    except Exception as e:
        return None, e


def extract_json_field(
    text: str,
    field: str,
) -> Optional[Any]:
    """Extract a field from a JSON string.

    Args:
        text: JSON string
        field: Field name to extract

    Returns:
        Field value or None if not found
    """
    value, error = try_json_parse(text)
    if error or not isinstance(value, dict):
        return None
    return value.get(field)


def extract_last_json_field(
    text: str,
    field: str,
) -> Optional[Any]:
    """Extract the last occurrence of a JSON field in text.

    Useful for extracting from logs or multi-line output.

    Args:
        text: Text potentially containing JSON
        field: Field name to extract

    Returns:
        Field value from last JSON containing the field, or None
    """
    lines = text.strip().split("\n")
    result = None

    for line in reversed(lines):
        value, _ = try_json_parse(line.strip())
        if isinstance(value, dict) and field in value:
            result = value[field]
            break

    return result