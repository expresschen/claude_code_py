"""API retry logic with exponential backoff and jitter.

Ported from: src/services/api/withRetry.ts

Key features:
- Exponential backoff with jitter (BASE_DELAY * 2^(attempt-1) + random)
- Max 10 retries default
- Retry on: 429, 401, 408, 5xx, connection errors
- 529 consecutive tracking with model fallback support
- Context overflow adjustment (max_tokens override)
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional, TypeVar

from claude_code_py.utils.abort_controller import AbortError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Retry constants (matching TypeScript)
DEFAULT_MAX_RETRIES = 10
BASE_DELAY_MS = 500
MAX_DELAY_MS = 32000
PERSISTENT_MAX_BACKOFF_MS = 300000  # 5 minutes
PERSISTENT_RESET_CAP_MS = 300000
HEARTBEAT_INTERVAL_MS = 30000  # 30 seconds
SHORT_RETRY_THRESHOLD_MS = 10000
DEFAULT_FAST_MODE_FALLBACK_HOLD_MS = 15000
MIN_COOLDOWN_MS = 5000
MAX_529_RETRIES = 3
FLOOR_OUTPUT_TOKENS = 4096


@dataclass
class RetryContext:
    """Context carried across retry attempts."""

    model: str
    thinking_config: Optional[Any] = None
    max_tokens_override: Optional[int] = None
    fast_mode: Optional[bool] = None


@dataclass
class RetryOptions:
    """Options for retry behavior."""

    max_retries: int = DEFAULT_MAX_RETRIES
    model: str = ""
    signal: Optional[Any] = None  # AbortSignal
    query_source: Optional[str] = None
    fallback_model: Optional[str] = None
    thinking_config: Optional[Any] = None
    initial_consecutive_529_errors: int = 0


class CannotRetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, original_error: Exception, context: RetryContext):
        super().__init__(f"Cannot retry after exhausting attempts: {original_error}")
        self.original_error = original_error
        self.retry_context = context


class FallbackTriggeredError(Exception):
    """Raised when consecutive 529 errors trigger model fallback."""

    def __init__(self, primary_model: str, fallback_model: str):
        super().__init__(
            f"Model fallback triggered: {primary_model} -> {fallback_model}"
        )
        self.primary_model = primary_model
        self.fallback_model = fallback_model


def get_retry_delay(
    attempt: int,
    retry_after_header: Optional[str] = None,
    max_delay_ms: int = MAX_DELAY_MS,
) -> float:
    """Calculate retry delay with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (1-based)
        retry_after_header: Optional Retry-After header value in seconds
        max_delay_ms: Maximum delay in milliseconds

    Returns:
        Delay in milliseconds
    """
    if retry_after_header:
        try:
            seconds = int(retry_after_header)
            if seconds > 0:
                return seconds * 1000
        except (ValueError, TypeError):
            pass

    base_delay = min(BASE_DELAY_MS * (2 ** (attempt - 1)), max_delay_ms)
    jitter = random.random() * 0.25 * base_delay
    return base_delay + jitter


def should_retry(status_code: Optional[int], error_type: Optional[str] = None) -> bool:
    """Determine if an API error should be retried.

    Args:
        status_code: HTTP status code
        error_type: Error type string (e.g., "connection", "timeout")

    Returns:
        True if the error should be retried
    """
    if error_type == "connection":
        return True
    if error_type == "timeout":
        return True

    if not status_code:
        return False

    # Retry on these status codes
    retryable_codes = {
        401,  # Auth error (clear cache and retry)
        408,  # Request timeout
        409,  # Lock timeout
        429,  # Rate limit
        500,  # Internal server error
        502,  # Bad gateway
        503,  # Service unavailable
        529,  # Overloaded
    }
    return status_code in retryable_codes or status_code >= 500


def is_529_error(status_code: Optional[int], error_message: Optional[str] = None) -> bool:
    """Check if error is a 529 (overloaded) error.

    Args:
        status_code: HTTP status code
        error_message: Error message string

    Returns:
        True if this is a 529 error
    """
    if status_code == 529:
        return True
    if error_message and "overloaded" in error_message.lower():
        return True
    return False


def is_foreground_query_source(query_source: Optional[str]) -> bool:
    """Check if query source should retry on 529 errors.

    Foreground sources (main thread, SDK, agents, compact) retry.
    Background sources (suggestions, classifiers) bail immediately.

    Args:
        query_source: Query source string

    Returns:
        True if this source should retry on 529
    """
    from claude_code_py.constants import is_main_thread_source

    if not query_source:
        return True

    foreground_sources = {
        "repl_main_thread",
        "sdk",
        "agent",
        "compact",
        "subagent",
    }
    return query_source in foreground_sources or is_main_thread_source(query_source)


async def with_retry(
    operation: Callable[[int, RetryContext], Any],
    options: RetryOptions,
) -> Any:
    """Execute an operation with retry logic.

    Wraps an async operation with configurable retry behavior including
    exponential backoff, jitter, 529 tracking, and model fallback.

    Args:
        operation: Async callable taking (attempt, context) -> result
        options: Retry configuration

    Returns:
        Result of the operation

    Raises:
        CannotRetryError: When all retries are exhausted
        FallbackTriggeredError: When 529 errors trigger model fallback
        AbortError: When the abort signal fires
    """
    max_retries = options.max_retries
    context = RetryContext(
        model=options.model,
        thinking_config=options.thinking_config,
    )

    consecutive_529_errors = options.initial_consecutive_529_errors
    last_error: Optional[Exception] = None
    persistent_attempt = 0

    for attempt in range(1, max_retries + 2):  # +2 because first attempt is not a retry
        # Check abort signal
        if options.signal and options.signal.aborted:
            raise AbortError("Operation aborted during retry")

        try:
            return await operation(attempt, context)

        except AbortError:
            raise  # Don't retry user aborts

        except FallbackTriggeredError:
            raise  # Don't retry fallback triggers

        except CannotRetryError:
            raise  # Don't retry exhaustion

        except Exception as error:
            last_error = error

            # Extract status code and error info
            status_code = getattr(error, "status_code", None)
            error_message = str(error)

            # Check for 529 errors
            if is_529_error(status_code, error_message):
                # Non-foreground sources bail immediately on 529
                if not is_foreground_query_source(options.query_source):
                    raise CannotRetryError(error, context)

                consecutive_529_errors += 1
                if consecutive_529_errors >= MAX_529_RETRIES:
                    if options.fallback_model:
                        raise FallbackTriggeredError(options.model, options.fallback_model)

            # Check if we should retry
            if attempt > max_retries:
                raise CannotRetryError(error, context)

            # Determine error type for should_retry
            error_type = None
            if isinstance(error, (ConnectionError, OSError)):
                error_type = "connection"
            elif isinstance(error, asyncio.TimeoutError):
                error_type = "timeout"

            if not should_retry(status_code, error_type):
                raise CannotRetryError(error, context)

            # Calculate delay
            retry_after = getattr(error, "headers", {})
            if isinstance(retry_after, dict):
                retry_after = retry_after.get("retry-after")
            else:
                retry_after = None

            delay_ms = get_retry_delay(attempt, retry_after)

            # Log retry attempt
            logger.debug(
                f"Retrying API call (attempt {attempt}/{max_retries}) "
                f"after {delay_ms:.0f}ms: {classify_api_error(error)}"
            )

            # Sleep with abort check
            await _sleep_with_abort_check(delay_ms, options.signal)


def classify_api_error(error: Exception) -> str:
    """Classify an API error for logging.

    Args:
        error: The exception

    Returns:
        Human-readable error classification
    """
    status_code = getattr(error, "status_code", None)
    if status_code:
        if status_code == 429:
            return "rate_limit"
        if status_code == 529:
            return "overloaded"
        if status_code == 401:
            return "auth_error"
        if status_code == 408:
            return "timeout"
        if status_code >= 500:
            return f"server_error_{status_code}"

    if isinstance(error, (ConnectionError, OSError)):
        return "connection_error"
    if isinstance(error, asyncio.TimeoutError):
        return "timeout"

    return type(error).__name__


async def _sleep_with_abort_check(
    delay_ms: float,
    signal: Optional[Any] = None,
) -> None:
    """Sleep for the specified duration, checking abort signal periodically.

    Args:
        delay_ms: Sleep duration in milliseconds
        signal: Optional abort signal to check

    Raises:
        AbortError: If the signal fires during sleep
    """
    delay_seconds = delay_ms / 1000.0
    check_interval = 0.5  # Check every 500ms

    elapsed = 0.0
    while elapsed < delay_seconds:
        if signal and signal.aborted:
            raise AbortError("Operation aborted during retry sleep")

        sleep_time = min(check_interval, delay_seconds - elapsed)
        await asyncio.sleep(sleep_time)
        elapsed += sleep_time