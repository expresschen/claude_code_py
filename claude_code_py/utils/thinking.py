"""Thinking configuration utilities.

Based on TypeScript implementation in utils/thinking.ts and utils/context.ts.

This module provides:
- ThinkingConfig type for configuring extended thinking
- Model support detection for thinking features
- Default thinking budget calculation
- Support for 3rd party model capability overrides
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional

from claude_code_py.utils.model import get_canonical_name, get_api_provider


# =============================================================================
# Thinking Config Type
# =============================================================================


@dataclass
class ThinkingConfig:
    """Configuration for extended thinking mode.

    Attributes:
        type: "adaptive", "enabled", or "disabled"
        budget_tokens: Token budget for "enabled" mode (must be < max_tokens)
    """

    type: Literal["adaptive", "enabled", "disabled"] = "disabled"
    budget_tokens: Optional[int] = None

    def to_api_param(self) -> Optional[dict]:
        """Convert to API parameter format.

        Returns:
            Dict for API thinking parameter, or None if disabled
        """
        if self.type == "disabled":
            return {"type": "disabled"}
        elif self.type == "adaptive":
            return {"type": "adaptive"}
        elif self.type == "enabled":
            result = {"type": "enabled"}
            if self.budget_tokens is not None:
                result["budget_tokens"] = self.budget_tokens
            return result
        return None


# =============================================================================
# 3P Model Capability Override
# =============================================================================

# Environment variable tiers for 3P model capability overrides
_TIERS = [
    {
        "model_env_var": "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "capabilities_env_var": "ANTHROPIC_DEFAULT_OPUS_MODEL_SUPPORTED_CAPABILITIES",
    },
    {
        "model_env_var": "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "capabilities_env_var": "ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES",
    },
    {
        "model_env_var": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "capabilities_env_var": "ANTHROPIC_DEFAULT_HAIKU_MODEL_SUPPORTED_CAPABILITIES",
    },
]


def get_3p_model_capability_override(
    model: str,
    capability: str,
) -> Optional[bool]:
    """Check whether a 3P model capability override is set.

    This allows configuring thinking support for non-Claude models via
    environment variables.

    Args:
        model: Model name
        capability: Capability name (e.g., "thinking", "adaptive_thinking")

    Returns:
        True/False if override is set, None otherwise
    """
    # Only apply for 3P providers (not firstParty)
    if get_api_provider() == "firstParty":
        return None

    model_lower = model.lower()

    for tier in _TIERS:
        pinned = os.environ.get(tier["model_env_var"])
        capabilities = os.environ.get(tier["capabilities_env_var"])

        if not pinned or capabilities is None:
            continue

        if model_lower != pinned.lower():
            continue

        # Parse capabilities
        caps = [c.strip().lower() for c in capabilities.split(",")]
        return capability.lower() in caps

    return None


# =============================================================================
# Model Support Detection
# =============================================================================


def model_supports_thinking(model: str) -> bool:
    """Check if a model supports extended thinking.

    Args:
        model: Model name (e.g., "claude-sonnet-4-6")

    Returns:
        True if the model supports thinking
    """
    # Check 3P override first
    supported_3p = get_3p_model_capability_override(model, "thinking")
    if supported_3p is not None:
        return supported_3p

    canonical = get_canonical_name(model)
    provider = get_api_provider()

    # 1P and Foundry: all Claude 4+ models (including Haiku 4.5)
    if provider in ("foundry", "firstParty"):
        return "claude-3-" not in canonical

    # 3P (Bedrock/Vertex): only Opus 4+ and Sonnet 4+
    return "sonnet-4" in canonical or "opus-4" in canonical


def model_supports_adaptive_thinking(model: str) -> bool:
    """Check if a model supports adaptive thinking.

    Adaptive thinking allows the model to dynamically allocate thinking
    budget without a fixed token limit.

    Args:
        model: Model name

    Returns:
        True if the model supports adaptive thinking
    """
    # Check 3P override first
    supported_3p = get_3p_model_capability_override(model, "adaptive_thinking")
    if supported_3p is not None:
        return supported_3p

    canonical = get_canonical_name(model)

    # Supported by a subset of Claude 4 models
    if "opus-4-6" in canonical or "sonnet-4-6" in canonical:
        return True

    # Exclude any other known legacy models (allowlist above catches 4-6 variants first)
    if "opus" in canonical or "sonnet" in canonical or "haiku" in canonical:
        return False

    # Default to true for unknown model strings on 1P and Foundry
    provider = get_api_provider()
    return provider in ("firstParty", "foundry")


# =============================================================================
# Default Thinking Configuration
# =============================================================================


def should_enable_thinking_by_default() -> bool:
    """Check if thinking should be enabled by default.

    Returns:
        True if thinking should be enabled by default
    """
    # Environment variable override
    max_thinking_tokens = os.environ.get("MAX_THINKING_TOKENS")
    if max_thinking_tokens:
        try:
            return int(max_thinking_tokens) > 0
        except ValueError:
            pass

    # Check settings (would need to be passed in)
    # For now, default to True (matching TS behavior)
    return True


def get_max_thinking_tokens_for_model(model: str) -> int:
    """Get the maximum thinking tokens for a model.

    The max thinking tokens should be strictly less than max output tokens.

    Args:
        model: Model name

    Returns:
        Maximum thinking budget tokens
    """
    from claude_code_py.utils.context import get_model_max_output_tokens

    upper_limit = get_model_max_output_tokens(model).get("upper_limit", 128000)
    return upper_limit - 1


# =============================================================================
# Ultrathink Detection
# =============================================================================


def has_ultrathink_keyword(text: str) -> bool:
    """Check if text contains the "ultrathink" keyword.

    Args:
        text: Text to check

    Returns:
        True if ultrathink keyword is present
    """
    return bool(re.search(r"\bultrathink\b", text, re.IGNORECASE))


def find_thinking_trigger_positions(text: str) -> list[dict]:
    """Find positions of "ultrathink" keyword in text.

    Args:
        text: Text to search

    Returns:
        List of dicts with word, start, end positions
    """
    positions = []
    for match in re.finditer(r"\bultrathink\b", text, re.IGNORECASE):
        positions.append({
            "word": match.group(0),
            "start": match.start(),
            "end": match.end(),
        })
    return positions


# =============================================================================
# Build Thinking Parameter for API
# =============================================================================


def build_thinking_param(
    thinking_config: ThinkingConfig,
    model: str,
    max_output_tokens: int,
) -> Optional[dict]:
    """Build the thinking parameter for API call.

    Directly uses the thinking_config without automatic model-based overrides.
    Only checks for global disable via environment variable.

    Args:
        thinking_config: Thinking configuration
        model: Model name (unused, kept for API compatibility)
        max_output_tokens: Max output tokens for the request

    Returns:
        Thinking parameter dict or None
    """
    # Check if thinking is disabled
    if thinking_config.type == "disabled":
        return None

    # Check environment variable override
    if _is_env_truthy("CLAUDE_CODE_DISABLE_THINKING"):
        return None

    # Directly use config type
    if thinking_config.type == "adaptive":
        return {"type": "adaptive"}

    # # type == "enabled"
    # thinking_budget = thinking_config.budget_tokens or get_max_thinking_tokens_for_model(model)

    # # Ensure budget < max_tokens (API constraint)
    # thinking_budget = min(thinking_budget, max_output_tokens - 1)

    return {
        "type": "enabled",
        # "budget_tokens": thinking_budget,
    }


def _is_env_truthy(name: str) -> bool:
    """Check if an environment variable is truthy.

    Args:
        name: Environment variable name

    Returns:
        True if the variable is set to a truthy value
    """
    value = os.environ.get(name, "")
    if not value:
        return False
    return value.lower() in ("true", "1", "yes", "on")
