"""Model utilities for Claude API.

Based on TypeScript implementation in utils/model/model.ts.

This module provides:
- Model name canonicalization
- API provider detection
- Model capability detection
"""

from __future__ import annotations

import os
from typing import Optional


# =============================================================================
# Model Name Canonicalization
# =============================================================================


# Model aliases mapping
MODEL_ALIASES = {
    # Short names
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    # Version aliases
    "claude-4-opus": "claude-opus-4-6",
    "claude-4-sonnet": "claude-sonnet-4-6",
    "claude-4-haiku": "claude-haiku-4-5",
    # Legacy aliases
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
}


def get_canonical_name(model: str) -> str:
    """Get the canonical name for a model.

    This normalizes model names to a consistent format.

    Args:
        model: Model name (may be alias or partial name)

    Returns:
        Canonical model name
    """
    model_lower = model.lower().strip()

    # Check aliases
    if model_lower in MODEL_ALIASES:
        return MODEL_ALIASES[model_lower]

    # Already canonical
    if model_lower.startswith("claude-"):
        return model_lower

    # Add claude- prefix if missing
    if not model_lower.startswith("claude"):
        return f"claude-{model_lower}"

    return model_lower


# =============================================================================
# API Provider Detection
# =============================================================================


class APIProvider:
    """API provider types."""

    FIRST_PARTY = "firstParty"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    FOUNDRY = "foundry"


def get_api_provider() -> str:
    """Get the current API provider.

    Determines the provider based on environment variables.

    Returns:
        Provider identifier string
    """
    # Check for Bedrock
    if os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"):
        return APIProvider.BEDROCK

    # Check for Vertex AI
    if os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return APIProvider.VERTEX

    # Check for Foundry (internal)
    if os.environ.get("ANTHROPIC_BASE_URL", "").find("foundry") >= 0:
        return APIProvider.FOUNDRY

    # Default to first party (direct API)
    return APIProvider.FIRST_PARTY


# =============================================================================
# Model Capability Detection
# =============================================================================


def is_claude_4_model(model: str) -> bool:
    """Check if model is Claude 4.x.

    Args:
        model: Model name

    Returns:
        True if Claude 4.x model
    """
    canonical = get_canonical_name(model)
    return "claude-4" in canonical or "opus-4" in canonical or "sonnet-4" in canonical or "haiku-4" in canonical


def is_claude_3_model(model: str) -> bool:
    """Check if model is Claude 3.x.

    Args:
        model: Model name

    Returns:
        True if Claude 3.x model
    """
    canonical = get_canonical_name(model)
    return "claude-3" in canonical


def get_model_family(model: str) -> str:
    """Get the model family (opus, sonnet, haiku).

    Args:
        model: Model name

    Returns:
        Model family name
    """
    canonical = get_canonical_name(model)

    if "opus" in canonical:
        return "opus"
    elif "sonnet" in canonical:
        return "sonnet"
    elif "haiku" in canonical:
        return "haiku"

    return "unknown"
