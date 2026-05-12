"""API configuration for Claude Code Python.

This module provides centralized configuration for API settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    """API configuration settings."""

    # Authentication
    auth_token: Optional[str] = None
    api_key: Optional[str] = None

    # Endpoint
    base_url: Optional[str] = None

    # Model
    model: str = "claude-sonnet-4-6"

    # Default models for different tasks
    default_sonnet: str = "claude-sonnet-4-6"
    default_haiku: str = "claude-haiku-4-5"

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load configuration from environment variables.

        Priority:
        1. ANTHROPIC_AUTH_TOKEN (for custom endpoints)
        2. ANTHROPIC_API_KEY (standard Anthropic API)

        Returns:
            APIConfig instance
        """
        # Auth token - highest priority for custom endpoints
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")

        # API key fallback
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        # Base URL - for custom endpoints
        base_url = os.environ.get("ANTHROPIC_BASE_URL")

        # Default: if no auth_token but base_url is set, use api_key as auth_token
        if base_url and not auth_token and api_key:
            auth_token = api_key

        # Model configuration
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

        # Override default models if specified
        default_sonnet = os.environ.get("ANTHROPIC_SONNET_MODEL", model)
        default_haiku = os.environ.get("ANTHROPIC_HAIKU_MODEL", "claude-haiku-4-5")

        return cls(
            auth_token=auth_token,
            api_key=api_key,
            base_url=base_url,
            model=model,
            default_sonnet=default_sonnet,
            default_haiku=default_haiku,
        )

    def get_auth_token(self) -> Optional[str]:
        """Get the authentication token to use.

        Returns:
            Auth token or None
        """
        # Prefer auth_token for custom endpoints
        if self.auth_token:
            return self.auth_token

        return self.api_key

    def is_valid(self) -> bool:
        """Check if configuration has valid auth.

        Returns:
            True if auth is configured
        """
        return bool(self.auth_token or self.api_key)

    def to_anthropic_kwargs(self) -> dict:
        """Convert to kwargs for Anthropic client.

        Returns:
            Dict of kwargs for anthropic.AsyncAnthropic()
        """
        kwargs = {}

        # Auth - use auth_token or api_key
        auth = self.get_auth_token()
        if auth:
            # For custom endpoints, auth_token goes in api_key parameter
            kwargs["api_key"] = auth

        # Base URL - for custom endpoints
        if self.base_url:
            kwargs["base_url"] = self.base_url

        return kwargs


# Global config instance
_config: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    """Get the global API configuration.

    Returns:
        APIConfig instance
    """
    global _config
    if _config is None:
        _config = APIConfig.from_env()
    return _config


def reset_api_config() -> None:
    """Reset the global API configuration."""
    global _config
    _config = None


def setup_default_anthropic(api_key: str) -> None:
    """Setup for default Anthropic API.

    Args:
        api_key: Anthropic API key
    """
    os.environ["ANTHROPIC_API_KEY"] = api_key
    os.environ.pop("ANTHROPIC_BASE_URL", None)
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

    # Reset config to pick up new env vars
    reset_api_config()