"""Managed environment variables from settings.

This module implements applying environment variables from settings.json
to process.env, with proper security filtering.

Based on TypeScript implementation in utils/managedEnv.ts.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Set

from claude_code_py.utils.settings import (
    SAFE_ENV_VARS,
    PROVIDER_MANAGED_ENV_VARS,
    SSH_TUNNEL_VARS,
    TRUSTED_SETTING_SOURCES,
    SettingSource,
    get_initial_settings,
    get_settings_for_source,
    get_settings_env,
)


# =============================================================================
# Global Config ( ~/.claude.json )
# =============================================================================


def get_global_config_path() -> str:
    """Get path to global config file.

    Returns:
        Path to ~/.claude.json
    """
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return os.path.join(config_dir, "claude.json")
    return os.path.expanduser("~/.claude.json")


def get_global_config() -> Dict[str, str]:
    """Load global config (~/.claude.json) if it exists.

    This is separate from settings.json - it's a simpler file
    for global preferences.

    Returns:
        Dict with config (including 'env' key if present)
    """
    import json

    path = get_global_config_path()
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if content.strip():
                return json.loads(content)
    except Exception:
        pass

    return {}


# =============================================================================
# Environment Filtering
# =============================================================================


def is_env_truthy(value: Optional[str]) -> bool:
    """Check if environment variable value is truthy.

    Args:
        value: Env var value

    Returns:
        True if truthy (1, true, yes, on)
    """
    if not value:
        return False
    return value.lower() in ("1", "true", "yes", "on")


def is_provider_managed_env_var(key: str) -> bool:
    """Check if env var is provider-managed.

    Args:
        key: Env var name

    Returns:
        True if provider-managed
    """
    return key.upper() in PROVIDER_MANAGED_ENV_VARS


def without_ssh_tunnel_vars(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Remove SSH tunnel vars from env dict.

    When using SSH remote, the host sets certain auth vars that
    the remote's settings.env MUST NOT clobber.

    Args:
        env: Environment dict

    Returns:
        Filtered env dict
    """
    if not env:
        return {}

    if not os.environ.get("ANTHROPIC_UNIX_SOCKET"):
        return env

    # Strip SSH tunnel vars
    result = {}
    for key, value in env.items():
        if key.upper() not in SSH_TUNNEL_VARS:
            result[key] = value

    return result


def without_host_managed_provider_vars(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Remove host-managed provider vars from env dict.

    When host manages provider (CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST=true),
    strip provider-selection vars so settings.json can't redirect.

    Args:
        env: Environment dict

    Returns:
        Filtered env dict
    """
    if not env:
        return {}

    if not is_env_truthy(os.environ.get("CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST")):
        return env

    result = {}
    for key, value in env.items():
        if not is_provider_managed_env_var(key):
            result[key] = value

    return result


# Snapshot of spawn env keys for CCD mode
_ccd_spawn_env_keys: Optional[Set[str]] = None


def without_ccd_spawn_env_keys(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Remove CCD spawn env keys from env dict.

    For Claude Desktop, capture keys present before settings.env is applied,
    so the host's operational vars (OTEL, etc.) are not overridden.

    Args:
        env: Environment dict

    Returns:
        Filtered env dict
    """
    if not env:
        return {}

    if _ccd_spawn_env_keys is None:
        return env

    result = {}
    for key, value in env.items():
        if key not in _ccd_spawn_env_keys:
            result[key] = value

    return result


def filter_settings_env(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Apply all filters to settings-sourced env.

    Args:
        env: Raw env from settings

    Returns:
        Filtered env dict
    """
    return without_ccd_spawn_env_keys(
        without_host_managed_provider_vars(
            without_ssh_tunnel_vars(env)
        )
    )


# =============================================================================
# Environment Application
# =============================================================================


def apply_safe_config_environment_variables(cwd: Optional[str] = None) -> None:
    """Apply environment variables from trusted sources.

    Called before the trust dialog so that user/enterprise env vars
    like ANTHROPIC_BASE_URL take effect during first-run/onboarding.

    For trusted sources (user, managed, CLI flags), ALL env vars are applied.
    For project-scoped sources, only SAFE_ENV_VARS are applied.

    Args:
        cwd: Working directory
    """
    # Capture CCD spawn-env keys before any settings.env is applied
    global _ccd_spawn_env_keys

    if _ccd_spawn_env_keys is None:
        entrypoint = os.environ.get("CLAUDE_CODE_ENTRYPOINT")
        if entrypoint == "claude-desktop":
            _ccd_spawn_env_keys = set(os.environ.keys())
        else:
            _ccd_spawn_env_keys = set()  # Empty set = no filtering

    # Apply global config env (user-controlled)
    global_config = get_global_config()
    if "env" in global_config and isinstance(global_config["env"], dict):
        filtered = filter_settings_env(global_config["env"])
        os.environ.update(filtered)

    # Apply ALL env vars from trusted sources (except policySettings - applied separately)
    for source in TRUSTED_SETTING_SOURCES:
        if source == SettingSource.POLICY_SETTINGS:
            continue  # Apply after eligibility check

        settings = get_settings_for_source(source, cwd)
        if settings and settings.env:
            filtered = filter_settings_env(settings.env.data)
            os.environ.update(filtered)

    # Apply policySettings env (highest priority, after eligibility)
    policy_settings = get_settings_for_source(SettingSource.POLICY_SETTINGS, cwd)
    if policy_settings and policy_settings.env:
        filtered = filter_settings_env(policy_settings.env.data)
        os.environ.update(filtered)

    # Apply only safe env vars from project-scoped sources
    # (merged settings includes projectSettings and localSettings)
    settings_env = get_settings_env(cwd)
    filtered = filter_settings_env(settings_env)

    for key, value in filtered.items():
        if key.upper() in SAFE_ENV_VARS:
            os.environ[key] = value


def apply_config_environment_variables(cwd: Optional[str] = None) -> None:
    """Apply ALL environment variables from settings.

    This applies potentially dangerous environment variables like
    LD_PRELOAD, PATH, etc. Should only be called after trust is established.

    Args:
        cwd: Working directory
    """
    # Apply global config env
    global_config = get_global_config()
    if "env" in global_config and isinstance(global_config["env"], dict):
        filtered = filter_settings_env(global_config["env"])
        os.environ.update(filtered)

    # Apply all settings env (including project-scoped)
    settings_env = get_settings_env(cwd)
    filtered = filter_settings_env(settings_env)
    os.environ.update(filtered)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_effective_env(cwd: Optional[str] = None) -> Dict[str, str]:
    """Get effective environment after settings.env is applied.

    This returns the current os.environ after settings have been applied.

    Args:
        cwd: Working directory

    Returns:
        Dict of current environment
    """
    # Apply settings env if not already done
    if not os.environ.get("_CLAUDE_SETTINGS_ENV_APPLIED"):
        apply_safe_config_environment_variables(cwd)
        os.environ["_CLAUDE_SETTINGS_ENV_APPLIED"] = "true"

    return dict(os.environ)


def setup_environment_from_settings(cwd: Optional[str] = None, trust_established: bool = False) -> None:
    """Setup environment from settings.

    Main entry point for setting up environment.

    Args:
        cwd: Working directory
        trust_established: Whether trust has been established
    """
    if trust_established:
        apply_config_environment_variables(cwd)
    else:
        apply_safe_config_environment_variables(cwd)

    # Mark as applied
    os.environ["_CLAUDE_SETTINGS_ENV_APPLIED"] = "true"


def clear_settings_env_marker() -> None:
    """Clear the marker that settings env has been applied.

    Useful for re-applying after settings change.
    """
    os.environ.pop("_CLAUDE_SETTINGS_ENV_APPLIED", None)


def reapply_settings_env(cwd: Optional[str] = None) -> None:
    """Re-apply settings environment after settings change.

    Args:
        cwd: Working directory
    """
    # Clear marker and cache
    clear_settings_env_marker()
    from claude_code_py.utils.settings import reset_settings_cache
    reset_settings_cache()

    # Re-apply with trust assumed (settings change happens after trust)
    apply_config_environment_variables(cwd)
    os.environ["_CLAUDE_SETTINGS_ENV_APPLIED"] = "true"