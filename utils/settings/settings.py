"""Settings loading and parsing.

This module implements loading settings from multiple sources:
- userSettings: ~/.claude/settings.json
- projectSettings: .claude/settings.json
- localSettings: .claude/settings.local.json
- policySettings: managed-settings.json (enterprise)

Based on TypeScript implementation in utils/settings/settings.ts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from claude_code_py.utils.settings.types import (
    SettingsJson,
    SettingSource,
    TRUSTED_SETTING_SOURCES,
)


# =============================================================================
# Paths
# =============================================================================


def get_claude_config_home_dir() -> Path:
    """Get Claude config home directory.

    Priority:
    1. CLAUDE_CONFIG_DIR environment variable
    2. ~/.claude (standard)

    Returns:
        Path to Claude config directory
    """
    env_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    return Path.home() / ".claude"


def get_user_settings_file_path() -> Path:
    """Get user settings file path.

    Returns:
        Path to ~/.claude/settings.json
    """
    return get_claude_config_home_dir() / "settings.json"


def get_project_settings_file_path(cwd: Optional[str] = None) -> Path:
    """Get project settings file path.

    Args:
        cwd: Working directory (defaults to current)

    Returns:
        Path to .claude/settings.json in project
    """
    work_dir = Path(cwd) if cwd else Path.cwd()
    return work_dir / ".claude" / "settings.json"


def get_local_settings_file_path(cwd: Optional[str] = None) -> Path:
    """Get local settings file path.

    Args:
        cwd: Working directory (defaults to current)

    Returns:
        Path to .claude/settings.local.json in project
    """
    work_dir = Path(cwd) if cwd else Path.cwd()
    return work_dir / ".claude" / "settings.local.json"


def get_managed_settings_file_path() -> Optional[Path]:
    """Get managed settings file path (enterprise policy).

    This checks common locations for managed-settings.json.

    Returns:
        Path to managed-settings.json or None
    """
    # Common locations for managed settings
    locations = [
        # Linux/macOS
        Path("/etc/claude/managed-settings.json"),
        Path("/usr/local/etc/claude/managed-settings.json"),
        # macOS specific
        Path("/Library/Application Support/Claude/managed-settings.json"),
        # Windows (if on Windows)
        Path("C:/ProgramData/Claude/managed-settings.json"),
        # Environment override
        os.environ.get("CLAUDE_MANAGED_SETTINGS_PATH"),
    ]

    for loc in locations:
        if loc and isinstance(loc, str):
            loc = Path(loc)
        if loc and loc.exists():
            return loc

    return None


def get_settings_file_path_for_source(
    source: SettingSource,
    cwd: Optional[str] = None,
) -> Optional[Path]:
    """Get settings file path for a source.

    Args:
        source: Settings source
        cwd: Working directory

    Returns:
        Path to settings file or None
    """
    if source == SettingSource.USER_SETTINGS:
        return get_user_settings_file_path()
    elif source == SettingSource.PROJECT_SETTINGS:
        return get_project_settings_file_path(cwd)
    elif source == SettingSource.LOCAL_SETTINGS:
        return get_local_settings_file_path(cwd)
    elif source == SettingSource.POLICY_SETTINGS:
        return get_managed_settings_file_path()
    elif source == SettingSource.FLAG_SETTINGS:
        # CLI flag settings are passed directly, not from file
        return None

    return None


# =============================================================================
# File Parsing
# =============================================================================


def parse_settings_file(path: Path) -> Tuple[Optional[SettingsJson], List[Dict[str, Any]]]:
    """Parse a settings file.

    Args:
        path: Path to settings file

    Returns:
        Tuple of (settings, errors) - settings is None if file doesn't exist or has errors
    """
    errors: List[Dict[str, Any]] = []

    if not path.exists():
        return None, errors

    try:
        content = path.read_text(encoding="utf-8")

        # Empty file = empty settings
        if content.strip() == "":
            return SettingsJson(), errors

        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            errors.append({
                "file": str(path),
                "message": f"JSON parse error: {e}",
                "line": e.lineno if hasattr(e, "lineno") else None,
            })
            return None, errors

        # Validate it's a dict
        if not isinstance(data, dict):
            errors.append({
                "file": str(path),
                "message": "Settings must be a JSON object (dict)",
            })
            return None, errors

        # Parse into SettingsJson
        settings = SettingsJson.from_dict(data)

        return settings, errors

    except PermissionError as e:
        errors.append({
            "file": str(path),
            "message": f"Permission denied: {e}",
        })
        return None, errors
    except Exception as e:
        errors.append({
            "file": str(path),
            "message": f"Error reading file: {e}",
        })
        return None, errors


# =============================================================================
# Settings Loading
# =============================================================================


# Cache for settings (invalidated on file change)
_settings_cache: Optional[Dict[str, Any]] = None
_settings_cache_key: Optional[str] = None


def get_settings_for_source(
    source: SettingSource,
    cwd: Optional[str] = None,
) -> Optional[SettingsJson]:
    """Get settings from a specific source.

    Args:
        source: Settings source
        cwd: Working directory

    Returns:
        SettingsJson or None if source doesn't exist
    """
    path = get_settings_file_path_for_source(source, cwd)
    if path is None:
        # FLAG_SETTINGS has no file - return None
        return None

    settings, _errors = parse_settings_file(path)
    return settings


def get_enabled_setting_sources() -> List[SettingSource]:
    """Get list of enabled setting sources.

    Sources can be disabled via environment variable.

    Returns:
        List of enabled sources in priority order (lowest to highest)
    """
    # Check environment for disabled sources
    disabled_sources_str = os.environ.get("CLAUDE_CODE_DISABLED_SETTING_SOURCES", "")
    disabled_sources = set(disabled_sources_str.split(",")) if disabled_sources_str else set()

    # Default sources in priority order
    all_sources = [
        SettingSource.USER_SETTINGS,
        SettingSource.PROJECT_SETTINGS,
        SettingSource.LOCAL_SETTINGS,
        SettingSource.POLICY_SETTINGS,
        SettingSource.FLAG_SETTINGS,
    ]

    return [s for s in all_sources if s.value not in disabled_sources]


def load_settings_from_disk(cwd: Optional[str] = None) -> Tuple[SettingsJson, List[Dict[str, Any]]]:
    """Load and merge settings from all sources.

    Priority order (lowest to highest):
    1. userSettings (~/.claude/settings.json)
    2. projectSettings (.claude/settings.json)
    3. localSettings (.claude/settings.local.json)
    4. policySettings (managed-settings.json)
    5. flagSettings (CLI --settings)

    Returns:
        Tuple of (merged_settings, all_errors)
    """
    all_errors: List[Dict[str, Any]] = []
    merged = SettingsJson()

    sources = get_enabled_setting_sources()

    for source in sources:
        settings = get_settings_for_source(source, cwd)

        if settings:
            merged = merged.merge_with(settings)

        # Also check for inline flag settings from environment
        if source == SettingSource.FLAG_SETTINGS:
            inline_settings_str = os.environ.get("CLAUDE_CODE_INLINE_SETTINGS")
            if inline_settings_str:
                try:
                    inline_data = json.loads(inline_settings_str)
                    if isinstance(inline_data, dict):
                        inline_settings = SettingsJson.from_dict(inline_data)
                        merged = merged.merge_with(inline_settings)
                except json.JSONDecodeError:
                    all_errors.append({
                        "file": "inline_settings",
                        "message": "Failed to parse CLAUDE_CODE_INLINE_SETTINGS",
                    })

    return merged, all_errors


def get_initial_settings(cwd: Optional[str] = None) -> SettingsJson:
    """Get merged settings from all sources.

    This is the main entry point for getting settings.

    Args:
        cwd: Working directory

    Returns:
        Merged SettingsJson (empty if no settings files exist)
    """
    global _settings_cache, _settings_cache_key

    # Check cache
    cache_key = f"{cwd}:{get_settings_cache_key()}"
    if _settings_cache is not None and _settings_cache_key == cache_key:
        cached = _settings_cache.get("settings")
        if cached:
            return cached

    # Load from disk
    settings, _errors = load_settings_from_disk(cwd)

    # Update cache
    _settings_cache = {"settings": settings}
    _settings_cache_key = cache_key

    return settings


def get_settings_cache_key() -> str:
    """Generate cache key based on file modification times.

    Returns:
        Cache key string
    """
    paths = [
        get_user_settings_file_path(),
        get_project_settings_file_path(),
        get_local_settings_file_path(),
        get_managed_settings_file_path(),
    ]

    mtimes = []
    for p in paths:
        if p and p.exists():
            try:
                mtimes.append(str(p.stat().st_mtime))
            except Exception:
                pass

    # Include inline settings if present
    inline = os.environ.get("CLAUDE_CODE_INLINE_SETTINGS")
    if inline:
        mtimes.append(inline)

    return ":".join(mtimes) if mtimes else "empty"


def reset_settings_cache() -> None:
    """Reset the settings cache."""
    global _settings_cache, _settings_cache_key
    _settings_cache = None
    _settings_cache_key = None


# =============================================================================
# Settings Update
# =============================================================================


def update_settings_for_source(
    source: SettingSource,
    settings_update: Dict[str, Any],
    cwd: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Update settings for a source.

    Merges the update into existing settings.

    Args:
        source: Settings source to update
        cwd: Working directory
        settings_update: Dict of settings to merge

    Returns:
        Tuple of (success, error_message)
    """
    # Cannot update policy or flag settings
    if source == SettingSource.POLICY_SETTINGS or source == SettingSource.FLAG_SETTINGS:
        return True, None  # Silently ignore

    path = get_settings_file_path_for_source(source, cwd)
    if path is None:
        return False, f"Cannot determine path for source {source}"

    # Read existing settings
    existing_settings, errors = parse_settings_file(path)

    # Handle case where file has JSON syntax error
    if existing_settings is None and errors:
        # Check if file exists with invalid JSON
        if path.exists():
            return False, f"Invalid JSON in settings file at {path}"

    # Create empty settings if None
    if existing_settings is None:
        existing_settings = SettingsJson()

    # Merge update
    update_settings = SettingsJson.from_dict(settings_update)
    merged = existing_settings.merge_with(update_settings)

    # Ensure parent directory exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Failed to create directory {path.parent}: {e}"

    # Write updated settings
    try:
        content = json.dumps(merged.to_dict(), indent=2, ensure_ascii=False) + "\n"
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        return False, f"Failed to write settings file: {e}"

    # Invalidate cache
    reset_settings_cache()

    return True, None


# =============================================================================
# Convenience Functions
# =============================================================================


def get_settings_env(cwd: Optional[str] = None) -> Dict[str, str]:
    """Get environment variables from merged settings.

    Args:
        cwd: Working directory

    Returns:
        Dict of environment variables
    """
    settings = get_initial_settings(cwd)
    return settings.get_env_dict()


def get_trusted_settings_env(cwd: Optional[str] = None) -> Dict[str, str]:
    """Get environment variables only from trusted sources.

    Trusted sources: userSettings, flagSettings, policySettings

    Args:
        cwd: Working directory

    Returns:
        Dict of environment variables from trusted sources
    """
    merged_env: Dict[str, str] = {}

    for source in TRUSTED_SETTING_SOURCES:
        settings = get_settings_for_source(source, cwd)
        if settings and settings.env:
            merged_env.update(settings.env.data)

    return merged_env