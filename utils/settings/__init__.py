"""Settings module for Claude Code Python.

Provides configuration management from multiple sources.
"""

from claude_code_py.utils.settings.types import (
    SettingsJson,
    SettingSource,
    EnvironmentVariables,
    PermissionsConfig,
    AttributionConfig,
    StatusLineConfig,
    SAFE_ENV_VARS,
    PROVIDER_MANAGED_ENV_VARS,
    SSH_TUNNEL_VARS,
    TRUSTED_SETTING_SOURCES,
)
from claude_code_py.utils.settings.settings import (
    get_initial_settings,
    get_settings_for_source,
    get_settings_env,
    get_trusted_settings_env,
    update_settings_for_source,
    reset_settings_cache,
    get_user_settings_file_path,
    get_project_settings_file_path,
    get_local_settings_file_path,
)

__all__ = [
    "SettingsJson",
    "SettingSource",
    "EnvironmentVariables",
    "PermissionsConfig",
    "AttributionConfig",
    "StatusLineConfig",
    "SAFE_ENV_VARS",
    "PROVIDER_MANAGED_ENV_VARS",
    "SSH_TUNNEL_VARS",
    "TRUSTED_SETTING_SOURCES",
    "get_initial_settings",
    "get_settings_for_source",
    "get_settings_env",
    "get_trusted_settings_env",
    "update_settings_for_source",
    "reset_settings_cache",
    "get_user_settings_file_path",
    "get_project_settings_file_path",
    "get_local_settings_file_path",
]