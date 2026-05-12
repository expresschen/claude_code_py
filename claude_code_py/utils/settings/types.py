"""Settings types and schema definitions.

This module defines the structure of settings.json files and related types.
Based on TypeScript implementation in utils/settings/types.ts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json


# =============================================================================
# Setting Sources
# =============================================================================


class SettingSource(str, Enum):
    """Source of settings configuration."""

    USER_SETTINGS = "userSettings"      # ~/.claude/settings.json
    PROJECT_SETTINGS = "projectSettings"  # .claude/settings.json
    LOCAL_SETTINGS = "localSettings"    # .claude/settings.local.json
    POLICY_SETTINGS = "policySettings"  # managed-settings.json (enterprise)
    FLAG_SETTINGS = "flagSettings"      # CLI --settings flag


# Trusted sources that can apply env vars before trust dialog
TRUSTED_SETTING_SOURCES = [
    SettingSource.USER_SETTINGS,
    SettingSource.FLAG_SETTINGS,
    SettingSource.POLICY_SETTINGS,
]


# =============================================================================
# Environment Variables Schema
# =============================================================================


# Safe environment variables that can be applied from project-scoped sources
# before trust is established
SAFE_ENV_VARS = {
    # Claude Code specific (safe)
    "CLAUDE_CODE_DEBUG",
    "CLAUDE_CODE_VERBOSE",
    "CLAUDE_CODE_DISABLE_COMPACT",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
    "CLAUDE_CODE_FORCE_THINKING",
    "CLAUDE_CODE_DISABLE_THINKING",
    "CLAUDE_CODE_OUTPUT_STYLE",
    "CLAUDE_CODE_LANGUAGE",

    # Logging/observability (safe - informational only)
    "OTEL_LOGS_EXPORTER",
    "OTEL_TRACES_EXPORTER",
    "OTEL_METRICS_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",

    # Model selection (safe - user preference)
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SONNET_MODEL",
    "ANTHROPIC_HAIKU_MODEL",

    # Debug/development (safe)
    "DEBUG",
    "VERBOSE",
    "LOG_LEVEL",

    # Terminal/display (safe)
    "TERM",
    "TERM_PROGRAM",
    "COLORTERM",
    "NO_COLOR",
}

# Provider managed env vars - should be stripped when host manages provider
PROVIDER_MANAGED_ENV_VARS = {
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SONNET_MODEL",
    "ANTHROPIC_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_MODEL",
    "CLAUDE_CODE_DEFAULT_MODEL",
}

# SSH tunnel vars - should be stripped from settings-sourced env
SSH_TUNNEL_VARS = {
    "ANTHROPIC_UNIX_SOCKET",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
}


# =============================================================================
# Settings JSON Schema
# =============================================================================


@dataclass
class EnvironmentVariables:
    """Environment variables from settings.

    Keys are env var names, values are strings (coerced from any type).
    """

    data: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EnvironmentVariables":
        """Create from dict, coercing values to strings.

        Args:
            data: Raw dict with any value types

        Returns:
            EnvironmentVariables instance
        """
        if not data:
            return cls()

        coerced: Dict[str, str] = {}
        for key, value in data.items():
            if value is None:
                continue
            # Coerce to string (same as z.coerce.string())
            if isinstance(value, bool):
                coerced[key] = "true" if value else "false"
            elif isinstance(value, (int, float)):
                coerced[key] = str(value)
            elif isinstance(value, str):
                coerced[key] = value
            elif isinstance(value, list):
                # Arrays join with comma (matching z.coerce behavior)
                coerced[key] = ",".join(str(v) for v in value)
            else:
                coerced[key] = str(value)

        return cls(data=coerced)


@dataclass
class PermissionsConfig:
    """Permissions section of settings."""

    allow: List[str] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)
    ask: List[str] = field(default_factory=list)
    default_mode: Optional[str] = None
    disable_bypass_permissions_mode: Optional[str] = None
    additional_directories: List[str] = field(default_factory=list)


@dataclass
class AttributionConfig:
    """Attribution settings for commits and PRs."""

    commit: Optional[str] = None
    pr: Optional[str] = None


@dataclass
class StatusLineConfig:
    """Custom status line configuration."""

    type: str = "command"
    command: Optional[str] = None
    padding: Optional[int] = None


@dataclass
class SettingsJson:
    """Settings JSON structure.

    This is the main settings configuration that can be stored in:
    - ~/.claude/settings.json (userSettings)
    - .claude/settings.json (projectSettings)
    - .claude/settings.local.json (localSettings)
    - managed-settings.json (policySettings)

    Backward compatibility: New optional fields should always use Optional.
    Invalid fields are preserved in the file (not stripped).
    """

    # Environment variables - main focus of this implementation
    env: Optional[EnvironmentVariables] = None

    # Authentication helpers
    api_key_helper: Optional[str] = None
    aws_credential_export: Optional[str] = None
    aws_auth_refresh: Optional[str] = None
    gcp_auth_refresh: Optional[str] = None

    # Permissions
    permissions: Optional[PermissionsConfig] = None

    # Model configuration
    model: Optional[str] = None
    available_models: List[str] = field(default_factory=list)
    model_overrides: Dict[str, str] = field(default_factory=dict)

    # Attribution
    attribution: Optional[AttributionConfig] = None
    include_co_authored_by: Optional[bool] = None
    include_git_instructions: Optional[bool] = None

    # MCP servers
    enable_all_project_mcp_servers: Optional[bool] = None
    enabled_mcpjson_servers: List[str] = field(default_factory=list)
    disabled_mcpjson_servers: List[str] = field(default_factory=list)
    allowed_mcp_servers: List[Dict[str, Any]] = field(default_factory=list)
    denied_mcp_servers: List[Dict[str, Any]] = field(default_factory=list)

    # Hooks
    hooks: Dict[str, Any] = field(default_factory=dict)
    disable_all_hooks: Optional[bool] = None
    allow_managed_hooks_only: Optional[bool] = None

    # Status line
    status_line: Optional[StatusLineConfig] = None

    # Cleanup
    cleanup_period_days: Optional[int] = None

    # Display/output
    output_style: Optional[str] = None
    language: Optional[str] = None
    syntax_highlighting_disabled: Optional[bool] = None
    spinner_tips_enabled: Optional[bool] = None

    # Thinking/effort
    always_thinking_enabled: Optional[bool] = None
    effort_level: Optional[str] = None
    fast_mode: Optional[bool] = None

    # Auto-memory
    auto_memory_enabled: Optional[bool] = None
    auto_memory_directory: Optional[str] = None

    # Shell
    default_shell: Optional[str] = None  # 'bash' or 'powershell'

    # Misc
    respect_gitignore: Optional[bool] = None
    file_suggestion: Optional[Dict[str, Any]] = None
    auto_updates_channel: Optional[str] = None  # 'latest' or 'stable'

    # Store any unknown fields for backward compatibility
    _extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SettingsJson":
        """Parse from raw dict.

        Args:
            data: Raw settings dict (from JSON parse)

        Returns:
            SettingsJson instance
        """
        # Extract known fields
        env_data = data.get("env")
        env = EnvironmentVariables.from_dict(env_data) if env_data else None

        permissions_data = data.get("permissions")
        permissions = None
        if permissions_data and isinstance(permissions_data, dict):
            permissions = PermissionsConfig(
                allow=permissions_data.get("allow", []),
                deny=permissions_data.get("deny", []),
                ask=permissions_data.get("ask", []),
                default_mode=permissions_data.get("defaultMode"),
                disable_bypass_permissions_mode=permissions_data.get("disableBypassPermissionsMode"),
                additional_directories=permissions_data.get("additionalDirectories", []),
            )

        attribution_data = data.get("attribution")
        attribution = None
        if attribution_data and isinstance(attribution_data, dict):
            attribution = AttributionConfig(
                commit=attribution_data.get("commit"),
                pr=attribution_data.get("pr"),
            )

        status_line_data = data.get("statusLine")
        status_line = None
        if status_line_data and isinstance(status_line_data, dict):
            status_line = StatusLineConfig(
                type=status_line_data.get("type", "command"),
                command=status_line_data.get("command"),
                padding=status_line_data.get("padding"),
            )

        # Collect known field names
        known_fields = {
            "env", "apiKeyHelper", "awsCredentialExport", "awsAuthRefresh",
            "gcpAuthRefresh", "permissions", "model", "availableModels",
            "modelOverrides", "attribution", "includeCoAuthoredBy",
            "includeGitInstructions", "enableAllProjectMcpServers",
            "enabledMcpjsonServers", "disabledMcpjsonServers",
            "allowedMcpServers", "deniedMcpServers", "hooks",
            "disableAllHooks", "allowManagedHooksOnly", "statusLine",
            "cleanupPeriodDays", "outputStyle", "language",
            "syntaxHighlightingDisabled", "spinnerTipsEnabled",
            "alwaysThinkingEnabled", "effortLevel", "fastMode",
            "autoMemoryEnabled", "autoMemoryDirectory", "defaultShell",
            "respectGitignore", "fileSuggestion", "autoUpdatesChannel",
            # Snake case variants (Python style)
            "api_key_helper", "aws_credential_export", "aws_auth_refresh",
            "gcp_auth_refresh", "available_models", "model_overrides",
            "include_co_authored_by", "include_git_instructions",
            "enable_all_project_mcp_servers", "enabled_mcpjson_servers",
            "disabled_mcpjson_servers", "allowed_mcp_servers", "denied_mcp_servers",
            "disable_all_hooks", "allow_managed_hooks_only", "status_line",
            "cleanup_period_days", "output_style", "syntax_highlighting_disabled",
            "spinner_tips_enabled", "always_thinking_enabled", "effort_level",
            "fast_mode", "auto_memory_enabled", "auto_memory_directory",
            "default_shell", "respect_gitignore", "file_suggestion",
            "auto_updates_channel",
        }

        # Store unknown fields
        extra = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            env=env,
            api_key_helper=data.get("apiKeyHelper") or data.get("api_key_helper"),
            aws_credential_export=data.get("awsCredentialExport") or data.get("aws_credential_export"),
            aws_auth_refresh=data.get("awsAuthRefresh") or data.get("aws_auth_refresh"),
            gcp_auth_refresh=data.get("gcpAuthRefresh") or data.get("gcp_auth_refresh"),
            permissions=permissions,
            model=data.get("model"),
            available_models=data.get("availableModels") or data.get("available_models") or [],
            model_overrides=data.get("modelOverrides") or data.get("model_overrides") or {},
            attribution=attribution,
            include_co_authored_by=data.get("includeCoAuthoredBy") or data.get("include_co_authored_by"),
            include_git_instructions=data.get("includeGitInstructions") or data.get("include_git_instructions"),
            enable_all_project_mcp_servers=data.get("enableAllProjectMcpServers") or data.get("enable_all_project_mcp_servers"),
            enabled_mcpjson_servers=data.get("enabledMcpjsonServers") or data.get("enabled_mcpjson_servers") or [],
            disabled_mcpjson_servers=data.get("disabledMcpjsonServers") or data.get("disabled_mcpjson_servers") or [],
            allowed_mcp_servers=data.get("allowedMcpServers") or data.get("allowed_mcp_servers") or [],
            denied_mcp_servers=data.get("deniedMcpServers") or data.get("denied_mcp_servers") or [],
            hooks=data.get("hooks") or {},
            disable_all_hooks=data.get("disableAllHooks") or data.get("disable_all_hooks"),
            allow_managed_hooks_only=data.get("allowManagedHooksOnly") or data.get("allow_managed_hooks_only"),
            status_line=status_line,
            cleanup_period_days=data.get("cleanupPeriodDays") or data.get("cleanup_period_days"),
            output_style=data.get("outputStyle") or data.get("output_style"),
            language=data.get("language"),
            syntax_highlighting_disabled=data.get("syntaxHighlightingDisabled") or data.get("syntax_highlighting_disabled"),
            spinner_tips_enabled=data.get("spinnerTipsEnabled") or data.get("spinner_tips_enabled"),
            always_thinking_enabled=data.get("alwaysThinkingEnabled") or data.get("always_thinking_enabled"),
            effort_level=data.get("effortLevel") or data.get("effort_level"),
            fast_mode=data.get("fastMode") or data.get("fast_mode"),
            auto_memory_enabled=data.get("autoMemoryEnabled") or data.get("auto_memory_enabled"),
            auto_memory_directory=data.get("autoMemoryDirectory") or data.get("auto_memory_directory"),
            default_shell=data.get("defaultShell") or data.get("default_shell"),
            respect_gitignore=data.get("respectGitignore") or data.get("respect_gitignore"),
            file_suggestion=data.get("fileSuggestion") or data.get("file_suggestion"),
            auto_updates_channel=data.get("autoUpdatesChannel") or data.get("auto_updates_channel"),
            _extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization.

        Returns:
            Dict representation
        """
        result: Dict[str, Any] = {}

        if self.env:
            result["env"] = self.env.data

        if self.api_key_helper:
            result["apiKeyHelper"] = self.api_key_helper

        if self.aws_credential_export:
            result["awsCredentialExport"] = self.aws_credential_export

        if self.aws_auth_refresh:
            result["awsAuthRefresh"] = self.aws_auth_refresh

        if self.gcp_auth_refresh:
            result["gcpAuthRefresh"] = self.gcp_auth_refresh

        if self.permissions:
            perms: Dict[str, Any] = {}
            if self.permissions.allow:
                perms["allow"] = self.permissions.allow
            if self.permissions.deny:
                perms["deny"] = self.permissions.deny
            if self.permissions.ask:
                perms["ask"] = self.permissions.ask
            if self.permissions.default_mode:
                perms["defaultMode"] = self.permissions.default_mode
            if self.permissions.disable_bypass_permissions_mode:
                perms["disableBypassPermissionsMode"] = self.permissions.disable_bypass_permissions_mode
            if self.permissions.additional_directories:
                perms["additionalDirectories"] = self.permissions.additional_directories
            if perms:
                result["permissions"] = perms

        if self.model:
            result["model"] = self.model

        if self.available_models:
            result["availableModels"] = self.available_models

        if self.model_overrides:
            result["modelOverrides"] = self.model_overrides

        if self.attribution:
            attr: Dict[str, Any] = {}
            if self.attribution.commit:
                attr["commit"] = self.attribution.commit
            if self.attribution.pr:
                attr["pr"] = self.attribution.pr
            if attr:
                result["attribution"] = attr

        if self.include_co_authored_by is not None:
            result["includeCoAuthoredBy"] = self.include_co_authored_by

        if self.include_git_instructions is not None:
            result["includeGitInstructions"] = self.include_git_instructions

        if self.enable_all_project_mcp_servers is not None:
            result["enableAllProjectMcpServers"] = self.enable_all_project_mcp_servers

        if self.enabled_mcpjson_servers:
            result["enabledMcpjsonServers"] = self.enabled_mcpjson_servers

        if self.disabled_mcpjson_servers:
            result["disabledMcpjsonServers"] = self.disabled_mcpjson_servers

        if self.allowed_mcp_servers:
            result["allowedMcpServers"] = self.allowed_mcp_servers

        if self.denied_mcp_servers:
            result["deniedMcpServers"] = self.denied_mcp_servers

        if self.hooks:
            result["hooks"] = self.hooks

        if self.disable_all_hooks is not None:
            result["disableAllHooks"] = self.disable_all_hooks

        if self.allow_managed_hooks_only is not None:
            result["allowManagedHooksOnly"] = self.allow_managed_hooks_only

        if self.status_line:
            sl: Dict[str, Any] = {"type": self.status_line.type}
            if self.status_line.command:
                sl["command"] = self.status_line.command
            if self.status_line.padding is not None:
                sl["padding"] = self.status_line.padding
            result["statusLine"] = sl

        if self.cleanup_period_days is not None:
            result["cleanupPeriodDays"] = self.cleanup_period_days

        if self.output_style:
            result["outputStyle"] = self.output_style

        if self.language:
            result["language"] = self.language

        if self.syntax_highlighting_disabled is not None:
            result["syntaxHighlightingDisabled"] = self.syntax_highlighting_disabled

        if self.spinner_tips_enabled is not None:
            result["spinnerTipsEnabled"] = self.spinner_tips_enabled

        if self.always_thinking_enabled is not None:
            result["alwaysThinkingEnabled"] = self.always_thinking_enabled

        if self.effort_level:
            result["effortLevel"] = self.effort_level

        if self.fast_mode is not None:
            result["fastMode"] = self.fast_mode

        if self.auto_memory_enabled is not None:
            result["autoMemoryEnabled"] = self.auto_memory_enabled

        if self.auto_memory_directory:
            result["autoMemoryDirectory"] = self.auto_memory_directory

        if self.default_shell:
            result["defaultShell"] = self.default_shell

        if self.respect_gitignore is not None:
            result["respectGitignore"] = self.respect_gitignore

        if self.file_suggestion:
            result["fileSuggestion"] = self.file_suggestion

        if self.auto_updates_channel:
            result["autoUpdatesChannel"] = self.auto_updates_channel

        # Include extra fields for backward compatibility
        result.update(self._extra)

        return result

    def get_env_dict(self) -> Dict[str, str]:
        """Get environment variables as dict.

        Returns:
            Dict of env vars, empty if none configured
        """
        if self.env:
            return self.env.data
        return {}

    def merge_with(self, other: "SettingsJson") -> "SettingsJson":
        """Merge this settings with another (other takes priority).

        Arrays are concatenated and deduplicated.
        Dicts are merged deeply.

        Args:
            other: Settings to merge on top (higher priority)

        Returns:
            New merged SettingsJson
        """
        # Merge env
        merged_env_data: Dict[str, str] = {}
        if self.env:
            merged_env_data.update(self.env.data)
        if other.env:
            merged_env_data.update(other.env.data)
        merged_env = EnvironmentVariables(data=merged_env_data) if merged_env_data else None

        # Merge permissions
        merged_permissions = None
        if self.permissions or other.permissions:
            merged_permissions = PermissionsConfig(
                allow=_merge_arrays(self.permissions and self.permissions.allow, other.permissions and other.permissions.allow),
                deny=_merge_arrays(self.permissions and self.permissions.deny, other.permissions and other.permissions.deny),
                ask=_merge_arrays(self.permissions and self.permissions.ask, other.permissions and other.permissions.ask),
                default_mode=(other.permissions and other.permissions.default_mode) or (self.permissions and self.permissions.default_mode),
                disable_bypass_permissions_mode=(other.permissions and other.permissions.disable_bypass_permissions_mode) or (self.permissions and self.permissions.disable_bypass_permissions_mode),
                additional_directories=_merge_arrays(
                    self.permissions and self.permissions.additional_directories,
                    other.permissions and other.permissions.additional_directories
                ),
            )

        # Merge arrays
        merged_available_models = _merge_arrays(self.available_models, other.available_models)
        merged_enabled_mcpjson_servers = _merge_arrays(self.enabled_mcpjson_servers, other.enabled_mcpjson_servers)
        merged_disabled_mcpjson_servers = _merge_arrays(self.disabled_mcpjson_servers, other.disabled_mcpjson_servers)
        merged_allowed_mcp_servers = _merge_dicts(self.allowed_mcp_servers, other.allowed_mcp_servers)
        merged_denied_mcp_servers = _merge_dicts(self.denied_mcp_servers, other.denied_mcp_servers)

        # Merge dicts
        merged_hooks = _merge_dicts(self.hooks, other.hooks)
        merged_model_overrides = _merge_dicts(self.model_overrides, other.model_overrides)

        # Merge extra
        merged_extra = _merge_dicts(self._extra, other._extra)

        return SettingsJson(
            env=merged_env,
            api_key_helper=other.api_key_helper or self.api_key_helper,
            aws_credential_export=other.aws_credential_export or self.aws_credential_export,
            aws_auth_refresh=other.aws_auth_refresh or self.aws_auth_refresh,
            gcp_auth_refresh=other.gcp_auth_refresh or self.gcp_auth_refresh,
            permissions=merged_permissions,
            model=other.model or self.model,
            available_models=merged_available_models,
            model_overrides=merged_model_overrides,
            attribution=other.attribution or self.attribution,
            include_co_authored_by=other.include_co_authored_by if other.include_co_authored_by is not None else self.include_co_authored_by,
            include_git_instructions=other.include_git_instructions if other.include_git_instructions is not None else self.include_git_instructions,
            enable_all_project_mcp_servers=other.enable_all_project_mcp_servers if other.enable_all_project_mcp_servers is not None else self.enable_all_project_mcp_servers,
            enabled_mcpjson_servers=merged_enabled_mcpjson_servers,
            disabled_mcpjson_servers=merged_disabled_mcpjson_servers,
            allowed_mcp_servers=merged_allowed_mcp_servers,
            denied_mcp_servers=merged_denied_mcp_servers,
            hooks=merged_hooks,
            disable_all_hooks=other.disable_all_hooks if other.disable_all_hooks is not None else self.disable_all_hooks,
            allow_managed_hooks_only=other.allow_managed_hooks_only if other.allow_managed_hooks_only is not None else self.allow_managed_hooks_only,
            status_line=other.status_line or self.status_line,
            cleanup_period_days=other.cleanup_period_days if other.cleanup_period_days is not None else self.cleanup_period_days,
            output_style=other.output_style or self.output_style,
            language=other.language or self.language,
            syntax_highlighting_disabled=other.syntax_highlighting_disabled if other.syntax_highlighting_disabled is not None else self.syntax_highlighting_disabled,
            spinner_tips_enabled=other.spinner_tips_enabled if other.spinner_tips_enabled is not None else self.spinner_tips_enabled,
            always_thinking_enabled=other.always_thinking_enabled if other.always_thinking_enabled is not None else self.always_thinking_enabled,
            effort_level=other.effort_level or self.effort_level,
            fast_mode=other.fast_mode if other.fast_mode is not None else self.fast_mode,
            auto_memory_enabled=other.auto_memory_enabled if other.auto_memory_enabled is not None else self.auto_memory_enabled,
            auto_memory_directory=other.auto_memory_directory or self.auto_memory_directory,
            default_shell=other.default_shell or self.default_shell,
            respect_gitignore=other.respect_gitignore if other.respect_gitignore is not None else self.respect_gitignore,
            file_suggestion=other.file_suggestion or self.file_suggestion,
            auto_updates_channel=other.auto_updates_channel or self.auto_updates_channel,
            _extra=merged_extra,
        )


def _merge_arrays(a: Optional[List[Any]], b: Optional[List[Any]]) -> List[Any]:
    """Merge arrays, concatenating and deduplicating."""
    result: List[Any] = []
    if a:
        result.extend(a)
    if b:
        for item in b:
            if item not in result:
                result.append(item)
    return result


def _merge_dicts(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge dicts, b taking priority."""
    result: Dict[str, Any] = {}
    if a:
        result.update(a)
    if b:
        result.update(b)
    return result