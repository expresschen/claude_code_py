"""Permissions package for Claude Code Python.

This provides the permission checking system including:
- Rule matching
- Auto mode classifier (LLM as judge)
- Dangerous pattern detection
- Denial tracking
"""

from __future__ import annotations

from .rules import (
    PermissionBehavior,
    PermissionRuleSource,
    PermissionRule,
    PermissionRuleContent,
    parse_permission_rule,
    rule_to_string,
    tool_matches_rule,
    tool_matches_rule_with_input,
    get_allow_rules,
    get_deny_rules,
    get_ask_rules,
    find_matching_rule,
    get_deny_rule_for_tool,
    get_ask_rule_for_tool,
    get_allow_rule_for_tool,
    ToolPermissionRulesBySource,
)

from .dangerous_patterns import (
    DangerCategory,
    DangerDetection,
    check_command_dangerous,
    check_file_write_dangerous,
    check_tool_input_dangerous,
    is_critical_danger,
    should_block_without_confirmation,
    requires_explicit_user_confirmation,
)

from .denial_tracking import (
    DenialTrackingState,
    DenialRecord,
    create_denial_tracking_state,
    record_denial,
    record_success,
    get_denial_count,
    get_last_denial_reason,
    get_denial_history_message,
    should_fallback_to_prompting,
    reset_denial_tracking,
    get_session_stats,
)

from .classifier import (
    ClassifierResult,
    ClassifierStage,
    TwoStageMode,
    ClassifierUsage,
    PromptLengths,
    TranscriptEntry,
    AutoModeRules,
    DenialTrackingState,
    get_default_auto_mode_rules,
    is_auto_mode_allowlisted_tool,
    classify_action,
    classify_and_decide,
    build_classifier_system_prompt,
    build_transcript_for_classifier,
    format_action_for_classifier,
    format_bash_for_classifier,
    format_write_for_classifier,
    format_edit_for_classifier,
    format_agent_for_classifier,
    get_classifier_model,
    get_two_stage_mode,
    is_iron_gate_closed,
    get_denial_tracking_state,
    record_denial,
    record_success,
    should_fallback_to_prompting,
    get_denial_history_message,
    update_denial_tracking_state,
    reset_denial_tracking_state,
    check_accept_edits_fast_path,
    build_claude_md_message,
    get_cached_claude_md_content,
    get_last_classifier_requests,
)


__all__ = [
    # Rules
    "PermissionBehavior",
    "PermissionRuleSource",
    "PermissionRule",
    "PermissionRuleContent",
    "parse_permission_rule",
    "rule_to_string",
    "tool_matches_rule",
    "tool_matches_rule_with_input",
    "get_allow_rules",
    "get_deny_rules",
    "get_ask_rules",
    "find_matching_rule",
    "get_deny_rule_for_tool",
    "get_ask_rule_for_tool",
    "get_allow_rule_for_tool",
    "ToolPermissionRulesBySource",

    # Dangerous patterns
    "DangerCategory",
    "DangerDetection",
    "check_command_dangerous",
    "check_file_write_dangerous",
    "check_tool_input_dangerous",
    "is_critical_danger",
    "should_block_without_confirmation",
    "requires_explicit_user_confirmation",

    # Denial tracking
    "DenialTrackingState",
    "DenialRecord",
    "create_denial_tracking_state",
    "record_denial",
    "record_success",
    "get_denial_count",
    "get_last_denial_reason",
    "get_denial_history_message",
    "should_fallback_to_prompting",
    "reset_denial_tracking",
    "get_session_stats",

    # Classifier
    "ClassifierResult",
    "ClassifierStage",
    "TwoStageMode",
    "ClassifierUsage",
    "PromptLengths",
    "TranscriptEntry",
    "AutoModeRules",
    "DenialTrackingState",
    "get_default_auto_mode_rules",
    "is_auto_mode_allowlisted_tool",
    "classify_action",
    "classify_and_decide",
    "build_classifier_system_prompt",
    "build_transcript_for_classifier",
    "format_action_for_classifier",
    "format_bash_for_classifier",
    "format_write_for_classifier",
    "format_edit_for_classifier",
    "format_agent_for_classifier",
    "get_classifier_model",
    "get_two_stage_mode",
    "is_iron_gate_closed",
    "get_denial_tracking_state",
    "record_denial",
    "record_success",
    "should_fallback_to_prompting",
    "get_denial_history_message",
    "update_denial_tracking_state",
    "reset_denial_tracking_state",
    "check_accept_edits_fast_path",
    "build_claude_md_message",
    "get_cached_claude_md_content",
    "get_last_classifier_requests",
]