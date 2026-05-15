"""Agent tool prompt generation.

This mirrors the TypeScript getPrompt function with enhancements for:
- Fork subagent feature (when enabled)
- "Don't race" and "Don't peek" guidance
- Fork-specific examples
- Directive-style prompt writing instructions

Ported from TypeScript: src/tools/AgentTool/prompt.ts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    from .types import AgentDefinition


# =============================================================================
# Constants
# =============================================================================

AGENT_TOOL_NAME = "Agent"
FILE_READ_TOOL_NAME = "Read"
FILE_WRITE_TOOL_NAME = "Write"
GLOB_TOOL_NAME = "Glob"
SEND_MESSAGE_TOOL_NAME = "SendMessage"


# =============================================================================
# Helper Functions
# =============================================================================


def get_tools_description(agent: "AgentDefinition") -> str:
    """Get description of tools available to an agent."""
    tools = agent.tools or []
    disallowed_tools = agent.disallowed_tools or []

    has_allowlist = tools and len(tools) > 0
    has_denylist = disallowed_tools and len(disallowed_tools) > 0

    if has_allowlist and has_denylist:
        # Both defined: filter allowlist by denylist
        deny_set = set(disallowed_tools)
        effective_tools = [t for t in tools if t not in deny_set]
        if not effective_tools:
            return "None"
        return ", ".join(effective_tools)
    elif has_allowlist:
        return ", ".join(tools)
    elif has_denylist:
        return f"All tools except {', '.join(disallowed_tools)}"
    else:
        return "All tools"


def format_agent_line(agent: "AgentDefinition") -> str:
    """Format one agent line for listing."""
    tools_desc = get_tools_description(agent)
    return f"- {agent.agent_type}: {agent.when_to_use} (Tools: {tools_desc})"


# =============================================================================
# When to Fork Section
# =============================================================================


WHEN_TO_FORK_SECTION = """

## When to fork

Fork yourself (omit `subagent_type`) when the intermediate tool output isn't worth keeping in your context. The criterion is qualitative — "will I need this output again" — not task size.
- **Research**: fork open-ended questions. If research can be broken into independent questions, launch parallel forks in one message. A fork beats a fresh subagent for this — it inherits context and shares your cache.
- **Implementation**: prefer to fork implementation work that requires more than a couple of edits. Do research before jumping to implementation.

Forks are cheap because they share your prompt cache. Don't set `model` on a fork — a different model can't reuse the parent's cache. Pass a short `name` (one or two words, lowercase) so the user can see the fork in the teams panel and steer it mid-run.

**Don't peek.** Background agents return results via notification — do not try to read intermediate files or poll for progress. You get a completion notification when they finish; trust it.

**Don't race.** After launching, you know nothing about what the fork found. Never fabricate or predict fork results in any format — not as prose, summary, or structured output. The notification arrives as a user-role message in a later turn; it is never something you write yourself. If the user asks a follow-up before the notification lands, tell them the fork is still running — give status, not a guess.

**Writing a fork prompt.** Since the fork inherits your context, the prompt is a *directive* — what to do, not what the situation is. Be specific about scope: what's in, what's out, what another agent is handling. Don't re-explain background.
"""


# =============================================================================
# Writing the Prompt Section
# =============================================================================


WRITING_PROMPT_SECTION_STANDARD = """

## Writing the prompt

Brief the agent like a smart colleague who just walked into the room — it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters.
- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context about the surrounding problem that the agent can make judgment calls rather than just following a narrow instruction.
- If you need a short response, say so ("report in under 200 words").
- Lookups: hand over the exact command. Investigations: hand over the question — prescribed steps become dead weight when the premise is wrong.

Terse command-style prompts produce shallow, generic work.

**Never delegate understanding.** Don't write "based on your findings, fix the bug" or "based on the research, implement it." Those phrases push synthesis onto the agent instead of doing it yourself. Write prompts that prove you understood: include file paths, line numbers, what specifically to change.
"""


WRITING_PROMPT_SECTION_FORK = """

## Writing the prompt

When spawning a fresh agent (with a `subagent_type`), it starts with zero context. Brief the agent like a smart colleague who just walked into the room — it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters.
- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context about the surrounding problem that the agent can make judgment calls rather than just following a narrow instruction.
- If you need a short response, say so ("report in under 200 words").
- Lookups: hand over the exact command. Investigations: hand over the question — prescribed steps become dead weight when the premise is wrong.

For fresh agents, terse command-style prompts produce shallow, generic work.

**Never delegate understanding.** Don't write "based on your findings, fix the bug" or "based on the research, implement it." Those phrases push synthesis onto the agent instead of doing it yourself. Write prompts that prove you understood: include file paths, line numbers, what specifically to change.
"""


# =============================================================================
# Fork Examples
# =============================================================================


FORK_EXAMPLES = """Example usage:

<example>
user: "What's left on this branch before we can ship?"
assistant: <thinking>Forking this — it's a survey question. I want the punch list, not the git output in my context.</thinking>
{AGENT_TOOL_NAME}({
  name: "ship-audit",
  description: "Branch ship-readiness audit",
  prompt: "Audit what's left before this branch can ship. Check: uncommitted changes, commits ahead of main, whether tests exist, whether the GrowthBook gate is wired up, whether CI-relevant files changed. Report a punch list — done vs. missing. Under 200 words."
})
assistant: Ship-readiness audit running.
<commentary>
Turn ends here. The coordinator knows nothing about the findings yet. What follows is a SEPARATE turn — the notification arrives from outside, as a user-role message. It is not something the coordinator writes.
</commentary>
[later turn — notification arrives as user message]
assistant: Audit's back. Three blockers: no tests for the new prompt path, GrowthBook gate wired but not in build_flags.yaml, and one uncommitted file.
</example>

<example>
user: "so is the gate wired up or not"
<commentary>
User asks mid-wait. The audit fork was launched to answer exactly this, and it hasn't returned. The coordinator does not have this answer. Give status, not a fabricated result.
</commentary>
assistant: Still waiting on the audit — that's one of the things it's checking. Should land shortly.
</example>

<example>
user: "Can you get a second opinion on whether this migration is safe?"
assistant: <thinking>I'll ask the code-reviewer agent — it won't see my analysis, so it can give an independent read.</thinking>
<commentary>
A subagent_type is specified, so the agent starts fresh. It needs full context in the prompt. The briefing explains what to assess and why.
</commentary>
{AGENT_TOOL_NAME}({
  name: "migration-review",
  description: "Independent migration review",
  subagent_type: "code-reviewer",
  prompt: "Review migration 0042_user_schema.sql for safety. Context: we're adding a NOT NULL column to a 50M-row table. Existing rows get a backfill default. I want a second opinion on whether the backfill approach is safe under concurrent writes — I've checked locking behavior but want independent verification. Report: is this safe, and if not, what specifically breaks?"
})
</example>
"""


CURRENT_EXAMPLES = """Example usage:

<example_agent_descriptions>
"test-runner": use this agent after you are done writing code to run tests
"greeting-responder": use this agent to respond to user greetings with a friendly joke
</example_agent_descriptions>

<example>
user: "Please write a function that checks if a number is prime"
assistant: I'm going to use the {FILE_WRITE_TOOL_NAME} tool to write the following code:
<code>
function isPrime(n) {
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {
    if (n % i === 0) return false
  }
  return true
}
</code>
<commentary>
Since a significant piece of code was written and the task was completed, now use the test-runner agent to run the tests
</commentary>
assistant: Uses the {AGENT_TOOL_NAME} tool to launch the test-runner agent
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the {AGENT_TOOL_NAME} tool to launch the greeting-responder agent"
</example>
"""


# =============================================================================
# Main Prompt Generation
# =============================================================================


def get_prompt(
    agent_definitions: List["AgentDefinition"],
    is_coordinator: bool = False,
    allowed_agent_types: Optional[List[str]] = None,
    fork_enabled: bool = False,
) -> str:
    """Generate the Agent tool prompt.

    Args:
        agent_definitions: List of available agent definitions
        is_coordinator: Whether running in coordinator mode
        allowed_agent_types: Optional filter for allowed agent types
        fork_enabled: Whether fork subagent feature is enabled

    Returns:
        Tool prompt string
    """
    # Filter agents by allowed types
    if allowed_agent_types:
        effective_agents = [
            a for a in agent_definitions
            if a.agent_type in allowed_agent_types
        ]
    else:
        effective_agents = agent_definitions

    # Format agent list
    agent_list_lines = [format_agent_line(a) for a in effective_agents]
    agent_list_section = "\n".join(agent_list_lines)

    # Fork section (when enabled)
    when_to_fork_section = WHEN_TO_FORK_SECTION if fork_enabled else ""

    # Writing prompt section
    writing_prompt_section = WRITING_PROMPT_SECTION_FORK if fork_enabled else WRITING_PROMPT_SECTION_STANDARD

    # Shared core prompt
    subagent_type_desc = (
        "When using the Agent tool, specify a subagent_type to use a specialized agent, "
        "or omit it to fork yourself — a fork inherits your full conversation context."
        if fork_enabled
        else "When using the Agent tool, specify a subagent_type parameter to select which agent type to use."
    )

    shared = f"""Launch a new agent to handle complex, multi-step tasks autonomously.

The Agent tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.

Available agent types and the tools they have access to:
{agent_list_section}

{subagent_type_desc}"""

    # Coordinator mode gets slim prompt
    if is_coordinator:
        return shared

    # When NOT to use section
    when_not_to_use = ""
    if not fork_enabled:
        when_not_to_use = """
When NOT to use the Agent tool:
- If you want to read a specific file path, use the Read tool or Glob tool instead, to find the match more quickly
- If you are searching for a specific class definition like "class Foo", use the Glob tool instead, to find the match more quickly
- If you are searching for code within a specific file or set of 2-3 files, use the Read tool instead, to find the match more quickly
- Other tasks that are not related to the agent descriptions above
"""

    # Full prompt with all sections
    return f"""{shared}{when_not_to_use}

Usage notes:
- Always include a short description (3-5 words) summarizing what the agent will do
- Launch multiple agents concurrently whenever possible, to maximize performance
- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
- You can optionally run agents in the background using the run_in_background parameter. When an agent runs in the background, you will be automatically notified when it completes — do NOT sleep, poll, or proactively check on its progress.
- **Foreground vs background**: Use foreground (default) when you need the agent's results before you can proceed. Use background when you have genuinely independent work to do in parallel.
- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.)
- If the agent description mentions that it should be used proactively, then you should try to use it without the user having to ask for it first.{when_to_fork_section}{writing_prompt_section}

{FORK_EXAMPLES if fork_enabled else CURRENT_EXAMPLES}
"""


# =============================================================================
# Description
# =============================================================================


DESCRIPTION = "Launches specialized agents to handle complex, multi-step tasks autonomously"