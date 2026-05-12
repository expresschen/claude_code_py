"""Compact service prompts.

This module provides prompts for conversation compaction, including:
- NO_TOOLS_PREAMBLE - preamble to prevent tool calls
- DETAILED_ANALYSIS_INSTRUCTION - instructions for <analysis> block
- BASE_COMPACT_PROMPT - full conversation compact
- PARTIAL_COMPACT_PROMPT - partial conversation compact
- PARTIAL_COMPACT_UP_TO_PROMPT - compact from retained context
- format_compact_summary() - strip <analysis> block and format

Ported from TypeScript src/services/compact/prompt.ts
"""

from __future__ import annotations

import re
from typing import Optional


# =============================================================================
# Constants
# =============================================================================

# Aggressive no-tools preamble. Prevents the model from attempting tool calls
# during compaction, which would waste the turn since maxTurns is 1.
NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

# Trailer to reinforce no-tools rule
NO_TOOLS_TRAILER = """
REMINDER: Do NOT call any tools. Respond with plain text only —
an <analysis> block followed by a <summary> block.
Tool calls will be rejected and you will fail the task.
"""


# =============================================================================
# Analysis Instructions
# =============================================================================

# Instructions for the <analysis> drafting scratchpad
# BASE version scopes to "the conversation"
DETAILED_ANALYSIS_INSTRUCTION_BASE = """Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly."""

# PARTIAL version scopes to "the recent messages"
DETAILED_ANALYSIS_INSTRUCTION_PARTIAL = """Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Analyze the recent messages chronologically. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly."""


# =============================================================================
# Base Compact Prompt
# =============================================================================

BASE_COMPACT_PROMPT_TEMPLATE = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

{analysis_instruction}

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating your summary. Examples of instructions include:
<example>
## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>
"""


# =============================================================================
# Partial Compact Prompt (from retained context)
# =============================================================================

PARTIAL_COMPACT_PROMPT_TEMPLATE = """Your task is to create a detailed summary of the RECENT portion of the conversation — the messages that follow earlier retained context. The earlier messages are being kept intact and do NOT need to be summarized. Focus your summary on what was discussed, learned, and accomplished in the recent messages only.

{analysis_instruction}

Your summary should include the following sections:

1. Primary Request and Intent: Capture the user's explicit requests and intents from the recent messages
2. Key Technical Concepts: List important technical concepts, technologies, and frameworks discussed recently.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List errors encountered and how they were fixed.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages from the recent portion that are not tool results.
7. Pending Tasks: Outline any pending tasks from the recent messages.
8. Current Work: Describe precisely what was being worked on immediately before this summary request.
9. Optional Next Step: List the next step related to the most recent work. Include direct quotes from the most recent conversation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Important Code Snippet]

4. Errors and fixes:
    - [Error description]:
      - [How you fixed it]

5. Problem Solving:
   [Description]

6. All user messages:
    - [Detailed non tool use user message]

7. Pending Tasks:
   - [Task 1]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the RECENT messages only (after the retained earlier context), following this structure and ensuring precision and thoroughness in your response.
"""


# =============================================================================
# Partial Compact Up-To Prompt
# =============================================================================

PARTIAL_COMPACT_UP_TO_PROMPT_TEMPLATE = """Your task is to create a detailed summary of this conversation. This summary will be placed at the start of a continuing session; newer messages that build on this context will follow after your summary (you do not see them here). Summarize thoroughly so that someone reading only your summary and then the newer messages can fully understand what happened and continue the work.

{analysis_instruction}

Your summary should include the following sections:

1. Primary Request and Intent: Capture the user's explicit requests and intents in detail
2. Key Technical Concepts: List important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List errors encountered and how they were fixed.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results.
7. Pending Tasks: Outline any pending tasks.
8. Work Completed: Describe what was accomplished by the end of this portion.
9. Context for Continuing Work: Summarize any context, decisions, or state that would be needed to understand and continue the work in subsequent messages.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Important Code Snippet]

4. Errors and fixes:
    - [Error description]:
      - [How you fixed it]

5. Problem Solving:
   [Description]

6. All user messages:
    - [Detailed non tool use user message]

7. Pending Tasks:
   - [Task 1]

8. Work Completed:
   [Description of what was accomplished]

9. Context for Continuing Work:
   [Key context, decisions, or state needed to continue the work]

</summary>
</example>

Please provide your summary following this structure, ensuring precision and thoroughness in your response.
"""


# =============================================================================
# Prompt Generation Functions
# =============================================================================


def get_base_compact_prompt(custom_instructions: Optional[str] = None) -> str:
    """Get base compact prompt for full conversation summarization.

    Args:
        custom_instructions: Optional custom instructions to append

    Returns:
        Complete compact prompt string
    """
    prompt = NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT_TEMPLATE.format(
        analysis_instruction=DETAILED_ANALYSIS_INSTRUCTION_BASE
    )

    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"

    prompt += NO_TOOLS_TRAILER

    return prompt


def get_partial_compact_prompt(
    custom_instructions: Optional[str] = None,
    direction: str = "from",
) -> str:
    """Get partial compact prompt for summarizing recent messages.

    Args:
        custom_instructions: Optional custom instructions to append
        direction: "from" for recent messages after retained context,
                   "up_to" for summary that precedes newer messages

    Returns:
        Complete partial compact prompt string
    """
    template = (
        PARTIAL_COMPACT_UP_TO_PROMPT_TEMPLATE
        if direction == "up_to"
        else PARTIAL_COMPACT_PROMPT_TEMPLATE
    )

    prompt = NO_TOOLS_PREAMBLE + template.format(
        analysis_instruction=(
            DETAILED_ANALYSIS_INSTRUCTION_BASE
            if direction == "up_to"
            else DETAILED_ANALYSIS_INSTRUCTION_PARTIAL
        )
    )

    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"

    prompt += NO_TOOLS_TRAILER

    return prompt


def format_compact_summary(summary: str) -> str:
    """Format the compact summary by stripping analysis and formatting tags.

    The <analysis> block is a drafting scratchpad that improves summary quality
    but has no informational value once the summary is written.

    Args:
        summary: Raw summary string potentially containing <analysis> and <summary> tags

    Returns:
        Formatted summary with analysis stripped and summary tags replaced
    """
    formatted = summary

    # Strip analysis section - it's just a drafting scratchpad
    formatted = re.sub(r"<analysis>[\s\S]*?</analysis>", "", formatted)

    # Extract and format summary section
    summary_match = re.search(r"<summary>([\s\S]*?)</summary>", formatted)
    if summary_match:
        content = summary_match.group(1) or ""
        formatted = re.sub(
            r"<summary>[\s\S]*?</summary>",
            f"Summary:\n{content.strip()}",
            formatted,
        )

    # Clean up extra whitespace between sections
    formatted = re.sub(r"\n\n+", "\n\n", formatted)

    return formatted.strip()


def get_compact_user_summary_message(
    summary: str,
    suppress_follow_up_questions: bool = False,
    transcript_path: Optional[str] = None,
    recent_messages_preserved: bool = False,
) -> str:
    """Generate the user-facing message for after compaction.

    Args:
        summary: The formatted summary content
        suppress_follow_up_questions: Whether to suppress questions and continue directly
        transcript_path: Optional path to full transcript
        recent_messages_preserved: Whether recent messages are preserved verbatim

    Returns:
        User-facing message string
    """
    formatted_summary = format_compact_summary(summary)

    base = f"""This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

{formatted_summary}"""

    if transcript_path:
        base += f"\n\nIf you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: {transcript_path}"

    if recent_messages_preserved:
        base += "\n\nRecent messages are preserved verbatim."

    if suppress_follow_up_questions:
        return f"""{base}
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened."""

    return base


# =============================================================================
# Legacy Simple Prompt (for backwards compatibility)
# =============================================================================


def get_simple_compact_prompt() -> str:
    """Get simple compact prompt (legacy, no analysis block).

    This is used for backwards compatibility with the old compact system.

    Returns:
        Simple compact prompt string
    """
    return """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created.
4. Errors and fixes: List all errors that you ran into, and how you fixed them.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary.
9. Optional Next Step: List the next step that you will take, if applicable.

Format your response as:
<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]

3. Files and Code Sections:
   - [File Name]
     - [Why important]
     - [Code snippet or changes]

4. Errors and fixes:
   - [Error]: [How fixed]

5. Problem Solving:
   [Description]

6. All user messages:
   - [Message 1]
   - [Message 2]

7. Pending Tasks:
   - [Task 1]

8. Current Work:
   [Description]

9. Optional Next Step:
   [Next step if applicable]
</summary>"""