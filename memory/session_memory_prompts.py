"""Session Memory prompts for extraction.

This provides the prompt templates for session memory extraction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# =============================================================================
# Constants (matching TypeScript)
# =============================================================================


# Maximum tokens per section
MAX_SECTION_LENGTH = 2000

# Maximum total session memory tokens
MAX_TOTAL_SESSION_MEMORY_TOKENS = 12000


# =============================================================================
# Default Template (matching TypeScript exactly)
# =============================================================================


DEFAULT_SESSION_MEMORY_TEMPLATE = """
# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output if not obvious?_

# Errors & Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed and should not be tried again?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key results
_If the user asked a specific output such as an answer to a question, a table, or other document, repeat the exact result here_

# Worklog
_Step by step, what was attempted, done? Very terse summary for each step_
"""


def load_session_memory_template() -> str:
    """Load the session memory template.

    Returns:
        Template content
    """
    # Check for custom template in project (matching TypeScript loadSessionMemoryTemplate)
    template_path = Path(os.path.expanduser("~/.claude/session-memory/config/template.md"))

    if template_path.exists():
        try:
            return template_path.read_text(encoding="utf-8")
        except Exception:
            pass

    # Use default template
    return DEFAULT_SESSION_MEMORY_TEMPLATE


# =============================================================================
# Extraction Prompts (matching TypeScript)
# =============================================================================


def get_default_update_prompt() -> str:
    """Get the default prompt for session memory update.

    Returns:
        Default update prompt (matching TypeScript getDefaultUpdatePrompt)
    """
    return """IMPORTANT: This message and these instructions are NOT part of the actual user conversation. Do NOT include any references to "note-taking", "session notes extraction", or these update instructions in the notes content.

Based on the user conversation above (EXCLUDING this note-taking instruction message as well as system prompt, claude.md entries, or any past session summaries), update the session notes file.

The file {{notesPath}} has already been read for you. Here are its current contents:
<current_notes_content>
{{currentNotes}}
</current_notes_content>

Your ONLY task is to use the Edit tool to update the notes file, then stop. You can make multiple edits (update every section as needed) - make all Edit tool calls in parallel in a single message. Do not call any other tools.

CRITICAL RULES FOR EDITING:
- The file must maintain its exact structure with all sections, headers, and italic descriptions intact
-- NEVER modify, delete, or add section headers (the lines starting with '#' like # Task specification)
-- NEVER modify or delete the italic _section description_ lines (these are the lines in italics immediately following each header - they start and end with underscores)
-- The italic _section descriptions_ are TEMPLATE INSTRUCTIONS that must be preserved exactly as-is - they guide what content belongs in each section
-- ONLY update the actual content that appears BELOW the italic _section descriptions_ within each existing section
-- Do NOT add any new sections, summaries, or information outside the existing structure
- Do NOT reference this note-taking process or instructions anywhere in the notes
- It's OK to skip updating a section if there are no substantial new insights to add. Do not add filler content like "No info yet", just leave sections blank/unedited if appropriate.
- Write DETAILED, INFO-DENSE content for each section - include specifics like file paths, function names, error messages, exact commands, technical details, etc.
- For "Key results", include the complete, exact output the user requested (e.g., full table, full answer, etc.)
- Do not include information that's already in the CLAUDE.md files included in the context
- Keep each section under ~{MAX_SECTION_LENGTH} tokens/words - if a section is approaching this limit, condense it by cycling out less important details while preserving the most critical information
- Focus on actionable, specific information that would help someone understand or recreate the work discussed in the conversation
- IMPORTANT: Always update "Current State" to reflect the most recent work - this is critical for continuity after compaction

Use the Edit tool with file_path: {{notesPath}}

STRUCTURE PRESERVATION REMINDER:
Each section has TWO parts that must be preserved exactly as they appear in the current file:
1. The section header (line starting with #)
2. The italic description line (the _italicized text_ immediately after the header - this is a template instruction)

You ONLY update the actual content that comes AFTER these two preserved lines. The italic description lines starting and ending with underscores are part of the template structure, NOT content to be edited or removed.

REMEMBER: Use the Edit tool in parallel and stop. Do not continue after the edits. Only include insights from the actual user conversation, never from these note-taking instructions. Do not delete or change section headers or italic _section descriptions_."""


def load_session_memory_prompt() -> str:
    """Load custom session memory prompt from file if it exists.

    Returns:
        Prompt content
    """
    prompt_path = Path(os.path.expanduser("~/.claude/session-memory/config/prompt.md"))

    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8")
        except Exception:
            pass

    return get_default_update_prompt()


def substitute_variables(template: str, variables: dict[str, str]) -> str:
    """Substitute variables in the prompt template using {{variable}} syntax.

    Args:
        template: Template string with {{variable}} placeholders
        variables: Dictionary of variable names to values

    Returns:
        Template with variables substituted
    """
    import re
    # Single-pass replacement to avoid $ backreference corruption and double-substitution
    return re.sub(
        r"\{\{(\w+)\}\}",
        lambda match: variables.get(match.group(1), match.group(0)),
        template,
    )


def analyze_section_sizes(content: str) -> dict[str, int]:
    """Parse the session memory file and analyze section sizes.

    Args:
        content: Session memory content

    Returns:
        Dictionary of section name -> token count
    """
    from claude_code_py.utils.context import rough_token_count_estimation

    sections: dict[str, int] = {}
    lines = content.split("\n")
    current_section = ""
    current_content: list[str] = []

    for line in lines:
        if line.startswith("# "):
            if current_section and current_content:
                section_content = "\n".join(current_content).strip()
                sections[current_section] = rough_token_count_estimation(section_content)
            current_section = line
            current_content = []
        else:
            current_content.append(line)

    if current_section and current_content:
        section_content = "\n".join(current_content).strip()
        sections[current_section] = rough_token_count_estimation(section_content)

    return sections


def generate_section_reminders(
    section_sizes: dict[str, int],
    total_tokens: int,
) -> str:
    """Generate reminders for sections that are too long.

    Args:
        section_sizes: Dictionary of section sizes
        total_tokens: Total token count

    Returns:
        Reminder string to append to prompt
    """
    over_budget = total_tokens > MAX_TOTAL_SESSION_MEMORY_TOKENS
    oversized_sections = sorted(
        [(section, tokens) for section, tokens in section_sizes.items() if tokens > MAX_SECTION_LENGTH],
        key=lambda x: x[1],
        reverse=True,
    )

    if not oversized_sections and not over_budget:
        return ""

    parts: list[str] = []

    if over_budget:
        parts.append(
            f"\n\nCRITICAL: The session memory file is currently ~{total_tokens} tokens, "
            f"which exceeds the maximum of {MAX_TOTAL_SESSION_MEMORY_TOKENS} tokens. "
            "You MUST condense the file to fit within this budget. Aggressively shorten oversized sections "
            "by removing less important details, merging related items, and summarizing older entries. "
            "Prioritize keeping \"Current State\" and \"Errors & Corrections\" accurate and detailed."
        )

    if oversized_sections:
        section_list = "\n".join(
            f'- "{section}" is ~{tokens} tokens (limit: {MAX_SECTION_LENGTH})'
            for section, tokens in oversized_sections
        )
        parts.append(
            f"\n\n{over_budget and 'Oversized sections to condense' or 'IMPORTANT: The following sections exceed the per-section limit and MUST be condensed'}:\n{section_list}"
        )

    return "".join(parts)


def build_session_memory_update_prompt(
    current_notes: str,
    notes_path: str,
) -> str:
    """Build the prompt for session memory extraction.

    Args:
        current_notes: Current session memory content
        notes_path: Path to the memory file

    Returns:
        Prompt for extraction
    """
    prompt_template = load_session_memory_prompt()

    # Analyze section sizes and generate reminders if needed
    section_sizes = analyze_section_sizes(current_notes)
    from claude_code_py.utils.context import rough_token_count_estimation
    total_tokens = rough_token_count_estimation(current_notes)
    section_reminders = generate_section_reminders(section_sizes, total_tokens)

    # Substitute variables in the prompt
    variables = {
        "currentNotes": current_notes,
        "notesPath": notes_path,
    }

    base_prompt = substitute_variables(prompt_template, variables)

    # Add section size reminders and/or total budget warnings
    return base_prompt + section_reminders


def build_session_memory_init_prompt(memory_path: str) -> str:
    """Build the prompt for initializing session memory.

    Args:
        memory_path: Path to the memory file

    Returns:
        Prompt for initialization
    """
    template = load_session_memory_template()

    return f"""Initialize the session memory file for this conversation.

## Template

The initial template should be:

```
{template}
```

## Instructions

1. Create the file at `{memory_path}` with the template content
2. Review the conversation so far
3. Add initial summary based on what has been discussed
4. Include any early decisions or context that's important to remember

## Output Format

Use the Write tool to create the file if it doesn't exist, or Edit to update if it does."""


def is_session_memory_empty(content: str) -> bool:
    """Check if the session memory content is essentially empty (matches the template).

    Args:
        content: Session memory content

    Returns:
        True if content matches template (no actual content extracted)
    """
    template = load_session_memory_template()
    # Compare trimmed content to detect if it's just the template
    return content.strip() == template.strip()


def truncate_session_memory_for_compact(content: str) -> tuple[str, bool]:
    """Truncate session memory sections that exceed the per-section token limit.

    Used when inserting session memory into compact messages to prevent
    oversized session memory from consuming the entire post-compact token budget.

    Args:
        content: Session memory content

    Returns:
        Tuple of (truncated content, was truncated flag)
    """
    lines = content.split("\n")
    max_chars_per_section = MAX_SECTION_LENGTH * 4  # roughTokenCountEstimation uses length/4
    output_lines: list[str] = []
    current_section_lines: list[str] = []
    current_section_header = ""
    was_truncated = False

    for line in lines:
        if line.startswith("# "):
            result = _flush_section(current_section_header, current_section_lines, max_chars_per_section)
            output_lines.extend(result["lines"])
            was_truncated = was_truncated or result["was_truncated"]
            current_section_header = line
            current_section_lines = []
        else:
            current_section_lines.append(line)

    # Flush the last section
    result = _flush_section(current_section_header, current_section_lines, max_chars_per_section)
    output_lines.extend(result["lines"])
    was_truncated = was_truncated or result["was_truncated"]

    return "\n".join(output_lines), was_truncated


def _flush_section(
    section_header: str,
    section_lines: list[str],
    max_chars_per_section: int,
) -> dict:
    """Flush a section to output, truncating if needed.

    Args:
        section_header: Section header line
        section_lines: Content lines
        max_chars_per_section: Maximum characters allowed

    Returns:
        Dict with 'lines' and 'was_truncated' keys
    """
    if not section_header:
        return {"lines": section_lines, "was_truncated": False}

    section_content = "\n".join(section_lines)
    if len(section_content) <= max_chars_per_section:
        return {"lines": [section_header] + section_lines, "was_truncated": False}

    # Truncate at a line boundary near the limit
    char_count = 0
    kept_lines: list[str] = [section_header]
    for line in section_lines:
        if char_count + len(line) + 1 > max_chars_per_section:
            break
        kept_lines.append(line)
        char_count += len(line) + 1
    kept_lines.append("\n[... section truncated for length ...]")
    return {"lines": kept_lines, "was_truncated": True}


# =============================================================================
# Session Memory Content Loading
# =============================================================================


async def get_session_memory_content(memory_path: Path) -> Optional[str]:
    """Get the current session memory content.

    Args:
        memory_path: Path to memory file

    Returns:
        Content or None if file doesn't exist
    """
    if not memory_path.exists():
        return None

    try:
        return memory_path.read_text(encoding="utf-8")
    except Exception:
        return None