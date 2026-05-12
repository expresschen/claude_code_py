"""Plan mode prompts.

This module provides all plan mode-related prompts including:
- EnterPlanMode tool prompt
- ExitPlanMode tool prompt
- Plan Workflow instructions (5-Phase and Interview Phase)
- Plan mode reentry/exit messages

Ported from TypeScript:
- src/tools/EnterPlanModeTool/prompt.ts
- src/tools/ExitPlanModeTool/prompt.ts
- src/utils/messages.ts (getPlanModeV2Instructions, getPlanModeInterviewInstructions)
"""

from __future__ import annotations

from typing import Optional


# =============================================================================
# Constants
# =============================================================================

# Agent types (used in workflow descriptions)
EXPLORE_AGENT_TYPE = "Explore"
PLAN_AGENT_TYPE = "Plan"

# Default agent counts (simplified - TypeScript has dynamic configuration)
DEFAULT_EXPLORE_AGENT_COUNT = 3
DEFAULT_PLAN_AGENT_COUNT = 1

# Tool names referenced in prompts
FILE_EDIT_TOOL_NAME = "Edit"
FILE_WRITE_TOOL_NAME = "Write"
FILE_READ_TOOL_NAME = "Read"
GLOB_TOOL_NAME = "Glob"
GREP_TOOL_NAME = "Grep"
ASK_USER_QUESTION_TOOL_NAME = "AskUserQuestion"
EXIT_PLAN_MODE_TOOL_NAME = "ExitPlanMode"


# =============================================================================
# Phase 4 Variants (simplified - TypeScript has trim/cut/cap variants)
# =============================================================================

PLAN_PHASE4_DEFAULT = """### Phase 4: Final Plan
Goal: Write your final plan to the plan file (the only file you can edit).
- Do NOT write a Context, Background, or Overview section. The user just told you what they want.
- Do NOT restate the user's request. Do NOT write prose paragraphs.
- List the paths of files to be modified and what changes in each (one bullet per file)
- Reference existing functions to reuse, with file:line
- End with the single verification command"""


# =============================================================================
# EnterPlanMode Tool Prompt
# =============================================================================


def get_enter_plan_mode_prompt() -> str:
    """Get EnterPlanMode tool prompt."""
    return """Use this tool proactively when you're about to start a non-trivial implementation task. Getting user sign-off on your approach before writing code prevents wasted effort and ensures alignment. This tool transitions you into plan mode where you can explore the codebase and design an implementation approach for user approval.

## When to Use This Tool

**Prefer using EnterPlanMode** for implementation tasks unless they're simple. Use it when ANY of these conditions apply:

1. **New Feature Implementation**: Adding meaningful new functionality
   - Example: "Add a logout button" - where should it go? What should happen on click?
   - Example: "Add form validation" - what rules? What error messages?

2. **Multiple Valid Approaches**: The task can be solved in several different ways
   - Example: "Add caching to the API" - could use Redis, in-memory, file-based, etc.
   - Example: "Improve performance" - many optimization strategies possible

3. **Code Modifications**: Changes that affect existing behavior or structure
   - Example: "Update the login flow" - what exactly should change?
   - Example: "Refactor this component" - what's the target architecture?

4. **Architectural Decisions**: The task requires choosing between patterns or technologies
   - Example: "Add real-time updates" - WebSockets vs SSE vs polling
   - Example: "Implement state management" - Redux vs Context vs custom solution

5. **Multi-File Changes**: The task will likely touch more than 2-3 files
   - Example: "Refactor the authentication system"
   - Example: "Add a new API endpoint with tests"

6. **Unclear Requirements**: You need to explore before understanding the full scope
   - Example: "Make the app faster" - need to profile and identify bottlenecks
   - Example: "Fix the bug in checkout" - need to investigate root cause

7. **User Preferences Matter**: The implementation could reasonably go multiple ways
   - If you would use AskUserQuestion to clarify the approach, use EnterPlanMode instead
   - Plan mode lets you explore first, then present options with context

## When NOT to Use This Tool

Only skip EnterPlanMode for simple tasks:
- Single-line or few-line fixes (typos, obvious bugs, small tweaks)
- Adding a single function with clear requirements
- Tasks where the user has given very specific, detailed instructions
- Pure research/exploration tasks (use the Agent tool with explore agent instead)

## What Happens in Plan Mode

In plan mode, you'll:
1. Thoroughly explore the codebase using Glob, Grep, and Read tools
2. Understand existing patterns and architecture
3. Design an implementation approach
4. Present your plan to the user for approval
5. Use AskUserQuestion if you need to clarify approaches
6. Exit plan mode with ExitPlanMode when ready to implement

## Important Notes

- This tool REQUIRES user approval - they must consent to entering plan mode
- If unsure whether to use it, err on the side of planning - it's better to get alignment upfront than to redo work
- Users appreciate being consulted before significant changes are made to their codebase
"""


# =============================================================================
# ExitPlanMode Tool Prompt
# =============================================================================


def get_exit_plan_mode_prompt() -> str:
    """Get ExitPlanMode tool prompt."""
    return """Use this tool when you are in plan mode and have finished writing your plan to the plan file and are ready for user approval.

## How This Tool Works
- You should have already written your plan to the plan file specified in the plan mode system message
- This tool does NOT take the plan content as a parameter - it will read the plan from the file you wrote
- This tool simply signals that you're done planning and ready for the user to review and approve
- The user will see the contents of your plan file when they review it

## When to Use This Tool
IMPORTANT: Only use this tool when the task requires planning the implementation steps of a task that requires writing code. For research tasks where you're gathering information, searching files, reading files or in general trying to understand the codebase - do NOT use this tool.

## Before Using This Tool
Ensure your plan is complete and unambiguous:
- If you have unresolved questions about requirements or approach, use AskUserQuestion first (in earlier phases)
- Once your plan is finalized, use THIS tool to request approval

**Important:** Do NOT use AskUserQuestion to ask "Is this plan okay?" or "Should I proceed?" - that's exactly what THIS tool does. ExitPlanMode inherently requests user approval of your plan.

## Examples

1. Initial task: "Search for and understand the implementation of vim mode in the codebase" - Do not use the exit plan mode tool because you are not planning the implementation steps of a task.
2. Initial task: "Help me implement yank mode for vim" - Use the exit plan mode tool after you have finished planning the implementation steps of the task.
3. Initial task: "Add a new feature to handle user authentication" - If unsure about auth method (OAuth, JWT, etc.), use AskUserQuestion first, then use exit plan mode tool after clarifying the approach.
"""


# =============================================================================
# Plan Mode Instructions (5-Phase Workflow)
# =============================================================================


def get_plan_mode_v2_instructions(
    plan_file_path: str,
    plan_exists: bool = False,
    explore_agent_count: int = DEFAULT_EXPLORE_AGENT_COUNT,
    plan_agent_count: int = DEFAULT_PLAN_AGENT_COUNT,
    is_interview_phase: bool = False,
) -> str:
    """Get plan mode v2 5-phase workflow instructions.

    This is the message that gets injected when entering plan mode.

    Args:
        plan_file_path: Path to the plan file
        plan_exists: Whether a plan file already exists
        explore_agent_count: Number of explore agents allowed
        plan_agent_count: Number of plan agents allowed
        is_interview_phase: Whether to use interview phase workflow

    Returns:
        Plan mode instructions string
    """
    if is_interview_phase:
        return get_plan_mode_interview_instructions(plan_file_path, plan_exists)

    plan_file_info = (
        f"A plan file already exists at {plan_file_path}. You can read it and make incremental edits using the {FILE_EDIT_TOOL_NAME} tool."
        if plan_exists
        else f"No plan file exists yet. You should create your plan at {plan_file_path} using the {FILE_WRITE_TOOL_NAME} tool."
    )

    # Phase 2 multiple agents section
    multiple_agents_section = ""
    if plan_agent_count > 1:
        multiple_agents_section = f"""
- **Multiple agents**: Use up to {plan_agent_count} agents for complex tasks that benefit from different perspectives

Examples of when to use multiple agents:
- The task touches multiple parts of the codebase
- It's a large refactor or architectural change
- There are many edge cases to consider
- You'd benefit from exploring different approaches

Example perspectives by task type:
- New feature: simplicity vs performance vs maintainability
- Bug fix: root cause vs workaround vs prevention
- Refactoring: minimal change vs clean architecture
"""

    return f"""Plan mode is active. The user indicated that they do not want you to execute yet -- you MUST NOT make any edits (with the exception of the plan file mentioned below), run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supercedes any other instructions you have received.

## Plan File Info:
{plan_file_info}
You should build your plan incrementally by writing to or editing this file. NOTE that this is the only file you are allowed to edit - other than this you are only allowed to take READ-ONLY actions.

## Plan Workflow

### Phase 1: Initial Understanding
Goal: Gain a comprehensive understanding of the user's request by reading through code and asking them questions. Critical: In this phase you should only use the {EXPLORE_AGENT_TYPE} subagent type.

1. Focus on understanding the user's request and the code associated with their request. Actively search for existing functions, utilities, and patterns that can be reused — avoid proposing new code when suitable implementations already exist.

2. **Launch up to {explore_agent_count} {EXPLORE_AGENT_TYPE} agents IN PARALLEL** (single message, multiple tool calls) to efficiently explore the codebase.
   - Use 1 agent when the task is isolated to known files, the user provided specific file paths, or you're making a small targeted change.
   - Use multiple agents when: the scope is uncertain, multiple areas of the codebase are involved, or you need to understand existing patterns before planning.
   - Quality over quantity - {explore_agent_count} agents maximum, but you should try to use the minimum number of agents necessary (usually just 1)
   - If using multiple agents: Provide each agent with a specific search focus or area to explore. Example: One agent searches for existing implementations, another explores related components, a third investigating testing patterns

### Phase 2: Design
Goal: Design an implementation approach.

Launch {PLAN_AGENT_TYPE} agent(s) to design the implementation based on the user's intent and your exploration results from Phase 1.

You can launch up to {plan_agent_count} agent(s) in parallel.

**Guidelines:**
- **Default**: Launch at least 1 Plan agent for most tasks - it helps validate your understanding and consider alternatives
- **Skip agents**: Only for truly trivial tasks (typo fixes, single-line changes, simple renames)
{multiple_agents_section}
In the agent prompt:
- Provide comprehensive background context from Phase 1 exploration including filenames and code path traces
- Describe requirements and constraints
- Request a detailed implementation plan

### Phase 3: Review
Goal: Review the plan(s) from Phase 2 and ensure alignment with the user's intentions.
1. Read the critical files identified by agents to deepen your understanding
2. Ensure that the plans align with the user's original request
3. Use {ASK_USER_QUESTION_TOOL_NAME} to clarify any remaining questions with the user

{PLAN_PHASE4_DEFAULT}

### Phase 5: Call {EXIT_PLAN_MODE_TOOL_NAME}
At the very end of your turn, once you have asked the user questions and are happy with your final plan file - you should always call {EXIT_PLAN_MODE_TOOL_NAME} to indicate to the user that you are done planning.
This is critical - your turn should only end with either using the {ASK_USER_QUESTION_TOOL_NAME} tool OR calling {EXIT_PLAN_MODE_TOOL_NAME}. Do not stop unless it's for these 2 reasons

**Important:** Use {ASK_USER_QUESTION_TOOL_NAME} ONLY to clarify requirements or choose between approaches. Use {EXIT_PLAN_MODE_TOOL_NAME} to request plan approval. Do NOT ask about plan approval in any other way - no text questions, no AskUserQuestion. Phrases like "Is this plan okay?", "Should I proceed?", "How does this plan look?", "Any changes before we start?", or similar MUST use {EXIT_PLAN_MODE_TOOL_NAME}.

NOTE: At any point in time through this workflow you should feel free to ask the user questions or clarifications using the {ASK_USER_QUESTION_TOOL_NAME} tool. Don't make large assumptions about user intent. The goal is to present a well researched plan to the user, and tie any loose ends before implementation begins.
"""


# =============================================================================
# Plan Mode Interview Phase Instructions
# =============================================================================


def get_plan_mode_interview_instructions(
    plan_file_path: str,
    plan_exists: bool = False,
) -> str:
    """Get iterative interview-based plan mode workflow instructions.

    Instead of forcing Explore/Plan agents, this workflow has the model:
    1. Read files and ask questions iteratively
    2. Build up the spec/plan file incrementally as understanding grows
    3. Use AskUserQuestion throughout to clarify and gather input

    Args:
        plan_file_path: Path to the plan file
        plan_exists: Whether a plan file already exists

    Returns:
        Interview phase instructions string
    """
    plan_file_info = (
        f"A plan file already exists at {plan_file_path}. You can read it and make incremental edits using the {FILE_EDIT_TOOL_NAME} tool."
        if plan_exists
        else f"No plan file exists yet. You should create your plan at {plan_file_path} using the {FILE_WRITE_TOOL_NAME} tool."
    )

    read_only_tools = f"{FILE_READ_TOOL_NAME}, {GLOB_TOOL_NAME}, {GREP_TOOL_NAME}"

    return f"""Plan mode is active. The user indicated that they do not want you to execute yet -- you MUST NOT make any edits (with the exception of the plan file mentioned below), run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supercedes any other instructions you have received.

## Plan File Info:
{plan_file_info}

## Iterative Planning Workflow

You are pair-planning with the user. Explore the code to build context, ask the user questions when you hit decisions you can't make alone, and write your findings into the plan file as you go. The plan file (above) is the ONLY file you may edit — it starts as a rough skeleton and gradually becomes the final plan.

### The Loop

Repeat this cycle until the plan is complete:

1. **Explore** — Use {read_only_tools} to read code. Look for existing functions, utilities, and patterns to reuse. You can use the {EXPLORE_AGENT_TYPE} agent type to parallelize complex searches without filling your context, though for straightforward queries direct tools are simpler.
2. **Update the plan file** — After each discovery, immediately capture what you learned. Don't wait until the end.
3. **Ask the user** — When you hit an ambiguity or decision you can't resolve from code alone, use {ASK_USER_QUESTION_TOOL_NAME}. Then go back to step 1.

### First Turn

Start by quickly scanning a few key files to form an initial understanding of the task scope. Then write a skeleton plan (headers and rough notes) and ask the user your first round of questions. Don't explore exhaustively before engaging the user.

### Asking Good Questions

- Never ask what you could find out by reading the code
- Batch related questions together (use multi-question {ASK_USER_QUESTION_TOOL_NAME} calls)
- Focus on things only the user can answer: requirements, preferences, tradeoffs, edge case priorities
- Scale depth to the task — a vague feature request needs many rounds; a focused bug fix may need one or none

### Plan File Structure
Your plan file should be divided into clear sections using markdown headers, based on the request. Fill out these sections as you go.
- Begin with a **Context** section: explain why this change is being made — the problem or need it addresses, what prompted it, and the intended outcome
- Include only your recommended approach, not all alternatives
- Ensure that the plan file is concise enough to scan quickly, but detailed enough to execute effectively
- Include the paths of critical files to be modified
- Reference existing functions and utilities you found that should be reused, with their file paths
- Include a verification section describing how to test the changes end-to-end (run the code, use MCP tools, run tests)

### When to Converge

Your plan is ready when you've addressed all ambiguities and it covers: what to change, which files to modify, what existing code to reuse (with file paths), and how to verify the changes. Call {EXIT_PLAN_MODE_TOOL_NAME} when the plan is ready for approval.

### Ending Your Turn

Your turn should only end by either:
- Using {ASK_USER_QUESTION_TOOL_NAME} to gather more information
- Calling {EXIT_PLAN_MODE_TOOL_NAME} when the plan is ready for approval

**Important:** Use {EXIT_PLAN_MODE_TOOL_NAME} to request plan approval. Do NOT ask about plan approval via text or AskUserQuestion.
"""


# =============================================================================
# Plan Mode Sparse Instructions (reminder)
# =============================================================================


def get_plan_mode_sparse_instructions(
    plan_file_path: str,
    is_interview_phase: bool = False,
) -> str:
    """Get sparse reminder for plan mode (used on subsequent turns).

    Args:
        plan_file_path: Path to the plan file
        is_interview_phase: Whether using interview phase workflow

    Returns:
        Sparse reminder string
    """
    workflow_description = (
        "Follow iterative workflow: explore codebase, interview user, write to plan incrementally."
        if is_interview_phase
        else "Follow 5-phase workflow."
    )

    return f"""Plan mode still active (see full instructions earlier in conversation). Read-only except plan file ({plan_file_path}). {workflow_description} End turns with {ASK_USER_QUESTION_TOOL_NAME} (for clarifications) or {EXIT_PLAN_MODE_TOOL_NAME} (for plan approval). Never ask about plan approval via text or AskUserQuestion."""


# =============================================================================
# Plan Mode Reentry/Exit Messages
# =============================================================================


def get_plan_mode_reentry_message(plan_file_path: str) -> str:
    """Get message for re-entering plan mode after having exited.

    Args:
        plan_file_path: Path to existing plan file

    Returns:
        Reentry message string
    """
    return f"""## Re-entering Plan Mode

You are returning to plan mode after having previously exited it. A plan file exists at {plan_file_path} from your previous planning session.

**Before proceeding with any new planning, you should:**
1. Read the existing plan file to understand what was previously planned
2. Ask the user if they want to continue with that plan or start fresh
3. If continuing, build on the existing plan; if starting fresh, clear it and begin anew

The same plan mode rules apply: you are read-only except for the plan file, and must use ExitPlanMode to request approval.
"""


def get_plan_mode_exit_message(
    plan_file_path: str,
    plan_exists: bool = False,
) -> str:
    """Get message for exiting plan mode.

    Args:
        plan_file_path: Path to the plan file
        plan_exists: Whether the plan file exists

    Returns:
        Exit message string
    """
    plan_reference = (
        f" The plan file is located at {plan_file_path} if you need to reference it."
        if plan_exists
        else ""
    )

    return f"""## Exited Plan Mode

You are no longer in plan mode. You can now:
- Make edits and run tools to implement the plan
- Write code, run tests, and execute changes{plan_reference}

Remember:
- Follow the plan you created - but adapt as needed if you discover new information
- Mark tasks as completed as you finish them
- If significant changes are needed, you can re-enter plan mode with EnterPlanMode
"""


# =============================================================================
# Plan Mode Behavior Instructions
# =============================================================================


def get_plan_mode_instructions() -> str:
    """Get instructions for behavior in plan mode.

    This is the simpler version used by EnterPlanMode tool output.

    Returns:
        Plan mode behavior instructions string
    """
    return """In plan mode, you should:
1. Thoroughly explore the codebase to understand existing patterns
2. Identify similar features and architectural approaches
3. Consider multiple approaches and their trade-offs
4. Use AskUserQuestion if you need to clarify the approach
5. Design a concrete implementation strategy
6. Write your plan to the plan file
7. When ready, use ExitPlanMode to present your plan for approval

Remember: DO NOT write or edit any files yet (except the plan file). This is a read-only exploration and planning phase."""


def get_plan_template() -> str:
    """Get template for writing plan.

    Returns:
        Plan template string
    """
    return """# Implementation Plan

## Summary
[Brief summary of what will be implemented]

## Implementation Steps

### Step 1: [Description]
- Files to modify: [list files]
- Details: [implementation details]
- Dependencies: [any prerequisites]

### Step 2: [Description]
- Files to modify: [list files]
- Details: [implementation details]

## Considerations
[Any edge cases, potential issues, or things to watch out for]

## Alternatives Considered
[List alternative approaches and why they were not chosen]

## Verification
[How to test the changes end-to-end]
"""