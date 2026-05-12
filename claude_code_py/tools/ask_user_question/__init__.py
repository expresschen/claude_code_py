"""AskUserQuestion tool implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from claude_code_py.tool.base import Tool
from claude_code_py.tool.context import ToolUseContext
from claude_code_py.tool.result import ToolResult, ToolError
from .constants import (
    ASK_USER_QUESTION_TOOL_NAME,
    MAX_QUESTIONS,
    MIN_QUESTIONS,
    MAX_OPTIONS,
    MIN_OPTIONS,
    CHIP_WIDTH,
)
from .prompt import get_ask_user_question_prompt, DESCRIPTION


class QuestionOption(BaseModel):
    """A single option for a question."""

    label: str = Field(
        description="The display text for this option (1-5 words)"
    )
    description: str = Field(
        description="Explanation of what this option means"
    )
    preview: Optional[str] = Field(
        default=None,
        description="Optional preview content (code snippet, mockup, etc.)"
    )


class Question(BaseModel):
    """A single question to ask."""

    question: str = Field(
        description="The complete question to ask the user"
    )
    header: str = Field(
        description=f"Short label displayed as chip (max {CHIP_WIDTH} chars)",
        max_length=CHIP_WIDTH,
    )
    options: list[QuestionOption] = Field(
        description="Available choices (2-4 options)",
        min_length=MIN_OPTIONS,
        max_length=MAX_OPTIONS,
    )
    multi_select: bool = Field(
        default=False,
        description="Allow multiple selections",
    )

    @field_validator("question")
    @classmethod
    def question_ends_with_question_mark(cls, v: str) -> str:
        if not v.strip().endswith("?"):
            raise ValueError("Question should end with a question mark")
        return v

    @field_validator("options")
    @classmethod
    def options_have_unique_labels(cls, v: list[QuestionOption]) -> list[QuestionOption]:
        labels = [opt.label for opt in v]
        if len(labels) != len(set(labels)):
            raise ValueError("Option labels must be unique within each question")
        return v


class QuestionAnnotation(BaseModel):
    """Annotation for a question response."""

    preview: Optional[str] = None
    notes: Optional[str] = None


class AskUserQuestionInput(BaseModel):
    """Input for AskUserQuestion tool."""

    questions: list[Question] = Field(
        description=f"Questions to ask ({MIN_QUESTIONS}-{MAX_QUESTIONS})",
        min_length=MIN_QUESTIONS,
        max_length=MAX_QUESTIONS,
    )
    answers: Optional[dict[str, str]] = Field(
        default=None,
        description="Pre-filled answers (for resubmission)",
    )
    annotations: Optional[dict[str, QuestionAnnotation]] = Field(
        default=None,
        description="Per-question annotations",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for tracking",
    )

    @model_validator(mode="after")
    def validate_unique_questions(self) -> "AskUserQuestionInput":
        """Ensure question texts are unique."""
        question_texts = [q.question for q in self.questions]
        if len(question_texts) != len(set(question_texts)):
            raise ValueError("Question texts must be unique")
        return self


@dataclass
class AskUserQuestionOutput:
    """Output from AskUserQuestion tool."""

    questions: list[dict[str, Any]]
    answers: dict[str, str]  # question text -> answer
    annotations: Optional[dict[str, QuestionAnnotation]] = None


class AskUserQuestionTool(Tool[AskUserQuestionInput, AskUserQuestionOutput, dict[str, Any]]):
    """Tool for asking user questions."""

    name = ASK_USER_QUESTION_TOOL_NAME
    aliases: list[str] = []
    input_schema = AskUserQuestionInput
    max_result_size_chars = 50_000
    search_hint = "ask the user a multiple-choice question"

    # Pending questions storage (for async UI handling)
    _pending_questions: Optional[AskUserQuestionInput] = None
    _pending_answers: Optional[dict[str, str]] = None

    async def call(
        self,
        args: AskUserQuestionInput,
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: Any,
        on_progress: Optional[Any] = None,
    ) -> ToolResult[AskUserQuestionOutput]:
        """Ask the user questions.

        Args:
            args: Question arguments
            context: Execution context
            can_use_tool: Permission function
            parent_message: Parent message
            on_progress: Progress callback

        Returns:
            Tool result with answers
        """
        # If answers already provided (resubmission case)
        if args.answers:
            output = AskUserQuestionOutput(
                questions=[q.model_dump() for q in args.questions],
                answers=args.answers,
                annotations=args.annotations,
            )
            return ToolResult(data=output)

        # Check if we're in a teammate context (swarm mode)
        from claude_code_py.utils.teammate_context import (
            get_current_agent_name,
            get_current_team_name,
            is_team_lead,
        )

        agent_name = get_current_agent_name()
        team_name = get_current_team_name()

        # If we're a teammate (not leader), send question request via mailbox
        if agent_name and team_name and not is_team_lead():
            return await self._call_as_teammate(args, agent_name, team_name)

        # Store pending questions for UI to handle
        self._pending_questions = args

        # In a real implementation, this would block until user responds
        # For now, we return a placeholder indicating questions are pending
        # The actual answers would come from a user interaction callback

        # Simulate getting answers (placeholder)
        # In production: await user_response_callback()
        default_answers = {}
        for q in args.questions:
            # Default to first option as placeholder
            if q.options:
                default_answers[q.question] = q.options[0].label

        output = AskUserQuestionOutput(
            questions=[q.model_dump() for q in args.questions],
            answers=default_answers,
        )

        return ToolResult(data=output)

    async def _call_as_teammate(
        self,
        args: AskUserQuestionInput,
        agent_name: str,
        team_name: str,
    ) -> ToolResult[AskUserQuestionOutput]:
        """Send question request to leader via mailbox and wait for response.

        This is used when the tool is called by a teammate agent (not leader).

        Args:
            args: Question arguments
            agent_name: Current agent name
            team_name: Team name

        Returns:
            Tool result with answers from leader
        """
        import asyncio
        import json
        import uuid
        from datetime import datetime
        from claude_code_py.utils.teammate_mailbox import (
            TeammateMessage,
            write_to_mailbox,
            read_mailbox,
            create_question_request_message,
            is_question_response,
            TEAM_LEAD_NAME,
        )

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Format questions for message
        questions_data = [q.model_dump() for q in args.questions]

        # Create question request message
        question_request = create_question_request_message(
            request_id=request_id,
            from_agent=agent_name,
            questions=questions_data,
        )

        # Send to leader's mailbox
        await write_to_mailbox(
            TEAM_LEAD_NAME,
            TeammateMessage(
                from_agent=agent_name,
                text=json.dumps({
                    "type": question_request.type,
                    "request_id": question_request.request_id,
                    "from_agent": question_request.from_agent,
                    "questions": question_request.questions,
                    "timestamp": question_request.timestamp,
                }),
                timestamp=datetime.now().isoformat(),
            ),
            team_name,
        )

        # Poll for response from leader
        poll_interval = 0.5  # seconds
        max_wait = 300  # 5 minutes max

        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < max_wait:
            await asyncio.sleep(poll_interval)

            # Check our mailbox for response
            messages = await read_mailbox(agent_name, team_name)
            for msg in messages:
                if not msg.read:
                    response = is_question_response(msg.text)
                    if response and response.request_id == request_id:
                        # Found matching response
                        if response.error:
                            # Error case - return empty answers with error
                            return ToolResult(
                                data=AskUserQuestionOutput(
                                    questions=questions_data,
                                    answers={},
                                ),
                                error=response.error,
                            )
                        # Success - return answers
                        return ToolResult(
                            data=AskUserQuestionOutput(
                                questions=questions_data,
                                answers=response.answers,
                                annotations=response.annotations,
                            )
                        )

        # Timeout - no response received
        return ToolResult(
            data=AskUserQuestionOutput(
                questions=questions_data,
                answers={},
            ),
            error="Timeout waiting for user response from leader",
        )

    async def description(
        self,
        input: AskUserQuestionInput,
        options: dict[str, Any],
    ) -> str:
        """Get description."""
        if input.questions:
            headers = [q.header for q in input.questions]
            return f"Asking about: {', '.join(headers)}"
        return "Asking user questions"

    async def prompt(self, options: dict[str, Any]) -> str:
        """Get tool prompt."""
        return get_ask_user_question_prompt()

    def is_concurrency_safe(self, input: AskUserQuestionInput) -> bool:
        """Asking questions is concurrency safe."""
        return True

    def is_read_only(self, input: AskUserQuestionInput) -> bool:
        """Asking questions is read-only."""
        return True

    def user_facing_name(self, input: Optional[AskUserQuestionInput]) -> str:
        """Get user-facing name."""
        if input and input.questions:
            return input.questions[0].header
        return "Question"

    def requires_user_interaction(self) -> bool:
        """Requires user interaction."""
        return True

    def get_pending_questions(self) -> Optional[AskUserQuestionInput]:
        """Get pending questions (for UI)."""
        return self._pending_questions

    def set_answers(self, answers: dict[str, str]) -> None:
        """Set answers from user response."""
        self._pending_answers = answers


# Create instance
ask_user_question_tool = AskUserQuestionTool()


def format_questions_for_display(questions: list[Question]) -> str:
    """Format questions for text display.

    Args:
        questions: List of questions

    Returns:
        Formatted string
    """
    lines = []
    for i, q in enumerate(questions, 1):
        lines.append(f"**{q.header}**: {q.question}")
        for j, opt in enumerate(q.options, 1):
            marker = "☐" if q.multi_select else f"{j}."
            lines.append(f"  {marker} {opt.label}")
            if opt.description:
                lines.append(f"     {opt.description}")
        lines.append("")  # Empty line between questions

    return "\n".join(lines)