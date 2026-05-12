"""Agent execution engine.

This handles running subagents and managing their lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

# Debug flag - controlled by environment variable CLAUDE_CODE_DEBUG_TEAMMATE
DEBUG_AGENT_RUNNER = os.environ.get("CLAUDE_CODE_DEBUG_TEAMMATE", "").lower() in ("1", "true", "yes")

# Import unified debug logger
from claude_code_py.utils.debug_log import debug_log

def _debug_print(msg: str) -> None:
    """Print debug message to console and log file."""
    debug_log("[AGENT_RUNNER]", msg, DEBUG_AGENT_RUNNER)

# Import async helpers for reliable task execution
from claude_code_py.utils.async_helpers import create_task_with_yield

if TYPE_CHECKING:
    from claude_code_py.tool.base import Tool
    from claude_code_py.tool.context import ToolUseContext
    from claude_code_py.core_types.message import Message
    from .types import AgentDefinition


@dataclass
class AgentRunConfig:
    """Configuration for running an agent."""

    agent_id: str
    agent_type: str
    prompt: str
    description: str
    model: Optional[str] = None
    tools: Optional[list[str]] = None
    disallowed_tools: Optional[list[str]] = None
    run_in_background: bool = False
    isolation: Optional[str] = None  # "worktree" or None
    cwd: Optional[str] = None
    max_turns: Optional[int] = None
    parent_context: Optional["ToolUseContext"] = None
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None  # For transcript path
    abort_controller: Optional["AbortController"] = None  # External abort controller (for Escape key)


@dataclass
class AgentRunResult:
    """Result from running an agent."""

    agent_id: str
    status: str  # 'completed', 'async_launched', 'error'
    output: Optional[str] = None
    messages: Optional[list["Message"]] = None
    error: Optional[str] = None
    duration_ms: int = 0
    token_usage: Optional[dict[str, int]] = None
    output_file: Optional[str] = None  # For background agents


@dataclass
class AgentProgress:
    """Progress update from running agent."""

    agent_id: str
    message: str
    tool_use: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class AgentRunner:
    """Manages agent execution lifecycle."""

    def __init__(self):
        self._running_agents: dict[str, asyncio.Task] = {}
        self._agent_results: dict[str, AgentRunResult] = {}
        self._progress_callbacks: dict[str, Callable[[AgentProgress], None]] = {}

    def create_agent_id(self) -> str:
        """Create a unique agent ID."""
        return f"agent_{uuid.uuid4().hex[:8]}"

    async def run_agent(
        self,
        config: AgentRunConfig,
        on_progress: Optional[Callable[[AgentProgress], None]] = None,
    ) -> AgentRunResult:
        """Run an agent with given configuration.

        Args:
            config: Agent run configuration
            on_progress: Optional progress callback

        Returns:
            Agent run result
        """
        start_time = datetime.now()

        _debug_print("=" * 70)
        _debug_print("AgentRunner.run_agent: STARTING")
        _debug_print(f"  agent_id: '{config.agent_id}'")
        _debug_print(f"  agent_type: '{config.agent_type}'")
        _debug_print(f"  description: '{config.description}'")
        _debug_print(f"  prompt preview: '{config.prompt[:100]}{'...' if len(config.prompt) > 100 else ''}'")
        _debug_print(f"  run_in_background: {config.run_in_background}")
        _debug_print(f"  isolation: '{config.isolation}'")
        _debug_print(f"  cwd: '{config.cwd}'")
        _debug_print(f"  model: '{config.model}'")
        _debug_print(f"  max_turns: {config.max_turns}")
        _debug_print("=" * 70)

        # Store progress callback
        if on_progress:
            self._progress_callbacks[config.agent_id] = on_progress

        try:
            # Emit progress
            self._emit_progress(config.agent_id, f"Starting {config.agent_type} agent")

            # Run agent
            if config.run_in_background:
                _debug_print("→ Running in background mode")
                result = await self._run_agent_background(config)
            else:
                _debug_print("→ Running in foreground (sync) mode")
                result = await self._run_agent_sync(config)

            # Calculate duration
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            result.duration_ms = duration_ms

            # Store result
            self._agent_results[config.agent_id] = result

            _debug_print("=" * 70)
            _debug_print("AgentRunner.run_agent: COMPLETED")
            _debug_print(f"  agent_id: '{config.agent_id}'")
            _debug_print(f"  status: '{result.status}'")
            _debug_print(f"  duration_ms: {duration_ms}")
            if result.error:
                _debug_print(f"  error: '{result.error}'")
            _debug_print("=" * 70)

            return result

        except Exception as e:
            _debug_print("=" * 70)
            _debug_print("AgentRunner.run_agent: EXCEPTION")
            _debug_print(f"  {type(e).__name__}: {e}")
            _debug_print("=" * 70)

            error_result = AgentRunResult(
                agent_id=config.agent_id,
                status="error",
                error=str(e),
                duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            )
            self._agent_results[config.agent_id] = error_result
            return error_result

        finally:
            # Cleanup progress callback
            self._progress_callbacks.pop(config.agent_id, None)

    async def _run_agent_sync(self, config: AgentRunConfig) -> AgentRunResult:
        """Run agent synchronously (foreground)."""
        _debug_print("→ _run_agent_sync: Starting")

        # Import here to avoid circular dependency
        from claude_code_py.engine.query import query, QueryParams
        from claude_code_py.tool.context import (
            ToolUseContext,
            ToolUseContextOptions,
            create_default_tool_use_context,
        )
        from claude_code_py.utils.abort_controller import AbortController

        # Handle worktree isolation
        worktree_info = None
        original_cwd = config.cwd

        if config.isolation == "worktree":
            _debug_print("   Worktree isolation requested")
            from claude_code_py.utils.worktree import (
                create_agent_worktree,
                remove_agent_worktree,
                has_worktree_changes,
            )
            from claude_code_py.storage.session import (
                write_agent_metadata,
                AgentMetadata,
            )

            # Create agent worktree
            slug = config.agent_id.replace("@", "-").replace("_", "-")
            _debug_print(f"   → Creating worktree with slug '{slug}'")
            worktree_info = await create_agent_worktree(slug)
            config.cwd = worktree_info["worktree_path"]
            _debug_print(f"   ✅ Worktree created at '{worktree_info['worktree_path']}'")

            # Write metadata for resume
            if config.session_id:
                await write_agent_metadata(
                    config.session_id,
                    config.agent_id,
                    AgentMetadata(
                        agent_type=config.agent_type,
                        worktree_path=worktree_info["worktree_path"],
                        description=config.description,
                    ),
                )
                _debug_print("   ✅ Metadata written")

        # Determine working directory
        cwd = config.cwd or original_cwd or "."
        _debug_print(f"   Working directory: '{cwd}'")

        # Get tools for agent
        _debug_print("   → Getting agent tools...")
        tools = self._get_agent_tools(config)
        _debug_print(f"   ✅ Got {len(tools)} tools")
        _debug_print(f"      Tool names: {[t.name for t in tools[:5]]}{'...' if len(tools) > 5 else ''}")

        # Use external abort controller if provided, otherwise create one
        if config.abort_controller:
            abort_controller = config.abort_controller
            _debug_print("   Using external abort controller")
        else:
            abort_controller = AbortController()
            _debug_print("   Created new abort controller")

        # Create tool use context
        _debug_print("   → Creating tool use context...")
        tool_use_context = create_default_tool_use_context(
            tools=tools,
            abort_controller=abort_controller,
            cwd=cwd,
        )

        # Set agent context
        tool_use_context.agent_id = config.agent_id
        tool_use_context.agent_type = config.agent_type
        _debug_print(f"   ✅ Tool use context created")
        _debug_print(f"      agent_id='{tool_use_context.agent_id}'")
        _debug_print(f"      agent_type='{tool_use_context.agent_type}'")

        # Build messages for agent
        _debug_print("   → Building agent messages...")
        messages = self._build_agent_messages(config)
        _debug_print(f"   ✅ Built {len(messages)} initial messages")

        # Determine query_source for agent (for recursion guard)
        # Matches TypeScript's getQuerySourceForAgent()
        # Built-in agents have known agent_types
        builtin_agent_types = {'general-purpose', 'Explore', 'Plan', 'verification', 'code-improver'}
        is_builtin = config.agent_type in builtin_agent_types
        if is_builtin:
            query_source = f"agent:builtin:{config.agent_type}"
        else:
            query_source = "agent:custom"
        _debug_print(f"   query_source: '{query_source}'")

        # Create query params
        params = QueryParams(
            messages=messages,
            system_prompt=config.system_prompt or "",
            user_context={},
            system_context={},
            can_use_tool=agent_can_use_tool,
            tool_use_context=tool_use_context,
            fallback_model=config.model,
            max_turns=config.max_turns,
            query_source=query_source,
        )

        try:
            # Run query and collect results
            _debug_print("   → Running query loop...")
            result_messages = []
            async for event in query(params):
                result_messages.append(event)

            _debug_print(f"   ✅ Query loop completed with {len(result_messages)} messages")

            # Extract output from final message
            output = self._extract_output(result_messages)
            _debug_print(f"   Output length: {len(output)} chars")
            _debug_print(f"   Output preview: '{output[:100]}{'...' if len(output) > 100 else ''}'")

            return AgentRunResult(
                agent_id=config.agent_id,
                status="completed",
                output=output,
                messages=result_messages,
            )
        finally:
            # Cleanup worktree if created
            if worktree_info and config.isolation == "worktree":
                _debug_print("   → Cleaning up worktree...")
                from claude_code_py.utils.worktree import remove_agent_worktree

                git_root = worktree_info.get("git_root")
                worktree_branch = worktree_info.get("worktree_branch")
                head_commit = worktree_info.get("head_commit")

                # Check if there were changes
                should_remove = True
                if head_commit:
                    has_changes = await has_worktree_changes(
                        worktree_info["worktree_path"],
                        head_commit,
                    )
                    # Keep worktree if there were changes (write operations)
                    should_remove = not has_changes
                    _debug_print(f"      has_changes={has_changes}, should_remove={should_remove}")

                if should_remove:
                    await remove_agent_worktree(
                        worktree_info["worktree_path"],
                        worktree_branch,
                        git_root,
                    )
                    _debug_print("   ✅ Worktree removed")
                else:
                    _debug_print("   ℹ️ Worktree kept (has changes)")

    async def _run_agent_background(self, config: AgentRunConfig) -> AgentRunResult:
        """Run agent in background.

        Uses create_task_with_yield to ensure the background agent starts
        immediately, solving the "create_task doesn't start until next iteration" problem.
        """
        # Create output file path
        output_dir = Path(os.environ.get("CLAUDE_CODE_OUTPUT_DIR", ".claude/output"))
        output_file = output_dir / f"agent_{config.agent_id}.json"

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create background task with immediate execution guarantee
        task = await create_task_with_yield(
            self._run_agent_background_task(config, output_file)
        )
        self._running_agents[config.agent_id] = task

        return AgentRunResult(
            agent_id=config.agent_id,
            status="async_launched",
            output_file=str(output_file),
        )

    async def _run_agent_background_task(
        self,
        config: AgentRunConfig,
        output_file: Path,
    ) -> None:
        """Background task that runs agent and writes result to file."""
        try:
            result = await self._run_agent_sync(config)

            # Write result to file
            result_data = {
                "agent_id": result.agent_id,
                "status": result.status,
                "output": result.output,
                "duration_ms": result.duration_ms,
                "timestamp": datetime.now().isoformat(),
            }
            output_file.write_text(json.dumps(result_data, indent=2))

        except Exception as e:
            # Write error to file
            error_data = {
                "agent_id": config.agent_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            output_file.write_text(json.dumps(error_data, indent=2))

    def _build_agent_messages(self, config: AgentRunConfig) -> list["Message"]:
        """Build initial messages for agent.

        Returns UserMessage objects (not raw dicts) as expected by QueryParams.
        """
        from claude_code_py.core_types.message import UserMessage

        return [
            UserMessage(
                type="user",
                message={
                    "role": "user",
                    "content": config.prompt,
                },
            )
        ]

    def _get_agent_tools(self, config: AgentRunConfig) -> list["Tool"]:
        """Get tools available to agent."""
        from claude_code_py.tools import get_all_base_tools

        all_tools = get_all_base_tools()

        # Apply allowlist
        if config.tools:
            if "*" in config.tools:
                # All tools allowed (except disallowed)
                allowed_tools = list(all_tools)
            else:
                allowed_tools = [t for t in all_tools if t.name in config.tools]
        else:
            allowed_tools = list(all_tools)

        # Apply denylist
        if config.disallowed_tools:
            deny_set = set(config.disallowed_tools)
            allowed_tools = [t for t in allowed_tools if t.name not in deny_set]

        return allowed_tools

    def _extract_output(self, messages: list) -> str:
        """Extract output from agent messages.

        Handles both Pydantic Message objects and raw dicts.
        """
        # Find last assistant message
        for msg in reversed(messages):
            # Handle Pydantic Message objects (AssistantMessage, UserMessage)
            if hasattr(msg, "type") and hasattr(msg, "message"):
                if msg.type == "assistant":
                    content = msg.message.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Extract text blocks
                        texts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                texts.append(block.get("text", ""))
                        return "\n".join(texts)
            # Handle raw dicts (fallback)
            elif isinstance(msg, dict):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Extract text blocks
                        texts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                texts.append(block.get("text", ""))
                        return "\n".join(texts)
        return ""

    def _emit_progress(self, agent_id: str, message: str, tool_use: Optional[str] = None) -> None:
        """Emit progress update."""
        callback = self._progress_callbacks.get(agent_id)
        if callback:
            progress = AgentProgress(
                agent_id=agent_id,
                message=message,
                tool_use=tool_use,
            )
            callback(progress)

    def stop_agent(self, agent_id: str) -> bool:
        """Stop a running background agent.

        Args:
            agent_id: Agent ID to stop

        Returns:
            True if agent was stopped, False if not found
        """
        task = self._running_agents.get(agent_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def get_agent_result(self, agent_id: str) -> Optional[AgentRunResult]:
        """Get result for a completed agent.

        Args:
            agent_id: Agent ID

        Returns:
            Agent result or None if not found
        """
        return self._agent_results.get(agent_id)

    def is_agent_running(self, agent_id: str) -> bool:
        """Check if an agent is still running.

        Args:
            agent_id: Agent ID

        Returns:
            True if agent is running, False otherwise
        """
        task = self._running_agents.get(agent_id)
        return task is not None and not task.done()


# Global agent runner instance
_agent_runner: Optional[AgentRunner] = None


async def agent_can_use_tool(
    tool: "Tool",
    input: Any,
    context: "ToolUseContext",
    assistant_message: Any,
    tool_use_id: Optional[str] = None,
    force_decision: Optional[str] = None,
) -> "PermissionResult":
    """Permission handler for subagents.

    Agents inherit parent's permission context but with relaxed rules
    for autonomous operation. For swarm workers, dangerous operations
    bubble up to the team leader for approval.

    Args:
        tool: Tool to check
        input: Tool input
        context: Tool use context
        assistant_message: Parent message
        tool_use_id: Tool use ID
        force_decision: Forced decision

    Returns:
        Permission result
    """
    from claude_code_py.core_types.permissions import PermissionResult, PermissionBehavior

    _debug_print("=" * 60)
    _debug_print("agent_can_use_tool: CHECKING PERMISSION")
    _debug_print(f"  tool: '{tool.name}'")
    _debug_print(f"  tool_use_id: '{tool_use_id}'")
    _debug_print(f"  force_decision: '{force_decision}'")
    _debug_print(f"  agent_id: '{context.agent_id}'")
    _debug_print(f"  agent_type: '{context.agent_type}'")

    # Check bypass mode from environment
    import os
    if os.environ.get("CLAUDE_CODE_ACCEPT_ALL", "").lower() == "true":
        _debug_print("  ✅ ALLOW: CLAUDE_CODE_ACCEPT_ALL=true")
        return PermissionResult.allow(updated_input=input)

    # Force decision override
    if force_decision == "allow":
        _debug_print("  ✅ ALLOW: force_decision='allow'")
        return PermissionResult.allow(updated_input=input)
    if force_decision == "deny":
        _debug_print("  ❌ DENY: force_decision='deny'")
        return PermissionResult.deny(reason="Force denied")

    # Get permission context
    perm_context = getattr(context.options, "tool_permission_context", None)
    mode = getattr(perm_context, "mode", "default") if perm_context else "default"
    _debug_print(f"  permission mode: '{mode}'")

    # Check if tool is read-only (safe to allow)
    try:
        if tool.is_read_only(input):
            _debug_print("  ✅ ALLOW: tool is read-only")
            return PermissionResult.allow(updated_input=input)
    except Exception as e:
        _debug_print(f"  ⚠️ is_read_only check failed: {e}")
        # If is_read_only fails, continue with other checks
        pass

    # In bypass/accept-all mode, allow everything
    if mode in ("accept-all", "bypass"):
        _debug_print(f"  ✅ ALLOW: mode is '{mode}'")
        return PermissionResult.allow(updated_input=input)

    # For agents, use relaxed permission - check danger but don't prompt
    # This allows agents to work autonomously within safe bounds
    try:
        from claude_code_py.utils.permissions.classifier import (
            classify_action,
            is_auto_mode_allowlisted_tool,
            TwoStageMode,
        )
        from claude_code_py.utils.swarm.permission_sync import (
            is_swarm_worker,
            request_permission_from_leader,
        )

        # Check if tool is in safe allowlist first
        if is_auto_mode_allowlisted_tool(tool.name):
            _debug_print(f"  ✅ ALLOW: tool '{tool.name}' is in auto mode allowlist")
            return PermissionResult.allow(updated_input=input)

        _debug_print(f"  → Running classifier to check danger...")

        # Run classifier to check danger
        messages = context.messages or []
        tools = context.options.tools if context.options else []

        # Get model from context
        model = getattr(context.options, "main_loop_model", "claude-sonnet-4-6")

        classification = await classify_action(
            tool_name=tool.name,
            tool_input=input,
            messages=messages,
            tools=tools,
            context=context,
            model=model,
            mode=TwoStageMode.FAST,  # Use fast mode for agents
        )

        _debug_print(f"  Classification result:")
        _debug_print(f"    should_block: {classification.should_block}")
        _debug_print(f"    reason: '{classification.reason}'")

        # Allow if not blocked
        if not classification.should_block:
            _debug_print(f"  ✅ ALLOW: classification says not blocked")
            return PermissionResult.allow(updated_input=input)

        # Check if this is a swarm worker - bubble to leader
        is_swarm = is_swarm_worker()
        _debug_print(f"  is_swarm_worker: {is_swarm}")

        if is_swarm:
            # Build description for permission request
            description = f"{tool.name}: {classification.reason or 'dangerous operation'}"

            _debug_print(f"  → Swarm worker: bubbling permission to leader...")
            _debug_print(f"    description: '{description}'")

            # Convert input to dict if needed
            input_dict = input
            if hasattr(input, "model_dump"):
                input_dict = input.model_dump()
            elif hasattr(input, "to_dict"):
                input_dict = input.to_dict()
            elif not isinstance(input, dict):
                input_dict = {"input": str(input)}

            # Request permission from leader (bubble up)
            result = await request_permission_from_leader(
                tool_name=tool.name,
                tool_use_id=tool_use_id or "",
                tool_input=input_dict,
                description=description,
                timeout_ms=60000,  # 60s timeout
            )

            _debug_print(f"  ← Leader permission result:")
            _debug_print(f"    behavior: '{result['behavior']}'")
            _debug_print(f"    message: '{result.get('message', 'N/A')}'")

            if result["behavior"] == "allow":
                _debug_print(f"  ✅ ALLOW: leader approved")
                return PermissionResult.allow(updated_input=result["updated_input"])
            else:
                _debug_print(f"  ❌ DENY: leader rejected")
                return PermissionResult.deny(reason=result["message"])

        # Not a swarm worker - block dangerous operations
        _debug_print(f"  ❌ DENY: Not swarm worker, blocking dangerous operation")
        return PermissionResult.deny(
            reason=f"Agent blocked from dangerous operation: {classification.reason}"
        )
    except ImportError as e:
        _debug_print(f"  ⚠️ ImportError: {e}")
        # Fallback if classifier not available
        # Check basic safety rules
        from claude_code_py.utils.permissions.dangerous_patterns import check_tool_input_dangerous

        danger_result = check_tool_input_dangerous(tool.name, input)
        _debug_print(f"  Fallback danger check:")
        _debug_print(f"    is_dangerous: {danger_result.is_dangerous}")
        _debug_print(f"    severity: '{danger_result.severity}'")

        if danger_result.is_dangerous and danger_result.severity == "critical":
            _debug_print(f"  ❌ DENY: critical danger detected")
            return PermissionResult.deny(reason=danger_result.reason)

        # For non-critical operations in agent context, allow
        _debug_print(f"  ✅ ALLOW: non-critical operation (fallback)")
        return PermissionResult.allow(updated_input=input)
    except Exception as e:
        _debug_print(f"  ❌ Exception in permission check: {type(e).__name__}: {e}")
        # Fallback on any error - allow read-only, deny others
        # Agents cannot prompt user for permissions, so we deny on error
        try:
            if tool.is_read_only(input):
                _debug_print(f"  ✅ ALLOW: read-only (error fallback)")
                return PermissionResult.allow(updated_input=input)
        except Exception:
            pass
        _debug_print(f"  ❌ DENY: permission check failed")
        return PermissionResult.deny(reason=f"Agent permission check failed: {str(e)}")


def get_agent_runner() -> AgentRunner:
    """Get the global agent runner instance."""
    global _agent_runner
    if _agent_runner is None:
        _agent_runner = AgentRunner()
    return _agent_runner


async def run_agent(
    config: AgentRunConfig,
    on_progress: Optional[Callable[[AgentProgress], None]] = None,
) -> AgentRunResult:
    """Run an agent with given configuration.

    This is the main entry point for running agents.

    Args:
        config: Agent run configuration
        on_progress: Optional progress callback

    Returns:
        Agent run result
    """
    runner = get_agent_runner()
    return await runner.run_agent(config, on_progress)


def stop_agent(agent_id: str) -> bool:
    """Stop a running background agent.

    Args:
        agent_id: Agent ID to stop

    Returns:
        True if agent was stopped, False if not found
    """
    runner = get_agent_runner()
    return runner.stop_agent(agent_id)


def get_agent_result(agent_id: str) -> Optional[AgentRunResult]:
    """Get result for a completed agent.

    Args:
        agent_id: Agent ID

    Returns:
        Agent result or None if not found
    """
    runner = get_agent_runner()
    return runner.get_agent_result(agent_id)