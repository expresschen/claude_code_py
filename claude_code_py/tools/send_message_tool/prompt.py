"""SendMessage Tool prompt."""


def get_send_message_tool_prompt() -> str:
    """Get the prompt for SendMessageTool.

    Returns:
        Tool prompt string
    """
    return """Send a message to a teammate or continue a running/stopped agent.

## Usage

### Route 1: Continue/Resume Agent

Use to send follow-up instructions to an agent spawned by the Agent tool:
- The `to` field is the agent's ID or name (from the Agent tool's result)
- If the agent is running, the message is queued for delivery at its next tool round
- If the agent is stopped, it is automatically resumed with your message
- Use this to correct direction, provide clarifications, or extend work

### Route 2: Swarm/Team Communication

Use to communicate with teammates in a swarm/team:
- The `to` field specifies the recipient:
  - A teammate name (e.g., "researcher") to send to that specific teammate
  - "*" to broadcast to all teammates (including the team lead)
  - "lead" to send to the team lead
- The `summary` field is a 5-10 word preview shown in the UI

## Parameters

- `to`: Recipient identifier (agent ID/name, teammate name, "*" for broadcast, or "lead")
- `summary`: Optional 5-10 word summary for preview (required when message is a string for swarm)
- `message`: Plain text message or structured message object

## Behavior

**For Agent Continuation:**
- Running agents receive messages at their next tool round
- Stopped agents are automatically resumed in background
- You'll be notified when a resumed agent finishes

**For Swarm Communication:**
- Messages are written to the recipient's inbox file
- Recipients see messages as attachments on their next turn
- Use to coordinate work, share findings, or request approvals

## Examples

Continue a stopped agent:
```
SendMessage({ to: "agent-abc123", message: "Fix the null pointer in validate.ts:42 instead..." })
```

Queue message for running agent:
```
SendMessage({ to: "researcher", message: "Also check the test coverage for auth module..." })
```

Send to teammate:
```
SendMessage({ to: "researcher", summary: "Auth findings ready", message: "I found the bug in validate.ts..." })
```
"""