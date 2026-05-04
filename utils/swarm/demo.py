#!/usr/bin/env python3
"""Agent Team Demo - 演示核心功能

运行方式:
    python claude_code_py/utils/swarm/demo.py
"""
import asyncio
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from claude_code_py.task.manager import (
    spawn_in_process_teammate,
    InProcessSpawnConfig,
    InProcessSpawnOutput,
)
from claude_code_py.utils.swarm.spawn_in_process import (
    generate_task_id,
    format_agent_id,
    set_session_id,
    generate_session_id,
)
from claude_code_py.utils.swarm.permission_sync import (
    create_permission_request,
    write_permission_request,
    read_pending_permissions,
    resolve_permission,
    PermissionResolution,
)
from claude_code_py.utils.teammate_mailbox import (
    write_to_mailbox,
    read_mailbox,
    mark_messages_as_read,
    TeammateMessage,
    create_idle_notification,
    is_idle_notification,
)
from claude_code_py.utils.team.team_file import (
    write_team_file,
    read_team_file,
    add_member_to_team,
    ensure_team_dir,
    TeamMember,
    TeamFile,
)


async def demo_team_creation():
    """演示创建团队"""
    print("\n=== 1. 创建团队 ===")

    team_name = "demo-team"

    # 确保团队目录存在
    ensure_team_dir(team_name)

    # 创建团队文件
    import time
    team = TeamFile(
        name=team_name,
        created_at=int(time.time() * 1000),
        lead_agent_id="team-lead@demo-team",
        members=[
            TeamMember(
                agent_id="researcher@demo-team",
                name="researcher",
                prompt="Search for information",
                color="blue",
            )
        ],
    )

    write_team_file(team_name, team)

    print(f"团队创建成功: {team_name}")
    print(f"Leader: {team.lead_agent_id}")
    print(f"成员: {[m.name for m in team.members]}")


async def demo_spawn_teammate():
    """演示 spawn teammate"""
    print("\n=== 2. Spawn Teammate ===")

    # 设置 session ID
    session_id = generate_session_id()
    set_session_id(session_id)
    print(f"Session ID: {session_id}")

    # 创建 spawn config
    config = InProcessSpawnConfig(
        name="researcher",
        team_name="demo-team",
        prompt="Search for information about Python async programming",
        color="blue",
        model="claude-sonnet-4-6",
    )

    # 创建 spawn context (dict format for task/manager)
    def mock_set_app_state(fn):
        print("  [AppState 更新]")

    context = {
        "set_app_state": mock_set_app_state,
        "tool_use_id": "tool_001",
        "session_id": session_id,
    }

    # Spawn teammate (async)
    result = await spawn_in_process_teammate(config, context)

    if result.success:
        print(f"Spawn 成功!")
        print(f"  Agent ID: {result.agent_id}")
        print(f"  Task ID: {result.task_id}")
    else:
        print(f"Spawn 失败: {result.error}")


async def demo_permission_flow():
    """演示权限请求流程"""
    print("\n=== 3. 权限请求流程 ===")

    # Worker 创建权限请求
    request = create_permission_request(
        tool_name="Bash",
        tool_use_id="tool_002",
        input={"command": "pip install requests"},
        description="Install requests package",
        team_name="demo-team",
        worker_id="researcher@demo-team",
        worker_name="researcher",
        worker_color="blue",
    )

    print(f"权限请求创建:")
    print(f"  Request ID: {request.id}")
    print(f"  Tool: {request.tool_name}")
    print(f"  Description: {request.description}")

    # 写入 pending
    await write_permission_request(request)
    print("  已写入 pending 目录")

    # Leader 读取 pending
    pending = await read_pending_permissions("demo-team")
    print(f"  Leader 看到 {len(pending)} 个待处理请求")

    # Leader 决议
    resolution = PermissionResolution(
        decision="approved",
        resolved_by="leader",
    )

    success = await resolve_permission(request.id, resolution, "demo-team")
    print(f"  决议结果: {'已批准' if success else '失败'}")


async def demo_mailbox_messaging():
    """演示 mailbox 消息传递"""
    print("\n=== 4. Mailbox 消息传递 ===")

    # Leader 发送消息给 researcher
    await write_to_mailbox(
        "researcher",
        TeammateMessage(
            from_agent="team-lead",
            text="Please search for async patterns",
            timestamp="2026-04-27T10:00:00",
            color="green",
        ),
        "demo-team",
    )
    print("Leader 发送消息到 researcher inbox")

    # Researcher 读取消息
    messages = await read_mailbox("researcher", "demo-team")
    print(f"Researcher 收到 {len(messages)} 条消息")
    for msg in messages:
        print(f"  From: {msg.from_agent}")
        print(f"  Text: {msg.text[:50]}...")

    # Researcher 发送 idle notification
    notification = create_idle_notification(
        agent_id="researcher",
        options={
            "idle_reason": "available",
            "summary": "Search completed",
        },
    )

    await write_to_mailbox(
        "team-lead",
        TeammateMessage(
            from_agent="researcher",
            text=json.dumps({
                "type": notification.type,
                "from": notification.from_agent,
                "idle_reason": notification.idle_reason,
                "summary": notification.summary,
            }),
            timestamp="2026-04-27T10:05:00",
            color="blue",
        ),
        "demo-team",
    )
    print("Researcher 发送 idle notification")

    # Leader 解析 idle notification
    leader_messages = await read_mailbox("team-lead", "demo-team")
    for msg in leader_messages:
        parsed = is_idle_notification(msg.text)
        if parsed:
            print(f"Leader 收到 idle notification:")
            print(f"  From: {parsed.from_agent}")
            print(f"  Reason: {parsed.idle_reason}")
            print(f"  Summary: {parsed.summary}")


async def demo_full_workflow():
    """完整工作流演示"""
    print("\n" + "=" * 50)
    print("Agent Team 完整工作流演示")
    print("=" * 50)

    # 1. 创建团队
    await demo_team_creation()

    # 2. Spawn teammate
    await demo_spawn_teammate()

    # 3. 权限请求
    await demo_permission_flow()

    # 4. 消息传递
    await demo_mailbox_messaging()

    print("\n" + "=" * 50)
    print("演示完成!")
    print("=" * 50)


def main():
    """主入口"""
    asyncio.run(demo_full_workflow())


if __name__ == "__main__":
    main()