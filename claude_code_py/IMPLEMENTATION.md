# Claude Code Python Implementation

## 项目结构

```
claude_code_py/
├── __init__.py              # 主入口，导出核心类
├── main.py                  # CLI 入口 (REPL/SDK模式)
├── pyproject.toml           # 项目配置
├── requirements.txt         # 依赖列表
├── README.md                # 使用文档
│
├── core_types/              # 核心类型定义
│   ├── ids.py              # AgentId, SessionId, TaskId
│   ├── message.py          # Message 类型 (User, Assistant, System, Progress)
│   ├── permissions.py      # PermissionMode, PermissionResult
│   └── tools.py            # ToolProgressData 类型
│
├── constants/               # 常量定义
│   └── __init__.py         # QuerySource, FeatureFlag, etc.
│
├── tool/                    # Tool 系统核心
│   ├── base.py             # Tool 基类, build_tool 工厂
│   ├── context.py          # ToolUseContext 定义
│   └── result.py           # ToolResult, ToolError
│
├── orchestration/           # 工具编排
│   ├── partition.py        # partition_tool_calls
│   ├── executor.py         # runTools, 并发/串行执行
│   └── progress.py         # MessageUpdate
│
├── state/                   # 状态管理
│   ├── store.py            # Store 类 (Zustand-like)
│   ├── app_state.py        # AppState 定义
│   └── context.py          # ToolPermissionContext
│
├── engine/                  # 查询引擎
│   ├── query_engine.py     # QueryEngine 类
│   ├── query.py            # query() 主循环
│   ├── transitions.py      # Terminal, Continue 状态
│   ├── process_input.py    # 用户输入处理
│   ├── coordinator_mode.py # Coordinator 模式
│   └── deps.py             # 依赖注入
│
├── task/                    # 任务系统
│   ├── base.py             # TaskStateBase, generate_task_id
│   ├── types.py            # TaskType, TaskStatus
│   ├── in_process_teammate.py  # In-process teammate runner
│   └── manager.py          # TaskManager
│
├── tools/                   # 内置工具实现 (18+)
│   ├── bash_tool.py        # BashTool
│   ├── file_read_tool.py   # FileReadTool
│   ├── file_write_tool.py  # FileWriteTool
│   ├── file_edit_tool.py   # FileEditTool
│   ├── glob_tool.py        # GlobTool
│   ├── grep_tool.py        # GrepTool
│   ├── agent_tool/         # AgentTool (子代理启动)
│   ├── plan_mode/          # EnterPlanMode/ExitPlanMode
│   ├── ask_user_question/  # AskUserQuestionTool
│   ├── worktree_tool/      # EnterWorktree/ExitWorktree
│   ├── send_message_tool/  # SendMessageTool
│   ├── task_tools/         # TaskCreate/Update/List/Get/Stop
│   └── team_tools/         # TeamCreate/TeamDelete
│
├── memory/                  # 内存系统
│   ├── memory_types.py     # SessionMemory, AgentMemoryScope
│   ├── memory_manager.py   # 内存管理器
│   ├── session_memory.py   # 会话内存
│   ├── session_memory_prompts.py  # 内存提取 prompts
│   ├── extract.py          # 内存提取逻辑
│   ├── find_relevant.py    # 相关内存查找
│   ├── memdir.py           # 内存目录结构
│   ├── paths.py            # 内存路径处理
│   └── agent_memory.py     # Agent 内存 scope
│
├── services/                # 服务层
│   ├── compact.py          # 会话 compact 服务
│   ├── compact_prompt.py   # Compact prompts
│   ├── compact_types.py    # CompactResult 类型
│   ├── micro_compact.py    # Micro-compact 优化
│   ├── session_memory_compact.py  # Session memory compact
│   └── cached_mc_state.py  # 缓存 MC 状态
│
├── storage/                 # 会话持久化
│   ├── session.py          # SessionStorage
│   ├── session_resume.py   # Session resume
│   └── disk_output.py      # Disk output 处理
│
├── utils/                   # 工具函数
│   ├── abort_controller.py # AbortController 实现
│   ├── async_helpers.py    # 异步辅助函数
│   ├── generators.py       # 异步生成器工具
│   ├── json_utils.py       # JSON 处理工具
│   ├── cache.py            # 缓存系统
│   ├── context.py          # Token warning, context 分析
│   ├── debug_log.py        # Debug 日志
│   ├── worktree.py         # Git worktree 处理
│   ├── teammate_mailbox.py # Teammate mailbox 通信
│   ├── teammate_context.py # Teammate 上下文
│   ├── forked_agent.py     # Forked agent 处理
│   ├── agent_resume.py     # Agent resume
│   ├── side_query.py       # Side query 处理
│   ├── permissions/        # 权限系统
│   │   ├── classifier.py   # Two-stage auto mode classifier
│   │   ├── rules.py        # Permission rules
│   │   ├── dangerous_patterns.py  # Dangerous pattern 检测
│   │   ├── denial_tracking.py     # Denial tracking
│   │   ├── permission_setup.py    # Permission setup
│   │   └── context.py      # Permission context
│   ├── settings/           # Settings 处理
│   │   └── ...
│   ├── swarm/              # Swarm/Teammate 系统
│   │   ├── inbox_poller.py       # Inbox poller for leader
│   │   ├── in_process_runner.py  # In-process teammate runner
│   │   ├── permission_sync.py    # Permission sync for teammates
│   │   ├── leader_permission_handler.py  # Leader permission handling
│   │   ├── permission_bridge.py  # Permission bridge
│   │   ├── spawn_in_process.py   # Spawn in-process teammate
│   │   └── constants.py          # Swarm constants
│   ├── task/               # Task V2 文件存储
│   │   ├── file_storage.py       # File-based task storage with locking
│   │   └── disk_output.py        # Disk output for tasks
│   └── team/               # Team 文件处理
│       └── team_file.py    # Team config 文件
│
├── mcp/                     # MCP 集成 (占位)
│   └── __init__.py
│
└── tests/                   # 测试
    ├── swarm/
    │   ├── test_collaboration.py  # Swarm collaboration tests
    │   └── test_spawn.py          # Spawn tests
    └── ...
```

## 核心架构映射

| TypeScript 原始 | Python 实现 | 状态 | 说明 |
|----------------|-------------|------|------|
| `Tool.ts` | `tool/base.py` | ✅ 完成 | Tool 基类和工厂函数 |
| `QueryEngine.ts` | `engine/query_engine.py` | ✅ 完成 | 查询引擎核心 |
| `query.ts` | `engine/query.py` | ✅ 完成 | 查询主循环 |
| `AppState.tsx` | `state/app_state.py` | ✅ 完成 | 应用状态定义 |
| `toolOrchestration.ts` | `orchestration/executor.py` | ✅ 完成 | 工具执行编排 |
| `memdir/` | `memory/` | ✅ 完成 | Auto/session/agent 内存 |
| `compact.ts` | `services/compact.py` | ✅ 完成 | 会话 compact 服务 |
| `sessionStorage.ts` | `storage/session.py` | ✅ 完成 | 会话持久化 |
| `yoloClassifier.ts` | `utils/permissions/classifier.py` | ✅ 完成 | Two-stage classifier |
| `tasks.ts` | `utils/task/file_storage.py` | ✅ 完成 | Task V2 文件存储 |
| `TaskCreateTool.ts` | `tools/task_tools/tool.py` | ✅ 完成 | Task CRUD 工具 |
| `permissionSetup.ts` | `utils/permissions/permission_setup.py` | ✅ 完成 | Permission setup |
| `EnterPlanModeTool.ts` | `tools/plan_mode/enter_plan_mode.py` | ✅ 完成 | Enter plan mode |
| `ExitPlanModeTool.ts` | `tools/plan_mode/exit_plan_mode.py` | ✅ 完成 | Exit plan mode |
| `swarm/inboxPoller.ts` | `utils/swarm/inbox_poller.py` | ✅ 完成 | Inbox poller |
| `swarm/inProcessRunner.ts` | `utils/swarm/in_process_runner.py` | ✅ 完成 | In-process runner |
| `teammateMailbox.ts` | `utils/teammate_mailbox.py` | ✅ 完成 | Teammate mailbox |
| `AskUserQuestionTool.tsx` | `tools/ask_user_question/` | ✅ 完成 | Ask user question |
| `swarm/` coordination | `task/in_process_teammate.py` | ⚠️ 部分 | Spawn via Agent tool |
| `permissionSync.ts` | `utils/swarm/permission_sync.py` | ⚠️ 部分 | User approval queue |
| `mcp/` | `mcp/__init__.py` | ❌ 未实现 | MCP 协议支持 |
| `sandbox-adapter.ts` | — | ❌ 未实现 | Sandbox isolation |

## 关键特性

### 1. Tool 系统
- 泛型基类 `Tool[InputT, OutputT, ProgressT]`
- `build_tool()` 工厂函数
- 自动权限检查 (`check_permissions`)
- 并发安全标记 (`is_concurrency_safe`)
- 只读标记 (`is_read_only`)
- 动态 prompt (`prompt()` 方法)

### 2. 状态管理
- Zustand 风格的 `Store` 类
- 订阅机制 (`subscribe`)
- 不可变更新模式

### 3. 查询引擎
- 异步生成器实现查询循环
- 自动工具分区和并发执行
- 支持中断 (`AbortController`)
- Compact 触发和状态恢复

### 4. 工具编排
- `partition_tool_calls`: 分区为并发/串行批次
- 并发执行只读工具
- 串行执行写操作
- 进度回调支持

### 5. Permission 系统
- Two-stage auto mode classifier
  - Stage 1: acceptEdits fast path (编辑类操作快速通过)
  - Stage 2: LLM classifier with iron gate
- Rule matching (allow/deny/ask rules)
- Dangerous pattern detection
- Denial tracking (防止 classifier 误判)
- Plan mode 权限限制

### 6. Memory 系统
- Auto memory: 跨会话持久记忆
- Session memory: 会话级内存
- Agent memory: Agent scope 内存
- Memory extraction: 自动提取和存储
- Compact integration: 内存 compact 优化

### 7. Task V2 系统
- File-based storage (`.claude/tasks/`)
- Cross-process file locking (fcntl/portalocker)
- High water mark (防止 ID reuse)
- Atomic `claim_task()` for swarm mode

### 8. Swarm/Teammate 系统
- In-process teammate runner
- Mailbox messaging (SendMessage tool)
- Inbox poller for leader
- Permission sync and delegation
- Idle notifications

### 9. Plan Mode
- EnterPlanMode/ExitPlanMode 工具
- Plan 文件管理
- 权限限制 (只读操作)
- 用户审批流程

## 内置工具列表

| 工具 | 功能 | 状态 |
|-----|------|------|
| `Bash` | 执行 shell 命令 | ✅ |
| `Read` | 读取文件 | ✅ |
| `Edit` | 编辑文件 (diff) | ✅ |
| `Write` | 写入文件 | ✅ |
| `Glob` | 文件模式搜索 | ✅ |
| `Grep` | 内容搜索 | ✅ |
| `Agent` | 启动子代理 | ✅ |
| `EnterPlanMode` | 进入计划模式 | ✅ |
| `ExitPlanMode` | 退出计划模式 | ✅ |
| `AskUserQuestion` | 向用户提问 | ✅ |
| `EnterWorktree` | 进入 git worktree | ✅ |
| `ExitWorktree` | 退出 worktree | ✅ |
| `SendMessage` | 发送消息给 teammate | ✅ |
| `TaskCreate` | 创建任务 | ✅ |
| `TaskUpdate` | 更新任务 | ✅ |
| `TaskList` | 列出任务 | ✅ |
| `TaskGet` | 获取任务详情 | ✅ |
| `TaskStop` | 停止任务 | ✅ |
| `TeamCreate` | 创建团队 | ✅ (experimental) |
| `TeamDelete` | 删除团队 | ✅ (experimental) |

## 使用示例

```python
from claude_code_py import QueryEngine, Tool, ToolResult
from claude_code_py.state import Store, AppState
from claude_code_py.tools import get_all_base_tools

# 创建自定义工具
class MyTool(Tool[MyInput, MyOutput, dict]):
    name = "my_tool"
    input_schema = MyInput

    async def call(self, args, context, can_use_tool, parent_message, on_progress=None):
        # 执行逻辑
        return ToolResult(data=MyOutput(result="done"))

# 创建状态存储
store = Store(AppState())

# 获取内置工具
tools = get_all_base_tools()
tools.append(MyTool())

# 创建查询引擎
engine = QueryEngine(
    cwd=".",
    tools=tools,
    commands=[],
    mcp_clients=[],
    agents=[],
    can_use_tool=async_permission_check,
    get_app_state=store.get_state,
    set_app_state=store.set_state,
)

# 提交消息
async for message in engine.submit_message("Hello"):
    print(message)
```

## 统计

- **Python 文件数**: 146
- **核心模块数**: 15
- **工具实现数**: 18+
- **代码行数**: ~47,500
- **测试文件**: 5+

## 下一步工作

### 待实现
- MCP client/server 集成
- Sandbox 系统 (bwrap/bubblewrap)
- Skills 系统 (shell-in-prompt)
- Coordinator mode 完善
- Bridge/Remote mode

### 待修复
- Hooks 系统完整实现
- Teammate 权限请求队列
- Plan approval 用户提示