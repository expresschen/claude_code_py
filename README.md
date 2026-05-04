# Claude Code Python Implementation

Python reimplementation of the core architecture from claude-code.

## 环境要求

- **Python 3.11+** (使用现代 type hints, async patterns)
- pydantic >= 2.0
- aiofiles >= 23.0
- httpx >= 0.25

## 安装

```bash
# 安装依赖
pip install pydantic aiofiles httpx

# 或使用 pip
pip install -e .

# 开发依赖
pip install pytest pytest-asyncio mypy ruff black
```

## 启动方式

### 方式一：交互式 REPL 模式

```bash
# 进入交互式命令行界面
python -m claude_code_py.main

# 或直接运行
python run_claude.py
```

启动后：
```
============================================================
Claude Code Python - Interactive Mode
============================================================
Working directory: /path/to/your/project
Tools available: 18
Permission mode: default
============================================================
Type your message and press Enter. Ctrl+C to exit.
============================================================

> 
```

### 方式二：SDK/命令行模式

```bash
# 单次查询
python -m claude_code_py.main "帮我修复这个 bug"

# 输出到文件
python -m claude_code_py.main -o output.txt "分析这个项目"

# 指定工作目录
python -m claude_code_py.main --cwd /path/to/project "你的问题"

# 接受所有工具调用（无权限提示）
python -m claude_code_py.main --accept-all "执行任务"
```

### 方式三：作为库使用

```python
import asyncio
from claude_code_py import QueryEngine, Store
from claude_code_py.tools import get_all_base_tools
from claude_code_py.engine.query_engine import QueryEngineConfig
from claude_code_py.types.permissions import PermissionResult

async def main():
    # 初始化状态存储
    store = Store()
    
    # 获取工具
    tools = get_all_base_tools()
    
    # 配置引擎
    config = QueryEngineConfig(
        cwd="/path/to/project",
        tools=tools,
        commands=[],
        mcp_clients=[],
        agents=[],
        can_use_tool=lambda *args, **kwargs: PermissionResult.allow(),
        get_app_state=store.get_state,
        set_app_state=store.set_state,
    )
    
    # 创建引擎
    engine = QueryEngine(config)
    
    # 发送消息
    response_parts = []
    async for event in engine.submit_message("你的问题"):
        if hasattr(event, "type") and event.type == "assistant":
            content = event.message.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    response_parts.append(block.get("text", ""))
    
    print("".join(response_parts))

asyncio.run(main())
```

## Architecture

```
CLI Layer -> Query/Agent Engine -> Tool/Permission Layer -> Memory/Persistence Layer -> MCP/Remote Extension Layer
```

### 核心模块

| 模块 | 功能 |
|-----|------|
| `tool/` | Tool 基类和上下文 |
| `engine/` | 查询引擎和主循环 |
| `state/` | Zustand-style 状态管理 |
| `memory/` | Auto/session/agent 内存系统 |
| `services/` | Compact, 缓存服务 |
| `storage/` | Session 持久化 |
| `utils/permissions/` | 权限分类和规则 |
| `utils/swarm/` | Teammate 协调系统 |
| `utils/task/` | Task V2 文件存储 |

## 可用工具

| 工具 | 功能 |
|-----|------|
| `Bash` | 执行 shell 命令 |
| `Read` | 读取文件 |
| `Edit` | 编辑文件 (diff-based) |
| `Write` | 写入文件 |
| `Glob` | 文件模式搜索 |
| `Grep` | 内容搜索 |
| `Agent` | 启动子代理 |
| `EnterPlanMode` | 进入计划模式 |
| `ExitPlanMode` | 退出计划模式 |
| `AskUserQuestion` | 向用户提问 |
| `EnterWorktree` | 进入 git worktree |
| `ExitWorktree` | 退出 worktree |
| `SendMessage` | 发送消息给 teammate |
| `TaskCreate` | 创建任务 |
| `TaskUpdate` | 更新任务 |
| `TaskList` | 列出任务 |
| `TaskGet` | 获取任务详情 |
| `TaskStop` | 停止任务 |
| `TeamCreate` | 创建团队 (experimental) |
| `TeamDelete` | 删除团队 (experimental) |

## REPL 命令

在交互式模式中可用的命令：

| 命令 | 功能 |
|-----|------|
| `/help` | 显示帮助 |
| `/tools` | 列出可用工具 |
| `/clear` | 清空对话历史 |
| `/mode` | 显示/更改权限模式 |
| `/exit` | 退出 REPL |

## 环境变量

```bash
# API Key（必需）
export ANTHROPIC_API_KEY=your-api-key

# 接受所有工具调用
export CLAUDE_CODE_ACCEPT_ALL=true

# Auto mode (LLM classifier)
export CLAUDE_CODE_AUTO_MODE=true

# 输出目录（后台任务）
export CLAUDE_CODE_OUTPUT_DIR=.claude/output

# Enable agent teams (experimental)
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=true
```

## 项目结构

```
claude_code_py/
├── main.py                 # CLI 入口
├── __init__.py             # 包导出
├── engine/                 # 查询引擎
│   ├── query_engine.py    # QueryEngine 类
│   ├── query.py           # 主循环
│   └── process_input.py   # 输入处理
├── tools/                  # 工具实现 (18+)
│   ├── bash_tool.py
│   ├── file_read_tool.py
│   ├── file_edit_tool.py
│   ├── file_write_tool.py
│   ├── glob_tool.py
│   ├── grep_tool.py
│   ├── agent_tool/        # 子代理
│   ├── plan_mode/         # 计划模式
│   ├── ask_user_question/ # 用户提问
│   ├── worktree_tool/     # Worktree
│   ├── send_message_tool/ # Teammate 通信
│   ├── task_tools/        # Task CRUD
│   └── team_tools/        # Team 管理
├── tool/                   # 工具基类
│   ├── base.py            # Tool[InputT, OutputT, ProgressT]
│   ├── context.py         # ToolUseContext
│   └── result.py          # ToolResult, ToolError
├── types/                  # 类型定义
│   ├── ids.py             # AgentId, SessionId, TaskId
│   ├── message.py         # Message types
│   └── permissions.py     # PermissionMode, PermissionResult
├── state/                  # 状态管理
│   ├── store.py           # Store (Zustand-like)
│   └── app_state.py       # AppState
├── memory/                 # 内存系统
│   ├── memory_types.py    # SessionMemory, AgentMemoryScope
│   ├── extract.py         # 内存提取
│   └── find_relevant.py   # 相关内存查找
├── services/               # 服务层
│   ├── compact.py         # 会话 compact
│   └── micro_compact.py   # Micro-compact
├── storage/                # 会话存储
│   ├── session.py         # SessionStorage
│   └── session_resume.py  # Session resume
└── utils/                  # 工具函数
    ├── permissions/       # 权限系统
    │   ├── classifier.py  # Two-stage classifier
    │   ├── rules.py       # Permission rules
    │   └── dangerous_patterns.py
    ├── swarm/             # Swarm/Teammate
    │   ├── inbox_poller.py
    │   ├── in_process_runner.py
    │   └── permission_sync.py
    ├── task/              # Task V2 存储
    │   └── file_storage.py
    └── team/              # Team 文件
        └── team_file.py
```

## Tests

```bash
# 运行测试
pytest claude_code_py/utils/task/test_file_storage.py
pytest claude_code_py/tests/swarm/test_collaboration.py
pytest --tb=short

# Lint/format
ruff check claude_code_py/
black claude_code_py/
mypy claude_code_py/ --ignore-missing-imports
```

## License

MIT