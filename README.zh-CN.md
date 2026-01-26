# 🚀🧠 Deep Agents

智能体正在不断提升解决长周期任务的能力，[其可完成任务长度每 7 个月翻一倍](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)。但长周期任务通常需要几十次工具调用，带来成本与可靠性挑战。像 [Claude Code](https://code.claude.com/docs) 和 [Manus](https://www.youtube.com/watch?v=6_BcCthVvb8) 等热门智能体使用一些共通原则来应对这些挑战，包括 **规划**（执行前计划）、**计算机访问**（让智能体可访问 shell 与文件系统）、以及 **子智能体委派**（隔离的任务执行）。`deepagents` 是一个简单的开源智能体运行框架，内置这些工具，并可轻松扩展自定义工具、指令与所选 LLM。

<img src=".github/images/deepagents-banner.png" alt="deep agent" width="100%"/>

## 📚 资源

- **[文档](https://docs.langchain.com/oss/python/deepagents/overview)** - 全面概览与 API 参考
- **[Quickstarts Repo](https://github.com/langchain-ai/deepagents-quickstarts)** - 示例与用例
- **[CLI](libs/deepagents-cli/)** - 带技能、记忆与 HITL 工作流的交互式命令行界面

## 🚀 快速开始

`deepagents` 支持自定义工具以及内置工具（见下文）。本示例将添加可选的 `tavily` 工具进行网络搜索。

```bash
pip install deepagents tavily-python

# using uv
uv init
uv add deepagents tavily-python
```

在环境中设置 `TAVILY_API_KEY`（[在此获取](https://www.tavily.com/)）：

```python
import os

from deepagents import create_deep_agent
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(query: str, max_results: int = 5):
    """Run a web search"""
    return tavily_client.search(query, max_results=max_results)

agent = create_deep_agent(
    tools=[internet_search],
    system_prompt="Conduct research and write a polished report.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is LangGraph?"}]})
```

通过 `create_deep_agent` 创建的 agent 是一个编译后的 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) `StateGraph`，因此可以像任何 LangGraph agent 一样使用流式输出、人类在回路（HITL）、记忆或 Studio。更多示例请查看我们的 [quickstarts repo](https://github.com/langchain-ai/deepagents-quickstarts)。

## CLI 配置

Deep Agents CLI 支持为与 OpenAI 或 Anthropic API 兼容的第三方服务配置自定义 base URL。

### `base_url`

- CLI 参数：`--base-url`（优先级高于环境变量）
- 环境变量：
  - `OPENAI_BASE_URL`
  - `ANTHROPIC_BASE_URL`
  - `GOOGLE_BASE_URL`
- 当模型名无法推断 provider 时，请使用 `--provider`。
- 注意：`langchain-google-genai` 可能不支持自定义 `base_url`，因此 Google 端点可能会被底层客户端忽略。

```bash
# 智谱开发者套餐（Anthropic 兼容端点）
export ANTHROPIC_API_KEY="34d9d991**************CXE1"
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
deepagents --model GLM-4.7 --provider anthropic

# OpenAI 兼容端点（示例）
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
deepagents --model deepseek-chat --provider openai

# 运行时覆盖
deepagents --model llama-3.1-70b --provider openai --base-url https://api.groq.com/openai/v1
```

## 自定义 Deep Agents

你可以向 [`create_deep_agent`](https://reference.langchain.com/python/deepagents/#deepagents.create_deep_agent) 传入多个参数。

### `model`

默认情况下，`deepagents` 使用 `claude-sonnet-4-5-20250929`。你可以传入任意 [LangChain 模型对象](https://docs.langchain.com/oss/python/integrations/providers/overview) 进行自定义。

```python
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

model = init_chat_model("openai:gpt-4o")
agent = create_deep_agent(
    model=model,
)
```

### `system_prompt`

你可以为 `create_deep_agent()` 提供 `system_prompt` 参数。该自定义提示词会被 **追加** 到中间件自动注入的默认指令之后。

编写自定义系统提示词时，建议：

- ✅ 定义领域专用流程（如研究方法、数据分析步骤）
- ✅ 提供适用于你的用例的具体示例
- ✅ 添加专项指导（例如“把相似研究任务批量合并为一个 TODO”）
- ✅ 定义停止条件与资源限制
- ✅ 解释工具之间的协作方式

**不要：**

- ❌ 重新解释标准工具的用途（中间件已覆盖）
- ❌ 复制中间件的工具使用说明
- ❌ 违背默认指令（要与其配合，而不是对抗）

```python
from deepagents import create_deep_agent

research_instructions = """your custom system prompt"""
agent = create_deep_agent(
    system_prompt=research_instructions,
)
```

更多示例请参见 [quickstarts repo](https://github.com/langchain-ai/deepagents-quickstarts)。

### `tools`

为你的 agent 提供自定义工具（除 [内置工具](#内置工具) 之外）：

```python
from deepagents import create_deep_agent

def internet_search(query: str) -> str:
    """Run a web search"""
    return tavily_client.search(query)

agent = create_deep_agent(tools=[internet_search])
```

你也可以通过 [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) 连接 MCP 工具：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

async def main():
    mcp_client = MultiServerMCPClient(...)
    mcp_tools = await mcp_client.get_tools()
    agent = create_deep_agent(tools=mcp_tools)

    async for chunk in agent.astream({"messages": [{"role": "user", "content": "..."}]}):
        chunk["messages"][-1].pretty_print()
```

### `middleware`

Deep agents 使用 [中间件](https://docs.langchain.com/oss/python/langchain/middleware) 进行扩展（默认见 [内置工具](#内置工具)）。你可以添加自定义中间件来注入工具、修改提示词或钩子到 agent 生命周期：

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

class WeatherMiddleware(AgentMiddleware):
    tools = [get_weather]

agent = create_deep_agent(middleware=[WeatherMiddleware()])
```

### `subagents`

主 agent 可以通过 `task` 工具委派工作给子 agent（见 [内置工具](#内置工具)）。你可以提供自定义子 agent 以实现上下文隔离与专用指令：

```python
from deepagents import create_deep_agent

research_subagent = {
    "name": "research-agent",
    "description": "Used to research in-depth questions",
    "system_prompt": "You are an expert researcher",
    "tools": [internet_search],
    "model": "openai:gpt-4o",  # Optional, defaults to main agent model
}

agent = create_deep_agent(subagents=[research_subagent])
```

更复杂场景可传入预构建的 LangGraph 图：

```python
from deepagents import CompiledSubAgent, create_deep_agent

custom_graph = create_agent(model=..., tools=..., system_prompt=...)

agent = create_deep_agent(
    subagents=[CompiledSubAgent(
        name="data-analyzer",
        description="Specialized agent for data analysis",
        runnable=custom_graph
    )]
)
```

更多详情请查看 [subagents 文档](https://docs.langchain.com/oss/python/deepagents/subagents)。

### `interrupt_on`

部分工具较为敏感，执行前可能需要人工批准。Deepagents 通过 LangGraph 的中断能力支持人机交互（HITL）工作流。你可以使用 checkpointer 配置哪些工具需要审批。

这些工具配置会传给预构建的 [HITL 中间件](https://docs.langchain.com/oss/python/langchain/middleware#human-in-the-loop)，使 agent 在执行配置工具前暂停并等待用户反馈。

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[get_weather],
    interrupt_on={
        "get_weather": {
            "allowed_decisions": ["approve", "edit", "reject"]
        },
    }
)
```

更多详情请查看 [human-in-the-loop 文档](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)。

### `backend`

Deep agents 使用可插拔后端来控制文件系统操作。默认情况下，文件存储在 agent 的临时状态中。你可以配置不同后端用于本地磁盘访问、跨对话持久化存储或混合路由。

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/path/to/project"),
)
```

可用后端包括：

- **`StateBackend`**（默认）：文件存储在 agent 状态中（临时）
- **`FilesystemBackend`**：在指定根目录下进行真实磁盘操作
- **`StoreBackend`**：使用 LangGraph Store 的持久化存储
- **`CompositeBackend`**：将不同路径路由到不同后端

更多详情请查看 [backends 文档](https://docs.langchain.com/oss/python/deepagents/backends)。

### 长期记忆

Deep agents 可以通过 `CompositeBackend` 将特定路径路由到持久化存储，从而跨对话保留长期记忆。

这使得混合记忆成为可能：工作文件保持临时，而关键数据（如用户偏好或知识库）可跨线程持久保存。

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(store=InMemoryStore())},
    ),
)
```

`/memories/` 下的文件会在所有对话中持续存在，其它路径仍然是临时的。典型用例包括：

- 跨会话保留用户偏好
- 从多次对话中构建知识库
- 基于反馈自我改进指令
- 保持研究进度跨线程延续

更多详情请查看 [long-term memory 文档](https://docs.langchain.com/oss/python/deepagents/long-term-memory)。

## 内置工具

<img src=".github/images/deepagents_tools.png" alt="deep agent" width="600"/>

所有通过 `create_deep_agent` 创建的 deep agent 都自带一组标准工具：

| 工具名称 | 描述 | 提供者 |
|-----------|-------------|-------------|
| `write_todos` | 创建并管理结构化任务列表，以跟踪复杂工作流进度 | `TodoListMiddleware` |
| `read_todos` | 读取当前任务列表状态 | `TodoListMiddleware` |
| `ls` | 列出目录中的所有文件（需要绝对路径） | `FilesystemMiddleware` |
| `read_file` | 读取文件内容（支持 offset/limit 分页参数） | `FilesystemMiddleware` |
| `write_file` | 创建新文件或完全覆盖已有文件 | `FilesystemMiddleware` |
| `edit_file` | 对文件进行精确字符串替换 | `FilesystemMiddleware` |
| `glob` | 按模式匹配文件（如 `**/*.py`） | `FilesystemMiddleware` |
| `grep` | 在文件中搜索文本模式 | `FilesystemMiddleware` |
| `execute`* | 在沙箱环境中运行 shell 命令 | `FilesystemMiddleware` |
| `task` | 将任务委派给具有隔离上下文的专用子 agent | `SubAgentMiddleware` |

`execute` 工具仅在后端实现 `SandboxBackendProtocol` 时可用。默认使用的是内存状态后端，不支持命令执行。如上所示，这些工具（以及其它能力）由默认中间件提供：

更多详情请查看 [agent harness 文档](https://docs.langchain.com/oss/python/deepagents/harness)。

## 内置中间件

`deepagents` 在内部使用中间件。以下是所用中间件列表。

| 中间件 | 作用 |
|------------|---------|
| **`TodoListMiddleware`** | 任务规划与进度跟踪 |
| **`FilesystemMiddleware`** | 文件操作与上下文卸载（自动保存大结果） |
| **`SubAgentMiddleware`** | 委派任务给隔离子 agent |
| **`SummarizationMiddleware`** | 上下文超过 170k token 时自动摘要 |
| **`AnthropicPromptCachingMiddleware`** | 缓存 system prompt 以降低成本（Anthropic 专用） |
| **`PatchToolCallsMiddleware`** | 修复中断导致的悬挂工具调用 |
| **`HumanInTheLoopMiddleware`** | 人类在回路审批（需 `interrupt_on` 配置） |

## 内置提示词

中间件会自动添加关于标准工具的指令。你的自定义指令应 **补充而非重复** 这些默认内容：

#### 来自 [`TodoListMiddleware`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/middleware/todo.py)

- 说明何时使用 `write_todos` 和 `read_todos`
- 任务列表管理的最佳实践
- 何时不应使用 todo 列表（简单任务）

#### 来自 [`FilesystemMiddleware`](libs/deepagents/deepagents/middleware/filesystem.py)

- 列出所有文件系统工具（`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`*）
- 说明文件路径必须以 `/` 开头
- 解释各工具用途与参数
- 关于大工具结果的上下文卸载说明

#### 来自 [`SubAgentMiddleware`](libs/deepagents/deepagents/middleware/subagents.py)

- 解释用于委派子 agent 的 `task()` 工具
- 何时使用子 agent、何时不应使用
- 并行执行的指导
- 子 agent 生命周期（spawn → run → return → reconcile）

## 安全注意事项

### 信任模型

Deepagents 采用与 Claude Code 相似的“信任 LLM”模型。智能体可以执行底层工具允许的任何操作。安全边界应在工具/沙箱层面强制执行，而不应指望 LLM 自我约束。
