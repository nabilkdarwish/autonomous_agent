# 🤖 ARIA — Autonomous Reasoning and Intelligence Agent

A production-level autonomous AI agent system built from scratch in Python.
ARIA accepts complex, multi-step goals, plans how to achieve them, executes
actions using pluggable tools, maintains layered memory, and self-evaluates
its own outputs before returning them to the user.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Task Planning** | Decomposes any natural-language goal into ordered steps |
| **Tool Calling** | Calculator, Web Search, File I/O, HTTP API, Code Executor, Memory |
| **Short-term Memory** | Sliding-window conversation buffer (configurable size) |
| **Long-term Memory** | FAISS vector store with semantic search; keyword fallback |
| **Self-Reflection Loop** | Agent scores its own output, revises if confidence is low |
| **Plugin System** | Drop a `.py` file in `plugins/` — it is auto-loaded at startup |
| **Retry / Error Recovery** | Every step retries up to N times with exponential back-off |
| **Multi-provider LLM** | Supports Anthropic (Claude) and OpenAI (GPT-4o) |
| **Clean CLI** | Colour-coded interactive shell with commands |

---

## 📁 Folder Structure

```
autonomous_agent/
│
├── core/                        # Brain, engine, LLM client, data models
│   ├── agent.py                 # Top-level Agent class (public API)
│   ├── agent_brain.py           # Planner · Reasoner · Reflector
│   ├── config.py                # AgentConfig dataclass + singleton
│   ├── execution_engine.py      # Orchestrates plan → steps → result
│   ├── llm_client.py            # Anthropic / OpenAI abstraction
│   ├── logger.py                # Centralised logging
│   └── models.py                # Shared data classes (Plan, Step, …)
│
├── tools/                       # Tool registry and built-in tools
│   ├── tool_manager.py          # BaseTool ABC + ToolManager registry
│   └── builtin_tools.py         # Calculator, WebSearch, File*, API, Code, Memory
│
├── memory/                      # Memory system
│   └── memory_manager.py        # ShortTermMemory · LongTermMemory · MemoryManager
│
├── interface/                   # User-facing interfaces
│   └── cli.py                   # Interactive CLI (colour-coded)
│
├── plugins/                     # Auto-loaded tool plugins
│   └── weather_plugin.py        # Example: live weather via Open-Meteo
│
├── examples/                    # Runnable example tasks
│   └── run_examples.py
│
├── tests/                       # Unit tests (21 tests, 0 external deps)
│   └── test_agent.py
│
├── data/                        # Runtime data (vector store, output files)
│   └── vector_store/
│
├── logs/                        # Agent execution logs
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / Interface                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ goal (natural language)
┌──────────────────────────▼──────────────────────────────────┐
│                         Agent                                │
│  (wires all subsystems; single public entry point)           │
└──────┬──────────────────┬───────────────────────────────────┘
       │                  │
       ▼                  ▼
┌─────────────┐   ┌───────────────────────────────────────────┐
│   Memory    │   │            Execution Engine                │
│  Manager   ◄────►  1. Pull memory context                   │
│             │   │  2. Call Brain.plan()  →  Plan             │
│ Short-term  │   │  3. For each Step:                         │
│ (messages)  │   │     a. Brain.verify_step()                 │
│             │   │     b. ToolManager.execute()               │
│ Long-term   │   │     c. Retry on failure                    │
│ (FAISS)     │   │  4. Brain.synthesize()                     │
└─────────────┘   │  5. Brain.reflect()  →  revise if needed  │
                  │  6. Store result in long-term memory       │
                  └───────────────┬───────────────────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │        Agent Brain        │
                    │                          │
                    │  Planner   → JSON plan   │
                    │  Reasoner  → synthesis   │
                    │  Reflector → self-eval   │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │       LLM Client          │
                    │  Anthropic / OpenAI       │
                    └──────────────────────────┘
                                  │ (tool calls)
                    ┌─────────────▼────────────┐
                    │       Tool Manager        │
                    │  calculator               │
                    │  web_search               │
                    │  file_reader/writer       │
                    │  api_caller               │
                    │  code_executor            │
                    │  remember                 │
                    │  [plugins…]               │
                    └──────────────────────────┘
```

### Core Design Decisions

**1. Separation of concerns**
Each layer has a single responsibility. The Brain never touches files;
the Execution Engine never talks directly to the LLM for planning; the
CLI never knows about memory internals.

**2. Fail-safe execution**
Every tool call is wrapped in a retry loop with exponential back-off.
A failed step is logged and the engine continues rather than crashing.

**3. Dual memory**
Short-term memory is a plain list — fast, O(1) appends, auto-trimmed.
Long-term memory is a FAISS vector index for semantic similarity search,
with automatic pickle persistence. If FAISS isn't installed, it falls
back to TF-IDF keyword overlap — no crash, just less accuracy.

**4. Self-reflection loop**
After synthesis, the Reflector sends the goal + output back to the LLM
for a structured critique (JSON). If confidence < threshold or issues
exist, the agent produces a revised answer. This is the "second pass"
that often catches incomplete or incorrect first drafts.

**5. Plugin system**
Any `.py` file dropped into `plugins/` that exposes `register(manager)`
is discovered and loaded at startup. No config changes needed.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd autonomous_agent
pip install -r requirements.txt
```

For vector-powered long-term memory (optional but recommended):

```bash
pip install faiss-cpu sentence-transformers
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your key:
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

### 3. Launch the CLI

```bash
# Anthropic (default)
python interface/cli.py

# OpenAI
python interface/cli.py --provider openai

# Override model
python interface/cli.py --provider anthropic --model claude-opus-4-5
```

### 4. Run example tasks

```bash
python examples/run_examples.py
```

### 5. Run tests

```bash
python tests/test_agent.py        # built-in unittest runner
# or
pytest tests/ -v                  # if pytest is installed
```

---

## 💬 CLI Commands

| Command | Description |
|---|---|
| (any text) | Submit a goal to the agent |
| `/tools` | List all available tools |
| `/stats` | Show agent statistics + memory info |
| `/memory` | Preview short-term and long-term memory |
| `/clear` | Clear short-term conversation buffer |
| `/remember <text>` | Manually store a fact in long-term memory |
| `/help` | Show help |
| `/exit` | Quit |

---

## 🛠️ Example Goals

```
# Math + reasoning
Calculate compound interest on $50,000 at 6% for 10 years compounded monthly.

# Multi-step research
Search for the latest Python web frameworks, compare their pros and cons,
and save a summary report to data/frameworks.txt

# Code execution
Write Python code to find all prime numbers up to 500 and show their count.

# File pipeline
Read the file data/notes.txt, summarise it in 3 bullet points,
and save the summary to data/summary.txt

# Complex chained task
Plan a 30-day fitness routine for a beginner, calculate total weekly exercise
hours, and save the plan as a markdown file.
```

---

## 🔌 Writing a Plugin

Create `plugins/my_tool.py`:

```python
from tools.tool_manager import BaseTool

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "What this tool does"
    parameters = {
        "input": {"type": "string", "description": "The input"}
    }

    def execute(self, input: str) -> str:
        return f"Processed: {input}"

def register(tool_manager):
    tool_manager.register(MyCustomTool())
```

That's it. The agent will discover and use it automatically on next start.

---

## ⚙️ Configuration

Edit `core/config.py` or pass a custom `AgentConfig`:

```python
from core.config import AgentConfig, set_config
from core.agent import Agent

config = AgentConfig(
    llm_provider="anthropic",          # or "openai"
    model_name="claude-sonnet-4-5",
    max_tokens=4096,
    short_term_max_messages=30,
    long_term_top_k=5,
    max_retries=3,
    enable_reflection=True,
    reflection_threshold=0.7,
    agent_name="MyAgent",
)
set_config(config)
agent = Agent(config)
```

---

## 🧪 Test Coverage

| Suite | Tests | Status |
|---|---|---|
| CalculatorTool | 5 | ✅ All pass |
| FileReaderTool / FileWriterTool | 3 | ✅ All pass |
| CodeExecutorTool | 4 | ✅ All pass |
| ShortTermMemory | 3 | ✅ All pass |
| LongTermMemory | 3 | ✅ All pass |
| Plan / Step models | 3 | ✅ All pass |
| **Total** | **21** | ✅ |

---

## 📦 Technology Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude / OpenAI GPT |
| Vector Search | FAISS + sentence-transformers |
| HTTP | urllib (stdlib, zero extra deps) |
| AST Safety | Python `ast` module |
| Logging | Python `logging` (stdlib) |
| CLI | Python `readline` + ANSI codes |
| Testing | Python `unittest` (stdlib) |

---

## 🗺️ Roadmap

- [ ] Async execution engine (asyncio)
- [ ] Web UI (FastAPI + React)
- [ ] Multi-agent collaboration
- [ ] Tool result caching
- [ ] LangChain / LlamaIndex integration
- [ ] Docker container + Compose file
- [ ] Streaming responses in CLI
