# ARIA — Autonomous Reasoning and Intelligence Agent

ARIA is an experimental Python prototype for tool-using AI agents. It accepts a goal, asks a configured language model to plan steps, executes registered tools, stores conversation and long-term memory, and can review the resulting answer.

## Capabilities

- Planning, tool selection, retry handling, and answer reflection.
- Anthropic and OpenAI provider adapters.
- Calculator, web search, local file read/write, HTTP API, memory, and restricted Python tools.
- Short-term conversation history and optional vector-backed long-term memory.
- A plugin convention for registering additional tools.
- An interactive command-line interface and example tasks.

## Security and scope

This is a learning prototype, not a production agent or a security boundary. Tools run in the same process as the CLI. The file tools can access paths available to that process, and the HTTP tool can contact URLs requested by the agent.

The Python runner now validates syntax through an AST allowlist, exposes a small set of built-ins, and limits source length, loop nesting, range sizes, and output. It blocks imports, attribute access, and filesystem or network APIs. It still executes inside the application process; use a disposable environment and trusted prompts, and do not treat it as an OS-level sandbox.

## Requirements and setup

Use a Python version supported by the packages in requirements.txt. Configure an API key for the provider you plan to use.

~~~bash
cd autonomous_agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
~~~

Add your provider key to the local environment or .env file. Do not commit API keys.

## Run

~~~bash
# Anthropic
python interface/cli.py --provider anthropic

# OpenAI
python interface/cli.py --provider openai

# Example tasks
python examples/run_examples.py
~~~

## CLI commands

| Command | Action |
|---|---|
| /tools | List registered tools |
| /stats | Show agent and memory information |
| /memory | Preview stored memory |
| /clear | Clear short-term conversation history |
| /remember <text> | Store a fact in long-term memory |
| /help | Show available commands |
| /exit | Exit the CLI |

## Architecture

- core/: agent orchestration, planning and reflection, provider adapters, configuration, and shared models.
- tools/: tool interface, registry, built-in tools, and plugin discovery.
- memory/: short-term and long-term memory implementations.
- interface/: command-line application.
- plugins/: example tool plugin.
- examples/: sample goals and runs.
- tests/: standard-library unittest suite.

## Tests

The repository includes unit tests in tests/test_agent.py. Run them from the package directory with:

~~~bash
python tests/test_agent.py
~~~

## Author

Nabil Khaled Darwish
