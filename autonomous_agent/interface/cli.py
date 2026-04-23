#!/usr/bin/env python3
"""
Interactive CLI for the Autonomous AI Agent.
Run: python interface/cli.py
"""

import os
import sys
import time
import textwrap
from typing import Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import AgentConfig, set_config
from core.agent import Agent
from core.logger import get_logger

logger = get_logger("cli")

# ── ANSI colour helpers ─────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"

def c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET

def banner():
    lines = [
        "",
        c("╔══════════════════════════════════════════════════╗", BOLD, CYAN),
        c("║      🤖  ARIA — Autonomous AI Agent  v1.0        ║", BOLD, CYAN),
        c("║  Reasoning • Planning • Memory • Tool Execution  ║", DIM, CYAN),
        c("╚══════════════════════════════════════════════════╝", BOLD, CYAN),
        "",
    ]
    print("\n".join(lines))

def print_response(response):
    """Pretty-print an AgentResponse."""
    width = 72

    # Header
    print()
    print(c("─" * width, CYAN))
    status = c("✓ SUCCESS", GREEN) if response.success else c("⚠ PARTIAL", YELLOW)
    duration = c(f"{response.duration:.1f}s", DIM)
    steps_info = c(f"{response.completed_steps}/{response.total_steps} steps", DIM)
    print(f" {status}  {steps_info}  {duration}")
    print(c("─" * width, CYAN))

    # Final answer
    print(c("\n📋 Answer:\n", BOLD))
    wrapped = textwrap.fill(response.final_answer, width=width, subsequent_indent="  ")
    print("  " + wrapped.replace("\n", "\n  "))

    # Plan summary
    if response.plan and response.plan.steps:
        print(c("\n📌 Execution Plan:\n", BOLD))
        for step in response.plan.steps:
            icon = {"completed": "✓", "failed": "✗", "skipped": "⏭", "pending": "○"}.get(
                step.status.value, "•"
            )
            color = {
                "completed": GREEN, "failed": RED,
                "skipped": YELLOW, "pending": DIM
            }.get(step.status.value, RESET)
            tool = f" [{step.tool_name}]" if step.tool_name else ""
            dur = f" ({step.duration():.1f}s)" if step.duration() else ""
            print(f"  {c(icon, color)}  Step {step.step_id}: {step.description[:55]}"
                  + c(tool, DIM) + c(dur, DIM))
            if step.error:
                print(c(f"       ⚡ {step.error[:70]}", RED))

    # Reflection
    if response.reflection:
        r = response.reflection
        conf_color = GREEN if r.confidence > 0.75 else YELLOW if r.confidence > 0.5 else RED
        print(c(f"\n🔍 Self-Reflection: confidence={r.confidence:.0%}", BOLD))
        for issue in r.issues[:3]:
            print(c(f"   ⚠  {issue}", YELLOW))

    print(c("─" * width, CYAN))
    print()

def print_stats(agent: Agent):
    stats = agent.stats()
    print(c("\n📊 Agent Stats:", BOLD, CYAN))
    print(f"  Name     : {stats['agent']}")
    print(f"  Provider : {stats['provider']} / {stats['model']}")
    print(f"  Tools    : {', '.join(stats['tools'])}")
    mem = stats["memory"]
    print(f"  Memory   : {mem['short_term']} | LTM entries: {mem['long_term']['total_entries']}")
    print()

COMMANDS = {
    "/help"    : "Show this help",
    "/tools"   : "List available tools",
    "/stats"   : "Show agent statistics",
    "/memory"  : "Show memory contents",
    "/clear"   : "Clear short-term memory",
    "/remember <text>": "Manually store a fact in long-term memory",
    "/exit"    : "Quit",
}

def print_help():
    print(c("\nCommands:", BOLD))
    for cmd, desc in COMMANDS.items():
        print(f"  {c(cmd, CYAN):<30} {desc}")
    print()


# ── Main loop ───────────────────────────────────────────────────────────────

def main(provider: str = "anthropic", model: Optional[str] = None):
    banner()

    # Config
    config = AgentConfig(llm_provider=provider)
    if model:
        config.model_name = model
    set_config(config)

    # Init agent
    print(c("Initialising agent...", DIM))
    try:
        agent = Agent(config)
    except Exception as e:
        print(c(f"\n❌  Failed to initialise agent: {e}", RED))
        print(c("Make sure your API key is set in the environment:", YELLOW))
        print(c("  export ANTHROPIC_API_KEY=sk-...", DIM))
        sys.exit(1)

    print(c(f"✓  {config.agent_name} is online!", GREEN))
    print(c("Type your goal, or /help for commands.\n", DIM))

    while True:
        try:
            user_input = input(c("You › ", BOLD, BLUE)).strip()
        except (KeyboardInterrupt, EOFError):
            print(c("\n\nGoodbye! 👋\n", CYAN))
            break

        if not user_input:
            continue

        # ── Built-in commands ──────────────────────────────

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print(c("\nGoodbye! 👋\n", CYAN))
            break

        elif user_input.lower() == "/help":
            print_help()

        elif user_input.lower() == "/stats":
            print_stats(agent)

        elif user_input.lower() == "/tools":
            tools = agent.tools.list_tools()
            print(c(f"\n🛠  Available Tools ({len(tools)}):", BOLD))
            for t in tools:
                tool = agent.tools.get(t)
                desc = tool.description[:60] if tool else ""
                print(f"  • {c(t, CYAN):<25} {desc}")
            print()

        elif user_input.lower() == "/memory":
            stm = agent.memory.short_term.get_messages()
            ltm_stats = agent.memory.long_term.stats()
            print(c(f"\n🧠 Short-term ({len(stm)} messages):", BOLD))
            for m in stm[-5:]:
                preview = m.content[:60].replace("\n", " ")
                print(f"  [{m.role.value}] {preview}")
            print(c(f"\n💾 Long-term: {ltm_stats['total_entries']} entries stored\n", BOLD))

        elif user_input.lower() == "/clear":
            agent.memory.clear_short_term()
            print(c("✓ Short-term memory cleared\n", GREEN))

        elif user_input.lower().startswith("/remember "):
            fact = user_input[10:].strip()
            agent.remember(fact, importance=1.0)
            print(c(f"✓ Stored: {fact[:60]}\n", GREEN))

        # ── Agent goal ────────────────────────────────────

        else:
            print(c(f"\n⚙  Processing...\n", DIM))
            try:
                response = agent.run(user_input)
                print_response(response)
            except KeyboardInterrupt:
                print(c("\n\n⚡ Interrupted.\n", YELLOW))
            except Exception as e:
                print(c(f"\n❌ Error: {e}\n", RED))
                logger.exception("Unhandled error during agent run")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARIA — Autonomous AI Agent CLI")
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--model", default=None,
                        help="Override model name")
    args = parser.parse_args()
    main(provider=args.provider, model=args.model)
