#!/usr/bin/env python3
"""
Example tasks demonstrating ARIA's capabilities.
Run: python examples/run_examples.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import AgentConfig, set_config
from core.agent import Agent

def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def run_example(agent: Agent, goal: str, label: str):
    print(f"\n🎯 Goal: {goal}\n")
    response = agent.run(goal)
    print(f"📋 Answer:\n{response.final_answer}")
    print(f"\n✅ Steps: {response.completed_steps}/{response.total_steps} | "
          f"Time: {response.duration:.1f}s | Success: {response.success}")
    if response.reflection:
        r = response.reflection
        print(f"🔍 Reflection: confidence={r.confidence:.0%}, issues={len(r.issues)}")
    print()


def main():
    config = AgentConfig(
        llm_provider="anthropic",        # change to "openai" if preferred
        enable_reflection=True,
        max_iterations=10,
    )
    set_config(config)
    agent = Agent(config)

    # ── Example 1: Pure reasoning ──────────────────────────
    section("Example 1: Multi-step Math Reasoning")
    run_example(
        agent,
        goal="A company had revenue of $2.4M in Q1 and grew 15% each quarter. "
             "What is the projected annual revenue? Also calculate the Q4 revenue specifically.",
        label="math"
    )

    # ── Example 2: Tool use — calculator ──────────────────
    section("Example 2: Calculator Tool")
    run_example(
        agent,
        goal="Calculate the compound interest on $10,000 at 7.5% annual rate over 5 years, "
             "compounded monthly. Show the formula and the final amount.",
        label="calculator"
    )

    # ── Example 3: Web search + synthesis ─────────────────
    section("Example 3: Web Search + Synthesis")
    run_example(
        agent,
        goal="Search for information about the Python programming language and summarize "
             "its key features, use cases, and why developers love it.",
        label="web_search"
    )

    # ── Example 4: File I/O ────────────────────────────────
    section("Example 4: File Operations")
    run_example(
        agent,
        goal="Create a file called 'data/report.txt' with a short markdown report about "
             "the benefits of autonomous AI agents. Include 3 key benefits with explanations.",
        label="file_io"
    )

    # ── Example 5: Multi-step complex task ────────────────
    section("Example 5: Complex Multi-step Task")
    run_example(
        agent,
        goal="I need to plan a Python learning path. Search for Python learning resources, "
             "create a 4-week study plan, calculate how many hours per day needed if studying "
             "5 days/week for 4 weeks with a total of 80 study hours, and save the plan to "
             "'data/python_study_plan.txt'.",
        label="complex"
    )

    # ── Example 6: Code execution ──────────────────────────
    section("Example 6: Code Execution")
    run_example(
        agent,
        goal="Write and execute Python code that generates the first 15 Fibonacci numbers "
             "and calculates their sum.",
        label="code"
    )

    # ── Example 7: Memory demonstration ───────────────────
    section("Example 7: Long-term Memory")
    agent.remember("The user's name is Alex and they are a senior Python developer")
    agent.remember("The user prefers concise, technical explanations")
    run_example(
        agent,
        goal="Given what you know about me, suggest 3 advanced Python topics I should study "
             "and explain why each would be valuable for my level.",
        label="memory"
    )

    print("\n" + "=" * 60)
    print("  All examples complete!")
    print(f"  Agent stats: {agent.stats()['memory']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
