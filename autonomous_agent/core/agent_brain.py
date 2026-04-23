"""
Agent Brain:
  - Planner   : decomposes goals into steps
  - Reasoner  : decides next action given context
  - Reflector : self-evaluates outputs
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import AgentConfig, get_config
from core.llm_client import LLMClient, create_llm_client
from core.logger import get_logger
from core.models import Message, MessageRole, Plan, ReflectionResult, Step

logger = get_logger("agent_brain")


# ──────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are {agent_name} — {agent_description}.

You are an autonomous AI agent. You think step-by-step, use tools when needed, 
and always strive to complete the user's goal accurately and efficiently.

## Available Tools
{tools_description}

## Memory Context
{memory_context}

## Guidelines
- Break complex goals into clear, actionable steps
- Use tools whenever they can improve accuracy
- Be concise but thorough
- Always verify your work before responding
- If a step fails, adapt your plan
"""

PLANNER_PROMPT = """Analyze the following goal and create a detailed, ordered execution plan.

GOAL: {goal}

CONTEXT: {context}

Respond ONLY with a valid JSON object in this exact format:
{{
  "steps": [
    {{
      "step_id": 1,
      "description": "What this step does",
      "tool_name": "tool_name_or_null",
      "tool_args": {{}}
    }}
  ],
  "reasoning": "Brief explanation of your approach"
}}

Available tools: {tool_names}
Use null for tool_name if no tool is needed (just reasoning/analysis).
"""

REASONER_PROMPT = """You are executing step {step_id} of {total_steps} for this goal:
GOAL: {goal}

CURRENT STEP: {step_description}
TOOL TO USE: {tool_name}
TOOL ARGUMENTS: {tool_args}

PREVIOUS RESULTS:
{previous_results}

If the tool arguments look correct, confirm them. 
If they need adjustment based on previous results, provide updated JSON args.
Respond with a JSON object:
{{
  "proceed": true/false,
  "updated_args": {{...}},
  "reasoning": "why"
}}
"""

REFLECTION_PROMPT = """Evaluate this agent output for quality and completeness.

ORIGINAL GOAL: {goal}
AGENT OUTPUT: {output}
EXECUTION SUMMARY: {summary}

Rate the output on:
1. Does it fully address the goal?
2. Is it accurate and well-reasoned?
3. Are there any errors or gaps?

Respond with JSON:
{{
  "is_satisfactory": true/false,
  "confidence": 0.0-1.0,
  "issues": ["issue1", ...],
  "suggestions": ["suggestion1", ...],
  "revised_output": "improved version if needed, else null"
}}
"""

SYNTHESIS_PROMPT = """Synthesize all execution results into a clear, helpful final answer.

GOAL: {goal}

EXECUTION STEPS AND RESULTS:
{step_results}

Provide a comprehensive, well-structured answer to the user's goal.
Be clear, accurate, and helpful. Format nicely if the content warrants it.
"""


# ──────────────────────────────────────────────────────────
# Planner
# ──────────────────────────────────────────────────────────

class Planner:
    def __init__(self, llm: LLMClient, config: Optional[AgentConfig] = None):
        self.llm = llm
        self.config = config or get_config()

    def create_plan(self, goal: str, tool_names: List[str],
                    context: str = "") -> Plan:
        logger.info(f"Planning goal: {goal[:80]}...")
        prompt = PLANNER_PROMPT.format(
            goal=goal,
            context=context or "No prior context",
            tool_names=", ".join(tool_names) if tool_names else "none",
        )
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = self.llm.complete(messages)
        text = self.llm.extract_text(response)

        plan = Plan(goal=goal, context=context)
        try:
            data = self._parse_json(text)
            for s in data.get("steps", []):
                plan.add_step(
                    description=s.get("description", ""),
                    tool_name=s.get("tool_name") if s.get("tool_name") != "null" else None,
                    tool_args=s.get("tool_args", {}),
                )
            logger.info(f"Plan created: {len(plan.steps)} steps | {data.get('reasoning','')[:60]}")
        except Exception as e:
            logger.warning(f"JSON plan parse failed ({e}), creating single-step plan")
            plan.add_step(description=goal, tool_name=None)

        return plan

    def _parse_json(self, text: str) -> Dict:
        # Try to extract JSON block even if surrounded by prose
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"(\{.*\})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        return json.loads(text)


# ──────────────────────────────────────────────────────────
# Reasoner
# ──────────────────────────────────────────────────────────

class Reasoner:
    def __init__(self, llm: LLMClient, config: Optional[AgentConfig] = None):
        self.llm = llm
        self.config = config or get_config()

    def verify_step(self, step: Step, plan: Plan,
                    previous_results: List[str]) -> Tuple[bool, Dict]:
        """Check whether a step's args should be adjusted before execution."""
        if not step.tool_name or not step.tool_args:
            return True, step.tool_args

        prompt = REASONER_PROMPT.format(
            step_id=step.step_id,
            total_steps=len(plan.steps),
            goal=plan.goal,
            step_description=step.description,
            tool_name=step.tool_name,
            tool_args=json.dumps(step.tool_args),
            previous_results="\n".join(previous_results[-3:]) or "None",
        )
        messages = [Message(role=MessageRole.USER, content=prompt)]
        try:
            response = self.llm.complete(messages, max_tokens=512)
            text = self.llm.extract_text(response)
            data = json.loads(self._extract_json(text))
            proceed = data.get("proceed", True)
            updated = data.get("updated_args", step.tool_args)
            logger.debug(f"Reasoner: proceed={proceed}, args updated={updated != step.tool_args}")
            return proceed, updated
        except Exception as e:
            logger.debug(f"Reasoner skipped (non-critical): {e}")
            return True, step.tool_args

    def synthesize(self, goal: str, steps: List[Step]) -> str:
        """Produce the final consolidated answer from all step results."""
        results_text = []
        for s in steps:
            status = "✓" if s.status.value == "completed" else "✗"
            results_text.append(f"Step {s.step_id} [{status}] {s.description}")
            if s.result:
                results_text.append(f"  Result: {s.result[:500]}")
            if s.error:
                results_text.append(f"  Error: {s.error}")

        prompt = SYNTHESIS_PROMPT.format(
            goal=goal,
            step_results="\n".join(results_text),
        )
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = self.llm.complete(messages)
        return self.llm.extract_text(response)

    def _extract_json(self, text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text


# ──────────────────────────────────────────────────────────
# Reflector (self-evaluation loop)
# ──────────────────────────────────────────────────────────

class Reflector:
    def __init__(self, llm: LLMClient, config: Optional[AgentConfig] = None):
        self.llm = llm
        self.config = config or get_config()

    def reflect(self, goal: str, output: str, plan_summary: str) -> ReflectionResult:
        logger.info("Running self-reflection...")
        prompt = REFLECTION_PROMPT.format(
            goal=goal,
            output=output,
            summary=plan_summary,
        )
        messages = [Message(role=MessageRole.USER, content=prompt)]
        try:
            response = self.llm.complete(messages, max_tokens=1024)
            text = self.llm.extract_text(response)
            data = json.loads(self._extract_json(text))
            result = ReflectionResult(
                is_satisfactory=data.get("is_satisfactory", True),
                confidence=float(data.get("confidence", 0.8)),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                revised_output=data.get("revised_output"),
            )
            logger.info(
                f"Reflection: satisfactory={result.is_satisfactory}, "
                f"confidence={result.confidence:.2f}, issues={len(result.issues)}"
            )
            return result
        except Exception as e:
            logger.warning(f"Reflection failed (non-critical): {e}")
            return ReflectionResult(is_satisfactory=True, confidence=0.9)

    def _extract_json(self, text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text


# ──────────────────────────────────────────────────────────
# Agent Brain (composes all three)
# ──────────────────────────────────────────────────────────

class AgentBrain:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or get_config()
        self.llm = create_llm_client(self.config)
        self.planner = Planner(self.llm, self.config)
        self.reasoner = Reasoner(self.llm, self.config)
        self.reflector = Reflector(self.llm, self.config) if self.config.enable_reflection else None
        logger.info(f"AgentBrain ready | reflection={self.config.enable_reflection}")

    def plan(self, goal: str, tool_names: List[str], context: str = "") -> Plan:
        return self.planner.create_plan(goal, tool_names, context)

    def verify_step(self, step: Step, plan: Plan, previous_results: List[str]) -> Tuple[bool, Dict]:
        return self.reasoner.verify_step(step, plan, previous_results)

    def synthesize(self, goal: str, steps: List[Step]) -> str:
        return self.reasoner.synthesize(goal, steps)

    def reflect(self, goal: str, output: str, plan_summary: str) -> Optional[ReflectionResult]:
        if self.reflector:
            return self.reflector.reflect(goal, output, plan_summary)
        return None

    def build_system_prompt(self, tools_description: str, memory_context: str) -> str:
        return SYSTEM_PROMPT.format(
            agent_name=self.config.agent_name,
            agent_description=self.config.agent_description,
            tools_description=tools_description,
            memory_context=memory_context or "No prior memories.",
        )
