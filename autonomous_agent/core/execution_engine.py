"""
Execution Engine:
  - Orchestrates plan execution
  - Handles retries + error recovery
  - Logs every step
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from core.config import AgentConfig, get_config
from core.logger import get_logger
from core.models import (AgentResponse, MessageRole, Plan, ReflectionResult,
                          Step, TaskStatus, ToolResult)
from core.agent_brain import AgentBrain
from memory.memory_manager import MemoryManager
from tools.tool_manager import ToolManager

logger = get_logger("execution_engine")


class ExecutionEngine:
    """
    Orchestrates the full agent lifecycle:
      1. Receive goal
      2. Build context (memory)
      3. Plan
      4. Execute each step (with retry)
      5. Synthesize result
      6. Self-reflect (optional)
      7. Return final answer
    """

    def __init__(
        self,
        brain: AgentBrain,
        tool_manager: ToolManager,
        memory: MemoryManager,
        config: Optional[AgentConfig] = None,
    ):
        self.brain = brain
        self.tools = tool_manager
        self.memory = memory
        self.config = config or get_config()
        logger.info("ExecutionEngine ready")

    # ──────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────

    def run(self, goal: str) -> AgentResponse:
        start_time = time.time()
        logger.info(f"=== New Goal ===  {goal[:100]}")

        # 1. Pull relevant memories
        memory_context = self.memory.get_context(goal)
        if memory_context:
            logger.debug(f"Retrieved memory context ({len(memory_context)} chars)")

        # 2. Create plan
        tool_names = self.tools.list_tools()
        plan = self.brain.plan(goal, tool_names, memory_context)

        if not plan.steps:
            return AgentResponse(
                goal=goal,
                final_answer="I was unable to create a plan for this goal.",
                success=False,
                duration=time.time() - start_time,
            )

        # 3. Execute steps
        previous_results: List[str] = []
        for step in plan.steps:
            self._execute_step(step, plan, previous_results)
            if step.result:
                previous_results.append(f"Step {step.step_id}: {step.result[:400]}")
            # Abort if critical step fails after all retries
            if step.status == TaskStatus.FAILED:
                logger.warning(f"Step {step.step_id} permanently failed — continuing plan")

        # 4. Synthesize
        logger.info("Synthesizing final answer...")
        final_answer = self.brain.synthesize(goal, plan.steps)

        # 5. Self-reflection
        reflection: Optional[ReflectionResult] = None
        if self.config.enable_reflection:
            reflection = self.brain.reflect(goal, final_answer, plan.summary())
            if reflection and reflection.revised_output:
                logger.info("Reflection produced a revised output")
                final_answer = reflection.revised_output

        # 6. Store result in long-term memory
        self.memory.store_fact(
            f"Goal: {goal[:200]}\nResult: {final_answer[:400]}",
            metadata={"type": "task_result"},
            importance=0.8,
        )

        duration = time.time() - start_time
        completed = sum(1 for s in plan.steps if s.status == TaskStatus.COMPLETED)
        logger.info(f"=== Done in {duration:.1f}s | {completed}/{len(plan.steps)} steps ===")

        return AgentResponse(
            goal=goal,
            final_answer=final_answer,
            plan=plan,
            reflection=reflection,
            success=not plan.has_failures(),
            total_steps=len(plan.steps),
            completed_steps=completed,
            duration=duration,
        )

    # ──────────────────────────────────────────────────────
    # Step execution + retry
    # ──────────────────────────────────────────────────────

    def _execute_step(self, step: Step, plan: Plan, previous_results: List[str]):
        logger.info(f"▶  Step {step.step_id}/{len(plan.steps)}: {step.description[:70]}")
        step.mark_started()

        for attempt in range(1, self.config.max_retries + 1):
            try:
                result = self._run_step(step, plan, previous_results)
                step.mark_completed(result)
                logger.info(f"   ✓ Step {step.step_id} completed in {step.duration():.2f}s")
                return
            except Exception as e:
                step.retries = attempt
                logger.warning(f"   ✗ Step {step.step_id} attempt {attempt}/{self.config.max_retries}: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * attempt)
                else:
                    step.mark_failed(str(e))
                    logger.error(f"   ✗ Step {step.step_id} permanently failed: {e}")

    def _run_step(self, step: Step, plan: Plan, previous_results: List[str]) -> str:
        # Steps without a tool — pure LLM reasoning
        if not step.tool_name:
            return self._reasoning_step(step, plan, previous_results)

        # Verify / adjust args with reasoner
        proceed, updated_args = self.brain.verify_step(step, plan, previous_results)
        if not proceed:
            logger.info(f"   ⏭  Step {step.step_id} skipped by reasoner")
            step.status = TaskStatus.SKIPPED
            return "Skipped by reasoner"

        # Execute tool
        tool_result: ToolResult = self.tools.execute(step.tool_name, **updated_args)
        if not tool_result.success:
            raise RuntimeError(tool_result.error)
        return tool_result.to_string()

    def _reasoning_step(self, step: Step, plan: Plan, previous_results: List[str]) -> str:
        """Ask the LLM to handle a pure-reasoning step."""
        from core.models import Message, MessageRole
        context_lines = [
            f"Goal: {plan.goal}",
            f"Current step: {step.description}",
            "Previous results:",
            *previous_results[-5:],
        ]
        prompt = "\n".join(context_lines)
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = self.brain.llm.complete(messages, max_tokens=1024)
        return self.brain.llm.extract_text(response)
