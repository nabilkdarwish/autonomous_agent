"""
Agent — top-level orchestrator that wires all subsystems.
"""

from typing import Optional

from core.config import AgentConfig, get_config
from core.agent_brain import AgentBrain
from core.execution_engine import ExecutionEngine
from core.logger import get_logger
from core.models import AgentResponse, MessageRole
from memory.memory_manager import MemoryManager
from tools.tool_manager import ToolManager
from tools.builtin_tools import create_default_tools

logger = get_logger("agent")


class Agent:
    """
    High-level interface.
    Instantiate once, call run(goal) repeatedly.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or get_config()

        # Memory
        self.memory = MemoryManager(self.config)

        # Tools
        self.tools = ToolManager()
        default_tools = create_default_tools(self.memory)
        self.tools.register_many(default_tools)

        # Load plugins
        if self.config.enable_plugins:
            self.tools.load_plugins(self.config.plugins_dir)

        # Brain
        self.brain = AgentBrain(self.config)

        # Execution engine
        self.engine = ExecutionEngine(self.brain, self.tools, self.memory, self.config)

        logger.info(
            f"{self.config.agent_name} ready | "
            f"tools={len(self.tools)} | "
            f"provider={self.config.llm_provider}"
        )

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def run(self, goal: str) -> AgentResponse:
        """Execute a goal end-to-end and return a structured response."""
        self.memory.add_message(MessageRole.USER, goal)
        response = self.engine.run(goal)
        self.memory.add_message(MessageRole.ASSISTANT, response.final_answer)
        return response

    def chat(self, message: str) -> str:
        """Convenience wrapper — returns just the final answer string."""
        response = self.run(message)
        return response.final_answer

    def remember(self, fact: str, importance: float = 1.0):
        """Manually store a fact in long-term memory."""
        self.memory.store_fact(fact, importance=importance)

    def add_tool(self, tool):
        """Register a custom tool at runtime."""
        self.tools.register(tool)
        logger.info(f"Custom tool added: {tool.name}")

    def stats(self) -> dict:
        return {
            "agent": self.config.agent_name,
            "provider": self.config.llm_provider,
            "model": self.config.model_name,
            "tools": self.tools.list_tools(),
            "memory": self.memory.stats(),
        }
