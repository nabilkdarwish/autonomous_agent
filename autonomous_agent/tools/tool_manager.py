"""
Tool system — base class + ToolManager registry.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from core.logger import get_logger
from core.models import ToolResult

logger = get_logger("tool_manager")


# ──────────────────────────────────────────────────────────
# Base tool
# ──────────────────────────────────────────────────────────

class BaseTool(ABC):
    """All tools inherit from this class."""

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}     # JSON-schema style

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Run the tool and return a result (any type)."""
        ...

    def run(self, **kwargs) -> ToolResult:
        """Wrapper that measures time and captures errors."""
        start = time.time()
        try:
            output = self.execute(**kwargs)
            elapsed = time.time() - start
            logger.debug(f"Tool '{self.name}' OK in {elapsed:.2f}s")
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output,
                execution_time=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"Tool '{self.name}' FAILED in {elapsed:.2f}s: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=str(e),
                execution_time=elapsed,
            )

    def to_schema(self) -> Dict[str, Any]:
        """Return JSON schema for use with LLM tool-calling APIs."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }


# ──────────────────────────────────────────────────────────
# Tool manager / registry
# ──────────────────────────────────────────────────────────

class ToolManager:
    """
    Central registry for all tools.
    Supports dynamic registration and plugin loading.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        logger.info("ToolManager initialised")

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name}")

    def register_many(self, tools: List[BaseTool]):
        for t in tools:
            self.register(t)

    def unregister(self, name: str):
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Tool unregistered: {name}")

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                output=None,
                error=f"Tool '{name}' not found. Available: {self.list_tools()}",
            )
        return tool.run(**kwargs)

    def schemas_for_anthropic(self) -> List[Dict]:
        return [t.to_schema() for t in self._tools.values()]

    def schemas_for_openai(self) -> List[Dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def load_plugins(self, plugins_dir: str):
        """Discover and register tools from the plugins directory."""
        import importlib.util, os
        plugins_path = os.path.abspath(plugins_dir)
        if not os.path.isdir(plugins_path):
            logger.warning(f"Plugins dir not found: {plugins_path}")
            return
        for fname in os.listdir(plugins_path):
            if fname.endswith(".py") and not fname.startswith("_"):
                fpath = os.path.join(plugins_path, fname)
                spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    # convention: each plugin exposes register(manager)
                    if hasattr(module, "register"):
                        module.register(self)
                        logger.info(f"Plugin loaded: {fname}")
                except Exception as e:
                    logger.error(f"Plugin load error ({fname}): {e}")

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolManager({list(self._tools.keys())})"
