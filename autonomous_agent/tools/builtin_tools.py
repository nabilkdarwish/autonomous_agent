"""
Built-in tools shipped with the agent.
"""

import ast
import json
import math
import os
import re
import urllib.request
import urllib.parse
from typing import Any, Dict, Optional

from tools.tool_manager import BaseTool


# ──────────────────────────────────────────────────────────
# 1. Calculator
# ──────────────────────────────────────────────────────────

class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression safely. "
        "Supports +, -, *, /, **, sqrt, sin, cos, log, etc."
    )
    parameters = {
        "expression": {
            "type": "string",
            "description": "The math expression to evaluate, e.g. '2 ** 10 + sqrt(144)'"
        }
    }

    # safe symbols
    _SAFE_NAMES = {
        k: getattr(math, k) for k in dir(math) if not k.startswith("_")
    }
    _SAFE_NAMES.update({"abs": abs, "round": round, "int": int, "float": float})

    # ast.Num was removed in Python 3.14; use ast.Constant universally
    _ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Call, ast.Name, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
        ast.Load,
    )

    def execute(self, expression: str) -> str:
        expr = expression.strip()
        try:
            tree = ast.parse(expr, mode="eval")
            self._check_ast(tree)
            result = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, self._SAFE_NAMES)
            return f"{expr} = {result}"
        except Exception as e:
            raise ValueError(f"Cannot evaluate '{expr}': {e}")

    def _check_ast(self, tree):
        for node in ast.walk(tree):
            if not isinstance(node, self._ALLOWED_NODES):
                raise ValueError(f"Unsafe expression node: {type(node).__name__}")


# ──────────────────────────────────────────────────────────
# 2. Web search (DuckDuckGo instant-answer API — no key needed)
# ──────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web and return a summary of results. "
        "Uses DuckDuckGo — no API key required."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "The search query string"
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default 5)"
        }
    }

    def execute(self, query: str, max_results: int = 5) -> str:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        headers = {"User-Agent": "AutonomousAgent/1.0"}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            return f"Search failed: {e}"

        parts = []
        if data.get("AbstractText"):
            parts.append(f"Summary: {data['AbstractText']}")
        if data.get("AbstractSource"):
            parts.append(f"Source: {data['AbstractSource']}")
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                parts.append(f"• {topic['Text']}")
        if not parts:
            return f"No results found for: {query}"
        return "\n".join(parts)


# ──────────────────────────────────────────────────────────
# 3. File reader / writer
# ──────────────────────────────────────────────────────────

class FileReaderTool(BaseTool):
    name = "file_reader"
    description = "Read the contents of a local text file."
    parameters = {
        "path": {
            "type": "string",
            "description": "Absolute or relative path to the file"
        },
        "max_chars": {
            "type": "integer",
            "description": "Maximum characters to read (default 4000)"
        }
    }

    def execute(self, path: str, max_chars: int = 4000) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars)
        truncated = " [truncated]" if os.path.getsize(path) > max_chars else ""
        return content + truncated


class FileWriterTool(BaseTool):
    name = "file_writer"
    description = "Write text content to a local file (creates or overwrites)."
    parameters = {
        "path": {
            "type": "string",
            "description": "Path to write to"
        },
        "content": {
            "type": "string",
            "description": "Text content to write"
        },
        "mode": {
            "type": "string",
            "description": "Write mode: 'w' (overwrite, default) or 'a' (append)"
        }
    }

    def execute(self, path: str, content: str, mode: str = "w") -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"


# ──────────────────────────────────────────────────────────
# 4. HTTP API caller
# ──────────────────────────────────────────────────────────

class APICallerTool(BaseTool):
    name = "api_caller"
    description = "Make an HTTP GET or POST request to any URL and return the response."
    parameters = {
        "url": {
            "type": "string",
            "description": "The full URL to call"
        },
        "method": {
            "type": "string",
            "description": "HTTP method: 'GET' or 'POST' (default GET)"
        },
        "headers": {
            "type": "object",
            "description": "Optional HTTP headers as a JSON object"
        },
        "body": {
            "type": "object",
            "description": "Optional JSON body for POST requests"
        }
    }

    def execute(self, url: str, method: str = "GET",
                headers: Optional[Dict] = None, body: Optional[Dict] = None) -> str:
        method = method.upper()
        req_headers = {"User-Agent": "AutonomousAgent/1.0", "Accept": "application/json"}
        if headers:
            req_headers.update(headers)

        data = None
        if body and method == "POST":
            data = json.dumps(body).encode()
            req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(raw)
                    return json.dumps(parsed, indent=2)[:3000]
                except json.JSONDecodeError:
                    return raw[:3000]
        except Exception as e:
            raise RuntimeError(f"HTTP {method} {url} failed: {e}")


# ──────────────────────────────────────────────────────────
# 5. Python code executor (sandboxed)
# ──────────────────────────────────────────────────────────

class CodeExecutorTool(BaseTool):
    name = "code_executor"
    description = (
        "Execute a Python code snippet and return stdout output. "
        "Useful for data processing, calculations, and transformations."
    )
    parameters = {
        "code": {
            "type": "string",
            "description": "Python code to execute"
        }
    }

    _BLOCKED = ["import os", "import sys", "subprocess", "open(", "__import__",
                "exec(", "eval(", "compile(", "shutil", "socket"]

    def execute(self, code: str) -> str:
        for blocked in self._BLOCKED:
            if blocked in code:
                raise PermissionError(f"Blocked: '{blocked}' is not allowed in code executor")
        import io, contextlib
        stdout_buf = io.StringIO()
        local_ns: Dict[str, Any] = {}
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, {"__builtins__": __builtins__}, local_ns)
            output = stdout_buf.getvalue()
            return output[:2000] if output else "(no output)"
        except Exception as e:
            raise RuntimeError(f"Code execution error: {e}")


# ──────────────────────────────────────────────────────────
# 6. Memory store tool (agent can save facts explicitly)
# ──────────────────────────────────────────────────────────

class MemoryStoreTool(BaseTool):
    name = "remember"
    description = "Store an important fact or result in long-term memory for future recall."
    parameters = {
        "fact": {
            "type": "string",
            "description": "The fact or information to remember"
        },
        "importance": {
            "type": "number",
            "description": "Importance score 0.0–1.0 (default 0.8)"
        }
    }

    def __init__(self, memory_manager=None):
        self._memory = memory_manager

    def set_memory(self, memory_manager):
        self._memory = memory_manager

    def execute(self, fact: str, importance: float = 0.8) -> str:
        if not self._memory:
            return f"Memory not available. Fact noted: {fact}"
        entry_id = self._memory.store_fact(fact, importance=importance)
        return f"Remembered with id={entry_id}: {fact}"


# ──────────────────────────────────────────────────────────
# Factory — returns all default tools
# ──────────────────────────────────────────────────────────

def create_default_tools(memory_manager=None) -> list:
    memory_tool = MemoryStoreTool(memory_manager)
    return [
        CalculatorTool(),
        WebSearchTool(),
        FileReaderTool(),
        FileWriterTool(),
        APICallerTool(),
        CodeExecutorTool(),
        memory_tool,
    ]
