"""
LLM client abstraction — supports Anthropic (Claude) and OpenAI.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.config import AgentConfig, get_config
from core.logger import get_logger
from core.models import Message, MessageRole

logger = get_logger("llm_client")


class LLMClient(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send messages and return the raw response dict."""
        ...

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract plain text from a response."""
        ...

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict]:
        """Extract tool/function calls from a response."""
        ...


# ──────────────────────────────────────────────────────────
# Anthropic client
# ──────────────────────────────────────────────────────────

class AnthropicClient(LLMClient):
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or get_config()
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            self.model = self.config.model_name
            logger.info(f"Anthropic client initialized | model={self.model}")
        except ImportError:
            raise ImportError("Run: pip install anthropic")

    def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        api_messages = [m.to_llm_dict() for m in messages
                        if m.role != MessageRole.SYSTEM]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": api_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools

        try:
            resp = self.client.messages.create(**kwargs)
            return self._parse(resp)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _parse(self, resp) -> Dict[str, Any]:
        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
        return {
            "text": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "stop_reason": resp.stop_reason,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        }

    def extract_text(self, response: Dict[str, Any]) -> str:
        return response.get("text", "")

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict]:
        return response.get("tool_calls", [])


# ──────────────────────────────────────────────────────────
# OpenAI client
# ──────────────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or get_config()
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.openai_api_key)
            self.model = self.config.model_name if "gpt" in self.config.model_name else "gpt-4o"
            logger.info(f"OpenAI client initialized | model={self.model}")
        except ImportError:
            raise ImportError("Run: pip install openai")

    def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            if m.role != MessageRole.SYSTEM:
                api_messages.append(m.to_llm_dict())

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": api_messages,
            "temperature": self.config.temperature,
        }
        if tools:
            oai_tools = [{"type": "function", "function": t} for t in tools]
            kwargs["tools"] = oai_tools

        try:
            resp = self.client.chat.completions.create(**kwargs)
            return self._parse(resp)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _parse(self, resp) -> Dict[str, Any]:
        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })
        return {
            "text": msg.content or "",
            "tool_calls": tool_calls,
            "stop_reason": resp.choices[0].finish_reason,
            "usage": {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        }

    def extract_text(self, response: Dict[str, Any]) -> str:
        return response.get("text", "")

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict]:
        return response.get("tool_calls", [])


# ──────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────

def create_llm_client(config: Optional[AgentConfig] = None) -> LLMClient:
    cfg = config or get_config()
    provider = cfg.llm_provider.lower()
    if provider == "anthropic":
        return AnthropicClient(cfg)
    elif provider == "openai":
        return OpenAIClient(cfg)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'anthropic' or 'openai'.")
