"""
Core configuration for the Autonomous AI Agent System.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    # LLM Settings
    llm_provider: str = "anthropic"          # "anthropic" or "openai"
    model_name: str = "claude-sonnet-4-5"
    max_tokens: int = 4096
    temperature: float = 0.7

    # API Keys (loaded from env)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Memory Settings
    short_term_max_messages: int = 20
    long_term_top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store_path: str = "data/vector_store"

    # Execution Settings
    max_iterations: int = 15
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0

    # Self-Reflection Settings
    enable_reflection: bool = True
    reflection_threshold: float = 0.7   # trigger if confidence below this

    # Plugin Settings
    plugins_dir: str = "plugins"
    enable_plugins: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/agent.log"

    # Agent Persona
    agent_name: str = "ARIA"
    agent_description: str = "Autonomous Reasoning and Intelligence Agent"


# Global config singleton
_config: Optional[AgentConfig] = None


def get_config() -> AgentConfig:
    global _config
    if _config is None:
        _config = AgentConfig()
    return _config


def set_config(config: AgentConfig):
    global _config
    _config = config
