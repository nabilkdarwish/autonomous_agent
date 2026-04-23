"""
Centralized logging for the agent system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class AgentLogger:
    _loggers: dict = {}

    @classmethod
    def get(cls, name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        if not logger.handlers:
            # Console handler
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.DEBUG)
            fmt = logging.Formatter(
                "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S"
            )
            console.setFormatter(fmt)
            logger.addHandler(console)

            # File handler
            if log_file:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_file)
                fh.setLevel(logging.DEBUG)
                file_fmt = logging.Formatter(
                    "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
                )
                fh.setFormatter(file_fmt)
                logger.addHandler(fh)

        cls._loggers[name] = logger
        return logger


def get_logger(name: str) -> logging.Logger:
    from core.config import get_config
    cfg = get_config()
    return AgentLogger.get(name, cfg.log_file, cfg.log_level)
