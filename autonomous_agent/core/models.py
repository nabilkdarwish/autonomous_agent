"""
Shared data models for the agent system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    def to_llm_dict(self) -> Dict[str, str]:
        """Format for LLM API calls."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class Step:
    step_id: int
    description: str
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def mark_started(self):
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def mark_completed(self, result: str):
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def mark_failed(self, error: str):
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()

    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class Plan:
    goal: str
    steps: List[Step] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    context: str = ""

    def add_step(self, description: str, tool_name: Optional[str] = None,
                 tool_args: Optional[Dict] = None) -> Step:
        step = Step(
            step_id=len(self.steps) + 1,
            description=description,
            tool_name=tool_name,
            tool_args=tool_args or {}
        )
        self.steps.append(step)
        return step

    def next_pending(self) -> Optional[Step]:
        for s in self.steps:
            if s.status == TaskStatus.PENDING:
                return s
        return None

    def is_complete(self) -> bool:
        return all(s.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for s in self.steps)

    def has_failures(self) -> bool:
        return any(s.status == TaskStatus.FAILED for s in self.steps)

    def summary(self) -> str:
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.status == TaskStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == TaskStatus.FAILED)
        return f"Plan: {done}/{total} completed, {failed} failed"


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_string(self) -> str:
        if self.success:
            return str(self.output)
        return f"[ERROR in {self.tool_name}]: {self.error}"


@dataclass
class ReflectionResult:
    is_satisfactory: bool
    confidence: float          # 0.0 - 1.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    revised_output: Optional[str] = None

    def needs_revision(self) -> bool:
        return not self.is_satisfactory or bool(self.issues)


@dataclass
class AgentResponse:
    goal: str
    final_answer: str
    plan: Optional[Plan] = None
    reflection: Optional[ReflectionResult] = None
    success: bool = True
    error: Optional[str] = None
    total_steps: int = 0
    completed_steps: int = 0
    duration: float = 0.0
