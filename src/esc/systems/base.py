"""Base class and result containers for the four experimental systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

from esc.benchmark.tasks import EpiDAGTask


class SystemResult(BaseModel):
    """Execution result for a single task run on an experimental system."""

    system_name: str
    task_id: str
    depth: int
    final_answer: str | None
    target_answer: str
    is_correct: bool
    intermediate_answers: dict[str, str] = Field(default_factory=dict)
    tokens_used: int = 0
    latency_ms: float = 0.0
    false_promotions: int = 0
    contract_violations: int = 0
    abstained: bool = False
    error_propagated: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class BaseSystem(ABC):
    """Abstract interface for all experimental agent conditions."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, task: EpiDAGTask) -> SystemResult:
        """Execute task through this system architecture."""
        pass
