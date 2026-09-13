"""Core epistemic types and schemas for ESC (Epistemic State Compilation).

Key architectural invariant:
Persistent epistemic state is strictly typed, bounded, and decoupled from
any LLM reasoning trajectory, scratchpad, or prompt history.
"""

from __future__ import annotations

import time
from typing import Any, Literal
from pydantic import BaseModel, Field

EpistemicLevel = Literal[
    "candidate",
    "supported",
    "verified",
    "disputed",
]

AblationMode = Literal[
    "none",              # Full system: fresh contexts, typed state, witness promotion
    "no_typing",         # Fresh contexts but no typing/contracts
    "shared_history",    # Typed contracts but prior reasoning history shared
    "no_verification",   # Isolation + typing, but no witness check
    "llm_verifier",      # Isolation + probabilistic LLM verifier
    "raw_summaries",     # Isolated contexts + raw natural language summaries instead of typed facts
]

LEVEL_ORDER: dict[EpistemicLevel, int] = {
    "disputed": -1,
    "candidate": 0,
    "supported": 1,
    "verified": 2,
}


def level_ok(current: EpistemicLevel, required: EpistemicLevel) -> bool:
    """Check if the current epistemic level satisfies or exceeds the required level."""
    if current == "disputed":
        return False
    return LEVEL_ORDER.get(current, 0) >= LEVEL_ORDER.get(required, 0)


class Evidence(BaseModel):
    """Grounding citation linking a claim to a verified source span."""

    source_id: str
    span: str


class Fact(BaseModel):
    """A canonical, typed epistemic fact committed to persistent state.

    Notice what isn't present:
      - reasoning_history
      - previous_chain_of_thought
      - critic_comments
      - previous_failed_attempts

    Those remain in the audit log. They never become canonical state.
    """

    key: str
    value: str
    level: EpistemicLevel
    evidence: list[Evidence] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepSpec(BaseModel):
    """Specification of an epistemic transition step in a dependency DAG."""

    step_id: str
    goal: str
    requires: list[str] = Field(default_factory=list)
    permitted_sources: list[str] = Field(default_factory=list)
    required_level: EpistemicLevel = "supported"
    expected_key: str | None = None
    witness_type: Literal["type_1", "type_2", "type_3"] = "type_2"


class StepResult(BaseModel):
    """Output from an atomic epistemic worker invocation."""

    status: Literal[
        "supported",
        "contradicted",
        "insufficient",
        "ambiguous",
    ]
    value: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    raw_output: str | None = None


class AuditEntry(BaseModel):
    """Full execution audit log entry.

    Records disposable cognition (RLM trajectory, reasoning tokens, scratchpad)
    without polluting persistent epistemic memory.
    """

    step_id: str
    goal: str
    input_facts: list[Fact]
    input_evidence_ids: list[str]
    result: StepResult
    committed: bool
    witness_passed: bool = False
    witness_detail: str | None = None
    trajectory: Any | None = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)
