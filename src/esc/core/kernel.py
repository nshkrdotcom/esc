"""StateKernel: Persistent epistemic state manager and context compiler.

The LM does not decide what prior linguistic material belongs in its context.
The StateKernel does.
"""

from __future__ import annotations

from typing import Any
from esc.core.answers import answers_equal
from esc.core.types import AuditEntry, EpistemicLevel, Fact, StepResult, StepSpec, level_ok
from esc.core.witness import WitnessResult


class StateKernel:
    """Deterministic state kernel and context compiler for epistemic state.

    Maintains canonical typed state across steps while discarding
    intermediate LLM reasoning traces.
    """

    def __init__(self) -> None:
        self.facts: dict[str, Fact] = {}
        self.audit_log: list[AuditEntry] = []

    def project(self, step: StepSpec) -> list[Fact]:
        """Context Compiler: Filter canonical state to only facts required by the step contract.

        Rejects facts below the contract's required epistemic level.
        """
        projected: list[Fact] = []
        for key in step.requires:
            if key in self.facts:
                fact = self.facts[key]
                if level_ok(fact.level, step.required_level):
                    projected.append(fact.model_copy(deep=True))
        return projected

    def commit(
        self,
        fact: Fact,
        witness: WitnessResult,
        required_level: EpistemicLevel = "supported",
    ) -> bool:
        """Commit a candidate fact to canonical persistent state if promotion rules pass."""
        if not witness.passed:
            return False

        # Apply the promoted level from witness
        fact = fact.model_copy(deep=True)
        fact.level = witness.promoted_level

        # Verify that the promoted level satisfies contract
        if not level_ok(fact.level, required_level):
            return False

        self.facts[fact.key] = fact
        return True

    def record_audit(
        self,
        step: StepSpec,
        input_facts: list[Fact],
        result: StepResult,
        committed: bool,
        witness: WitnessResult | None = None,
        trajectory: Any | None = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
    ) -> AuditEntry:
        """Record full execution trace into audit log without leaking into canonical state."""
        entry = AuditEntry(
            step_id=step.step_id,
            goal=step.goal,
            input_facts=input_facts,
            input_evidence_ids=step.permitted_sources,
            result=result,
            committed=committed,
            witness_passed=witness.passed if witness else False,
            witness_detail=witness.detail if witness else None,
            trajectory=trajectory,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )
        self.audit_log.append(entry)
        return entry

    def get_fact(self, key: str) -> Fact | None:
        fact = self.facts.get(key)
        return fact.model_copy(deep=True) if fact else None

    def total_tokens_used(self) -> int:
        return sum(e.tokens_used for e in self.audit_log)

    def false_promotion_count(self, ground_truth_facts: dict[str, str]) -> int:
        """Count how many committed facts disagree with ground truth (evaluator check)."""
        count = 0
        for k, f in self.facts.items():
            if k in ground_truth_facts:
                true_val = str(ground_truth_facts[k]).strip().lower()
                pred_val = str(f.value).strip().lower()
                if not answers_equal(pred_val, true_val):
                    count += 1
        return count

    def reset(self) -> None:
        self.facts.clear()
        self.audit_log.clear()
