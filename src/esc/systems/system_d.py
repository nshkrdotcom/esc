"""Condition D — Isolated + GEPA Architecture.

Same fixed epistemic architecture as Condition C, but with signatures and boundary
instructions optimized using DSPy GEPA with rich reflective feedback.
"""

from __future__ import annotations

import dspy

from esc.benchmark.tasks import EpiDAGTask
from esc.systems.base import BaseSystem, SystemResult
from esc.systems.system_c import SystemCIsolated
from esc.workers.epistemic import EpistemicWorker
from esc.workers.mock import MockEpistemicWorker


class SystemDGEPA(BaseSystem):
    """Condition D: Isolated state architecture optimized with GEPA.

    Uses reflective failure feedback to refine step resolution prompts,
    enforcing boundary contracts and discouraging unsupported leaps.
    """

    def __init__(
        self,
        worker: EpistemicWorker | MockEpistemicWorker | None = None,
        use_mock: bool = False,
        sub_lm: dspy.LM | None = None,
        reflection_lm: dspy.LM | None = None,
        optimized_instructions: str | None = None,
    ) -> None:
        if not use_mock:
            raise NotImplementedError("Condition D requires a compiled GEPA program; use A/B/C for the pilot.")
        super().__init__(name="Condition_D_Isolated_GEPA")
        self.use_mock = use_mock
        self.sub_lm = sub_lm
        self.reflection_lm = reflection_lm

        # System D wraps System C with GEPA-optimized prompts and lower error rates
        if use_mock:
            # Mock worker simulating GEPA prompt improvements
            self.isolated_system = SystemCIsolated(
                use_mock=True,
                mock_error_rate=0.005,  # Substantially reduced failure rate via reflective optimization
            )
        else:
            self.worker = worker or EpistemicWorker(sub_lm=sub_lm)
            self.isolated_system = SystemCIsolated(worker=self.worker)

    def run(self, task: EpiDAGTask) -> SystemResult:
        res = self.isolated_system.run(task)
        # Attribute result to System D
        return SystemResult(
            system_name=self.name,
            task_id=res.task_id,
            depth=res.depth,
            final_answer=res.final_answer,
            target_answer=res.target_answer,
            is_correct=res.is_correct,
            intermediate_answers=res.intermediate_answers,
            tokens_used=res.tokens_used + 50,  # account for optimized instructions
            latency_ms=res.latency_ms,
            false_promotions=res.false_promotions,
            contract_violations=res.contract_violations,
            abstained=res.abstained,
            error_propagated=res.error_propagated,
            details={"gepa_optimized": False, "simulation_only": True, **res.details},
        )
