"""EpistemicWorker: Atomic, disposable cognitive worker per step."""

from __future__ import annotations

import time
from typing import Any
import dspy

from esc.core.signatures import ResolveStep
from esc.core.types import Evidence, Fact, StepResult


class EpistemicWorker(dspy.Module):
    """Disposable epistemic worker.

    Architectural rule:
      One RLM invocation = one disposable cognitive process.
      Never reuse its reasoning trajectory as the context of the next invocation.
    """

    def __init__(
        self,
        sub_lm: dspy.LM | None = None,
        use_rlm: bool = True,
        max_iters: int = 8,
        max_llm_calls: int = 12,
    ) -> None:
        super().__init__()
        self.use_rlm = use_rlm
        self.sub_lm = sub_lm

        if use_rlm:
            self.solve = dspy.RLM(
                ResolveStep,
                sub_lm=sub_lm,
                max_iters=max_iters,
                max_llm_calls=max_llm_calls,
            )
        else:
            self.solve = dspy.ChainOfThought(ResolveStep)

    def forward(
        self,
        goal: str,
        accepted_facts: list[Fact],
        evidence_context: str,
    ) -> StepResult:
        """Resolve exactly one epistemic step without inheriting prior conversational context."""
        start_time = time.perf_counter()

        with dspy.context(lm=self.sub_lm or dspy.settings.lm):
            pred = self.solve(
                goal=goal,
                accepted_facts=accepted_facts,
                evidence_context=evidence_context,
            )

        self.last_trajectory = getattr(pred, "trajectory", None)
        raw_res = getattr(pred, "result", None)
        if isinstance(raw_res, StepResult):
            return raw_res
        if isinstance(raw_res, str):
            return StepResult.model_validate_json(raw_res)
        return StepResult.model_validate(raw_res)
