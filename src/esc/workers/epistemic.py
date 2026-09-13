"""EpistemicWorker: Atomic, disposable cognitive worker per step."""

from __future__ import annotations

import time
from typing import Any
import dspy
from pydantic import ValidationError
from esc.workers.errors import ModelOutputError

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
        max_iters: int = 4,
        max_llm_calls: int = 4,
        max_output_chars: int = 4000,
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
                max_output_chars=max_output_chars,
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
        try:
            if isinstance(raw_res, str):
                return StepResult.model_validate_json(raw_res)
            return StepResult.model_validate(raw_res)
        except ValidationError as exc:
            raise ModelOutputError(f"Invalid StepResult: {exc}") from exc
