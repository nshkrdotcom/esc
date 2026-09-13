"""ContinuousWorker: Monolithic reasoning worker for Conditions A and B."""

from __future__ import annotations

import dspy

from esc.core.signatures import ContinuousSolve


class ContinuousWorker(dspy.Module):
    """Monolithic worker that receives the entire task and runs a continuous reasoning trajectory.

    Condition A uses this directly.
    Condition B wraps this in independent rollouts sharing an optional episode allowance.
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
                ContinuousSolve,
                sub_lm=sub_lm,
                max_iters=max_iters,
                max_llm_calls=max_llm_calls,
                max_output_chars=max_output_chars,
            )
        else:
            self.solve = dspy.ChainOfThought(ContinuousSolve)

    def forward(self, task_description: str, corpus_context: str) -> dspy.Prediction:
        with dspy.context(lm=self.sub_lm or dspy.settings.lm):
            return self.solve(
                task_description=task_description,
                corpus_context=corpus_context,
            )
