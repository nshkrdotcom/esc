"""ContinuousWorker: Monolithic reasoning worker for Conditions A and B."""

from __future__ import annotations

import dspy

from esc.core.signatures import ContinuousSolve


class ContinuousWorker(dspy.Module):
    """Monolithic worker that receives the entire task and runs a continuous reasoning trajectory.

    Condition A uses this directly.
    Condition B wraps this in Best-of-N / retries matching compute budget.
    """

    def __init__(
        self,
        sub_lm: dspy.LM | None = None,
        use_rlm: bool = True,
        max_iters: int = 16,
        max_llm_calls: int = 24,
    ) -> None:
        super().__init__()
        self.use_rlm = use_rlm
        self.sub_lm = sub_lm

        if use_rlm:
            try:
                self.solve = dspy.RLM(
                    ContinuousSolve,
                    sub_lm=sub_lm,
                    max_iters=max_iters,
                    max_llm_calls=max_llm_calls,
                )
            except Exception:
                self.solve = dspy.ChainOfThought(ContinuousSolve)
        else:
            self.solve = dspy.ChainOfThought(ContinuousSolve)

    def forward(self, task_description: str, corpus_context: str) -> dspy.Prediction:
        return self.solve(
            task_description=task_description,
            corpus_context=corpus_context,
        )
