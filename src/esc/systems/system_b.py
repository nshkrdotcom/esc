"""Condition B — Search-Heavy Architecture.

Continuous worker with independent rollouts and majority voting. An optional
episode allowance is shared by all rollouts; actual spend is not matched to C.
Tests the trade-off:
  Spend compute on trajectories (Condition B)
  vs.
  Spend compute on boundaries (Condition C)
"""

from __future__ import annotations

from collections import Counter
import time
import dspy

from esc.benchmark.tasks import EpiDAGTask
from esc.config import RLMConfig
from esc.budget import EpisodeBudgetExhausted
from esc.core.answers import answers_equal, vote_key
from esc.systems.base import BaseSystem, SystemResult
from esc.systems.system_a import SystemAContinuous


class SystemBSearchHeavy(BaseSystem):
    """Condition B: Continuous context with Best-of-N search / retries.

    Allocates compute to rolling multiple trajectories rather than isolating state.
    """

    def __init__(
        self,
        n_samples: int = 3,
        use_mock: bool = False,
        mock_error_rate: float = 0.08,
        sub_lm: dspy.LM | None = None,
        rlm_config: RLMConfig | None = None,
    ) -> None:
        super().__init__(name="Condition_B_SearchHeavy")
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        self.n_samples = n_samples
        self.use_mock = use_mock
        self.mock_error_rate = mock_error_rate
        self.sub_lm = sub_lm
        self.sub_system = SystemAContinuous(use_mock=use_mock, mock_error_rate=mock_error_rate,
                                           sub_lm=sub_lm, rlm_config=rlm_config)

    def run(self, task: EpiDAGTask) -> SystemResult:
        start_time = time.perf_counter()
        target = task.target_answer()

        results: list[SystemResult] = []
        total_tokens = 0
        answers: list[str] = []
        exhausted = False

        for i in range(self.n_samples):
            try:
                sub_res = self.sub_system.run(task)
            except EpisodeBudgetExhausted:
                exhausted = True
                break
            results.append(sub_res)
            total_tokens += sub_res.tokens_used
            if sub_res.final_answer:
                answers.append(vote_key(sub_res.final_answer))

        # Majority vote / Best-of-N selection
        if answers:
            vote_counts = Counter(answers)
            best_answer, _ = vote_counts.most_common(1)[0]
        else:
            best_answer = None

        duration_ms = (time.perf_counter() - start_time) * 1000
        is_correct = answers_equal(best_answer, target)

        # Average false promotions across sample rollouts
        avg_false_promotions = int(sum(r.false_promotions for r in results) / len(results)) if results else 0
        error_propagated = any(r.error_propagated for r in results)

        return SystemResult(
            system_name=self.name,
            task_id=task.task_id,
            depth=task.depth,
            final_answer=best_answer,
            target_answer=target,
            is_correct=is_correct,
            intermediate_answers={"best_of_n_votes": str(dict(Counter(answers)))},
            tokens_used=total_tokens,
            latency_ms=duration_ms,
            false_promotions=avg_false_promotions,
            contract_violations=0,
            abstained=best_answer is None,
            budget_exhausted=exhausted,
            model_output_error=any(r.model_output_error for r in results),
            error_propagated=error_propagated,
            details={"rollouts_count": len(results), "rollouts": [r.model_dump() for r in results]},
        )
