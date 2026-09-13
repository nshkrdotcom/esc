"""Condition B — Search-Heavy Architecture.

Continuous worker with retries and Best-of-N sampling matching the compute
budget of Condition C. Tests the trade-off:
  Spend compute on trajectories (Condition B)
  vs.
  Spend compute on boundaries (Condition C)
"""

from __future__ import annotations

from collections import Counter
import time
import dspy

from esc.benchmark.tasks import EpiDAGTask
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
    ) -> None:
        super().__init__(name="Condition_B_SearchHeavy")
        self.n_samples = n_samples
        self.use_mock = use_mock
        self.mock_error_rate = mock_error_rate
        self.sub_lm = sub_lm

    def run(self, task: EpiDAGTask) -> SystemResult:
        start_time = time.perf_counter()
        target = task.target_answer()

        results: list[SystemResult] = []
        total_tokens = 0
        answers: list[str] = []

        for i in range(self.n_samples):
            sub_system = SystemAContinuous(
                use_mock=self.use_mock,
                mock_error_rate=self.mock_error_rate,
                sub_lm=self.sub_lm,
            )
            sub_res = sub_system.run(task)
            results.append(sub_res)
            total_tokens += sub_res.tokens_used
            if sub_res.final_answer:
                answers.append(sub_res.final_answer.strip())

        # Majority vote / Best-of-N selection
        if answers:
            vote_counts = Counter(answers)
            best_answer, _ = vote_counts.most_common(1)[0]
        else:
            best_answer = None

        duration_ms = (time.perf_counter() - start_time) * 1000
        is_correct = (
            best_answer is not None
            and best_answer.lower() == target.strip().lower()
        )

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
            abstained=False,
            error_propagated=error_propagated,
            details={"rollouts_count": len(results)},
        )
