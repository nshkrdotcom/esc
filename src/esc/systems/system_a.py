"""Condition A — Continuous Context Architecture.

One worker receives the entire task and solves the complete dependency chain
in a continuous reasoning context.
"""

from __future__ import annotations

import time
from typing import Any
import dspy

from esc.benchmark.tasks import EpiDAGTask
from esc.systems.base import BaseSystem, SystemResult
from esc.workers.continuous import ContinuousWorker
from esc.workers.mock import MockContinuousWorker


class SystemAContinuous(BaseSystem):
    """Condition A: Monolithic continuous reasoning context.

    Features:
      - Continuous context accumulates thoughts, intermediate deductions, and noise
      - Vulnerable to cascading error propagation (wrong assumption -> narrate -> rationalize)
      - No typed boundary protection
    """

    def __init__(
        self,
        worker: ContinuousWorker | MockContinuousWorker | None = None,
        use_mock: bool = False,
        mock_error_rate: float = 0.05,
        sub_lm: dspy.LM | None = None,
    ) -> None:
        super().__init__(name="Condition_A_Continuous")
        self.use_mock = use_mock
        if worker is not None:
            self.worker = worker
        elif use_mock:
            self.worker = MockContinuousWorker(error_rate=mock_error_rate)
        else:
            self.worker = ContinuousWorker(sub_lm=sub_lm)

    def run(self, task: EpiDAGTask) -> SystemResult:
        start_time = time.perf_counter()
        corpus = task.get_corpus()
        full_corpus = corpus.full_context()

        node_keys = [node.step_spec.expected_key or node.node_id for node in task.nodes]
        target = task.target_answer()

        step_answers: dict[str, str] = {}
        error_propagated = False
        final_answer: str | None = None
        tokens_used = 1500 + task.depth * 300

        if isinstance(self.worker, MockContinuousWorker):
            res = self.worker.solve_dag(
                task_prompt=task.question,
                ground_truth=task.ground_truth_map,
                node_ids=node_keys,
                injected_error_node=task.injected_error_node_id,
            )
            step_answers = res["step_answers"]
            final_answer = res["final_answer"]
            error_propagated = res["corrupted"]
        else:
            try:
                pred = self.worker.forward(
                    task_description=task.question,
                    corpus_context=full_corpus,
                )
                final_answer = getattr(pred, "final_answer", str(pred))
            except Exception as ex:
                final_answer = None

        duration_ms = (time.perf_counter() - start_time) * 1000
        is_correct = (
            final_answer is not None
            and final_answer.strip().lower() == target.strip().lower()
        )

        false_promotions = sum(
            1 for k, v in step_answers.items()
            if k in task.ground_truth_map and str(v).strip().lower() != str(task.ground_truth_map[k]).strip().lower()
        )

        return SystemResult(
            system_name=self.name,
            task_id=task.task_id,
            depth=task.depth,
            final_answer=final_answer,
            target_answer=target,
            is_correct=is_correct,
            intermediate_answers=step_answers,
            tokens_used=tokens_used,
            latency_ms=duration_ms,
            false_promotions=false_promotions,
            contract_violations=0,  # Continuous has no contracts
            abstained=False,
            error_propagated=error_propagated,
        )
