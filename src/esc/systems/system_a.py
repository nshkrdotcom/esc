"""Condition A — Continuous Context Architecture.

One worker receives the entire task and solves the complete dependency chain
in a continuous reasoning context.
"""

from __future__ import annotations

import time
from typing import Any
import dspy

from esc.benchmark.tasks import EpiDAGTask
from esc.config import RLMConfig
from esc.core.answers import answers_equal
from esc.systems.base import BaseSystem, SystemResult
from esc.workers.continuous import ContinuousWorker
from esc.workers.usage import invoke_with_usage
from esc.workers.errors import ModelOutputError
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
        rlm_config: RLMConfig | None = None,
    ) -> None:
        super().__init__(name="Condition_A_Continuous")
        self.use_mock = use_mock
        if worker is not None:
            self.worker = worker
        elif use_mock:
            self.worker = MockContinuousWorker(error_rate=mock_error_rate)
        else:
            self.worker = ContinuousWorker(sub_lm=sub_lm, **(rlm_config or RLMConfig()).model_dump())

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
        details = {"usage_kind": "simulated", "injected_error": bool(task.injected_error_node_id)}
        if task.injected_error_node_id and not isinstance(self.worker, MockContinuousWorker):
            raise NotImplementedError("Live injection into a continuous RLM trajectory is not implemented")

        if isinstance(self.worker, MockContinuousWorker):
            res = self.worker.solve_dag(
                task_prompt=task.question,
                ground_truth=task.ground_truth_map,
                node_ids=node_keys,
                injected_error_node=(
                    (task.node_by_id(task.injected_error_node_id).step_spec.expected_key or task.injected_error_node_id)
                    if task.injected_error_node_id else None
                ),
            )
            step_answers = res["step_answers"]
            final_answer = res["final_answer"]
            if task.injected_error_node_id:
                injected = task.node_by_id(task.injected_error_node_id)
                descendants = {injected.step_spec.expected_key or injected.node_id}
                for node in task.nodes:
                    key = node.step_spec.expected_key or node.node_id
                    if any(parent in descendants for parent in node.step_spec.requires):
                        descendants.add(key)
                        if key in step_answers and not answers_equal(step_answers[key], task.ground_truth_map[key]):
                            error_propagated = True
        else:
            try:
                pred, usage = invoke_with_usage(
                    self.worker, task_description=task.public_description(), corpus_context=full_corpus,
                )
                if not hasattr(pred, "final_answer"):
                    raise ModelOutputError("Continuous worker omitted final_answer", usage)
                final_answer = pred.final_answer
                if final_answer is not None and not isinstance(final_answer, str):
                    raise ModelOutputError("Continuous worker returned an invalid final_answer type", usage)
            except ModelOutputError as exc:
                return SystemResult(system_name=self.name, task_id=task.task_id, depth=task.depth,
                    final_answer=None, target_answer=target, is_correct=False, abstained=True,
                    model_output_error=True, tokens_used=exc.usage["total_tokens"],
                    latency_ms=(time.perf_counter()-start_time)*1000,
                    details={"usage_kind":"measured", "usage":exc.usage,
                             "error_type":"ModelOutputError", "error":str(exc)})
            final_answer = (final_answer.strip() or None) if final_answer is not None else None
            tokens_used = usage["total_tokens"]
            details.update(usage_kind="measured", usage=usage,
                           trajectory=getattr(pred, "trajectory", None),
                           reasoning_steps=getattr(pred, "reasoning_steps", None))

        duration_ms = (time.perf_counter() - start_time) * 1000
        is_correct = answers_equal(final_answer, target)

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
            abstained=final_answer is None,
            error_propagated=error_propagated,
            details=details,
        )
