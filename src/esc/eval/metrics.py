"""Evaluation metrics for ESC experiments.

Evaluates the multi-dimensional vector R = (A, C, E, F, V, T):
  A = final accuracy
  C = repeated-run consistency (pass^k)
  E = error propagation coefficient (EPC_k)
  F = false-promotion rate
  V = abstention / coverage behavior
  T = token / computation cost
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from pydantic import BaseModel, Field

from esc.systems.base import SystemResult


class EvalVector(BaseModel):
    """The multi-dimensional evaluation vector R = (A, C, E, F, V, T)."""

    system_name: str
    depth: int | str
    accuracy: float = Field(description="A: Final task accuracy")
    consistency: float = Field(description="C: Repeated-run consistency (pass^k)")
    pass_at_k: float = Field(description="Searchable capability (pass@k)")
    error_propagation: float = Field(description="E: EPC_k P(downstream wrong | upstream wrong)")
    false_promotion_rate: float = Field(description="F: Rate of invalid claims crossing boundary")
    abstention_rate: float = Field(description="V: Rate of safe abstention / Insufficient return")
    avg_tokens: float = Field(description="T: Average total token expenditure per task")
    avg_latency_ms: float = Field(description="Average latency in ms")
    total_runs: int = 0
    raw_metrics: dict[str, Any] = Field(default_factory=dict)


def compute_pass_at_k(runs_per_task: list[list[bool]], k: int) -> float:
    """Calculate pass@k (unbiased estimator).

    pass@k = 1 - E[ C(n - c, k) / C(n, k) ]
    where n is total trials and c is correct trials for a task.
    """
    scores: list[float] = []
    for trials in runs_per_task:
        n = len(trials)
        c = sum(trials)
        if n < k:
            continue
        if n - c < k:
            scores.append(1.0)
        else:
            # comb(n-c, k) / comb(n, k)
            comb_fail = math.comb(n - c, k)
            comb_total = math.comb(n, k)
            scores.append(1.0 - (comb_fail / comb_total))
    return float(np.mean(scores)) if scores else 0.0


def compute_pass_pow_k(runs_per_task: list[list[bool]], k: int) -> float:
    """Calculate pass^k (consistency across all k repeated runs).

    pass^k = Fraction of tasks where ALL k trials were successful.
    Captures composability and deterministic repeatability.
    """
    scores: list[float] = []
    for trials in runs_per_task:
        if len(trials) < k:
            continue
        # Take first k runs
        subset = trials[:k]
        scores.append(1.0 if all(subset) else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_eval_vector(
    system_name: str,
    results_by_task: dict[str, list[SystemResult]],
    depth: int | str = "all",
    k: int = 3,
) -> EvalVector:
    """Compute the full evaluation vector R = (A, C, E, F, V, T) for a set of results."""
    all_runs: list[SystemResult] = [
        res for run_list in results_by_task.values() for res in run_list
    ]

    if not all_runs:
        return EvalVector(
            system_name=system_name,
            depth=depth,
            accuracy=0.0,
            consistency=0.0,
            pass_at_k=0.0,
            error_propagation=0.0,
            false_promotion_rate=0.0,
            abstention_rate=0.0,
            avg_tokens=0.0,
            avg_latency_ms=0.0,
            total_runs=0,
        )

    # 1. Final Accuracy A
    accuracy = float(np.mean([1.0 if r.is_correct else 0.0 for r in all_runs]))

    # Group binary success per task for pass@k and pass^k
    runs_per_task = [
        [r.is_correct for r in task_results]
        for task_results in results_by_task.values()
    ]
    min_runs = min(len(t) for t in runs_per_task) if runs_per_task else 0
    eff_k = min(k, min_runs) if min_runs > 0 else 1

    pass_k = compute_pass_at_k(runs_per_task, k=eff_k)
    pass_pow = compute_pass_pow_k(runs_per_task, k=eff_k)

    # 2. Error Propagation EPC_k (E)
    # Measured over runs where an upstream error was deliberately injected
    error_runs = [r for r in all_runs if r.details.get("injected_error") or r.error_propagated or "CORRUPT" in str(r.final_answer)]
    if error_runs:
        error_propagation = float(np.mean([1.0 if r.error_propagated else 0.0 for r in error_runs]))
    else:
        error_propagation = float(np.mean([1.0 if not r.is_correct and not r.abstained else 0.0 for r in all_runs]))

    # 3. False Promotion Rate (F)
    total_steps = sum(len(r.intermediate_answers) or 1 for r in all_runs)
    total_false_promotions = sum(r.false_promotions for r in all_runs)
    false_promotion_rate = float(total_false_promotions / max(1, total_steps))

    # 4. Abstention / Coverage Behavior (V)
    abstention_rate = float(np.mean([1.0 if r.abstained else 0.0 for r in all_runs]))

    # 5. Token & Cost (T)
    avg_tokens = float(np.mean([r.tokens_used for r in all_runs]))
    avg_latency = float(np.mean([r.latency_ms for r in all_runs]))

    return EvalVector(
        system_name=system_name,
        depth=depth,
        accuracy=accuracy,
        consistency=pass_pow,
        pass_at_k=pass_k,
        error_propagation=error_propagation,
        false_promotion_rate=false_promotion_rate,
        abstention_rate=abstention_rate,
        avg_tokens=avg_tokens,
        avg_latency_ms=avg_latency,
        total_runs=len(all_runs),
        raw_metrics={"eff_k": eff_k},
    )
