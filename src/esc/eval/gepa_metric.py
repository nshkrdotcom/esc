"""GEPA natural-language feedback metric for reflective prompt and boundary optimization."""

from __future__ import annotations

from typing import Any
import dspy


def epistemic_gepa_metric(
    example: Any,
    pred: Any,
    trace: Any | None = None,
    pred_name: str | None = None,
    pred_trace: Any | None = None,
) -> dspy.Prediction:
    """GEPA reflective failure feedback metric.

    Returns numeric score along with rich natural language diagnostic feedback
    for the reflection LM to evolve signatures and instructions.
    """
    target = getattr(example, "target_answer", "") or getattr(example, "answer", "")
    pred_ans = getattr(pred, "final_answer", None) or getattr(pred, "value", None) or str(pred)

    target_str = str(target).strip().lower()
    pred_str = str(pred_ans).strip().lower()

    is_correct = (pred_str == target_str) or (target_str in pred_str)
    score = 1.0 if is_correct else 0.0

    failures: list[str] = []

    # 1. Final correctness check
    if not is_correct:
        failures.append(
            f"Final answer incorrect: expected '{target}', but received '{pred_ans}'. "
            "Review intermediate factual extractions and verify reasoning steps."
        )

    # 2. Boundary crossing & false promotion check
    false_promotions = getattr(pred, "false_promotions", 0)
    if false_promotions > 0:
        failures.append(
            f"{false_promotions} incorrect claims crossed an epistemic boundary into canonical state. "
            "Enforce stricter witness validation rules before accepting candidates."
        )

    # 3. Contract violation check
    contract_violations = getattr(pred, "contract_violations", 0)
    if contract_violations > 0:
        failures.append(
            f"{contract_violations} contract violations occurred: downstream modules consumed "
            "state below their required assurance level. Ensure dependencies are satisfied before invocation."
        )

    # 4. Token budget overflow check
    token_budget = getattr(example, "token_budget", 5000)
    token_count = getattr(pred, "tokens_used", 0)
    if token_count > token_budget:
        failures.append(
            f"Program exceeded matched compute budget ({token_count} tokens > {token_budget} limit)."
        )

    # 5. Hallucination / Unsupported assumptions
    assumptions = getattr(pred, "assumptions", [])
    if assumptions:
        failures.append(
            f"Worker made ungrounded assumptions: {', '.join(assumptions[:3])}."
        )

    feedback_text = (
        "\n".join(failures)
        if failures
        else "Step executed with zero boundary violations, grounded witnesses, and correct final state."
    )

    return dspy.Prediction(
        score=score,
        feedback=feedback_text,
    )
