"""Epistemic witness system: Non-oracle promotion gates for state compilation.

Three witness classes:
  Type I   - Deterministic witness (arithmetic, code execution, exact span verification, date/numeric comparison)
             -> Promotes to `verified`
  Type II  - Externally grounded witness (multi-source provenance, citation verification against corpus)
             -> Promotes to `supported`
  Type III - Purely semantic inference (no deterministic witness available)
             -> Remains `candidate` (or requires independent adjudication)
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field

from esc.core.types import EpistemicLevel, Evidence, Fact, StepResult, StepSpec
from esc.core.lookup import verify_lookup


class WitnessResult(BaseModel):
    """Evaluation outcome from an epistemic witness check."""

    passed: bool
    promoted_level: EpistemicLevel
    witness_type: str
    detail: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def verify_text_span(evidence: list[Evidence], corpus_map: dict[str, str]) -> tuple[bool, str]:
    """Verify that cited evidence spans actually exist in the allowed source documents."""
    if not evidence:
        return False, "No evidence spans cited."

    for ev in evidence:
        doc = corpus_map.get(ev.source_id)
        if doc is None:
            return False, f"Source ID '{ev.source_id}' not found in accessible corpus."
        cleaned_span = ev.span.strip()
        if not cleaned_span:
            return False, f"Empty span for source '{ev.source_id}'."
        if cleaned_span not in doc:
            # Try normalized whitespace check
            norm_span = " ".join(cleaned_span.split())
            norm_doc = " ".join(doc.split())
            if norm_span not in norm_doc:
                return False, f"Span citation not grounded in document '{ev.source_id}'."

    return True, f"All {len(evidence)} evidence spans verified against corpus."


def verify_deterministic_arithmetic(
    value: str,
    goal: str,
    input_facts: list[Fact],
) -> tuple[bool, str]:
    """Type I deterministic witness: recompute arithmetic calculations from accepted facts.

    Supports operations like:
      - ratio (A / B)
      - difference (A - B)
      - sum (A + B)
      - product (A * B)
      - threshold comparison
    """
    import math

    try:
        claimed = float(value.strip())
        values = [float(f.value.strip()) for f in input_facts]
    except ValueError:
        return False, "Arithmetic requires scalar numeric values, not prose."
    if not math.isfinite(claimed) or not all(math.isfinite(v) for v in values):
        return False, "Non-finite numeric value."
    goal_lower = goal.lower()
    expected = None
    if "ratio" in goal_lower and len(values) == 2 and values[1] != 0:
        expected = values[0] / values[1]
    elif any(w in goal_lower for w in ["sum", "total", "combined"]) and len(values) >= 2:
        expected = sum(values)
    elif any(w in goal_lower for w in ["difference", "delta"]) and len(values) == 2:
        expected = values[0] - values[1]
    elif "product" in goal_lower and len(values) >= 2:
        expected = math.prod(values)
    elif "average" in goal_lower and values:
        expected = sum(values) / len(values)
    if expected is not None and math.isclose(claimed, expected, rel_tol=0, abs_tol=0.00005):
        return True, f"Recomputed ordered arithmetic result {expected}."
    return False, f"Claim {value!r} does not match the specified operation on {values}."


def evaluate_witness(
    step: StepSpec,
    result: StepResult,
    input_facts: list[Fact],
    corpus_map: dict[str, str],
) -> WitnessResult:
    """Evaluate candidate output through non-oracle promotion rules."""
    if result.status != "supported" or not result.value:
        return WitnessResult(
            passed=False,
            promoted_level="candidate",
            witness_type=step.witness_type,
            detail=f"Result status is '{result.status}' (requires 'supported' and non-empty value).",
        )

    if result.assumptions:
        return WitnessResult(passed=False, promoted_level="candidate", witness_type=step.witness_type,
                             detail="Unresolved assumptions cannot cross a promotion boundary.")
    corpus_map = {key: value for key, value in corpus_map.items() if key in step.permitted_sources}

    if step.lookup is not None:
        passed, detail = verify_lookup(step, result, input_facts, corpus_map)
        return WitnessResult(passed=passed, promoted_level="verified" if passed else "candidate",
                             witness_type="type_1", detail=detail)

    # Type I: Deterministic Witness
    if step.witness_type == "type_1":
        goal_lower = step.goal.lower()

        # Only the explicitly supported greater-than relation is verified here.
        if any(w in goal_lower for w in ["threshold", "exceeds", "compare", "greater", "less"]) and "recommendation" not in goal_lower:
            import math
            threshold = re.search(r"exceeds (?:threshold )?([-+]?(?:\d*\.\d+|\d+))", goal_lower)
            passed = False
            expected = None
            if threshold and len(input_facts) == 1:
                try:
                    number = float(input_facts[0].value.strip())
                    limit = float(threshold.group(1))
                    if math.isfinite(number) and math.isfinite(limit):
                        expected = "true" if number > limit else "false"
                        passed = result.value.strip().lower() == expected
                except ValueError:
                    pass
            return WitnessResult(passed=passed, promoted_level="verified" if passed else "candidate",
                                 witness_type="type_1", detail=f"Strict threshold comparison expected {expected!r}.")

        # Exact, public transition rules used by this benchmark.
        if any(w in goal_lower for w in ["recommendation", "compliance check stage"]):
            expected = None
            if len(input_facts) == 1:
                parent = input_facts[0].value.strip().lower()
                if "recommendation" in goal_lower and parent in {"true", "false"}:
                    expected = "approved" if parent == "true" else "rejected"
                stage = re.search(r"compliance check stage (\d+)", goal_lower)
                if stage:
                    n = int(stage.group(1))
                    positive = "approved" if n == 9 else f"stage_{n-1}_cleared"
                    negative = "rejected" if n == 9 else f"stage_{n-1}_halted"
                    if parent in {positive, negative}:
                        expected = f"stage_{n}_cleared" if parent == positive else f"stage_{n}_halted"
            passed = expected is not None and result.value.strip().lower() == expected
            return WitnessResult(passed=passed, promoted_level="verified" if passed else "candidate",
                                 witness_type="type_1", detail=f"Exact transition expected {expected!r}.")

        # 3. Check arithmetic calculations
        if any(w in goal_lower for w in ["ratio", "sum", "difference", "calculate", "average", "compute", "product"]):
            ok, msg = verify_deterministic_arithmetic(result.value, step.goal, input_facts)
            if ok:
                return WitnessResult(
                    passed=True,
                    promoted_level="verified",
                    witness_type="type_1",
                    detail=f"Deterministic arithmetic witness verified: {msg}",
                )
            return WitnessResult(
                passed=False,
                promoted_level="candidate",
                witness_type="type_1",
                detail=f"Deterministic arithmetic verification failed: {msg}",
            )

        # Check exact text span verification
        span_ok, span_msg = verify_text_span(result.evidence, corpus_map)
        if span_ok and result.value.strip().lower() in [ev.span.strip().lower() for ev in result.evidence]:
            return WitnessResult(
                passed=True,
                promoted_level="verified",
                witness_type="type_1",
                detail=f"Exact span matching verified: {span_msg}",
            )

        # If span exists but not exact match
        if span_ok and claim_in_evidence(result):
            return WitnessResult(
                passed=True,
                promoted_level="supported",
                witness_type="type_1",
                detail=f"Grounded span verified (promoted to supported): {span_msg}",
            )

        return WitnessResult(
            passed=False,
            promoted_level="candidate",
            witness_type="type_1",
            detail=f"Type I verification failed: {span_msg}",
        )

    # Type II: Grounded Witness (multi-source or verified document span)
    if step.witness_type == "type_2":
        span_ok, span_msg = verify_text_span(result.evidence, corpus_map)
        if span_ok and claim_in_evidence(result):
            return WitnessResult(
                passed=True,
                promoted_level="supported",
                witness_type="type_2",
                detail=f"Type II grounded witness satisfied: {span_msg}",
            )
        return WitnessResult(
            passed=False,
            promoted_level="candidate",
            witness_type="type_2",
            detail=f"Type II grounding failed: {span_msg}",
        )

    # Type III: Purely semantic inference
    # Remains candidate unless explicit adjudication
    return WitnessResult(
        passed=True,
        promoted_level="candidate",
        witness_type="type_3",
        detail="Type III semantic inference accepted as candidate state (no deterministic witness).",
    )


def claim_in_evidence(result: StepResult) -> bool:
    """Conservative extractive grounding; this does not establish semantic entailment."""
    value = " ".join((result.value or "").casefold().split())
    pattern = r"(?<!\w)" + re.escape(value) + r"(?!\w)"
    return bool(value) and any(re.search(pattern, " ".join(ev.span.casefold().split())) for ev in result.evidence)
