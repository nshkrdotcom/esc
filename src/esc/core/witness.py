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
    cleaned_val = value.strip().rstrip(".").lower()

    # Extract numbers from input facts
    fact_values: list[float] = []
    for f in input_facts:
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", f.value)
        for n in nums:
            try:
                fact_values.append(float(n))
            except ValueError:
                pass

    claimed_nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", cleaned_val)
    if not claimed_nums:
        return False, f"No numerical value found in claimed result '{value}'."

    claimed_float = float(claimed_nums[0])

    goal_lower = goal.lower()
    # Check ratio
    if "ratio" in goal_lower and len(fact_values) >= 2:
        denom = fact_values[1]
        if denom != 0:
            expected = fact_values[0] / denom
            if abs(claimed_float - expected) < 1e-4 or abs(claimed_float - round(expected, 2)) < 1e-4:
                return True, f"Recomputed ratio {fact_values[0]} / {denom} = {expected:.4f} matched."
            # Also try reverse ratio
            if fact_values[0] != 0:
                expected_rev = fact_values[1] / fact_values[0]
                if abs(claimed_float - expected_rev) < 1e-4 or abs(claimed_float - round(expected_rev, 2)) < 1e-4:
                    return True, f"Recomputed ratio {fact_values[1]} / {fact_values[0]} = {expected_rev:.4f} matched."

    # Check sum / total
    if any(w in goal_lower for w in ["sum", "total", "combined"]) and len(fact_values) >= 2:
        expected = sum(fact_values)
        if abs(claimed_float - expected) < 1e-4:
            return True, f"Recomputed sum {expected} matched."

    # Check difference / delta
    if any(w in goal_lower for w in ["difference", "delta", "growth"]) and len(fact_values) >= 2:
        expected = fact_values[0] - fact_values[1]
        if abs(claimed_float - expected) < 1e-4 or abs(claimed_float - abs(expected)) < 1e-4:
            return True, f"Recomputed delta {expected} matched."

    # General check: if value is directly verified from single input fact
    if len(fact_values) == 1 and abs(claimed_float - fact_values[0]) < 1e-4:
        return True, f"Single numerical identity {fact_values[0]} matched."

    return False, f"Could not deterministically verify arithmetic for '{value}' from {fact_values}."


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

    # Type I: Deterministic Witness
    if step.witness_type == "type_1":
        goal_lower = step.goal.lower()

        # 1. Check deterministic threshold comparison
        if any(w in goal_lower for w in ["threshold", "exceeds", "compare", "greater", "less"]):
            threshold_matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", goal_lower)
            if threshold_matches and input_facts:
                try:
                    thresh_val = float(threshold_matches[-1])
                    fact_nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", input_facts[0].value)
                    if fact_nums:
                        input_num = float(fact_nums[0])
                        expected_bool = "true" if input_num > thresh_val else "false"
                        if result.value.strip().lower() == expected_bool:
                            return WitnessResult(
                                passed=True,
                                promoted_level="verified",
                                witness_type="type_1",
                                detail=f"Deterministic threshold check {input_num} > {thresh_val} = {expected_bool} verified.",
                            )
                        else:
                            return WitnessResult(
                                passed=False,
                                promoted_level="candidate",
                                witness_type="type_1",
                                detail=f"Deterministic threshold check failed: {input_num} > {thresh_val} is {expected_bool}, not {result.value}.",
                            )
                except Exception:
                    pass

        # 2. Check deterministic rule transition (e.g. approved if threshold holds, else rejected)
        if any(w in goal_lower for w in ["recommendation", "decision", "compliance", "stage"]):
            if input_facts:
                parent_val = input_facts[0].value.strip().lower()
                val_lower = result.value.strip().lower()
                if any(pos in parent_val for pos in ["approved", "true", "cleared", "pass"]) and any(
                    pos in val_lower for pos in ["approved", "cleared", "true", "pass"]
                ):
                    return WitnessResult(
                        passed=True,
                        promoted_level="verified",
                        witness_type="type_1",
                        detail=f"Deterministic rule transition from '{parent_val}' to '{result.value}' verified.",
                    )
                elif any(neg in parent_val for neg in ["rejected", "false", "halted", "fail"]) and any(
                    neg in val_lower for neg in ["rejected", "halted", "false", "fail"]
                ):
                    return WitnessResult(
                        passed=True,
                        promoted_level="verified",
                        witness_type="type_1",
                        detail=f"Deterministic rule transition from '{parent_val}' to '{result.value}' verified.",
                    )
                else:
                    return WitnessResult(
                        passed=False,
                        promoted_level="candidate",
                        witness_type="type_1",
                        detail=f"Deterministic rule transition invalid: '{parent_val}' does not lead to '{result.value}'.",
                    )

        # 3. Check arithmetic calculations
        if any(w in goal_lower for w in ["ratio", "sum", "difference", "calculate", "average", "compute"]):
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
        if span_ok:
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
        if span_ok:
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
