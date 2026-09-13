"""DSPy Signatures for atomic epistemic steps and continuous tasks."""

from __future__ import annotations

import dspy
from esc.core.types import Fact, StepResult


class ResolveStep(dspy.Signature):
    """Resolve exactly one epistemic transition.

    Use only supplied accepted facts and permitted evidence.
    Do not assume unsupported propositions.
    Return insufficient rather than inventing missing state.
    """

    goal: str = dspy.InputField(desc="Specific sub-goal to resolve for this step.")
    accepted_facts: list[Fact] = dspy.InputField(
        desc="Canonical facts verified from previous steps.",
    )
    evidence_context: str = dspy.InputField(
        desc="Text snippets from permitted sources for this step.",
    )

    result: StepResult = dspy.OutputField(
        desc="Structured outcome including status (supported, contradicted, insufficient, ambiguous), value, and evidence.",
    )


class ContinuousSolve(dspy.Signature):
    """Resolve an entire multi-step dependency problem in a single continuous reasoning trajectory.

    All intermediate reasoning, calculations, and deductions occur in this single context.
    """

    task_description: str = dspy.InputField(desc="Complete multi-step task and final goal.")
    corpus_context: str = dspy.InputField(desc="All available corpus documents and reference materials.")

    reasoning_steps: list[str] = dspy.OutputField(
        desc="Step-by-step reasoning trajectory and intermediate deductions.",
    )
    final_answer: str = dspy.OutputField(desc="Final consolidated answer.")
