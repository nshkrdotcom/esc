"""DSPy Signatures for atomic epistemic steps and continuous tasks."""

from __future__ import annotations

import dspy
from esc.core.types import Fact, StepResult


class ResolveStep(dspy.Signature):
    """Resolve exactly one epistemic transition.

    Use only supplied accepted facts and permitted evidence.
    Do not assume unsupported propositions.
    Return insufficient rather than inventing missing state.
    Return only the requested scalar or entity name in value (no units or prose).
    Evidence context is a JSON list of documents with source_id, title, and content.
    Parse the list and search each document's content. Cite the source_id of the
    document containing the matching span, never the first document by default.
    Copy the span from that document's decoded content, not the JSON-escaped wrapper.
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
    Corpus context is a JSON list of documents with source_id, title, and content.
    Return null for final_answer if you cannot determine the answer. Do not use
    the strings "None" or "null" as refusal markers.
    """

    task_description: str = dspy.InputField(desc="Complete multi-step task and final goal.")
    corpus_context: str = dspy.InputField(desc="All available corpus documents and reference materials.")

    reasoning_steps: list[str] = dspy.OutputField(
        desc="Step-by-step reasoning trajectory and intermediate deductions.",
    )
    final_answer: str | None = dspy.OutputField(desc="Final consolidated answer, or null to abstain.")
