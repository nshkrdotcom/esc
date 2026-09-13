"""Mock and simulation workers for offline validation, tests, and error propagation studies."""

from __future__ import annotations

import random
from typing import Any

from esc.core.types import Evidence, Fact, StepResult, StepSpec


class MockEpistemicWorker:
    """Deterministic or stochastic mock worker for offline testing of epistemic isolation."""

    def __init__(
        self,
        error_rate: float = 0.0,
        hallucinate_on_corrupt_input: bool = False,
        seed: int = 42,
    ) -> None:
        self.error_rate = error_rate
        self.hallucinate_on_corrupt_input = hallucinate_on_corrupt_input
        self.rng = random.Random(seed)

    def forward(
        self,
        step: StepSpec,
        accepted_facts: list[Fact],
        evidence_context: str,
        ground_truth: dict[str, str],
        true_evidence: list[Evidence] | None = None,
    ) -> StepResult:
        # Check if required facts are missing
        if step.requires:
            present_keys = {f.key for f in accepted_facts}
            missing = [req for req in step.requires if req not in present_keys]
            if missing:
                return StepResult(
                    status="insufficient",
                    value=None,
                    evidence=[],
                    assumptions=[f"Missing required facts: {missing}"],
                )

        # Check for corrupted upstream facts
        corrupted_upstream = any("CORRUPT" in f.value for f in accepted_facts)
        if corrupted_upstream:
            if self.hallucinate_on_corrupt_input:
                # Rationalize the error into a new corrupted conclusion
                return StepResult(
                    status="supported",
                    value=f"CORRUPT_DERIVED_FROM_{[f.value for f in accepted_facts]}",
                    evidence=[],
                    assumptions=["Rationalized corrupted input"],
                )
            else:
                # Refuse or return insufficient
                return StepResult(
                    status="insufficient",
                    value=None,
                    evidence=[],
                    assumptions=["Cannot proceed with corrupt or conflicting facts"],
                )

        # Check if stochastic error strikes this step
        if self.error_rate > 0.0 and self.rng.random() < self.error_rate:
            return StepResult(
                status="supported",
                value=f"CORRUPT_STOCHASTIC_ERROR_{step.step_id}",
                evidence=[],
                assumptions=["Hallucinated stochastic error"],
            )

        # Normal true answer lookup from ground truth
        expected_key = step.expected_key or step.step_id
        true_val = ground_truth.get(expected_key, "mock_success_value")

        # Provide evidence citations if available
        ev = true_evidence if true_evidence is not None else []
        return StepResult(
            status="supported",
            value=true_val,
            evidence=ev,
            assumptions=[],
        )


class MockContinuousWorker:
    """Mock worker simulating continuous context behavior.

    In continuous reasoning, when an error occurs upstream, it gets narrated into
    the shared reasoning trajectory, rationalized, and contaminates downstream nodes.
    """

    def __init__(
        self,
        error_rate: float = 0.05,
        contamination_decay: float = 0.85,  # Probability that an upstream error ruins the whole trajectory
        seed: int = 42,
    ) -> None:
        self.error_rate = error_rate
        self.contamination_decay = contamination_decay
        self.rng = random.Random(seed)

    def solve_dag(
        self,
        task_prompt: str,
        ground_truth: dict[str, str],
        node_ids: list[str],
        injected_error_node: str | None = None,
    ) -> dict[str, Any]:
        """Simulate solving a DAG in a continuous trajectory."""
        trajectory: list[str] = []
        answers: dict[str, str] = {}
        corrupted = False

        for node_id in node_ids:
            true_val = ground_truth.get(node_id, "ok")

            if node_id == injected_error_node or (self.error_rate > 0 and self.rng.random() < self.error_rate):
                corrupted = True
                curr_val = f"CORRUPT_{true_val}"
            elif corrupted:
                # Contamination cascades down continuous context!
                if self.rng.random() < self.contamination_decay:
                    curr_val = f"CORRUPT_CASCADED_{true_val}"
                else:
                    curr_val = true_val
            else:
                curr_val = true_val

            answers[node_id] = curr_val
            trajectory.append(f"Step {node_id}: deduced {curr_val}")

        final_key = node_ids[-1]
        return {
            "final_answer": answers.get(final_key),
            "step_answers": answers,
            "trajectory": trajectory,
            "corrupted": corrupted,
        }
