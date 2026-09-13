"""Condition C — Isolated Epistemic Architecture (EpiDSPy).

Fresh worker for every semantic transition; only typed canonical state crosses boundaries.
RLM reasoning trajectories are discarded from ongoing state.
"""

from __future__ import annotations

import time
from typing import Any
import dspy

from esc.benchmark.tasks import EpiDAGTask
from esc.core.kernel import StateKernel
from esc.core.types import AblationMode, Fact, StepResult
from esc.core.witness import WitnessResult, evaluate_witness
from esc.systems.base import BaseSystem, SystemResult
from esc.workers.epistemic import EpistemicWorker
from esc.workers.mock import MockEpistemicWorker


class SystemCIsolated(BaseSystem):
    """Condition C: Isolated Epistemic State Compilation.

    Features:
      - Clean disposable worker per step
      - Kernel project() context compiler (only required, certified facts enter prompt)
      - Non-oracle Type I / II / III witness evaluation
      - Full amnesia regarding previous trajectory scratchpads
    """

    def __init__(
        self,
        worker: EpistemicWorker | MockEpistemicWorker | None = None,
        use_mock: bool = False,
        mock_error_rate: float = 0.0,
        sub_lm: dspy.LM | None = None,
        ablation_mode: AblationMode = "none",
    ) -> None:
        super().__init__(name=f"Condition_C_Isolated{f'_{ablation_mode}' if ablation_mode != 'none' else ''}")
        self.use_mock = use_mock
        self.ablation_mode = ablation_mode
        if worker is not None:
            self.worker = worker
        elif use_mock:
            self.worker = MockEpistemicWorker(error_rate=mock_error_rate)
        else:
            self.worker = EpistemicWorker(sub_lm=sub_lm)

    def run(self, task: EpiDAGTask) -> SystemResult:
        start_time = time.perf_counter()
        kernel = StateKernel()
        corpus = task.get_corpus()
        corpus_map = corpus.as_dict()

        contract_violations = 0
        abstained = False
        step_answers: dict[str, str] = {}
        total_tokens = 0
        rolling_history: list[str] = []

        for node in task.nodes:
            step = node.step_spec

            # 1. Context compiler: project accepted facts required by contract
            if self.ablation_mode == "no_typing":
                # Ablation 1: fresh contexts but no typing / contract enforcement
                local_state = list(kernel.facts.values())
            elif self.ablation_mode == "raw_summaries":
                # Ablation 6: raw summaries instead of typed state
                local_state = []
            else:
                local_state = kernel.project(step)

            # Contract check
            if self.ablation_mode not in ["no_typing", "raw_summaries"]:
                missing_requires = [req for req in step.requires if req not in [f.key for f in local_state]]
                if missing_requires:
                    contract_violations += 1
                    abstained = True
                    kernel.record_audit(
                        step=step,
                        input_facts=local_state,
                        result=StepResult(
                            status="insufficient",
                            value=None,
                            evidence=[],
                            assumptions=[f"Contract violation: missing valid facts {missing_requires}"],
                        ),
                        committed=False,
                    )
                    break

            # 2. Context compiler: project permitted document snippets
            evidence_context = corpus.project(step.permitted_sources)

            if self.ablation_mode == "shared_history" and rolling_history:
                # Ablation 2: contracts present, but prior reasoning history shared into context
                evidence_context = f"{evidence_context}\n\n--- Prior Trajectory History ---\n" + "\n".join(rolling_history)

            if self.ablation_mode == "raw_summaries" and kernel.facts:
                summary_text = "Previously deduced facts: " + "; ".join(f"{f.key}={f.value}" for f in kernel.facts.values())
                evidence_context = f"{evidence_context}\n\n{summary_text}"

            # 3. Disposable worker invocation
            step_tokens = 250  # estimate / baseline token charge
            if isinstance(self.worker, MockEpistemicWorker):
                # If error injection was requested at this node
                if node.injected_error_value is not None:
                    prediction = StepResult(
                        status="supported",
                        value=node.injected_error_value,
                        evidence=[],
                        assumptions=["Injected corrupted candidate"],
                    )
                else:
                    prediction = self.worker.forward(
                        step=step,
                        accepted_facts=local_state,
                        evidence_context=evidence_context,
                        ground_truth=task.ground_truth_map,
                        true_evidence=node.true_evidence,
                    )
            else:
                try:
                    prediction = self.worker.forward(
                        goal=step.goal,
                        accepted_facts=local_state,
                        evidence_context=evidence_context,
                    )
                    step_tokens = 450
                except Exception as ex:
                    prediction = StepResult(
                        status="insufficient",
                        value=None,
                        evidence=[],
                        assumptions=[f"Worker error: {ex}"],
                    )

            total_tokens += step_tokens

            if prediction.status == "insufficient":
                abstained = True
                kernel.record_audit(
                    step=step,
                    input_facts=local_state,
                    result=prediction,
                    committed=False,
                )
                break

            # 4. Non-oracle witness evaluation
            if self.ablation_mode == "no_verification":
                # Ablation 3: isolation + typing, but no verification check
                witness = WitnessResult(
                    passed=True,
                    promoted_level="verified",
                    witness_type="none",
                    detail="Ablation: direct promotion without witness verification",
                )
            else:
                witness = evaluate_witness(
                    step=step,
                    result=prediction,
                    input_facts=local_state,
                    corpus_map=corpus_map,
                )

            # Record trajectory for shared_history ablation
            rolling_history.append(f"Step {step.step_id} ({step.goal}) -> Result: {prediction.value} (status: {prediction.status})")

            # 5. Kernel commit: candidate promoted only if witness passed
            fact_key = step.expected_key or step.step_id
            candidate_fact = Fact(
                key=fact_key,
                value=prediction.value or "",
                level=witness.promoted_level,
                evidence=prediction.evidence,
                parents=step.requires,
            )

            committed = kernel.commit(
                fact=candidate_fact,
                witness=witness,
                required_level=step.required_level,
            )

            kernel.record_audit(
                step=step,
                input_facts=local_state,
                result=prediction,
                committed=committed,
                witness=witness,
                tokens_used=step_tokens,
            )

            if committed:
                step_answers[fact_key] = candidate_fact.value
            else:
                # Promotion failed - candidate remains unpromoted
                abstained = True
                break

        duration_ms = (time.perf_counter() - start_time) * 1000
        final_fact = kernel.get_fact(task.final_node_id)
        final_answer = final_fact.value if final_fact else None
        target = task.target_answer()
        is_correct = (
            final_answer is not None
            and final_answer.strip().lower() == target.strip().lower()
        )

        false_promotions = kernel.false_promotion_count(task.ground_truth_map)

        # Error propagation check: if an upstream error was injected, did it reach downstream?
        error_propagated = False
        if task.injected_error_node_id:
            # Check if any fact committed AFTER the injected node is corrupt
            for k, f in kernel.facts.items():
                if k != task.injected_error_node_id and "CORRUPT" in f.value:
                    error_propagated = True

        return SystemResult(
            system_name=self.name,
            task_id=task.task_id,
            depth=task.depth,
            final_answer=final_answer,
            target_answer=target,
            is_correct=is_correct,
            intermediate_answers=step_answers,
            tokens_used=total_tokens,
            latency_ms=duration_ms,
            false_promotions=false_promotions,
            contract_violations=contract_violations,
            abstained=abstained,
            error_propagated=error_propagated,
            details={"audit_log_length": len(kernel.audit_log)},
        )
