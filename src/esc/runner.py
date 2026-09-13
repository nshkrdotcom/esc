"""Experiment runner coordinating multi-system evaluation across depths and repetitions."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import dspy

from esc.benchmark.generator import generate_epidag_task, generate_task_suite
from esc.benchmark.tasks import EpiDAGTask
from esc.eval.decay import HorizonDecayResult, check_hypothesis_h1, fit_horizon_decay
from esc.eval.metrics import EvalVector, compute_eval_vector
from esc.systems.base import BaseSystem, SystemResult
from esc.systems.system_a import SystemAContinuous
from esc.systems.system_b import SystemBSearchHeavy
from esc.systems.system_c import SystemCIsolated
from esc.systems.system_d import SystemDGEPA


@dataclass
class ExperimentResult:
    """Consolidated outcome of Experiment 1 across all systems and depths."""

    eval_vectors_by_system_and_depth: dict[str, dict[int, EvalVector]]
    overall_eval_vectors: dict[str, EvalVector]
    horizon_decays: dict[str, HorizonDecayResult]
    h1_hypothesis: dict[str, Any]
    all_runs: list[SystemResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_eval_vectors": {
                sys: v.model_dump() for sys, v in self.overall_eval_vectors.items()
            },
            "horizon_decays": {
                sys: dec.model_dump() for sys, dec in self.horizon_decays.items()
            },
            "h1_hypothesis": self.h1_hypothesis,
        }


def run_experiment_1(
    depths: list[int] = [2, 4, 8, 16],
    tasks_per_depth: int = 3,
    repetitions: int = 3,
    use_mock: bool = True,
    sub_lm: dspy.LM | None = None,
    reflection_lm: dspy.LM | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> ExperimentResult:
    """Execute Experiment 1: Does epistemic isolation change horizon scaling?

    Compares:
      - Condition A: Continuous
      - Condition B: Search-Heavy (Best-of-N)
      - Condition C: Isolated (EpiDSPy)
      - Condition D: Isolated + GEPA
    """
    systems: dict[str, BaseSystem] = {
        "Condition_A_Continuous": SystemAContinuous(use_mock=use_mock, sub_lm=sub_lm),
        "Condition_B_SearchHeavy": SystemBSearchHeavy(use_mock=use_mock, sub_lm=sub_lm),
        "Condition_C_Isolated": SystemCIsolated(use_mock=use_mock, sub_lm=sub_lm),
        "Condition_D_Isolated_GEPA": SystemDGEPA(use_mock=use_mock, sub_lm=sub_lm, reflection_lm=reflection_lm),
    }

    # Generate test tasks across depths
    task_suite = generate_task_suite(
        depths=depths,
        tasks_per_depth=tasks_per_depth,
        seed=seed,
        inject_errors=False,
    )

    # Generate paired error-injection tasks to compute EPC_k
    error_task_suite = generate_task_suite(
        depths=depths,
        tasks_per_depth=tasks_per_depth,
        seed=seed + 500,
        inject_errors=True,
    )

    all_tasks = task_suite + error_task_suite
    results_by_system_and_task: dict[str, dict[str, list[SystemResult]]] = {
        sys_name: defaultdict(list) for sys_name in systems
    }
    all_runs: list[SystemResult] = []

    for task in all_tasks:
        for rep in range(repetitions):
            for sys_name, system in systems.items():
                run_res = system.run(task)
                results_by_system_and_task[sys_name][task.task_id].append(run_res)
                all_runs.append(run_res)

    # Compute evaluation vectors per system and per depth
    eval_vectors_by_system_and_depth: dict[str, dict[int, EvalVector]] = defaultdict(dict)
    overall_eval_vectors: dict[str, EvalVector] = {}
    accuracies_by_system_and_depth: dict[str, dict[int, float]] = defaultdict(dict)

    for sys_name in systems:
        # Overall vector
        overall_vec = compute_eval_vector(
            system_name=sys_name,
            results_by_task=results_by_system_and_task[sys_name],
            depth="overall",
            k=repetitions,
        )
        overall_eval_vectors[sys_name] = overall_vec

        # Per-depth vectors
        for d in depths:
            tasks_at_d = {
                t.task_id: results_by_system_and_task[sys_name][t.task_id]
                for t in all_tasks
                if t.depth == d
            }
            depth_vec = compute_eval_vector(
                system_name=sys_name,
                results_by_task=tasks_at_d,
                depth=d,
                k=repetitions,
            )
            eval_vectors_by_system_and_depth[sys_name][d] = depth_vec
            accuracies_by_system_and_depth[sys_name][d] = depth_vec.accuracy

    # Fit horizon decay for each system
    horizon_decays: dict[str, HorizonDecayResult] = {}
    for sys_name in systems:
        decay = fit_horizon_decay(
            accuracies_by_depth=accuracies_by_system_and_depth[sys_name],
            system_name=sys_name,
        )
        horizon_decays[sys_name] = decay

    # Test hypothesis H1 (Condition C vs Condition A)
    h1 = check_hypothesis_h1(
        decay_c=horizon_decays["Condition_C_Isolated"],
        decay_a=horizon_decays["Condition_A_Continuous"],
    )

    exp_res = ExperimentResult(
        eval_vectors_by_system_and_depth=eval_vectors_by_system_and_depth,
        overall_eval_vectors=overall_eval_vectors,
        horizon_decays=horizon_decays,
        h1_hypothesis=h1,
        all_runs=all_runs,
    )

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "experiment_1_summary.json", "w") as f:
            json.dump(exp_res.to_dict(), f, indent=2)

    return exp_res
