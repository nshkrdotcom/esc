"""Experiment runner coordinating multi-system evaluation across depths and repetitions."""

from __future__ import annotations

import json
import os
import platform
import time
from importlib.metadata import version
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import dspy

from esc.benchmark.generator import generate_task_suite
from esc.benchmark.relational import generate_relational_suite
from esc.config import RLMConfig
from esc.budget import BudgetedLM, EpisodeLedger, EpisodeBudgetExhausted
from esc.telemetry import RunTelemetry
from esc.eval.decay import HorizonDecayResult, fit_horizon_decay
from esc.eval.metrics import EvalVector, compute_eval_vector
from esc.systems.base import BaseSystem, SystemResult
from esc.systems.system_a import SystemAContinuous
from esc.systems.system_b import SystemBSearchHeavy
from esc.systems.system_c import SystemCIsolated


@dataclass
class ExperimentResult:
    """Consolidated outcome of Experiment 1 across all systems and depths."""

    eval_vectors_by_system_and_depth: dict[str, dict[int, EvalVector]]
    overall_eval_vectors: dict[str, EvalVector]
    horizon_decays: dict[str, HorizonDecayResult]
    h1_hypothesis: dict[str, Any]
    all_runs: list[SystemResult] = field(default_factory=list)
    configuration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "eval_vectors_by_system_and_depth": {
                name: {depth: vector.model_dump() for depth, vector in vectors.items()}
                for name, vectors in self.eval_vectors_by_system_and_depth.items()
            },
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
    rlm_config: RLMConfig | None = None,
    n_samples: int = 3,
    benchmark: str = "legacy",
    split: str = "dev",
    world_width: int = 16,
    episode_token_budget: int | None = None,
) -> ExperimentResult:
    """Execute Experiment 1: Does epistemic isolation change horizon scaling?

    Compares:
      - Condition A: Continuous
      - Condition B: Search-Heavy (Best-of-N)
      - Condition C: Isolated (EpiDSPy)

    This is a clean A/B/C pipeline pilot; H1 is not statistically tested.
    """
    if not depths or len(set(depths)) != len(depths) or any(d not in {2, 4, 8, 16} for d in depths):
        raise ValueError("depths must be distinct task sizes from 2, 4, 8, 16")
    if tasks_per_depth < 1 or repetitions < 1:
        raise ValueError("tasks_per_depth and repetitions must be positive")
    if benchmark not in {"legacy", "relational_v2"}:
        raise ValueError("benchmark must be legacy or relational_v2")
    if split not in {"train", "validation", "test", "dev"} or world_width < 2:
        raise ValueError("Invalid split or world_width (must be >= 2)")
    if benchmark == "legacy" and split != "dev":
        raise ValueError("Legacy benchmark has no split isolation; use relational_v2")
    rlm_config = rlm_config or RLMConfig()
    if episode_token_budget is not None:
        if episode_token_budget < 1 or use_mock or not isinstance(sub_lm, BudgetedLM):
            raise ValueError("Episode budget requires positive allowance and a live BudgetedLM")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if reflection_lm is not None:
        raise NotImplementedError("GEPA compilation is not implemented; pilot runs A/B/C only")
    if not use_mock and (sub_lm is None or sub_lm.cache):
        raise ValueError("Live runs require an explicit task LM with cache=False")
    configuration = {
        "task_sizes": depths, "tasks_per_depth": tasks_per_depth,
        "repetitions": repetitions, "seed": seed, "use_mock": use_mock,
        "model": getattr(sub_lm, "model", None),
        "lm_parameters": {key: getattr(sub_lm, "kwargs", {}).get(key) for key in
                          ("temperature", "max_tokens", "num_ctx", "reasoning_effort", "timeout", "top_p", "top_k")},
        "rlm": rlm_config.model_dump(), "n_samples": n_samples,
        "adapter": "ChatAdapter (automatic JSON retry disabled)",
        "python": platform.python_version(), "dspy": version("dspy"),
        "compute_matched": False, "error_injection": False,
        "episode_token_budget": episode_token_budget,
        "budget_policy": "soft_generation_reservation_v1" if episode_token_budget else None,
        "scope": "pipeline pilot; no confirmatory H1 inference",
        "benchmark": benchmark, "split": split,
        "world_width": world_width if benchmark == "relational_v2" else None,
    }
    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)
        if any((out_path / name).exists() for name in
               ("experiment_1_summary.json", "runs.jsonl", "manifest.json")):
            raise FileExistsError(f"Run artifacts already exist in {out_path}; choose a new output directory")
        with (out_path / "manifest.json").open("x") as handle:
            json.dump(configuration, handle, indent=2, allow_nan=False)

    systems: dict[str, BaseSystem] = {
        "Condition_A_Continuous": SystemAContinuous(use_mock=use_mock, sub_lm=sub_lm, rlm_config=rlm_config),
        "Condition_B_SearchHeavy": SystemBSearchHeavy(use_mock=use_mock, sub_lm=sub_lm, rlm_config=rlm_config, n_samples=n_samples),
        "Condition_C_Isolated": SystemCIsolated(use_mock=use_mock, sub_lm=sub_lm, rlm_config=rlm_config),
    }

    # Generate test tasks across depths
    if benchmark == "relational_v2":
        task_suite = generate_relational_suite(depths=depths, tasks_per_depth=tasks_per_depth,
                                              seed=seed, split=split, width=world_width)
    else:
        task_suite = generate_task_suite(depths=depths, tasks_per_depth=tasks_per_depth,
                                         seed=seed, inject_errors=False)

    # Clean tasks only: live A/B injection hooks are not implemented.
    all_tasks = task_suite
    if out_path:
        with (out_path / "tasks.json").open("x") as handle:
            json.dump([task.model_dump(mode="json") for task in all_tasks], handle, indent=2)
    if use_mock:
        for offset, system in enumerate(systems.values()):
            worker = system.sub_system.worker if isinstance(system, SystemBSearchHeavy) else system.worker
            worker.rng.seed(seed + offset)
    results_by_system_and_task: dict[str, dict[str, list[SystemResult]]] = {
        sys_name: defaultdict(list) for sys_name in systems
    }
    all_runs: list[SystemResult] = []
    telemetry = RunTelemetry(out_path / "events.jsonl") if out_path else None
    callbacks = list(dspy.settings.callbacks or []) + ([telemetry] if telemetry else [])

    for task in all_tasks:
        for rep in range(repetitions):
            for sys_name, system in systems.items():
                episode = {"task_id": task.task_id, "system": sys_name, "repetition": rep}
                ledger = EpisodeLedger(episode_token_budget) if episode_token_budget else None
                started = time.perf_counter()
                if telemetry:
                    telemetry.episode = episode
                    telemetry.record("episode_start")
                if not use_mock:
                    print(f"Starting {len(all_runs)+1}/{len(all_tasks)*repetitions*len(systems)}: "
                          f"{sys_name}, {task.task_id}, repetition {rep+1}", flush=True)
                try:
                    with dspy.context(callbacks=callbacks,
                                      esc_ledger=ledger,
                                      adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
                        run_res = system.run(task)
                except EpisodeBudgetExhausted:
                    run_res = SystemResult(system_name=sys_name, task_id=task.task_id,
                        depth=task.dependency_depth, final_answer=None,
                        target_answer=task.target_answer(), is_correct=False,
                        abstained=True, budget_exhausted=True,
                        latency_ms=(time.perf_counter()-started)*1000)
                except (Exception, KeyboardInterrupt) as exc:
                    if telemetry:
                        telemetry.record("episode_failed", error_type=type(exc).__name__, error=str(exc))
                    if out_path:
                        with (out_path / "failure.json").open("w") as handle:
                            json.dump({"task_id": task.task_id, "system": sys_name,
                                       "repetition": rep, "error_type": type(exc).__name__,
                                       "error": str(exc),
                                       "ledger": ledger.snapshot() if ledger else None}, handle, indent=2)
                    raise
                if ledger:
                    state = ledger.snapshot()
                    run_res.tokens_used = state["measured_tokens"]
                    run_res.budget_exhausted = state["exhausted"]
                    run_res.details["ledger"] = state
                    if telemetry:
                        telemetry.record("episode_budget", **state)
                run_res.depth = task.dependency_depth
                run_res.details.update(repetition=rep, nominal_task_size=task.depth,
                                       use_mock=use_mock, benchmark=benchmark,
                                       world_id=task.metadata.get("world_id"), split=split)
                results_by_system_and_task[sys_name][task.task_id].append(run_res)
                all_runs.append(run_res)
                if telemetry:
                    telemetry.record("episode_end", tokens=run_res.tokens_used, correct=run_res.is_correct)
                if not use_mock:
                    print(f"Completed: {run_res.tokens_used} tokens, "
                          f"{run_res.latency_ms/1000:.1f}s, correct={run_res.is_correct}", flush=True)
                if out_path:
                    with (out_path / "runs.jsonl").open("a") as handle:
                        handle.write(run_res.model_dump_json() + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())

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
        for d in sorted({task.dependency_depth for task in all_tasks}):
            tasks_at_d = {
                t.task_id: results_by_system_and_task[sys_name][t.task_id]
                for t in all_tasks
                if t.dependency_depth == d
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

    h1 = {
        "h1_supported": None,
        "beta_A": horizon_decays["Condition_A_Continuous"].beta,
        "beta_C": horizon_decays["Condition_C_Isolated"].beta,
        "interpretation": "Not tested: pilot compute is not matched; mock outcomes are scripted. Slopes are descriptive only.",
    }

    exp_res = ExperimentResult(
        eval_vectors_by_system_and_depth=eval_vectors_by_system_and_depth,
        overall_eval_vectors=overall_eval_vectors,
        horizon_decays=horizon_decays,
        h1_hypothesis=h1,
        all_runs=all_runs,
        configuration=configuration,
    )

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        temporary_summary = out_path / "experiment_1_summary.json.tmp"
        with temporary_summary.open("w") as f:
            json.dump(exp_res.to_dict(), f, indent=2, allow_nan=False)
        temporary_summary.replace(out_path / "experiment_1_summary.json")

    return exp_res
