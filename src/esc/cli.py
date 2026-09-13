"""Command-line interface for ESC (Epistemic State Compilation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from esc.benchmark.generator import generate_epidag_task, generate_task_suite
from esc.eval.gepa_metric import epistemic_gepa_metric
from esc.runner import run_experiment_1

app = typer.Typer(
    name="esc",
    help="ESC: Epistemic State Compilation - Separating persistent epistemic state from LLM cognition.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    depths: str = typer.Option("2,4,8,16", help="Comma-separated depths to evaluate (e.g. 2,4,8,16)"),
    tasks_per_depth: int = typer.Option(3, help="Number of unique tasks per depth level"),
    repetitions: int = typer.Option(3, help="Number of repetitions per task (for pass@k and pass^k)"),
    mock: bool = typer.Option(True, help="Use deterministic/stochastic mock workers for instant offline evaluation"),
    model: Optional[str] = typer.Option("ollama_chat/qwen3:14b", help="Task LM for RLM and workers"),
    reflection_model: Optional[str] = typer.Option(None, help="Reflection LM for GEPA optimization"),
    output_dir: str = typer.Option("outputs/experiment_1", help="Directory to save experiment results"),
) -> None:
    """Run Experiment 1: Does epistemic isolation change horizon scaling?"""
    depth_list = [int(d.strip()) for d in depths.split(",") if d.strip()]
    console.rule("[bold cyan]ESC Experiment 1: Epistemic State Compilation Horizon Scaling[/bold cyan]")
    rprint(f"[bold]Evaluating depths:[/bold] {depth_list}")
    rprint(f"[bold]Tasks per depth:[/bold] {tasks_per_depth} | [bold]Repetitions (k):[/bold] {repetitions}")
    rprint(f"[bold]Execution mode:[/bold] {'MOCK (offline deterministic / stochastic)' if mock else f'REAL LM ({model})'}")

    sub_lm = None
    ref_lm = None
    if not mock:
        import dspy
        lm_name = model or "ollama_chat/qwen3:14b"
        rprint(f"[yellow]Initializing DSPy Task LM: {lm_name}...[/yellow]")
        api_base = "http://localhost:11434" if "ollama" in lm_name else None
        sub_lm = dspy.LM(lm_name, api_base=api_base)
        dspy.configure(lm=sub_lm)

        if reflection_model:
            rprint(f"[yellow]Initializing DSPy Reflection LM: {reflection_model}...[/yellow]")
            ref_lm = dspy.LM(reflection_model)

    exp_result = run_experiment_1(
        depths=depth_list,
        tasks_per_depth=tasks_per_depth,
        repetitions=repetitions,
        use_mock=mock,
        sub_lm=sub_lm,
        reflection_lm=ref_lm,
        output_dir=output_dir,
    )

    # 1. Display Overall Multi-Dimensional Evaluation Vector Table R = (A, C, E, F, V, T)
    table = Table(title="Overall Evaluation Vector: R = (A, C, E, F, V, T)")
    table.add_column("System Condition", style="bold cyan")
    table.add_column("Accuracy (A)", justify="right")
    table.add_column("Consistency pass^k (C)", justify="right")
    table.add_column("Search pass@k", justify="right")
    table.add_column("Error Prop EPC (E)", justify="right")
    table.add_column("False Prom Rate (F)", justify="right")
    table.add_column("Abstention (V)", justify="right")
    table.add_column("Avg Tokens (T)", justify="right")

    for sys_name, vec in exp_result.overall_eval_vectors.items():
        table.add_row(
            sys_name,
            f"{vec.accuracy:.1%}",
            f"{vec.consistency:.1%}",
            f"{vec.pass_at_k:.1%}",
            f"{vec.error_propagation:.1%}",
            f"{vec.false_promotion_rate:.3f}",
            f"{vec.abstention_rate:.1%}",
            f"{int(vec.avg_tokens):,}",
        )
    console.print(table)

    # 2. Display Horizon Decay β Table
    decay_table = Table(title="Horizon Scaling Decay: log P(success) = α - β * d")
    decay_table.add_column("System Condition", style="bold cyan")
    decay_table.add_column("Intercept (α)", justify="right")
    decay_table.add_column("Decay Rate (β)", justify="right", style="bold magenta")
    decay_table.add_column("R² Fit", justify="right")
    decay_table.add_column("Acc @ d=2", justify="right")
    decay_table.add_column("Acc @ d=16", justify="right")

    for sys_name, decay in exp_result.horizon_decays.items():
        acc_2 = decay.accuracies_by_depth.get(2, 0.0)
        acc_16 = decay.accuracies_by_depth.get(16, 0.0)
        decay_table.add_row(
            sys_name,
            f"{decay.alpha:.3f}",
            f"{decay.beta:.4f}",
            f"{decay.r_squared:.3f}",
            f"{acc_2:.1%}",
            f"{acc_16:.1%}",
        )
    console.print(decay_table)

    # 3. Hypothesis H1 summary
    h1 = exp_result.h1_hypothesis
    console.rule("[bold green]Hypothesis H1 Verification[/bold green]")
    rprint(f"[bold]Hypothesis H1 Supported:[/bold] [{'green' if h1['h1_supported'] else 'red'}]{h1['h1_supported']}[/]")
    rprint(f"[bold]β_A (Continuous):[/bold] {h1['beta_A']:.4f}")
    rprint(f"[bold]β_C (Isolated):[/bold]   {h1['beta_C']:.4f}")
    rprint(f"[bold]Result:[/bold] {h1['interpretation']}")
    rprint(f"\n[green]✓ Results and audit logs saved to: {output_dir}/experiment_1_summary.json[/green]")


@app.command()
def bench(
    depth: int = typer.Option(4, help="Depth of synthetic task DAG to inspect (2, 4, 8, 16)"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Generate and inspect an EpiDAG synthetic benchmark task."""
    task = generate_epidag_task(task_id=f"demo_d{depth}", depth=depth, seed=seed)
    console.rule(f"[bold cyan]EpiDAG Task Inspection: Depth {depth}[/bold cyan]")
    rprint(f"[bold]Question:[/bold] {task.question}")
    rprint(f"[bold]Final Target Answer:[/bold] {task.target_answer()}")
    rprint(f"[bold]Node Count:[/bold] {len(task.nodes)}")

    table = Table(title=f"Hidden DAG Nodes (Depth {depth})")
    table.add_column("Node ID", style="bold yellow")
    table.add_column("Witness Type", style="cyan")
    table.add_column("Dependencies (Requires)", style="green")
    table.add_column("Goal", style="white")
    table.add_column("Ground Truth", style="magenta")

    for n in task.nodes:
        table.add_row(
            n.node_id,
            n.witness_type,
            ", ".join(n.step_spec.requires) or "None",
            n.step_spec.goal,
            n.true_value,
        )
    console.print(table)


@app.command()
def gepa_demo() -> None:
    """Demonstrate the GEPA reflective failure feedback metric."""
    import dspy
    from esc.core.types import StepResult

    console.rule("[bold cyan]DSPy GEPA Epistemic Metric Demonstration[/bold cyan]")

    class MockExample:
        target_answer = "Target_MicroVantage"
        token_budget = 2000

    class MockPrediction:
        final_answer = "CORRUPT_Target_Wrong"
        false_promotions = 2
        contract_violations = 1
        tokens_used = 2800
        assumptions = ["Unverified assumption that Alpha acquired Beta in 2020"]

    res = epistemic_gepa_metric(MockExample(), MockPrediction())
    rprint(f"[bold]Score:[/bold] {res.score}")
    rprint("[bold]Reflective Feedback provided to GEPA Reflection LM:[/bold]")
    console.print(f"[yellow]{res.feedback}[/yellow]")


if __name__ == "__main__":
    app()
