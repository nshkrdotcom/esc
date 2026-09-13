"""Command-line interface for ESC (Epistemic State Compilation)."""

from __future__ import annotations

from typing import Optional
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from esc.benchmark.generator import generate_epidag_task
from esc.benchmark.relational import generate_relational_suite
from esc.eval.gepa_metric import epistemic_gepa_metric
from esc.runner import run_experiment_1
from esc.config import DEFAULT_MODEL, RLMConfig, task_lm

app = typer.Typer(
    name="esc",
    help="ESC: Epistemic State Compilation - Separating persistent epistemic state from LLM cognition.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    depths: str = typer.Option("2,4,8,16", help="Depths for relational_v2; nominal node counts for legacy"),
    benchmark: str = typer.Option("legacy", help="legacy or relational_v2 (controlled relational chains)"),
    split: str = typer.Option("dev", help="dev, train, validation, test; namespaced worlds for relational_v2"),
    world_width: int = typer.Option(16, min=2, help="Entities per relational world, fixed across depths"),
    tasks_per_depth: int = typer.Option(3, min=1, help="Number of unique tasks per depth level"),
    repetitions: int = typer.Option(3, min=1, help="Number of repetitions per task (for pass@k and pass^k)"),
    mock: bool = typer.Option(True, help="Use deterministic/stochastic mock workers for instant offline evaluation"),
    model: str = typer.Option(DEFAULT_MODEL, help="Task LM for RLM and workers"),
    reflection_model: Optional[str] = typer.Option(None, help="Reserved; rejected until GEPA compilation is implemented"),
    seed: int = typer.Option(42, help="Corpus and offline mock random seed"),
    max_tokens: int = typer.Option(1024, min=1, help="Maximum generated tokens per LM call, not episode"),
    num_ctx: int = typer.Option(8192, min=1024, help="Ollama context window per call"),
    temperature: float = typer.Option(0.6, min=0.0, help="Sampling temperature"),
    max_iters: int = typer.Option(4, min=1, help="REPL iterations per invocation; may add one final extraction call"),
    max_subcalls: int = typer.Option(4, min=0, help="Recursive LM calls per invocation"),
    max_output_chars: int = typer.Option(4000, min=1, help="REPL output characters visible per entry"),
    n_samples: int = typer.Option(3, min=1, help="Continuous rollouts in condition B"),
    request_timeout: float = typer.Option(120.0, min=1.0, help="Timeout in seconds per backend request"),
    output_dir: str = typer.Option("outputs/experiment_1", help="Directory to save experiment results"),
) -> None:
    """Run Experiment 1: Does epistemic isolation change horizon scaling?"""
    try:
        depth_list = [int(d.strip()) for d in depths.split(",")]
    except ValueError as exc:
        raise typer.BadParameter("Use comma-separated task sizes: 2,4,8,16") from exc
    if not depth_list or len(set(depth_list)) != len(depth_list) or any(d not in {2, 4, 8, 16} for d in depth_list):
        raise typer.BadParameter("Use distinct task sizes from 2,4,8,16")
    if reflection_model:
        raise typer.BadParameter("GEPA compilation is not implemented; omit --reflection-model")
    if benchmark not in {"legacy", "relational_v2"}:
        raise typer.BadParameter("--benchmark must be legacy or relational_v2")
    if split not in {"dev", "train", "validation", "test"} or (benchmark == "legacy" and split != "dev"):
        raise typer.BadParameter("Use a valid split; legacy supports dev only")
    if model.startswith("ollama_chat/") and num_ctx <= max_tokens:
        raise typer.BadParameter("--num-ctx must exceed --max-tokens")
    console.rule("[bold cyan]ESC A/B/C Pipeline Pilot[/bold cyan]")
    rprint(f"[bold]Nominal task sizes:[/bold] {depth_list}")
    rprint(f"[bold]Tasks per depth:[/bold] {tasks_per_depth} | [bold]Repetitions (k):[/bold] {repetitions}")
    rprint(f"[bold]Execution mode:[/bold] {'MOCK (offline deterministic / stochastic)' if mock else f'REAL LM ({model})'}")

    sub_lm = None
    if not mock:
        import dspy
        lm_name = model
        rprint(f"[yellow]Initializing DSPy Task LM: {lm_name}...[/yellow]")
        sub_lm = task_lm(lm_name, max_tokens=max_tokens, num_ctx=num_ctx,
                         temperature=temperature, request_timeout=request_timeout)
        dspy.configure(lm=sub_lm)

    exp_result = run_experiment_1(
        depths=depth_list,
        seed=seed,
        tasks_per_depth=tasks_per_depth,
        repetitions=repetitions,
        use_mock=mock,
        sub_lm=sub_lm,
        output_dir=output_dir,
        rlm_config=RLMConfig(max_iters=max_iters, max_llm_calls=max_subcalls,
                             max_output_chars=max_output_chars),
        n_samples=n_samples,
        benchmark=benchmark, split=split, world_width=world_width,
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
            f"{vec.error_propagation:.1%}" if vec.error_propagation is not None else "N/A",
            f"{vec.false_promotion_rate:.3f}" if vec.false_promotion_rate is not None else "N/A",
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
    measured_depths = sorted(next(iter(exp_result.horizon_decays.values())).accuracies_by_depth)
    min_depth, max_depth = measured_depths[0], measured_depths[-1]
    decay_table.add_column(f"Acc @ d={min_depth}", justify="right")
    decay_table.add_column(f"Acc @ d={max_depth}", justify="right")

    for sys_name, decay in exp_result.horizon_decays.items():
        acc_min = decay.accuracies_by_depth[min_depth]
        acc_max = decay.accuracies_by_depth[max_depth]
        decay_table.add_row(
            sys_name,
            f"{decay.alpha:.3f}",
            f"{decay.beta:.4f}",
            f"{decay.r_squared:.3f}",
            f"{acc_min:.1%}",
            f"{acc_max:.1%}",
        )
    console.print(decay_table)

    # 3. Hypothesis H1 summary
    h1 = exp_result.h1_hypothesis
    console.rule("[bold green]Descriptive Pilot Comparison[/bold green]")
    rprint("[bold]Hypothesis H1:[/bold] Not tested (pipeline pilot)")
    rprint(f"[bold]β_A (Continuous):[/bold] {h1['beta_A']:.4f}")
    rprint(f"[bold]β_C (Isolated):[/bold]   {h1['beta_C']:.4f}")
    rprint(f"[bold]Result:[/bold] {h1['interpretation']}")
    rprint(f"\n[green]✓ Summary, individual runs, and available trajectories saved to: {output_dir}/[/green]")


@app.command()
def bench(
    depth: int = typer.Option(4, help="Depth of synthetic task DAG to inspect (2, 4, 8, 16)"),
    seed: int = typer.Option(42, help="Random seed"),
    benchmark: str = typer.Option("legacy", help="legacy or relational_v2"),
    world_width: int = typer.Option(16, min=2, help="Entities per relational world"),
) -> None:
    """Generate and inspect an EpiDAG synthetic benchmark task."""
    if benchmark == "relational_v2":
        if depth not in {2, 4, 8, 16}:
            raise typer.BadParameter("Use depth 2, 4, 8, or 16")
        task = generate_relational_suite(depths=[depth], tasks_per_depth=1, seed=seed, width=world_width)[0]
    elif benchmark == "legacy":
        task = generate_epidag_task(task_id=f"demo_d{depth}", depth=depth, seed=seed)
    else:
        raise typer.BadParameter("--benchmark must be legacy or relational_v2")
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
