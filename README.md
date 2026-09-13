# ESC: Epistemic State Compilation

MIT License © 2026 nshkrdotcom.

ESC explores whether keeping persistent typed state separate from each language-model invocation improves multi-step reliability. DSPy RLM supplies disposable reasoning contexts; a state kernel projects accepted facts and checks candidate outputs against corpus evidence or deterministic arithmetic rules.

**Current status: pipeline prototype.** The pre-experiment review fixed execution and measurement bugs, but this is not yet a controlled horizon-scaling study. Read [the review and remaining limitations](docs/PREFLIGHT_REVIEW.md) and [the pilot runbook](docs/RUNBOOK.md) before spending inference compute.

## Implemented pilot

| Condition | Execution |
|---|---|
| A — Continuous | One RLM invocation receives the corpus and public step specifications |
| B — Search-heavy | Three independent A rollouts, with majority voting and measured total usage |
| C — Isolated | One fresh RLM interpreter per step; only projected typed state crosses boundaries |
| D — GEPA | Deferred; live use is rejected until compilation/loading is implemented |

B's budget is not matched to C. Mock workers intentionally use hidden labels and scripted error behavior; their results cannot support the hypothesis. Live failures stop the run instead of counting as abstentions. Missing token usage is an error.

## Setup and offline checks

Use `uv` for all Python commands:

```bash
uv sync --locked
uv run pytest
uv run esc --help
uv run esc bench --depth 4
```

The local task model selected for this workspace is **Qwen3-14B Q4_K_M**, served by Ollama. The CLI default is `ollama_chat/qwen3:14b`; the Hugging Face name can be supplied explicitly. Deno/Pyodide is also required for RLM. No amount of free VRAM is guaranteed: weights, KV cache, runtime buffers, and desktop usage share the 16GB GPU.

The [runbook](docs/RUNBOOK.md) contains the future live pilot command. Defaults use uncached sampling, temperature 0.6, a 2,048-token output limit per call, and an 8,192-token local context window with thinking disabled. These are pilot settings; the output limit is not an episode budget.

## Outputs and interpretation

Completed episodes are flushed to `runs.jsonl`, including available trajectories and C's audit records. The output directory also includes a reproducible task corpus, manifest, and a summary on successful completion. Existing runs cannot be overwritten. Failed runs retain completed episodes but do not automatically resume.

The summary reports accuracy, repeated success, abstention, measured tokens, and descriptive horizon fits. Unsupported/inapplicable EPC and false-promotion metrics are N/A. H₁ remains untested until compute matching and benchmark controls are implemented.

The CLI's original `--depths 2,4,8,16` selects node counts. Actual dependency depths are 2, 4, 7, and 15; these are used in the report. Some edges are procedural and the long tasks repeat simple rules, so this prototype does not isolate semantic dependency depth.

## Research objective

The intended study compares the degradation of success probability with dependency depth at equal model, information and inference budget:

$$\log P(\mathrm{success}) = \alpha - \beta d.$$

The hypothesis is that isolated state produces a shallower degradation curve. A valid test requires controlling compute, task difficulty, decomposition and source selection; the initial mock harness cannot establish it.

## Documentation

- [Runbook](docs/RUNBOOK.md): setup, future pilot commands and artifacts.
- [Pre-experiment review](docs/PREFLIGHT_REVIEW.md): fixes and outstanding study blockers.
- [Architecture](docs/ARCHITECTURE.md): state projection and promotion mechanics.
- [EpiDAG](docs/EPIDAG_BENCHMARK.md): actual graph topology and limitations.
- [Ablations](docs/ABLATIONS.md): partial modes and missing implementations.
- [Experiment specification](docs/EXPERIMENT_SPEC.md): intended future study.

`uv run esc gepa-demo` demonstrates metric feedback only; it does not run optimization. Flex and external benchmark integrations are future work.
