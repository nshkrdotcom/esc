# ESC: Epistemic State Compilation

MIT License © 2026 nshkrdotcom.

**Resuming work? Read [HANDOFF.md](HANDOFF.md)** for completed validation, exact
next commands, pending checks, and the ordered implementation roadmap.

ESC explores whether keeping persistent typed state separate from each language-model invocation improves multi-step reliability. DSPy RLM supplies disposable reasoning contexts; a state kernel projects accepted facts and checks candidate outputs against corpus evidence or deterministic arithmetic rules.

Start with [RESEARCH_NOTE.md](RESEARCH_NOTE.md) for the hypothesis, actual evidence, four planned figures and current study status. [EXPERIMENT_1_PLAN.md](docs/EXPERIMENT_1_PLAN.md) tracks implementation gates. Experiment 1 has a validated runtime and an opt-in controlled relational benchmark; the matched-compute study and GEPA remain unfinished.

**Current status: pipeline prototype.** The pre-experiment review fixed execution and measurement bugs, but this is not yet a controlled horizon-scaling study. Read [the review and remaining limitations](docs/PREFLIGHT_REVIEW.md) and [the pilot runbook](docs/RUNBOOK.md) before spending inference compute.

The new [frozen study workflow](docs/STUDY_RUNBOOK.md) implements common public-row
parsing, candidate/accepted-state intervention hooks, explicit ablations,
randomized plans, independent accounting replay, and four analysis figures.
Development validation is ongoing; a repeated held-out study has not completed.
Its cost policy is quality versus **measured** cost at soft allowance bands,
not an equal-compute H₁ claim. Use `esc study-*` for this workflow; the historical
`esc run` pilot remains separate.

## Implemented pilot

| Condition | Execution |
|---|---|
| A — Continuous | One RLM invocation receives the corpus and public step specifications |
| B — Search-heavy | Three independent A rollouts, with majority voting and measured total usage |
| C — Isolated | One fresh RLM interpreter per step; only projected typed state crosses boundaries |
| D — GEPA | Deferred; live use is rejected until compilation/loading is implemented |

B's actual spend is not matched to C. An optional shared soft episode allowance records exhaustion and continues the sweep; prompt costs can overshoot. Mock workers intentionally use hidden labels and scripted error behavior; their results cannot support the hypothesis. Backend/accounting failures stop the run; missing token usage is an error.
Malformed model outputs are flagged unsuccessful attempts with measured usage,
so they do not silently remove the rest of a sweep. Budgeted runs now include a
[request journal and offline auditor](docs/REQUEST_JOURNAL.md).

## Setup and offline checks

Use `uv` for all Python commands:

```bash
uv sync --locked
uv run pytest
uv run esc --help
uv run esc bench --depth 4
```

The local task model selected for this workspace is **Qwen3-14B Q4_K_M**, served by Ollama. The CLI default is `ollama_chat/esc-qwen3:14b-nothink`. Create this alias using `ollama create esc-qwen3:14b-nothink -f models/Modelfile.qwen3-nothink`; it reuses the installed Hugging Face weights with a corrected non-thinking template. Deno/Pyodide is also required for RLM. No amount of free VRAM is guaranteed: weights, KV cache, runtime buffers, and desktop usage share the 16GB GPU.

The [runbook](docs/RUNBOOK.md) contains the live pilot command. Defaults use uncached sampling, temperature 0.6, a 1,024-token output limit per call, and an 8,192-token local context window with thinking disabled. RLM limits are 4 iterations and 4 recursive calls per invocation, with a possible final extraction call. Backend requests time out after 120 seconds. These are pilot settings; the output limit is not an episode budget.

## Outputs and interpretation

Completed episodes are flushed to `runs.jsonl`, including available trajectories and C's audit records. The output directory also includes a reproducible task corpus, manifest, and a summary on successful completion. Existing runs cannot be overwritten. Failed runs retain completed episodes but do not automatically resume.

The summary reports accuracy, repeated success, abstention, measured tokens, and descriptive horizon fits. Unsupported/inapplicable EPC and false-promotion metrics are N/A. H₁ remains untested until compute matching and benchmark controls are implemented.
The exported `check_hypothesis_h1` helper also leaves `h1_supported=None`; it only
compares point estimates on matching depth grids. Ratios are omitted for
nonpositive slopes or incomparable depth grids and do not establish significance.

For the default legacy family, `--depths 2,4,8,16` selects node counts with actual depths 2, 4, 7, and 15. The new `--benchmark relational_v2` family has exact depths 2/4/8/16, fixed corpus size across paired depths, opaque shuffled sources, and full corpus access for every condition. It controls relational dependency depth; broader semantic reasoning and matched compute remain untested. See [the relational benchmark](docs/RELATIONAL_BENCHMARK.md).

## Research objective

The intended study compares the degradation of success probability with dependency depth at equal model, information and inference budget:

$$\log P(\mathrm{success}) = \alpha - \beta d.$$

The hypothesis is that isolated state produces a shallower degradation curve. A valid test requires controlling compute, task difficulty, decomposition and source selection; the initial mock harness cannot establish it.

## Documentation

- [Runbook](docs/RUNBOOK.md): setup, future pilot commands and artifacts.
- [Runtime configuration fix](docs/RUNTIME_CONFIGURATION.md): the thinking-template bug, corrected model alias, and completed local validation.
- [Pre-experiment review](docs/PREFLIGHT_REVIEW.md): fixes and outstanding study blockers.
- [Architecture](docs/ARCHITECTURE.md): state projection and promotion mechanics.
- [EpiDAG](docs/EPIDAG_BENCHMARK.md): actual graph topology and limitations.
- [Ablations](docs/ABLATIONS.md): partial modes and missing implementations.
- [Experiment specification](docs/EXPERIMENT_SPEC.md): intended future study.

`uv run esc gepa-demo` demonstrates metric feedback only; it does not run optimization. Flex and external benchmark integrations are future work.
