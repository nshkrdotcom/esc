# ESC handoff: frozen relational study workflow

## Thesis and evidence boundary

**Can stochastic reasoning be made composable without first making every local
reasoner reliable?** ESC separates persistent epistemic state from linguistic
history. Each isolated transition receives projected typed state and public
evidence; a witness governs promotion. The continuous baseline retains its native
RLM history/interpreter. We want to measure depth-dependent reliability, error
propagation, repeatability, and which primitives cause any difference.

The original H₁ requires equal model, information, and compute. This backend
currently provides a **soft** allowance with measured prompt reconciliation, not
a hard total-token bound. The new study uses prespecified allowance bands and
**quality versus actual measured cost**. `compute_matched=false` stays explicit.
This narrower exploratory study cannot confirm equal-compute H₁ by relabeling
the allowance. No positive outcome is assumed or required.

The common parser, transition hooks, explicit ablations, frozen randomized plans,
independent accounting replay, world-cluster analysis, and four figure exports
are implemented. Development validation is ongoing. A full held-out study has
**not** completed. GEPA/Flex remain deferred.

## Implemented and tested

- Local Qwen3-14B Q4_K_M with corrected non-thinking Ollama template, uncached
  DSPy RLM root/subcalls, context-scoped shared ledger and durable request journal.
- `relational_v2`: exact depths 2/4/8/16, paired worlds, fixed equally accessible
  corpus, opaque sources, strict Type I public-row witnesses. This is computational
  pointer following, not broad semantic inference.
- Every study condition gets `read_rows()`: all public rows with exact source/span,
  no task-specific selection and no evaluator labels.
- Native continuous RLMs call `emit` inside their trajectory; isolated variants
  call the same protocol at the worker boundary. Seeded alternate valid entities,
  reached/applied records, receipt compliance, and separate candidate versus
  accepted-state corruption modes are implemented.
- Seven variants: `continuous`, `continuous_emit`, `search_emit`, `isolated_raw`,
  `isolated_typed`, `shared_verified`, `isolated_verified`. Shared variants retain
  one native interpreter/history. Isolated variants create a new RLM per hop.
  Typed-unverified facts retain evidence with candidate assurance. Historical
  pilot ablation names remain prototypes; use these new variants for comparisons.
- Frozen model/runtime/source archive, randomized complete episode schedule,
  fsynced results/requests, no overwrite, no silent resume, incomplete-batch rejection.
- Paired world bootstrap, binomial log-link decay/contrasts, pass@5/pass⁵, actual
  cost/coverage/errors, downstream denominators, four PNG/SVG figures and a real
  clean/intervened trace when available. Degenerate intervals and unidentifiable
  slopes are unavailable rather than falsely precise.
- Last full verification: **169 tests passed** using
  `ESC_TEST_RLM=1 uv run pytest -q`. Tests include actual Deno/Pyodide receipts,
  retained continuous scratch state, fresh contexts, candidate/state corruption,
 label-poisoned public-input solvers, and accounting/integrity failures.

The follow-up submission guard keeps premature `SUBMIT` attempts inside the same
RLM history/interpreter as public protocol errors. Recovery uses only remaining
iterations and the shared allowance; extraction still cannot bypass receipts.
Rejected submissions are recorded separately from terminal violations. New plans
record this policy and root/recursive request roles explicitly.

## Live evidence and current processes

Historical checks are in [M2_VALIDATION.md](docs/M2_VALIDATION.md) and
[INTERFACE_CONTRACT.md](docs/INTERFACE_CONTRACT.md). They exposed source-header
parsing, wrong-source citations, malformed output and incomplete search. They
established accounting, not an ESC advantage.

`outputs/study_dev_001` completed six episodes on one depth-2 development world:
**43,684 tokens**, independent request audit passed. A typed-isolated candidate
corruption crossed the unverified boundary and its descendant was wrong; a
separate verified episode rejected a corrupted candidate. The continuous
intervened episode skipped all receipts, so it **cannot support comparable live
EPC**. A clean verified episode had a measured formatting failure. Four development
figures and a paired trace exist locally in `analysis/`.

`outputs/study_dev_all_001` was interrupted after 13/52 episodes: no completion
marker, excluded from analysis. Preserve it.
`outputs/study_dev_all_002` is the replacement all-variant development gate:
two worlds, depth 2, one repetition, two allowance bands, and clean/corrupt pairs
(52 episodes). Check process status and artifacts before starting another run.
An existing plan or partial `runs.jsonl` is not a completed study.

A serial validation queue waits for that batch, audits/analyzes it, then runs
`outputs/study_dev_receipts_001` (12 guarded episodes on dev seed 86) and
`outputs/study_dev_accepted_001` (six accepted-state episodes on dev seed 84).
Queue output is `/tmp/esc-validation-queue.log`; processes run in detached sessions
so chat interruptions do not terminate them. The queue stops on failure. Check
processes and completion markers before starting anything else; no held-out run
is queued automatically.

## Commands and artifacts

Read [STUDY_RUNBOOK.md](docs/STUDY_RUNBOOK.md) for semantics, variants, gates,
background execution, and analysis limits. All Python uses `uv`.

```bash
uv sync --locked
ESC_TEST_RLM=1 uv run pytest -q
uv run esc study-freeze configs/study_dev_all.json outputs/study_dev_next
uv run esc study-run outputs/study_dev_next
uv run esc study-audit outputs/study_dev_next
uv run esc study-analyze outputs/study_dev_next
```

Choose a new directory. Freeze reads local model metadata but generates no tokens.
Changed code, lockfile, model or template requires a new plan. `source.zip`
preserves the actual frozen source; analysis records its own code hash.
Do not edit old plans to evade identity checks.

The held-out template `configs/study_heldout.json` has **1,040 episodes**: two
worlds, four depths, five outer repeats, two bands, all variants/pairs. It is a
template, not a completed or automatically justified study. Two clusters provide
weak uncertainty; choose adequate world count from development evidence before
freeze. User authorization to proceed exists, but interpretation still requires
protocol feasibility and complete accounting.

| Work | Main files |
|---|---|
| Public parser | `src/esc/benchmark/public_rows.py` |
| Interventions | `src/esc/study_protocol.py` |
| Architectures | `src/esc/study_systems.py` |
| Frozen plans/execution | `src/esc/study.py`, `configs/` |
| Request/episode audit | `src/esc/study_audit.py` |
| Statistics/figures/trace | `src/esc/study_analysis.py` |
| CLI | `src/esc/cli.py` |

## Remaining work, in order

1. Finish the all-variant development gate; audit every request and episode.
   Preserve the interrupted predecessor. Inspect both bands, each variant,
   emissions, witness decisions, errors, exhaustion and actual cost.
2. Quantify whether continuous variants actually reach and obey receipts. If
   compliance fails, report the limitation, fix the common interface on disjoint
   development worlds, and retain previous attempts. Missing downstream emissions
   are not evidence of fault containment.
3. Freeze sufficient held-out sampling and cost analysis from development evidence.
   The policy is measured quality/cost, not equal-compute H₁. Candidate and
   accepted-state interventions use separate plans. Retain the uninstrumented
   comparator to measure the effect of mandatory receipts.
4. Execute every planned held-out episode with at least five outer repeats, audit,
   then analyze. Backend/unknown-consumption failures invalidate a batch; no
   convenient completed-prefix analysis. Formatting errors and exhaustion are
   measured outcomes, not dropped samples.
5. Review intervals, saturation, adherence, coverage, cost, promotions and
   intervention denominators. Rendering four figures does not establish a
   research result. Retain valid negative or inconclusive findings.
6. Update [RESEARCH_NOTE.md](RESEARCH_NOTE.md) only with qualifying results and a
   real paired trace. Keep domain and verification scope explicit. GEPA/Flex and
   external-corpus replication follow valid fixed-architecture work.

Full local `outputs/` are ignored and absent from clones; compact evidence belongs
in the repo. Preserve existing work/artifacts. Keep the thesis and actual evidence
boundary clear in subsequent handoffs.
