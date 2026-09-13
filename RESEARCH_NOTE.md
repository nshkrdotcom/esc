# ESC: Can Epistemic State Compilation Make Stochastic LM Programs Composable?

**Status: Experiment 1 infrastructure and benchmark development. No evidence for the central hypothesis yet.**

Long-horizon LM programs often carry a model's linguistic history forward as state. ESC tests an alternative: discard reasoning history after each transition, persist typed externally validated state, and compile a fresh context for the next transition.

**Hypothesis:** at equal model, available information, and inference compute, this reduces the rate at which final reliability decays with dependency depth.

```text
A — Continuous (one RLM invocation)
task + corpus → REPL step → accumulated history → REPL step → answer

C — ESC (one RLM invocation per transition)
canonical state → project → fresh RLM → candidate → witness → commit
       ↑_____________________________________________________|
                         repeat; final state → answer

B — Search-heavy
independent A rollouts → label-blind majority vote → answer
```

The implemented separation is real: the default isolated worker gets copied typed facts and permitted evidence; the RLM interpreter is fresh at each invocation; trajectories are audit data, not inputs to subsequent default workers. This does not itself demonstrate a reliability advantage.

## Results at a glance

There is **no controlled result table yet**. Dashes denote unmeasured outcomes, not zero performance. The illustrative numbers and improvements in the proposed research pitch are not observations.

| System | Final accuracy | Decay β ↓ | pass⁵ ↑ | EPC₃ ↓ | Matched tokens/task |
|---|---:|---:|---:|---:|---:|
| Continuous RLM | — | — | — | — | — |
| Search-heavy RLM | — | — | — | — | — |
| ESC | — | — | — | — | — |
| ESC + GEPA (deferred) | — | — | — | — | — |

**What has actually run:** Qwen3-14B Q4_K_M on a 16GB GPU completed one legacy two-node task in A, B, and C, one repetition each. All three final answers were correct. Token counts were 3,647 / 17,803 / 8,625; latencies were 13.9 / 53.5 / 23.2 seconds. B includes three rollouts. All 16 LM calls stopped normally after a serving-template fix. This is a runtime check with unequal cost, not a controlled experiment. The committed [runtime report](docs/RUNTIME_CONFIGURATION.md) records it; raw local artifacts are in the ignored `outputs/config_validation_001/` directory.

**New benchmark compatibility check:** one `relational_v2` development task at depth 2, one outer repeat, completed in all three conditions. A and B answered correctly; C abstained after its generated parser repeatedly found no rows. A/B/C used 11,822 / 48,763 / 14,426 tokens and 36.6 / 160.4 / 42.3 seconds. B's three rollouts required final extraction after reaching the iteration limit. These unequal-cost episodes are not headline results. A compact [evidence snapshot](docs/evidence/relational_v2_validation_001.json) accompanies the code; full local traces remain in ignored `outputs/relational_v2_validation_001/`.

**Earlier shared-ledger compatibility check:** one further depth-2 development task run used a nonbinding allowance of 100,000 tokens per condition. A/B/C spent 16,932 / 47,208 / 4,874 tokens; A/B were correct and C was incorrect. All reservations settled with no exhaustion or unknown usage. The [saved snapshot](docs/evidence/budget_generous_001.json) supports these accounting observations. Subsequent binding and generous checks pass request-journal replay; [M2_VALIDATION.md](docs/M2_VALIDATION.md) records the new results and the [handoff](HANDOFF.md) states the next gates.

These smoke checks establish neither a Pareto winner nor a reliability effect. None has repeated-trial uncertainty. C's abstention is a failed task, not an observed containment benefit.

## The four figures that will decide the claim

### 1. Reliability versus dependency depth — pending

Plot final success probability for A/B/C at depths 2, 4, 8, and 16, at a preregistered common episode budget. Include task-cluster uncertainty intervals and actual token distributions. Fit `log P(success) = α − βd` with an analysis that handles zero successes without an arbitrary numerical floor. Report paired differences in slopes and success at each depth; report a ratio of slopes only when the denominator is identifiable and positive.

A fixed accuracy improvement is weaker evidence than an advantage that grows with depth. B is essential: if matched-budget search reproduces C's reliability, the isolation claim loses support. An equal maximum allowance is not equal realized expenditure; both must be reported, with quality-versus-cost comparisons where spending differs.

The legacy company benchmark does not support this plot: task size changes answer type, some dependencies are unnecessary, and its longest tasks repeat approval transformations. The new opt-in `relational_v2` family fixes several controls: each step applies a fresh arbitrary relation to the preceding entity, source IDs carry no roles, and every condition sees the same complete corpus. Worlds are paired across depths and the corpus size remains fixed. This tests **relational/computational dependency depth**, not the full proposed semantic inference setting. A continuous RLM may write a short Python loop that solves the chain perfectly; that is an informative baseline, not a reason to prohibit code.

### 2. Error propagation versus downstream distance — pending

Intervene on a reached intermediate candidate and measure downstream wrong answers at each distance, conditional on the intervention actually being applied. Keep wrong, correct, abstained, and unreached outcomes separate. Report containment together with final coverage so refusing everything cannot look like improved reliable composition.

A/B need a comparable intervention inside their persistent RLM invocation. Replacing an answer only after the invocation has finished cannot test propagation. The current C hook and label-reading mock behaviors do not supply this figure. Candidate corruption before promotion and corruption of already accepted state are different interventions and must be separate experiments.

### 3. Searchable success versus repeated success — pending

Evaluate each fixed task at least five times with uncached sampling. Report pass@5 (at least one success) and pass⁵ (all five successful) from complete groups, with uncertainty over tasks. If more than five repeats are collected, specify the estimators in advance. Keep B's internal rollout count distinct from the outer repetition count. A task budget includes every internal B rollout, every C transition, recursive calls, formatting retries, and extraction calls.

Existing metric functions alone are not evidence: the live validation has one outer repeat. Partial or failed runs must not silently change k or select easier survivors.

### 4. Which primitive matters? — pending

| Intended variant | Fresh contexts | Typed state | Witness promotion | Current status |
|---|---:|---:|---:|---|
| Continuous | no | no | no | A implemented |
| Isolation only | yes | no | no | Not implemented faithfully |
| Shared history with contracts | no | yes | yes | Current mode carries summaries, not full history |
| Isolation + typing | yes | yes | no | Prototype `no_verification`; needs controlled validation |
| Full ESC | yes | yes | yes | C implemented; witnesses have limited scope |

The current `no_typing` mode still uses typed facts and witnesses; `raw_summaries` breaks typed arithmetic inputs. They cannot stand in for the proposed ablations. Attribute effects only from controlled contrasts, not additive percentages inferred from separate accuracy deltas. Interactions between isolation, state format, and promotion may dominate.

## One trace — not yet the requested paired failure

The preserved legacy runtime trace shows A choosing a document-header line, failing a revenue regex twice, then correcting its line selection and answering correctly. That is an actual self-correction, **not** a demonstration of global contamination. No paired live A/C failure with a shared intervention exists yet. We will publish a linked, reproducible pair with its task, seed, model settings, intervention and witness decision when the intervention machinery is ready. We will not substitute a scripted mock cascade or an imagined narrative.

## Scope, falsification, and next gate

A [shared soft episode allowance](docs/EPISODE_BUDGET.md) now accounts for all
worker calls and records exhaustion without aborting the sweep. Prompt costs
can overshoot; this does not yet establish matched compute or clear M2.
Live binding and generous checks now pass independent request-journal replay.
At a 1,000-token allowance, prompt costs caused 130–142% overshoot in the
journaled binding check. A four-depth development attempt also exposed malformed
model output aborting a batch; that is now recorded as a failed attempt, with the
historical aborted batch preserved. See [the validation report](docs/M2_VALIDATION.md)
for outcomes and the remaining cost-policy gate.
The completed four-depth development calibration also passes accounting replay,
but uses only one paired world and one repetition. C exited early in every task;
costs differed substantially. This is feasibility evidence, not support for H₁.
A subsequent [versioned interface check](docs/INTERFACE_CONTRACT.md) introduced
explicit null abstention and a shared JSON corpus representation. Its accounting
passed, but C still searched incompletely and abstained; it supplies no positive
reliability evidence.

The primary claim remains untested until compute, corpus access, depth manipulation, repetitions and uncertainty are controlled. Citation existence is not semantic verification. The new relational family uses a deterministic authoritative-row witness, so any eventual benefit there is scoped to Type I checkable transitions; it would not establish the effect for Type II/III reasoning.

No positive finding is required. Similar slopes, a search-heavy baseline that matches ESC, loss of coverage, or advantages explained by source filtering or verifier strength are useful negative results. GEPA and structural boundary discovery are deferred until the fixed architecture experiment clears these controls.

See [the staged implementation plan](docs/EXPERIMENT_1_PLAN.md) for acceptance gates and current completion status. This note is the entry point for the eventual results, not a claim that those results already exist.
