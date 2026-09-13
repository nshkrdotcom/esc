# Experiment 1: implementation gates and research fidelity

[HANDOFF.md](../HANDOFF.md) is the operational entry point: exact pending command,
current evidence, completion boundaries, and file-level implementation checklist.

## Audit of the proposed study

The runtime and first controlled-domain benchmark milestones are complete; the confirmatory study is not. The legacy and relational development smoke checks each contain one task and one outer repetition. They cannot estimate horizon decay, pass⁵, EPCₖ, ablation effects, or confidence intervals. No GEPA compile or Flex search has occurred. There is no basis for a numeric percentage-complete claim: the execution framework is working, but the experimental controls are the central remaining work.

| Requirement | Current status | Gate |
|---|---|---|
| Fresh RLM + typed projected state + separate audit | Implemented and interpreter-tested | M0 complete |
| Local serving, provider tokens, failure diagnostics | Live validated | M0 complete |
| 90-second research entry point | Root RESEARCH_NOTE.md implemented | Complete; figures pending |
| Comparable tasks at actual depths 2/4/8/16 | Exact relational depth implemented; broader semantic depth untested | M1 complete within scope |
| Equal information without answer-aware source routing | Full corpus access in relational_v2; legacy remains confounded | M1 complete within scope |
| Shared episode accounting | Soft allowance implemented; hard/matched policy pending | M2 partial |
| Matched live A/B/C intervention | Not implemented | M3 |
| Faithful primitive ablations | Partial names/behaviors only | M4 |
| Repeated held-out study and uncertainty | Not performed | M5 |
| Four figures and paired failure trace | No qualifying data | M5 |
| GEPA/Flex | Stubs/design only | After Experiment 1 |

## M1 — controlled relational benchmark (implemented and validated)

Implement an opt-in `relational_v2` family while preserving legacy runs. A world contains opaque entity identifiers and independently sampled permutation relations. Each transition must look up the outgoing object of the preceding entity under the next relation. The same world, start entity, ordered relations and corpus are reused at all depths; shorter tasks ask for a prefix of the same chain.

Acceptance checks:

- Requested node count and longest dependency depth both equal 2/4/8/16.
- Every intermediate value changes the suffix result when replaced with another valid entity; permutations prevent accidental path merging.
- Source IDs/order are shuffled independently of the selected path and depth.
- Every step's permitted sources equal the full corpus available to A/B. No projected source list identifies the answer document.
- Final output is always an opaque entity; corpus size, answer vocabulary, local operation and branching width stay fixed within a world.
- The lookup witness uses only public source rows, the public relation and supplied input state. It binds subject, relation and object, not just occurrence of a value somewhere in a citation.
- Deterministic reference solving and mutation tests validate the generator. Mock scores remain plumbing checks.
- Train/validation/test world seeds are namespaced separately. Paired depths from one world never cross splits.

Limits that remain: this is relational pointer following, not an independently calibrated semantic inference benchmark. Type I checking is deliberately strong. A Python program can solve it without a stochastic LM decision at each hop. A future semantic family and verifier ablation are required for broad claims. Depth changes how many rows on a fixed corpus are relevant; total corpus access is controlled, relevant/irrelevant ratio is not independent.

Validation: 83 tests passed with the opt-in Deno interpreter test enabled. The live depth-2 development check completed: A/B correct, C abstained on a parsing failure; respective token counts were 11,822 / 48,763 / 14,426. See the [saved evidence snapshot](evidence/relational_v2_validation_001.json). This establishes runtime compatibility, not comparative reliability. M2 is the next implementation gate.

## M2 — shared episode budget (partial; next validation in HANDOFF.md)

**Implementation update:** a first shared soft allowance is implemented; see
[exact semantics and remaining limitations](EPISODE_BUDGET.md). It reserves
generation only and reconciles measured prompt costs. It does not implement the
validated preflight tokenizer below, claim bounded overshoot, or clear the
matched-compute gate. The following remains the stricter study target.

Binding and generous live checks now pass independent request-journal replay.
The request journal covers rejected and concurrent responses without relying on
DSPy history. A read-only `esc audit-budget` command checks it against saved
episodes and ledger totals. The 139-test suite includes real batched recursive
calls and malformed-output handling. A four-depth development sweep exposed a
model-formatting failure that previously aborted batches; the new recorded
failure policy counts it as an unsuccessful attempt. See [M2_VALIDATION.md](M2_VALIDATION.md)
and the [handoff acceptance criteria](../HANDOFF.md). The subsequent complete
four-depth calibration (`budget_depth_dev_003`) passed request replay for all 12
episodes and 200,393 tokens. Four episodes exhausted; one recorded a model-output
error without aborting the batch. Unequal realized costs and C's early provenance
failures keep the policy and feasibility gates open. One paired development world
is not a held-out reliability study.

The follow-up [interface contracts](INTERFACE_CONTRACT.md) add explicit typed-null
abstention and identical JSON document serialization for all conditions. They
address refusal ambiguity and source-header parsing without changing the witness
or granting privileged source access. Manifest versions separate these changes
from earlier runs. Cost audits now expose per-depth spend disparities explicitly.

Implement one accounting owner around the actual LM backend, shared by root/subcalls, all C steps, and all B rollouts. Do not implement independent per-step caps and call them a matched episode budget.

For hard enforcement, before each dispatch reserve prompt plus allowed generation; use a tokenizer compatible with the exact served model or a provider tokenization endpoint, validate it against reported usage, and reject an unverifiable budget configuration. Set the output cap from remaining allowance; reconcile actual usage on completion. Parallel recursive calls must reserve atomically. Include extraction and adapter retries. Exhaustion is a recorded episode outcome, not a backend failure or silently dropped sample. An interrupted worker cannot supply an accepted answer; B can still vote over earlier completed rollouts. A backend failure invalidates the batch and preserves diagnostic usage separately.

Calibrate budget bands on development worlds only. B votes over the rollouts it can afford, without labels. Report common allowance, actual tokens, cache status, calls, truncation and exhaustion. Early exits mean actual spend can differ; report quality/cost frontiers and preregister the matching tolerance or allocation policy. Do not pad outputs with meaningless tokens to manufacture equal expenditure.

## M3 — comparable live fault intervention

Add an explicit result-emission hook inside A/B's persistent RLM invocation, shared with C's candidate boundary. The benchmark public schedule requires emitting each transition before advancing; record deviations. Preserve A's interpreter/history between emissions. Confirm the hook does not introduce C's projection or validation semantics into A. Keep the uninstrumented A baseline and quantify instrumentation effects.

Select intervention site/value from public world structure and a fixed seed; choose a different valid entity, not a `CORRUPT_` sentinel. Apply at the same reached transition, after emission and before the next transition. In C's candidate-intervention arm, this is before witness promotion. Do not conflate it with accepted-state corruption. Save clean/intervened runs paired by world, condition, outer repeat and site.

Measure EPC at each graph distance over actually applied interventions. Also report correct, wrong, abstained and unreached fractions, attempted/applied intervention counts, false promotions and final coverage. B must report both selected-trajectory behavior and per-rollout effects, not `any(corrupted)` mislabeled as the ensemble outcome. No oracle labels enter tools or voting.

## M4 — real ablations

Implement the five variants in RESEARCH_NOTE.md behind explicit architecture configurations. Verify context contents using capture workers: which raw history, summaries, typed facts and witness information crosses each transition. Give all variants the same task, source access, LM, public operation rules and episode budget. Separate schema enforcement from assurance checks. Reject unsupported combinations instead of aliasing names to incomplete behavior.

## M5 — analysis and results artifact

Freeze implementation, model digest/template, budget policy, corpus seeds/splits and analysis before held-out evaluation. First run a small development pilot across all depths to inspect saturation, cost, failure types and feasibility; choose sample size using observed uncertainty/compute, not desired effect direction. No billion-token study before controls pass.

Collect at least five uncached outer repeats on held-out worlds for pass@5/pass⁵. Treat the world as the cluster: bootstrap worlds with paired conditions, depths and repetitions retained. Fit a prespecified binomial model with a suitable log-probability parameterization or likelihood-based contrast; handle zero/all-success cells and separation explicitly. Report β intervals/differences, absolute reliability curves and count data; do not substitute clipped log-accuracy OLS for inference. If the model is unidentifiable or data saturate, report that rather than a fabricated slope advantage.

Generate the four figures, machine-readable aggregates, and one actual paired failure trace from saved records. For matched-cost claims show actual spend and budget exhaustion. Update page-one results only from this analysis; keep smoke/mock data out. Headline only what the intervals, ablations and scope support. A deterministic relational result does not establish a general semantic composability result.

## After Experiment 1

Only then implement D with saved GEPA programs, disjoint training/validation/test worlds and separate reflection cost. Structural Flex search and external-corpus replication follow a valid fixed-architecture comparison. They cannot repair a confounded initial benchmark after the fact.
