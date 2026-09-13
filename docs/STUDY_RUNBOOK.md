# Frozen relational study: execution and interpretation

The original question remains whether separating persistent epistemic state from
reasoning history makes stochastic programs more composable. This implementation
tests a narrow public-row domain. It does **not** assume that ESC wins.

## What changed and what the study can claim

`esc study-*` is a separate workflow from the historical `esc run` pilot. Every
variant has the same `read_rows()` tool, which parses every public document into
subject/relation/object/source_id/span records. The tool has no task argument,
does no answer-specific filtering, and sees no evaluator labels. Models must
select rows and retain provenance themselves. Calls record how many rows and
documents were returned; that is availability, not proof the model searched all
of them. Witnesses still require the exact row in the cited document.

The cost decision is **quality versus measured cost at prespecified soft allowance
bands**, not hard-token matching. Generation reservations and actual prompt
reconciliation remain `soft_generation_reservation_v1`. `compute_matched=false`
throughout. Report success and realized cost jointly; do not discard exhausted,
expensive, or unsuccessful episodes. No padding, oracle vote selection, free
repair, or post-hoc overshoot cutoff is permitted. This is an exploratory
cost/reliability study, not confirmation of the original equal-compute H₁.

## Architectures

| Variant | Context/state across transitions | Witness | Role |
|---|---|---|---|
| `continuous` | One RLM history/interpreter; no required emissions | None | Uninstrumented baseline with common parser |
| `continuous_emit` | One RLM history/interpreter, transition receipts | None | Instrumented continuous baseline |
| `search_emit` | Up to three independent `continuous_emit` rollouts sharing one allowance | None | Label-blind majority vote |
| `isolated_raw` | Fresh RLM each hop; only the previous raw value crosses | None | Isolation without semantic state schema |
| `isolated_typed` | Fresh RLM each hop; previous candidate Fact including evidence | None | Isolation plus typing |
| `shared_verified` | One RLM history/interpreter; typed verified facts returned in receipts | Public-row witness | Contracts without amnesia |
| `isolated_verified` | Fresh RLM each hop; previous verified Fact only | Public-row witness | Full ESC |

Transport outputs necessarily have types; `isolated_raw` removes the epistemic
Fact/StepResult/assurance contract, not JSON/string transport. Its inputs omit
formal assurance requirements. Typed-unverified state stays `candidate` and its
input requirement is candidate; it is never mislabeled verified. All variants
retain the same public lookup operation and corpus. Shared and isolated variants
have different aggregate invocation limits; calls and invocations are reported
alongside tokens rather than falsely claiming equal call counts.
New frozen plans also tag request starts as `root` or `recursive`. Both use
identical LM configuration and the same episode ledger; the tag is local journal
metadata, not a provider option. Root includes action generation and final
extraction. Earlier development plans lack these tags and remain unclassified.

## Comparable fault interventions

Instrumented continuous variants must call
`emit(step_id, input_value, value, evidence)` before advancing. Their native RLM
history and interpreter persist. They must use the returned value as the next
subject. Isolated variants call the same protocol at the worker boundary.

The protocol checks order and input continuity without supplying correct values.
Skipping a receipt, ignoring its returned value, or finalizing an incomplete chain
records a protocol violation and invalidates that output. The uninstrumented
baseline measures the effect of this extra requirement. A model can still compute
ahead invisibly; receipts establish observable compliance, not access to private
reasoning or proof that no anticipatory computation happened.

Intervention values are seeded different valid entities selected from the public
corpus, with the same seed mapping for each paired task/repetition and rollout.
If original proposals differ, the replacement can differ; record both rather than
claiming the initial mistakes were identical. No target labels enter the hook.

- `candidate`: replace the emitted value before any witness promotion.
- `accepted_state`: replace a value after acceptance, deliberately corrupting
  trusted state. This is a separate sensitivity experiment, not candidate rejection.

Use a separate frozen plan for each mode. Configurable sites must precede the
shallowest final node; default site is N1. Clean/intervened episodes share worlds,
condition, budget, and outer repetition. Model sampling remains stochastic; pairing
does not imply identical random LM draws. The analysis conditions on interventions
that were applied and actually made the site wrong (evaluated afterward). It
reports unreached, abstained, correct and wrong descendants, candidate errors,
violations, and B's per-rollout/selected-rollout identities separately. EPC values
with protocol violations are unavailable, not zero.

## Commands

All Python uses `uv`. Start with:

```bash
uv sync --locked
ESC_TEST_RLM=1 uv run pytest -q
uv run esc study-freeze configs/study_dev_all.json outputs/study_dev_next
uv run esc study-run outputs/study_dev_next
uv run esc study-audit outputs/study_dev_next
uv run esc study-analyze outputs/study_dev_next
```

Choose a new directory; neither outputs nor frozen plans are overwritten. Freeze
reads local model metadata but does not generate tokens. It records the exact
Ollama model digest/template/parameters, runtime, code/lockfile hash, public worlds,
randomized episode schedule, budget and analysis settings. `source.zip` preserves
the actual source, including uncommitted changes, and is checksummed in the plan.
Execution rejects a different current source or model. Complete batches can be
analyzed later; the report records the analysis code hash separately.

The development-all config has 52 episodes: two worlds, one depth, two budget
bands, seven variants, with clean/intervened pairs except the uninstrumented
baseline. It is a feasibility check, not an uncertainty study. The held-out
template has 1,040 episodes: two worlds, four depths, five repetitions, two bands,
and the same variants/pairs. Two worlds are a minimal repeatability check, not a
credible precision target. Select adequate world count and budget bands from
development evidence before freezing the actual study; never tune on test worlds.

For a long run that must survive a chat disconnection:

```bash
setsid -f sh -c 'exec uv run esc study-run outputs/study_dev_next > outputs/study_dev_next/console.log 2>&1 < /dev/null'
```

Do not start a second copy. Inspect `console.log`, `runs.jsonl`, and process status.
The runner fsyncs each result and request. A backend/unknown-consumption failure
invalidates the batch and writes `failure.json`; malformed model output and budget
exhaustion are measured outcomes. There is no resume command. An interrupted
in-flight batch without `complete.json` is incomplete even if some results exist.
Preserve it; do not select its convenient prefix or silently refund its requests.

## Analysis and four figures

The independent auditor verifies complete episode membership, journal order,
reservations, actual provider usage, settled ledger totals, frozen tasks/scoring,
and the plan/archive checksums before analysis. It trusts provider token reports,
not an independent GPU measurement. Evidence authenticity is not cryptographically
guaranteed against an actor rewriting both data and checksums.

`analysis/report.json` contains world-cluster bootstrap intervals with paired
condition/depth/budget draws; binomial log-link decay fits and paired contrasts;
pass@5 and pass⁵ only with complete groups of at least five repeats; cost, coverage,
exhaustion, invocation/call counts, false promotions, and intervention denominators.
Saturated or unidentifiable fits and degenerate empirical intervals are unavailable.
With few worlds, even nondegenerate intervals may be unstable. Do not report a
two-world result as precise evidence of a scaling law.

The command writes four PNG/SVG figures (depth reliability, downstream propagation,
repeatability, and ablation reliability-versus-depth), plus a supplemental
`cost_frontier` figure comparing measured cost across allowance bands at each
fixed depth. It never connects different depths as a cost frontier. It also writes
`paired_trace.json`, a real
clean/intervened pair when an intervention was applied. An empty trace file means
no qualifying pair. These are descriptive artifacts unless the sampling and
protocol gates pass; rendering a figure does not establish a research result.

## Gates before interpreting held-out results

1. All variants pass offline context/provenance tests and complete development
   execution without unexplained accounting failures.
2. Development data establishes whether continuous agents actually reach and obey
   emissions; report the compliance rate and incomplete sites. A failed protocol
   prevents a clean live EPC comparison; do not force an apparent containment win.
3. Freeze cost bands, sample size, model/runtime/source, and analysis before test
   access. Report cost frontiers; equal-compute H₁ remains untested by this policy.
4. Collect all planned episodes and at least five repeats; retain negative and
   inconclusive outcomes. Low power, saturation, and poor coverage are findings,
   not reasons to invent a slope or remove failures.
5. Update the research note only with qualifying data. GEPA/Flex remain deferred;
   they cannot repair a confounded fixed-architecture comparison retrospectively.
