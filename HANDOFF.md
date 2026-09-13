# Handoff — paused after initial M2 implementation

## Start here

The shared **soft** episode allowance is implemented and offline-tested. Experiment
1 is still a development pipeline, not a valid H₁ study. Do not start a large
sweep, GEPA, or publish comparative claims yet. No experiment is running at this
handoff. The next action is the small binding-budget check below, when the owner
is ready to spend local inference time.

Read [budget semantics](docs/EPISODE_BUDGET.md), then the
[research gates](docs/EXPERIMENT_1_PLAN.md). [RESEARCH_NOTE.md](RESEARCH_NOTE.md)
is the external-facing entry point; its headline results intentionally remain empty.

## Thesis and intended contribution

**Can stochastic reasoning be made composable without first making every local
reasoner reliable?** ESC proposes that persistent epistemic state should be
separate from linguistic reasoning history. Each transition receives a fresh
context compiled from typed accepted facts and permitted evidence; externally
checkable witnesses govern what becomes persistent state. Reasoning trajectories
remain in an audit log rather than becoming the next invocation's assumptions.

The primary hypothesis is that, at equal model, available information and
defensibly controlled inference cost, ESC has a shallower decline in final success
as dependency depth grows than continuous RLM reasoning. In
`log P(success) = alpha - beta * depth`, the intended finding is a smaller beta
for ESC, with uncertainty that supports the difference. A search-heavy baseline
must test whether extra trajectories reproduce any gain. No positive finding is
assumed or required.

The supporting questions are whether local errors stop at state boundaries,
whether repeated-run success improves alongside searchable success, and which
primitive—fresh context, typed state, or witness promotion—causes any effect.
Measure coverage alongside containment: refusing every task is not reliable
composition. Similar slopes, a search baseline matching ESC, or gains explained
by verifier strength are useful negative results.

The budget work serves that thesis by making actual episode cost and exhaustion
visible across architectures. It is infrastructure for a fair comparison, not
evidence for the hypothesis. So far we have working mechanics and narrow live
checks; we have **not** demonstrated improved reliability scaling or fault containment.

## Completed and verified

- M0: local Qwen3-14B Q4_K_M serving template, fresh RLM execution, usage/events,
  diagnostics and persistent per-episode results.
- M1: `relational_v2`, actual relational depths 2/4/8/16, paired worlds, opaque
  source IDs, full corpus access for A/B/C, public-row deterministic witnesses.
- M2 initial implementation: locked context-scoped ledger; synchronous/async LM
  interception; generation reservations; measured prompt reconciliation; shared
  B rollouts/C steps; durable exhaustion after swallowed REPL errors; partial B
  votes and C audit preservation; runner continues exhausted episodes.
- `ESC_TEST_RLM=1 uv run pytest -q`: **94 passed**. Includes an actual scripted
  Deno/Pyodide recursive-call exhaustion test; no inference needed for these tests.
- Nonbinding live check `outputs/budget_generous_001`: one development depth-2
  world, one outer repeat, allowance 100,000 per condition. A/B/C measured
  **16,932 / 47,208 / 4,874 tokens**, with 5/13/2 dispatched calls. A/B correct;
  C incorrect. All ledger reservations settled, no unknown usage or exhaustion.
  This run started before the final reduced-cap truncation handling adjustment;
  that adjustment is offline-tested and was not exercised by this nonbinding run.
- Compact evidence is versioned in
  [docs/evidence/budget_generous_001.json](docs/evidence/budget_generous_001.json).
  Full local output directories are ignored by Git. Do not expect them in a clone.

## Exact next commands

Use `uv` exclusively for Python. The local Ollama model alias is
`esc-qwen3:14b-nothink`; setup is in [RUNTIME_CONFIGURATION.md](docs/RUNTIME_CONFIGURATION.md).
Do not substitute another model/template silently. Deno is required for RLM.

```bash
uv sync --locked
ESC_TEST_RLM=1 uv run pytest -q
ollama list
uv run esc run --benchmark relational_v2 --split dev --depths 2 \
  --tasks-per-depth 1 --repetitions 1 --no-mock \
  --episode-token-budget 1000 --output-dir outputs/budget_binding_001
```

The binding run has **not been performed**. It executes A/B/C, not a single LM
call. Prompt usage can exceed 1,000; that is an expected property of this soft
policy, not proof of hard enforcement. Use a fresh output directory if that name
exists. The runner rejects overwrites and has **no resume command**.

Check `experiment_1_summary.json`, `runs.jsonl`, `events.jsonl`, and any
`failure.json`. Accept only if all three episodes are persisted, at least one
exhausts, no call starts after durable exhaustion (already-dispatched concurrent
requests may finish), measured tokens reconcile, pending/unknown reservations are
zero, and the sweep finishes. Inspect budget-reduced truncation handling. Preserve
failures; do not retry until success and omit the earlier attempt. Add a compact
evidence snapshot and update this handoff after validation.

## Granular remaining roadmap, in order

1. **M2 validation:** run the binding check; compare ledger prompt/completion totals
   with available provider events and the nonbinding worker usage records. Confirm
   root, recursive and extraction paths are counted once. Diagnose discrepancies
   before further runs. Backend/unknown-usage failures invalidate a batch, while
   exhaustion is a recorded episode outcome. B may retain a correct completed vote.
2. **M2 study policy:** choose and freeze an acceptable cost allocation policy on
   development worlds. Current policy reserves generation only; prompt overshoot
   has no guaranteed bound. Investigate a validated exact-template tokenizer or
   conservative reservation if hard total-token enforcement is required. Version
   any changed policy, test concurrency again, and document actual versus allowed
   cost. Keep `compute_matched=false` until a defensible matching criterion exists.
3. **M2 calibration:** inspect small development runs across depths, costs,
   saturation, refusal/truncation and call counts. Select budget bands without
   held-out labels. Per-invocation iteration caps differ in aggregate for A/C;
   report this second resource axis rather than claiming call parity.
4. **M3 intervention plumbing:** emit intermediate results inside A/B's persistent
   RLM, preserving its history; instrument C's candidate boundary comparably.
   Test that instrumentation does not give A C's state projection/verification.
   Retain an uninstrumented baseline to measure instrumentation effects.
5. **M3 measurements:** paired clean/intervened worlds with a seeded valid alternate
   entity. Separate candidate corruption from accepted-state corruption. Record
   attempted/reached/applied interventions, downstream graph distance, wrong,
   correct, abstained and unreached outcomes. Separate B rollout effects from the
   selected ensemble answer. No hidden labels in model tools or voting.
6. **M4 faithful ablations:** implement isolation-only, shared-history contracts,
   isolation+typing without witnesses, and full ESC alongside A. Capture actual
   inputs in tests. Existing `no_typing` and summary modes are not faithful substitutes.
7. **M5 execution integrity:** freeze model digest/template, code, world splits,
   budget policy and analysis. Add deterministic randomized/interleaved scheduling
   and explicit incomplete-batch handling before a long sweep; do not analyze a
   convenient completed prefix. Resume support, if added, must validate the full
   manifest and avoid duplicated episodes. Choose sample size from dev uncertainty.
8. **M5 analysis:** at least five uncached outer repeats on held-out worlds;
   paired world-cluster uncertainty, appropriate binomial treatment of zero/all
   success cells, identifiable slope contrasts, pass@5/pass⁵ with complete groups,
   actual cost/exhaustion distributions. Current descriptive clipped slopes are
   not inferential results. Report saturation instead of manufacturing a slope.
9. **Results artifact:** generate the four requested figures and an actual paired
   failure trace; fill the research-note table only with qualifying observations.
   Relational pointer-following with strong Type I witnesses is a narrow domain;
   broaden semantic tasks/verifier ablations before general composability claims.
10. **Only after valid fixed-architecture results:** implement D/GEPA with saved
    programs and disjoint train/validation/test worlds; account reflection cost
    separately. Flex boundary search and external-corpus replication follow.

## Working rules for the next agent

Preserve existing artifacts and owner edits. No running background job is expected.
Do not treat the smoke outcomes as architecture evidence, mock outputs as real
results, soft allowances as equal spend, or citation existence as semantic truth.
Keep patches small and tested. This handoff intentionally stops before the pending
live validation; no large experiment is authorized by merely reading this document.
