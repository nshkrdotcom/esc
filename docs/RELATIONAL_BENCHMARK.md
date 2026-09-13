# Relational EpiDAG v2

`relational_v2` is an opt-in controlled benchmark for the next stage of Experiment 1. It replaces the legacy revenue/approval task family for future controls; it does not retroactively make legacy results valid.

## World construction

A world has 16 opaque entity identifiers and 16 independently sampled permutation relations by default. Its 256 authoritative JSON rows map `(subject, relation)` to `object`. The rows are shuffled, grouped eight per document, assigned opaque source IDs, and document order is shuffled. There is no path document, answer document, or depth encoded in an ID.

Choose one starting entity and an ordered relation sequence per world. A task of depth d asks for the entity reached after the first d relations. The output of transition i is the subject of transition i+1. There are exactly d nodes and longest-path depth d. Since each relation is injective, substituting a different valid intermediate entity changes the final answer at every later distance; errors do not disappear through path merging.

Depth variants use the same world, start, relation ordering, vocabulary, document contents and order. The corpus always contains the full 16 relations, even for depth 2. Thus total available corpus size is held fixed. The number of relevant relations changes with depth; relevant/irrelevant density is not an independent manipulation in this first family. Relations may include self-maps by chance, but no constant identity/approval operation is appended just to increase node count.

## Inputs and witness

All A/B/C conditions receive the same complete corpus. A/B receive the full public task and step specifications; each C invocation receives that step's public goal and the single required parent fact (or the public starting subject for the first step). Every step's permitted source list is the complete corpus, so the compiler does not route C to the answer's document.

`LookupSpec` contains the public relation and either a literal initial subject or an input-fact key. It contains no correct answer, row number, or evidence-source hint. The Type I witness checks the supplied subject/relation/object against authoritative rows and requires an exact full-row citation in its permitted source. Conflicting rows, missing/low-assurance parents, and citations to other subjects/relations fail. Evaluator labels are used only after execution. Regression tests poison all labels and confirm that a public-input reference worker still produces the same final state.

This is a deterministic database-witness family, not semantic verification. The verifier is intentionally powerful for this domain. A/B have all the same rows and can recompute lookups in their REPL. Results would need verifier ablations and semantic-family replication before supporting broader epistemic claims.

## Splits and reproducibility

World seeds are derived from a SHA-256 namespace containing benchmark version, base seed, split and world index. `dev`, `train`, `validation`, and `test` use distinct namespaces. All depths of one world belong to the same split and share a `world_id`. Analysis must cluster on world, not treat depth variants as independent tasks.

Changing the requested depth list or its order does not change generated worlds. `--tasks-per-depth N` means N worlds observed at each depth. `--world-width` changes the answer vocabulary and corpus width for the entire run; freeze it before a held-out study. Each run saves world IDs, source policy, split and corpus hashes with tasks; individual results retain world IDs and split for later paired analysis.

## Commands

```bash
# No model calls: inspect one task, including evaluator labels
uv run esc bench --benchmark relational_v2 --depth 4

# Offline pipeline plumbing only; mock workers read labels
uv run esc run --benchmark relational_v2 --depths 2,4,8,16 \
  --tasks-per-depth 1 --repetitions 1 --mock \
  --output-dir outputs/relational_v2_mock_001

# Small live development compatibility check (not matched-compute study)
uv run esc run --benchmark relational_v2 --split dev --depths 2 \
  --tasks-per-depth 1 --repetitions 1 --no-mock \
  --output-dir outputs/relational_v2_dev_001
```

The CLI/API default remains `legacy` for backward compatibility. Select `relational_v2` explicitly for this family. The runner continues to mark H₁ untested. Legacy split labels other than `dev` are rejected because its generator does not implement split namespaces.

## What this does not establish

The initial live compatibility check completed on one depth-2 development world: A/B correct, C abstained after its generated parser found no rows. Costs were unequal (11,822 / 48,763 / 14,426 tokens). See the [evidence snapshot](evidence/relational_v2_validation_001.json). All 83 tests, including the opt-in interpreter test, passed. A model's unsuccessful episode is retained rather than retried until it succeeds.

An arbitrary relation chain requires dependent data lookups, but not necessarily a stochastic semantic decision at every hop. A continuous RLM can write a deterministic loop. If that baseline saturates, report saturation and design a richer semantic family; do not artificially ban loops to favor per-step LM invocation.

This implementation completes the first controlled-domain benchmark milestone only. Shared episode budgets, equivalent live fault interventions, faithful ablations, and uncertainty-aware held-out analysis remain on the [Experiment 1 plan](EXPERIMENT_1_PLAN.md).
