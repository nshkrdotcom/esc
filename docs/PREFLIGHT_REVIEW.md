# Pre-experiment review

The initial implementation is a pipeline prototype, not yet a controlled test of H₁. No live model experiment was run during this review. Changes are intentionally uncommitted.

## Fixed before the pilot

- RLM construction and backend failures now raise errors. They cannot silently select ChainOfThought or count infrastructure failure as model abstention.
- Live token totals come from DSPy's provider usage tracker across root and recursive calls. Missing usage is an error. Mock token totals are explicitly simulated.
- CLI sampling disables DSPy caching, uses temperature 0.6, and sets a 2,048-token output limit per call. Local Ollama calls request an 8,192-token context and non-thinking mode. These are recorded in the manifest; they are not an episode token budget.
- Search-heavy rollouts retain their mock RNG instead of restarting it for every sample. Live rollouts use uncached sampling. B uses majority vote without evaluator feedback.
- The pilot includes A/B/C only. D's live constructor rejects use until a compiled GEPA program exists; `--reflection-model` is rejected instead of ignored.
- Clean and injected task IDs are distinct. The default runner uses only clean tasks. Injection comparisons and EPC are not silently mixed into clean accuracy.
- Ratio prompts no longer embed operand answers. Owner/target prompts no longer embed hidden upstream entities. A/B receive the same public step specifications and source lists available to C, without ground-truth labels.
- Threshold tasks include both positive and negative outcomes. Compliance steps specify exact output rules to all conditions.
- Reported depth is calculated from dependencies: task sizes 2/4/8/16 currently have depths 2/4/7/15.
- Witnesses reject reversed ratios, arbitrary numeric fragments, unrelated cited values, forbidden sources, unresolved assumptions, and loose approval-word matches. Grounding remains a conservative extractive check, not semantic entailment.
- Canonical state is copied on insertion and projection to prevent workers from mutating prior accepted facts through object references.
- Completed episodes are flushed to `runs.jsonl`, including available trajectories and C's audit entries. Failures produce `failure.json` and stop execution. Existing result directories cannot be silently overwritten.
- Per-depth vectors, model settings, package version, corpus, and task specifications are persisted. Unmeasured EPC and inapplicable false-promotion rates are null, displayed as N/A. Constant accuracy curves produce finite JSON values.
- Numeric answer scoring consistently accepts four-decimal rounding and equivalent formatting. B groups equivalent numeric spellings when voting. GEPA feedback no longer rewards substrings or missing targets.
- DSPy is pinned to the locally verified 3.3.1 release in pyproject.toml and uv.lock.

## Still required for the scientific study

1. **Matched compute:** current RLM iteration/subcall limits and B's three rollouts do not enforce equal total episode tokens. Compare pilot costs descriptively; implement a shared budget policy before testing H₁. Provider retries/transport failures are not fully cost-accounted, and partial failed episodes are excluded from summary metrics.
2. **Real dependency-depth manipulation:** several graph edges are procedural prerequisites rather than necessary semantic dependencies. The last eight nodes repeat a simple approval rule; increasing their count does not establish increasing reasoning difficulty. Varying task sizes also changes task type and answer space. A redesigned benchmark is needed for horizon-scaling conclusions.
3. **Information controls:** public step specifications equalize access to the supplied decomposition, but C still receives compiler-filtered facts and sources. Source IDs are informative, distractors are easy, and ablations are incomplete.
4. **Live error intervention:** A/B cannot yet inject a result inside an RLM trajectory. C has a candidate replacement hook, and mocks exercise containment mechanics. Neither is a measured live EPCₖ comparison; the pilot reports N/A.
5. **Promotion validity:** a source span containing the claim is not proof that the claim answers the question. Type II is only supported, and Type III sealed adjudication is not implemented. This prototype is not an oracle verifier.
6. **GEPA and ablations:** D has no compilation/loading path. `llm_verifier` now raises; other non-default modes are partial prototypes, described in ABLATIONS.md.
7. **Statistical inference:** slopes are descriptive log-linear fits with clipping at zero accuracy, not calibrated significance tests. The runner sets `h1_supported` to null. Mock outputs encode assumed failure behavior and cannot test the hypothesis.
8. **Recovery:** completed episodes survive a failure, but automatic resume and recovery of an in-progress episode are not implemented. Use a new output directory when restarting.

The next authorized live run should be a small pipeline pilot to inspect generated outputs, refusals, measured tokens, and latency. It should not be presented as evidence for H₁.

## Verification completed

- `ESC_TEST_RLM=1 uv run pytest -q`: **61 passed**, including the actual Deno/Pyodide RLM path with scripted DSPy DummyLM responses. No local or cloud model inference was requested.
- `uv sync --locked --offline`: lockfile and installed environment verified.
- `git diff --check`: passed.

The regression tests cover malformed outputs, architectural fallback, source restrictions, arithmetic direction, exact transition rules, protected canonical state, answer leakage, both outcome branches, actual DAG depth, usage aggregation, cache rejection, task-ID separation, unsupported features, result persistence, failure recovery of completed episodes, and CLI rendering/validation.
