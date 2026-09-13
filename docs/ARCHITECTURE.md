# ESC architecture

ESC tests whether separating persistent epistemic state from transient reasoning contexts can improve reliability. State contamination is a hypothesis, not an established finding of this repository.

## Default isolated execution

1. `StateKernel.project(step)` returns copies of required facts whose assurance satisfies the step's input contract.
2. Missing inputs block the step; they are not counted as consumed invalid state.
3. The corpus compiler supplies only permitted source documents.
4. `EpistemicWorker` invokes DSPy RLM, whose default interpreter factory creates a fresh interpreter per invocation. The Python worker object is reusable; interpreter variables and RLM history are not shared across invocations.
5. The worker returns a validated `StepResult`. Malformed model outputs become flagged unsuccessful attempts with known usage; backend/accounting failures still raise and stop the batch.
6. Supported, nonempty outputs with no unresolved assumptions pass to a witness. Rejected, ambiguous or insufficient outputs halt the default pipeline.
7. A successful promotion stores a copy of the fact in canonical state. The audit record contains the step output, witness decision, usage and available trajectory. Audit data is saved but is not supplied to the next default invocation.

The normal kernel projection includes only facts in `requires`, preserving their order for arithmetic operations. Both insertion and projection copy objects so a worker cannot mutate canonical facts by reference.

## Witness scope

| Class | Implemented check | Maximum assurance |
|---|---|---|
| Type I | Authoritative relational row lookup; ordered scalar arithmetic, explicit greater-than comparisons, exact approval/compliance transitions, exact quoted value | verified |
| Type II | Citations in allowed sources with spans containing the claimed value | supported |
| Type III | No sealed adjudicator | candidate |

Type II is a conservative extractive provenance check, not proof of entailment or multi-source corroboration. Type I text checks verify the quote, not arbitrary semantic claims about it. Code execution as a claim verifier is not implemented. No live witness reads the hidden target answer.

Disputed facts never satisfy a contract. Otherwise assurance orders candidate < supported < verified. The current commit API also requires the new fact to meet `step.required_level`; separating input and output assurance policies is future work.

## Continuous and search-heavy systems

A receives the question, public step specifications and full corpus. B repeats A up to three times by default and selects by majority vote, with stable first-seen tie-breaking. An exhausted episode votes only over completed rollouts. It cannot inspect evaluator correctness when selecting an answer. C receives compiler-selected facts. In `relational_v2` every C step receives the same full corpus as A/B; source filtering is restricted to the legacy family. The public decomposition is available to all conditions.

All live calls share the task LM. A and each B rollout use up to 4 iterations and 4 recursive calls per invocation; C uses the same limits per step. The CLI can override these limits for all workers together, and records them in the manifest. A final extraction call may follow iteration exhaustion. These are not equal-token budgets. The provider usage tracker records prompt/completion tokens and total LM call count, including recursive calls. Root/subcall counts are not separately labeled.

## Failures and persistence

RLM initialization cannot silently fall back to another architecture. Backend/accounting errors stop execution and produce a runner failure record. Optional soft-budget exhaustion instead records an episode outcome and continues the sweep; B can retain completed votes and C preserves committed state and audit records. Completed episodes are persisted individually. Available RLM trajectories are audit artifacts; they do not become canonical reasoning history. A failed in-progress episode is not recoverable from the current runner. See [episode budget semantics](EPISODE_BUDGET.md), including prompt overshoot and telemetry limitations.

See [the review](PREFLIGHT_REVIEW.md) for unresolved experimental controls and [ablation status](ABLATIONS.md) for non-default behavior.
