# Shared episode allowance: initial M2 implementation

Use `--episode-token-budget N --no-mock` to enable policy
`soft_generation_reservation_v1`. The default remains unmetered. This is a
**soft allowance, not a hard total-token cap or a compute-matched study**.

Each task/condition/outer repetition owns a fresh locked ledger in DSPy context.
All B rollouts and C steps share it, including recursive threads, extraction,
and synchronous/asynchronous calls through BudgetedLM. Callbacks only observe.
Caching and provider retries are rejected when budgeting is active.

Before dispatch, reserve up to the remaining allowance for generation and pass
the reduced maximum to the backend. After dispatch, reconcile provider-reported
prompt plus completion tokens. Prompt length is **not estimated**: the custom
Ollama template has no validated preflight tokenizer in this implementation.
Therefore prompt work and concurrent in-flight requests can overshoot. There is
no claimed `num_ctx` overshoot bound. Reservations prevent concurrent generation
allocations from spending the same remaining allowance; they do not bound total
prompt work. No tokenizer calibration has been performed.

An exceeded allowance or denied reservation durably exhausts the episode.
Already-dispatched calls finish and are charged; further calls are blocked.
Worker boundaries inspect ledger state even if the REPL swallowed an exception.
Outputs from an exhausted worker are discarded. A length-truncated response
whose generation cap was reduced by the ledger also marks exhaustion.
Exactly spending the allowance can still return a completed answer; requesting
another call then exhausts it. No arbitrary minimum generation floor is used.

B consumes greedily, up to its configured rollout count, voting only over
completed rollouts. Ties retain the existing first-seen rule. Exhaustion can
coexist with a correct B answer from earlier completed rollouts. With zero
completed answers B abstains. C preserves committed facts and the audit of its
interrupted step, but cannot promote that step's result. A has no partial vote.
Budget exhaustion is recorded and the sweep continues. Secondary per-call and
per-invocation iteration limits remain unchanged and are not equal aggregate
call limits across architectures.

Ledger snapshots record measured prompt/completion tokens, overshoot, pending
reservations, unknown reserved consumption, dispatched/blocked calls, and terminal
flags. Runner token totals come from the ledger, including interrupted work.
Unknown consumption after a request failure is **not** represented as measured
tokens or refunded. Backend/accounting failures still invalidate the batch and
write `failure.json` with its ledger. Ordinary malformed output and non-budget
truncation retain existing failure behavior; they are not automatically reclassified
as exhaustion. Raw events retain available provider output and finish reasons.

Example development check (choose an unused output directory):

```bash
uv run esc run --benchmark relational_v2 --split dev --depths 2 \
  --tasks-per-depth 1 --repetitions 1 --no-mock \
  --episode-token-budget 100000 --output-dir outputs/budget_dev
```

M2 remains partial: validate these mechanics in development, then establish an
acceptable allocation/overshoot policy before a controlled study. A conservative
prompt reservation or validated tokenizer would be a separate policy version.
Keep `compute_matched=false`; report allowance and actual costs separately. Live
EPC, faithful ablations, held-out repetitions, and uncertainty remain future gates.
