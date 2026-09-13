# Request journal and offline budget audit

Budgeted runs with an output directory now write `requests.jsonl` independently
of DSPy callback history. Enable the existing `--episode-token-budget N` option;
the manifest records `request_journal_version: 1`. Unmetered and mock runs do not
create this journal. The runner refuses to reuse an existing journal file.

Each record includes task, condition, outer repetition and a per-episode sequence
number. The ledger emits records under its lock, preserving reservation/settlement
order across recursive threads. Writes flush and fsync. A write failure marks the
accounting invalid and stops execution; a failed start-record write prevents the
wrapper from entering the backend call.

| Event | Meaning | Recorded information |
|---|---|---|
| `request_start` | A generation reservation succeeded immediately before calling the DSPy backend | Request ID, model, prompt/messages input mode, requested and capped generation |
| `request_end`, measured | The backend returned usable token counts | Same request ID, prompt/completion usage, finish reasons, budget truncation and exhaustion state |
| `request_end`, unknown_usage | Backend failure or unusable response metadata | Same request ID, reserved generation and error type; no fabricated measured cost |
| `request_blocked` | The ledger rejected a new request | Budget-exhausted or accounting-failed reason; no backend invocation follows |

`request_start` records the application's backend boundary, not proof that the
server accepted a network request. Transport failures may have unknown remote
consumption. Successful responses are recorded before exhaustion is raised, so
their usage survives even when DSPy never adds them to LM history. Request IDs
are scoped to the episode; never join records by request ID alone.

The journal has no model reasoning text, prompts, API keys, or evaluator labels.
It is separate from `events.jsonl`, which retains diagnostic model/interpreter
outputs. Input mode is descriptive and is **not** a root/subcall classification.
Neither output changes the public context supplied to the model.

## Offline audit

```bash
uv run esc audit-budget outputs/budget_journal_binding_001
uv run esc audit-budget outputs/budget_journal_generous_001
```

The command makes no model calls and writes no files. It rejects failed or
incomplete batches, missing/duplicate episodes, unsupported journal versions,
sequence gaps, unmatched/duplicate responses, invalid reservations, dispatch after
exhaustion, unknown consumption, provider generation-cap violations, and journal /
ledger / run-total mismatches. It checks episode membership against saved tasks
and repetitions and checks the summary configuration against the manifest.

This is independent arithmetic replay of the request records, not independent
measurement of server/GPU work. The journal and ledger use the same provider usage
reports; a provider misreport can still affect both. Auditing old directories
without a journal intentionally fails: historical accounting is not backfilled or
invented. Accuracy labels and statistical significance are outside this auditor's
scope. `valid: true` certifies this accounting check only.
The report also includes `cost_by_depth`: mean tokens per condition, the
largest/smallest mean ratio (null if the minimum is zero), and maximum episode
overshoot. `compute_matched` is always false for this soft policy; a ratio close
to one is not itself a preregistered matching criterion.

## Validation scope

Tests exercise sync/async requests, concurrent responses, a real Deno/Pyodide
batched recursive call, response rejection after actual spend, corrupted records,
and disk-write failure before dispatch. Live binding and generous checks are
recorded in [M2_VALIDATION.md](M2_VALIDATION.md).

This closes the missing request-attribution mechanism. The current soft policy
still reserves generation only; prompt costs may overshoot without a guaranteed
bound. See [EPISODE_BUDGET.md](EPISODE_BUDGET.md). A journal does not turn that
policy into matched compute or complete the scientific study.
