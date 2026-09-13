# M2 development validation

These are execution and accounting checks on development worlds, not estimates
of comparative reliability or evidence for H₁. The task model remains local
Qwen3-14B Q4_K_M through `ollama_chat/esc-qwen3:14b-nothink`; per-call output/context
limits remain 1,024/8,192, with four RLM iterations and four subcalls per invocation.

## Completed initial checks

| Local run | Allowance per episode | A tokens | B tokens | C tokens | Exhausted episodes |
|---|---:|---:|---:|---:|---:|
| `budget_binding_001` (before journal) | 1,000 | 2,455 | 2,535 | 2,348 | 3/3 |
| `budget_journal_binding_001` | 1,000 | 2,419 | 2,356 | 2,300 | 3/3 |
| `budget_journal_generous_001` | 100,000 | 8,128 | 45,944 | 7,650 | 0/3 |

Each run uses one depth-2 development world and one outer repetition across A/B/C.
All runs completed and persisted all three episodes with zero pending/unknown
reservations. Both journaled runs passed `esc audit-budget`, reconciling 7,075
and 61,722 measured tokens respectively. Full artifacts remain local in ignored
`outputs/`; compact evidence snapshots are in `docs/evidence/`.

The binding runs test refusal to continue spending, not successful task solving:
all three conditions exhausted and abstained. In the generous journaled run, A
was correct, B was incorrect, and C abstained. B had one correct rollout and two
literal `None` responses that won its vote. C proposed the correct first entity
with a citation to the wrong document, which the witness rejected. Neither is
a paired intervention or evidence of improved fault containment.

## Cost-policy decision

The 1,000-token check overshot by 130–142% in the journaled run because the first
prompt alone cost about 2,000 tokens. This directly rules out treating the present
policy as an enforced 1,000-total-token cap. Preserve
`soft_generation_reservation_v1` as a development policy, keep
`compute_matched=false`, and report actual cost as well as nominal allowance.

The calibration uses 24,000 tokens: enough for several measured root calls
from these runtime checks, chosen before observing the new depth sweep. It uses
one paired development world across depths 2/4/8/16, one outer repetition, and all
three conditions (12 episodes). This inspects overshoot, failure types and
feasibility; it cannot identify a reliable slope or settle a matching tolerance.

The first attempt, `budget_depth_dev_001`, aborted at B/depth 4 after four completed
episodes. The LM returned `Code:` rather than the required field marker; all
11,863 tokens in that interrupted episode were measured and its request journal
was retained. There was no transport failure. This batch has no complete summary
and must not be used as a selected prefix of successful observations.

That failure exposed the need to distinguish malformed model output from backend
failure. `record_failed_attempt_v1` now records `model_output_error=true`, counts
the known tokens, and continues the sweep. A records an unsuccessful result; B
counts the failed rollout against its configured slots and votes over valid
answers from other completed rollouts; C preserves previous facts and records the
failed step without promoting it. No repair call or free retry is added. A new
directory, `budget_depth_dev_002`, records the changed policy explicitly.

The second attempt was interrupted after 10 of 12 episodes. Its process ended
without a summary; the final B/depth-16 episode has an unfinished request in the
journal. Preserve it as an incomplete batch, not a completed calibration. Its
[snapshot](evidence/budget_depth_dev_002_interrupted.json) records the completed
prefix solely for diagnosis. There is no resume support. The fresh full attempt
uses `budget_depth_dev_003` with the same configuration and failure policy.

## Completed four-depth calibration

`budget_depth_dev_003` completed all 12 episodes and passed `esc audit-budget`:
200,393 measured tokens, four exhausted episodes, one episode containing a model
output error, and zero pending or unknown reservations. Its
[versioned snapshot](evidence/budget_depth_dev_003.json) includes the manifest,
episode accounting, and all request-journal records.

| Depth | A tokens / correct | B tokens / correct | C tokens / correct |
|---|---|---|---|
| 2 | 11,803 / yes | 28,581 / yes | 4,887 / no |
| 4 | 15,086 / yes | 24,740 / no | 7,607 / no |
| 8 | 11,787 / yes | 27,431 / no | 8,159 / no |
| 16 | 12,963 / yes | 24,751 / no | 22,598 / no |

All four B episodes exhausted the 24,000-token soft allowance. Overshoot ranged
from 740 to 4,581 tokens (3.1–19.1%); this is observed overshoot, not a guaranteed
bound. B/depth 2 retained a correct completed vote. B/depth 8 recorded a malformed
rollout and the batch continued, exercising the new failure policy live.

C abstained at every depth. At depths 2/4/8 it proposed the correct first-hop
entity with the wrong source ID, so the witness rejected it. At depth 16 it
committed the first hop and returned insufficient at the second. A solved all
four tasks. These observations show early-exit and ceiling behavior on one world;
they cannot estimate comparative reliability, a decay slope, or repeated-run
consistency. They provide no evidence of an ESC advantage. Do not use the CLI's
descriptive clipped slopes as inferential results.

The immediate development priorities are explicit refusal/voting semantics,
public provenance handling, and a defensible cost-comparison policy. Preserve
the strict public-row witness and equal source access; do not make C pass by
accepting a citation to a document that lacks the row. Any prompt/interface change
needs a new versioned development run, not retrospective rescoring.

Before a held-out study, choose a validated prompt reservation/total-budget policy
or explicitly reformulate the primary analysis as quality versus measured cost.
No silent trimming of over-budget episodes, padding, or post-hoc removal of
unsuccessful runs is acceptable. A soft allowance and an independent journal do
not remove unequal realized-cost confounding by themselves.

## Remaining implementation implications

- Root/subcall/extraction categories and model digest/template identity should be
  made explicit in the frozen study manifest; input-mode heuristics are insufficient.
- Continuous output currently treats the literal string `None` as an incorrect
  answer rather than typed abstention. Define the refusal/voting contract before
  the study, apply it consistently, and preserve current runs under their original
  semantics. Do not silently rescore the observed majority vote.
- Correct values with wrong provenance can reduce C's coverage. Keep that cost
  visible; a rejected true claim is not evidence that incorrect claims were contained.
- Comparable live interventions, faithful ablations, held-out repetitions, and
  uncertainty remain later gates in [EXPERIMENT_1_PLAN.md](EXPERIMENT_1_PLAN.md).
