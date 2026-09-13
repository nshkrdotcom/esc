# Public corpus and answer contracts

New runs record `corpus_interface=json_documents_v2` and
`answer_contract=nullable_answer_v2` in their manifest. Earlier artifacts keep
their original interface, prompts, and scoring; do not pool versions as one study.

## Corpus and provenance

Every condition receives a JSON list of documents. Each object contains
`source_id`, `title`, and the unchanged `content` string. Decode the outer JSON,
search each document's content, and retain that document's source ID when citing
a span. In relational tasks the content itself contains JSON rows; these are
parsed after decoding the enclosing document. Cite the decoded original row,
not the escaped representation in the outer JSON.

The same serializer supplies A/B's full corpus and C's projection. For
`relational_v2`, each permitted-source list still contains the entire corpus in
the same order. No answer-dependent filtering, labels, or canonical solver are
added. The witness still rejects a correct row attributed to the wrong document.
This addresses header-parsing fragility without relaxing provenance requirements.

The change affects both benchmark families. Public-input tests solve all 16
relational transitions with evaluator labels poisoned, verify exact corpus
equality across conditions, and exercise the unchanged witness. This establishes
interface integrity, not that the model will use it correctly.

## Answers and voting

Continuous output is `final_answer: str | None`. A model that cannot answer must
return typed null. A null or blank answer is a charged abstention, consumes its
rollout slot, and contributes no B vote. A missing field or wrong type is a
measured model-output error. B still votes over completed nonempty answers, with
first-seen tie breaking; no label enters selection and no free repair is added.

The strings `None` and `null` are not silently converted: they remain literal
answers, so legitimate categorical values are not erased by a global heuristic.
The signature explicitly instructs the model to use typed null for abstention.
Actual adherence must be checked on development worlds. Old literal-string votes
are never retrospectively rescored. C already has typed status/value output and
retains its existing supported/insufficient/ambiguous/contradicted contract.

## Cost implications and next gate

The live compatibility run `outputs/interface_v2_dev_001` used dev seed 43,
depth 2, one repetition and the unchanged 24,000 soft allowance. All three episodes
completed and passed accounting replay: A 15,655 tokens / incorrect, B 25,538 /
correct with exhaustion, C 10,589 / abstained. Total cost was 51,782 tokens; no
model-output errors or unknown usage occurred. The largest/smallest episode cost
ratio was 2.41. See the [saved evidence](evidence/interface_v2_dev_001.json).

C initially accessed row fields on the document wrapper, then searched only the
first document's content and returned insufficient. Thus the serializer works,
but model adherence remains inadequate in this observation. This new-world check
does not establish improvement over the earlier interface. Do not repeatedly tune
against this one world until it passes and present that as validation. Next assess
the common interface on disjoint development worlds, explicitly report how many
documents/hops were reached, and distinguish parsing/search failures from witness
rejections. Any stronger common parsing helper is an additional intervention that
must be supplied equally to A/B/C and versioned before comparison.

Serialization and signature changes alter prompt cost. Compare versions using
their recorded actual usage; improvement on a development task is not a causal
architecture result. `esc audit-budget` now reports per-depth mean tokens for
each condition, the largest/smallest mean ratio, and maximum episode overshoot.
These are descriptive diagnostics, not an automatic matching threshold.

The policy still reserves generation only and `compute_matched` remains false.
Next, freeze a defensible allocation and quality-versus-cost analysis on additional
development worlds. If a hard total-token bound is required, first validate a
prompt reservation mechanism against the exact served template, including
concurrent calls; do not infer a bound from a few small overshoots. Live
interventions and faithful ablations remain required before a held-out study.
