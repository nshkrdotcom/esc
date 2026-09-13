# External benchmark replication — deferred plan

No external benchmark adapter or run is implemented. This is a planning document,
not a claim about verified dataset availability, publication metadata, or measured
replication results. Complete the fixed-architecture study gates in
[EXPERIMENT_1_PLAN.md](EXPERIMENT_1_PLAN.md) first.

## Purpose

The synthetic relational family tests a narrow, deterministically checkable task.
External replication should ask whether any observed effect survives richer
retrieval, semantic ambiguity, and tool use. It must retain the thesis's controls:
the same model, available information, and defensibly comparable inference cost.

## Candidate families and prerequisites

A fixed-corpus retrieval benchmark, with BrowseComp-Plus as a candidate to evaluate,
could reduce changes in the available evidence between runs. Before choosing it,
verify the primary dataset documentation, exact release/version, license, corpus
access, labels, scoring, and retrieval protocol. A fixed corpus alone does not
eliminate differences in retrieval policy or task difficulty. No publication year
or venue is asserted here without verification.

GAIA is another candidate for external validity after supported tools/modalities
are implemented. Difficulty labels and tool-call counts are not measured dependency
depth. Do not assign numerical depths to its levels or use them directly as the
x-axis of a horizon-decay fit. Either annotate and validate an actual dependency
structure or report difficulty-stratified accuracy without claiming depth scaling.

## Implementation and acceptance steps

1. Freeze a verified dataset version and retrieval/tool protocol, with hashes and
   explicit split ownership. Keep evaluator labels out of runtime tools.
2. Build a reproducible adapter and score a reference fixture before model runs.
   Record unavailable evidence and tool failures; do not silently discard tasks.
3. Give A/B/C equal corpus/tool access and apply the validated episode cost policy.
   Keep decomposition and source filtering visible as controlled variables.
4. Run a small development compatibility check before choosing a held-out sample.
   Use repeated trials, paired comparisons, and reported coverage/cost.
5. Compare final reliability, repeatability, and intervention outcomes only where
   the task structure supports those measurements. Report limitations or negative
   replication results without extrapolating the synthetic slope.
6. Add optimized D only after GEPA exists and training/test separation and reflection
   costs are recorded. Structural boundary discovery remains a later experiment.

No commands are provided because these integrations do not exist yet. The current
executable next step remains in [HANDOFF.md](../HANDOFF.md).
