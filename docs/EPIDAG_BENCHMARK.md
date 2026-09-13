# EpiDAG prototype

The generator builds synthetic company profiles, founder biographies, earnings reports and simple distractors. It keeps true values in evaluator fields. Live A/B receive the question, public step specifications, and corpus; C receives each public step's goal, permitted sources and projected canonical facts.

## Task sizes and actual depths

| `--depths` selection (nominal node count) | Measured dependency depth | Final task |
|---|---|---|
| 2 | 2 | Revenue exceeds 150: true/false |
| 4 | 4 | Target revenue divided by Alpha revenue |
| 8 | 7 | Ratio exceeds a sampled threshold: approved/rejected |
| 16 | 15 | Eight further rule-based compliance transitions |

Depth is computed from `requires`, not node count or stored display labels. Ratios return four decimal places. Threshold generation includes positive and negative answers. Compliance rules and exact required output strings are public instructions.

The eight-node graph has edges N1→N2, N1/N2→N3, N3→N4, N4/N5→N6, N6→N7, N7→N8; N5 is an independent revenue lookup. The longest path contains seven nodes.

## Important limits

Some prerequisites are procedurally enforced but not necessary to infer the answer. Source names reveal document roles. Distractors are obvious. The sixteen-node case extends a decision through repetitive string transformations. Task size changes answer type and difficulty as well as graph depth. This generator is suitable for pipeline checks, not yet a controlled horizon-scaling benchmark.

Type II witnesses verify allowed source spans and extractive claim presence, not semantic entailment. Type I witnesses recompute ordered arithmetic and explicit threshold/decision rules from accepted state. Neither reads hidden answers. Mock workers intentionally read hidden answers and cannot supply evidence for an architectural advantage.

## Error injection

Clean and injected suites use distinct task IDs; using the same seed pairs their corpora. Node-level injected values remain a harness facility. C can replace a candidate before promotion; live A/B reject injected tasks because no equivalent within-trajectory intervention exists. The default runner executes only clean tasks, so EPC is N/A. Distance-conditioned live EPCₖ remains unimplemented.
