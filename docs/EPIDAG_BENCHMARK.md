# EpiDAG: Synthetic Dependency Benchmark

## 1. Overview

EpiDAG is a controlled synthetic benchmark designed to evaluate composability, dependency depth scaling, and error containment in autonomous language agents.

Unlike static black-box benchmarks, EpiDAG:
- Generates precise dependency DAGs from a hidden knowledge graph.
- Supports depths $d \in \{2, 4, 8, 16\}$.
- Integrates distractor documents with confounding entities, dates, and financial metrics.
- Provides non-oracle witness criteria (Type I arithmetic/comparison, Type II corpus span verification).
- Deliberately injects upstream errors to compute the **Error Propagation Coefficient ($EPC_k$)**.

---

## 2. DAG Topology Across Depths

### Depth 2
- **N1**: Retrieve enterprise revenue from corporate profile (Type II grounded).
- **N2**: Compare revenue against statutory threshold condition (Type I deterministic).

### Depth 4
- **N1**: Retrieve enterprise revenue (Type II grounded).
- **N2**: Identify principal founder/owner (Type II grounded, requires N1).
- **N3**: Retrieve top-line revenue of subsidiary/target enterprise (Type II grounded, requires N2).
- **N4**: Deterministically calculate ratio of target revenue to parent revenue (Type I arithmetic, requires N1, N3).

### Depth 8 (Paper-Sized Topology)
```
N1 (Identify owner) ────────┐
                             ├── N3 (Identify target) ──┐
N2 (Founding year of owner) ─┘                          │
                                                        ├── N6 (Ratio) ── N7 (Threshold) ── N8 (Final Decision)
N4 (Retrieve target 2024 revenue) ──────────────────────┤
N5 (Retrieve Alpha 2024 revenue) ───────────────────────┘
```

### Depth 16
Extends the depth-8 topology with an 8-stage chained auditing and regulatory compliance cascade ($N_9 \dots N_{16}$), testing extreme horizon bounds.

---

## 3. Error Propagation Coefficient ($EPC_k$)

To measure how far stochastic mistakes travel down a reasoning chain:

$$EPC_k = P(N_{i+k} \text{ wrong} \mid N_i \text{ wrong})$$

### Behavior Under Test:
- **Rolling Continuous Agent (Condition A & B)**:
  An injected upstream error ($N_i = \text{wrong}$) enters the reasoning narrative, is rationalized by the model in subsequent steps, and contaminates downstream conclusions ($EPC_k \to 1.0$).
- **Isolated State Kernel (Condition C & D)**:
  An unverified candidate claim fails witness validation, fails to commit to canonical state, and causes downstream consumers to cleanly return `insufficient` without propagating corrupted values ($EPC_k \to 0.0$).
