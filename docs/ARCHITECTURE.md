# ESC System Architecture

## 1. Foundational Principle

In standard autonomous agent architectures, long-horizon failure is frequently diagnosed as an "intelligence deficit." ESC tests an alternative formulation:

> **Long-horizon failure in rolling agents is fundamentally state contamination.**

When an LLM operates over a continuous context, every intermediate step produces:
- Verbose chain-of-thought traces
- Tentative conjectures
- Minor numerical discrepancies
- Failed sub-tool explorations

In a monolithic session, these intermediate traces become prompt tokens for all future steps. The LLM naturally treats its own prior generated words as accepted contextual truth, rationalizes early deviations, and amplifies stochastic mistakes across the dependency chain.

ESC resolves this by enforcing two separations:

$$\boxed{\text{Variable space} \neq \text{Token space}} \quad \text{(Recursive Language Models / RLM)}$$

$$\boxed{\text{Epistemic state} \neq \text{Reasoning context}} \quad \text{(Epistemic State Compilation / ESC)}$$

---

## 2. The Three Architectural Pillars

```
+-----------------------------------------------------------------------------+
|                               StateKernel                                   |
|   Canonical State: dict[str, Fact]          Audit Log: list[AuditEntry]     |
|   - N1_alpha_rev: 170M (Supported)         - RLM Trajectories (Discarded)   |
|   - N2_owner: Marcus Chen (Supported)       - Scratchpads & Code REPL traces|
|   - N4_ratio: 3.0882 (Verified)            - Diagnostic Error Logs          |
+----------------------+------------------------------------------------------+
                       |
               project(step) [Context Compiler]
                       |
                       v
         +----------------------------+
         |      Step Spec Contract    |
         |  goal: calculate ratio     |
         |  requires: [N1, N3]        |
         |  required_level: Supported |
         +-------------+--------------+
                       |
                       v
         +----------------------------+
         |     EpistemicWorker        |
         |  (One RLM = One Process)   |
         |  - Fresh REPL interpreter  |
         |  - Max 8 iters / 12 calls  |
         +-------------+--------------+
                       |
                       v
         +----------------------------+
         |     Candidate Output       |
         |  value: "3.0882"           |
         |  status: "supported"       |
         +-------------+--------------+
                       |
                       v
         +----------------------------+
         |    evaluate_witness()      |
         |  Type I: Recompute Math    |
         |  Type II: Corpus Provenance|
         |  Type III: Semantic        |
         +-------------+--------------+
                       |
              commit(candidate, witness)
                       |
                       v
             [Canonical State Update]
```

### Pillar I: Disposable Cognition (RLM Invocations)
Every epistemic step executes in an independent, newly initialized worker instance. 
- The worker receives only the certified inputs required by the current step contract.
- Trajectory scratchpads, REPL variables, and reasoning strings are recorded in the audit log and immediately discarded.
- No conversational tokens bleed from step $i$ into step $i+1$.

### Pillar II: The StateKernel Context Compiler
The language model never decides what prior material enters its context. The StateKernel compiles the context deterministically:
```python
def project(self, step: StepSpec) -> list[Fact]:
    return [
        self.facts[k]
        for k in step.requires
        if k in self.facts and level_ok(self.facts[k].level, step.required_level)
    ]
```
If an upstream dependency was not committed or failed to reach the contract's required assurance level, `project()` omits it. The worker immediately returns `insufficient`, cleanly halting error propagation.

### Pillar III: Non-Oracle Witness Promotion Hierarchy
To avoid circular evaluations where the running system queries the ground-truth benchmark, ESC defines three realistic promotion classes:

1. **Type I — Deterministic Witness (`Verified`)**:
   - Arithmetic results: Deterministically recomputed from accepted inputs.
   - Code claims: Executed against unit tests.
   - Threshold checks & Boolean conditions: Deterministically validated.
   - Quoted text spans: Verified to match source documents character-by-character.
2. **Type II — Externally Grounded Witness (`Supported`)**:
   - Multi-source provenance checks.
   - Exact text span existence within permitted corpus documents.
3. **Type III — Purely Semantic Inference (`Candidate`)**:
   - No deterministic witness exists.
   - Remains in `Candidate` status unless corroborated by sealed multi-agent adjudication.

---

## 3. Epistemic Assurance Ordering

Assurance levels follow a strict lattice:

$$\text{disputed} (-1) < \text{candidate} (0) < \text{supported} (1) < \text{verified} (2)$$

A step requesting `required_level="verified"` cannot consume a `candidate` or `supported` fact, guaranteeing that strict boundaries are maintained across cognitive stages.
