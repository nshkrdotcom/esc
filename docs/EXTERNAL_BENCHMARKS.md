# External Benchmark Replication Guide

Following initial verification on the synthetic EpiDAG benchmark, the ESC architecture can be ported to external real-world benchmarks to evaluate external validity.

---

## 1. BrowseComp-Plus (ACL 2026)

### Why BrowseComp-Plus?
Ordinary BrowseComp relies on dynamic web environments where websites change, search results fluctuate, and rate limits hinder reproducibility.

**BrowseComp-Plus** (published at ACL 2026) addresses this by pairing questions with a **fixed, human-verified web corpus**. This provides:
- A standardized, static text corpus ($C$).
- Multi-hop retrieval and reasoning dependencies.
- Perfect disentanglement between agent reasoning and network variance.

### RLM Formulation on BrowseComp-Plus:
```python
import dspy
from esc.core.types import Fact, StepResult
from esc.workers.epistemic import EpistemicWorker

class BrowseCompStep(dspy.Signature):
    """Resolve one retrieval and deduction step over the BrowseComp-Plus corpus."""
    goal: str = dspy.InputField()
    accepted_facts: list[Fact] = dspy.InputField()
    corpus: str = dspy.InputField(desc="Fixed human-verified document corpus")
    
    result: StepResult = dspy.OutputField()
```

### Comparison Suite:
1. **Monolithic RLM**: Entire corpus loaded into external variable space; single continuous trajectory.
2. **Search-Heavy RLM**: Continuous RLM evaluated over $N$ parallel rollouts (Best-of-$N$).
3. **ESC Epistemic RLM**: Step-by-step dependency resolution with canonical fact persistence and amnesia.
4. **GEPA-Optimized ESC**: Epistemic boundaries tuned with reflective error feedback.

---

## 2. GAIA Benchmark (General AI Assistants)

### Why GAIA?
GAIA evaluates multimodal and tool-use capabilities across three distinct difficulty levels:
- **Level 1**: Short horizon (1–3 tool calls).
- **Level 2**: Medium horizon (4–8 tool calls).
- **Level 3**: Long horizon (9+ chained tool executions).

The three difficulty levels map directly onto our depth variable $d$:
$$d_{\text{Level 1}} \approx 2, \quad d_{\text{Level 2}} \approx 6, \quad d_{\text{Level 3}} \ge 10$$

This allows testing whether the horizon decay curve $\log P(\text{success}) = \alpha - \beta d$ observed on EpiDAG replicates on multi-modal tool-use agents.
