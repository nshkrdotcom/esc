# ESC Ablation Studies

The central claim of ESC is that long-horizon degradation is driven by state contamination, which can be mitigated through strict epistemic boundaries. To disentangle which primitives contribute to this effect, ESC defines seven controlled ablations.

---

## The Seven Core Ablations

| # | Ablation Condition | Implementation in ESC | What It Isolates |
|---|:---|:---|:---|
| **1** | **Fresh contexts, no typing** | `ablation_mode="no_typing"` | Tests whether disposable contexts alone prevent degradation without contractual type boundaries. |
| **2** | **Typing, shared history** | `ablation_mode="shared_history"` | Enforces typed contracts, but appends prior step trajectories into context. Tests if contracts suffice without amnesia. |
| **3** | **Isolation + typing, no verification** | `ablation_mode="no_verification"` | Commits candidate facts immediately without running witness checks. Isolates the value of promotion gates. |
| **4** | **Isolation + deterministic promotion** | Type I deterministic witnesses only | Measures the specific contribution of hard executable witnesses (recomputed math, unit tests). |
| **5** | **Isolation + LLM verifier** | `ablation_mode="llm_verifier"` | Replaces deterministic check with an LLM self-evaluator. Measures regress introduced by probabilistic verification. |
| **6** | **Isolated contexts + raw summaries** | `ablation_mode="raw_summaries"` | Passes natural language textual summaries instead of structured `Fact` objects. Tests if text summaries reintroduce contamination. |
| **7** | **Full ESC System** | `ablation_mode="none"` (Default) | Combines fresh contexts, typed contracts, amnesia, and non-oracle witness validation. |

---

## Scientific Predictions & Interpretations

### Potential Finding 1: $\text{Fresh Contexts Alone} \gg \text{Continuous Context}$
If Ablation 1 drastically reduces horizon decay compared to Condition A, then long-horizon failure in rolling agents is largely an artifact of **distractor token accumulation** rather than model capacity limits.

### Potential Finding 2: $\text{Typed State} > \text{Raw Summaries}$
If the full system significantly outperforms Ablation 6, then traditional "memory summarizers" (which convert conversational history into rolling paragraphs) fail because they reproduce linguistic ambiguity and allow ungrounded assertions to slip through.

### Potential Finding 3: $\text{Deterministic Witnesses} \gg \text{LLM Verifiers}$
If Ablation 4 beats Ablation 5, it confirms that using an LLM to verify an LLM creates circular error reinforcement, whereas software witnesses (Type I) ground the reasoning process in external ground truth.
