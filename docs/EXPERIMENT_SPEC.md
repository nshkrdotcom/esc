# ESC Experiment Specifications

## Experiment 1: Does Epistemic Isolation Change Horizon Scaling?

### 1. The Core Question
Does isolating persistent epistemic state from LLM cognition yield a fundamentally shallower horizon degradation curve than maintaining a rolling, continuous trajectory?

### 2. Hypothesis Formulation
$$H_1: \quad \left| \frac{d\log P(\text{success})}{d\,\text{dependency depth}} \right|_{\text{isolated}} < \left| \frac{d\log P(\text{success})}{d\,\text{dependency depth}} \right|_{\text{continuous}}$$

Model performance as exponential horizon decay:
$$\log P(\text{success}) = \alpha - \beta d$$

The hypothesis is supported if:
$$\beta_C \ll \beta_A \quad \text{at matched compute budget.}$$

### 3. The Experimental Conditions

| Condition | Description | Compute Allocation |
| :--- | :--- | :--- |
| **A — Continuous** | Monolithic RLM over full task | 1 continuous rollout per task |
| **B — Search-Heavy** | Continuous RLM + Best-of-$N$ | $N$ continuous rollouts; matched tokens to Condition C |
| **C — Isolated** | Clean disposable RLM per step + StateKernel | 1 isolated invocation per DAG step + witness checks |
| **D — Isolated + GEPA** | Condition C with GEPA-optimized prompts | Same structure as C, compiled with reflective feedback |

### 4. Controlled Token Budget & Hardware Execution
- **Task LM**: Qwen3-14B (Dense), Q4_K_M GGUF.
- **Hardware**: Single 16GB GPU (RTX 5060 Ti). 8.5GB weights, 7.5GB KV cache headroom.
- **Metrics recorded**:
  - Prefill tokens
  - Generated tokens
  - RLM root calls & subcalls
  - Wall-clock latency (ms)

---

## Experiment 2: Let GEPA Discover the Boundaries (DSPy Flex)

### 1. Motivation
Rather than imposing hand-crafted epistemic boundaries, does a program optimizer spontaneously learn to insert state boundaries as dependency depth increases?

### 2. DSPy Flex Library
Constrained primitive library exposed to GEPA:
```text
Predict
RLM
fresh_context()
project_state()
verify()
promote()
fork()
merge()
abstain()
```

### 3. Objective Function
$$U = \text{robust correctness} - \lambda \cdot \text{compute\_cost}$$

GEPA optimizes the program structure against multi-depth EpiDAG instances without being told that isolation is advantageous.

### 4. Falsification Criteria
The epistemic isolation hypothesis is falsified if:
1. $\beta_C \ge \beta_A$ after compute matching.
2. Condition B matches or exceeds Condition C consistency ($pass^k$) at equal tokens.
3. Condition C requires oracle-level witnesses to outperform Condition A.
4. Flex consistently removes state boundaries in favor of shared continuous contexts under multi-objective optimization.
