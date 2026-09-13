# ESC Experiment Runbook

This runbook provides end-to-end operational instructions for executing the Epistemic State Compilation (ESC) experimental campaign across its three planned stages.

---

## 1. Prerequisites & Environment Check

Verify that the local environment and GPU resources are ready:

```bash
# Verify Python virtualenv and packages
uv sync

# Run the test suite
uv run pytest

# Check GPU memory headroom
nvidia-smi

# Check Ollama model availability
ollama list
```

---

## 2. Stage 1: Tiny Proof-of-Concept

Goal: Fast validation of the horizon decay curve ($\beta_C \ll \beta_A$) with minimal token expenditure.

* **Configuration**:
  * Depths: $d \in \{2, 4, 8, 16\}$
  * Tasks per depth: 3
  * Stochastic repetitions: 3 ($k=3$ for $pass@k$ and $pass^k$)
  * Total episodes: $4 \text{ depths} \times 3 \text{ tasks} \times 3 \text{ reps} \times 4 \text{ systems} = 144$ episodes.
  * Expected token budget: ~1–3 million tokens.

### Fast Dry-Run (Mock Mode)
```bash
uv run esc run \
  --depths 2,4,8,16 \
  --tasks-per-depth 3 \
  --repetitions 3 \
  --mock \
  --output-dir outputs/stage_1_mock
```

### Live Local LM Run (Qwen3-14B on 16GB GPU)
```bash
uv run esc run \
  --depths 2,4,8 \
  --tasks-per-depth 2 \
  --repetitions 2 \
  --no-mock \
  --model ollama_chat/qwen3:14b \
  --output-dir outputs/stage_1_qwen3
```

---

## 3. Stage 2: Credible Statistical Study

Goal: Rigorous statistical separation across the multi-dimensional evaluation vector $R = (A, C, E, F, V, T)$.

* **Configuration**:
  * Depths: $d \in \{2, 4, 8, 16\}$
  * Tasks per depth: 25
  * Repetitions: 5
  * Total episodes: ~2,000 task executions.

```bash
uv run esc run \
  --depths 2,4,8,16 \
  --tasks-per-depth 25 \
  --repetitions 5 \
  --no-mock \
  --model ollama_chat/qwen3:14b \
  --output-dir outputs/stage_2_study
```

---

## 4. Stage 3: Error Propagation Measurement ($EPC_k$)

Measures:
$$EPC_k = P(N_{i+k} \text{ wrong} \mid N_i \text{ wrong})$$

The runner automatically generates paired error-injection tasks where an incorrect upstream fact is deliberately planted at $N_i$:
- **Condition A (Continuous)**: Observes rationalization cascade where the error infects all downstream descendants.
- **Condition C (Isolated)**: Observes that promotion gates reject the candidate, causing downstream modules to safely emit `insufficient` without propagating corruption.

The resulting metric is reported under the `Error Prop EPC (E)` column in the output table.

---

## 5. Stage 4: DSPy GEPA Reflective Optimization

To optimize step resolution signatures and boundary contracts using natural language feedback:

```bash
# Run GEPA reflective diagnostic demonstration
uv run esc gepa-demo
```

### Programmatic GEPA Optimization Script:
```python
import dspy
from esc.core.signatures import ResolveStep
from esc.eval.gepa_metric import epistemic_gepa_metric

# 1. Initialize models
task_lm = dspy.LM("ollama_chat/qwen3:14b", api_base="http://localhost:11434")
reflection_lm = dspy.LM("openai/gpt-4o") # or gemini-3.6-flash

dspy.configure(lm=task_lm)

# 2. Define GEPA optimizer
optimizer = dspy.GEPA(
    metric=epistemic_gepa_metric,
    reflection_lm=reflection_lm,
    max_metric_calls=200,
    num_threads=4,
)

# 3. Compile optimized epistemic program
# optimized_worker = optimizer.compile(epistemic_program, trainset=train_dags)
```

---

## 6. Output Artifacts & Analysis

All results are automatically persisted to JSON:
- `outputs/<run_name>/experiment_1_summary.json`

Contains:
1. `overall_eval_vectors`: $(A, C, E, F, V, T)$ per system.
2. `horizon_decays`: $\alpha, \beta, R^2$, and depth-by-depth accuracy points.
3. `h1_hypothesis`: Statistical comparison of $\beta_C$ vs. $\beta_A$ and ratio of decay rates.
