# ESC: Epistemic State Compilation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![DSPy 3.3.1](https://img.shields.io/badge/DSPy-3.3.1-orange.svg)](https://github.com/stanfordnlp/dspy)

> **"RLM made context external to cognition. EpiDSPy makes history external to cognition."**

**ESC** (Epistemic State Compilation / EpiDSPy) is a framework designed to test whether stochastic reasoning can be made composable without requiring the underlying language model itself to be perfectly reliable.

Instead of maintaining a continuous reasoning trajectory where unverified assumptions get narrated, rationalized, and inevitably contaminate downstream conclusions, ESC enforces **strict epistemic isolation**:
1. **Disposable Cognition**: Each RLM (Recursive Language Model) invocation is an ephemeral cognitive process (`max_iters=8`, `max_llm_calls=12`).
2. **State Decoupling**: Persistent state consists *only* of canonical, typed, verified facts. Reasoning traces, scratchpads, and failed attempts stay in the audit log and never enter persistent state.
3. **Context Compilation**: The program—not the language model—projects only contractually required, certified facts into each step's prompt (`kernel.project()`).
4. **Non-Oracle Witness Gates**: Candidate claims must clear Type I (deterministic execution/arithmetic), Type II (grounded multi-source provenance), or Type III (semantic adjudication) promotion gates before committing to canonical state.

---

## The Central Hypothesis ($H_1$)

Do isolated epistemic boundaries change long-horizon scaling?

$$
H_1: \quad
\left| \frac{d\log P(\text{success})}{d\,\text{dependency depth}} \right|_{\text{isolated}}
<
\left| \frac{d\log P(\text{success})}{d\,\text{dependency depth}} \right|_{\text{continuous}}
$$

At equal model and inference budget, **isolated typed state should make task success degrade more slowly with dependency depth** ($\beta_C \ll \beta_A$) than continuous-context reasoning.

---

## Hardware & Model Strategy (16GB VRAM Allocation)

Running RLM effectively on a local **16GB GPU** (e.g., RTX 5060 Ti) requires balancing weight footprint against KV cache headroom for REPL round-trips and recursive subcalls:

| Role | Target Model | Format / Quantization | Memory Footprint | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Task LM** (Worker) | **Qwen3-14B (Dense)** | `Q4_K_M` GGUF | **~8.5–9.0 GB** | Outperforms prior 14B coder checkpoints on STEM/reasoning benchmarks while leaving **7GB+ VRAM** strictly reserved for RLM's working REPL context and recursive subcalls. |
| **Reflection LM** (GEPA) | **Qwen3.6-27B** or **Gemini 3.6 Flash** / Frontier API | API or dedicated load | N/A (Cloud) or sequential | GEPA prompt reflection runs far less frequently; benefits from frontier capability without eating continuous VRAM. |

```bash
# Pull Task LM via Ollama from Hugging Face:
ollama pull hf.co/Qwen/Qwen3-14B-GGUF:Q4_K_M
```

---

## The Four Systems Under Study

| Condition | Architecture | Description |
| :--- | :--- | :--- |
| **A — Continuous** | Monolithic RLM | One continuous session receives the whole task; trajectory accumulates across the dependency chain. |
| **B — Search-Heavy** | Continuous + Best-of-$N$ | Continuous RLM with rollouts/retries matching the total token budget of Condition C. Tests: *Spend compute on trajectories vs. spend compute on boundaries.* |
| **C — Isolated** | Epistemic State Kernel | Fresh RLM per transition; state compiler projects typed inputs; Type I/II/III witness gates before commit. |
| **D — Isolated + GEPA** | Optimized Boundaries | Same architecture as C, but signature instructions optimized via DSPy GEPA with reflective failure feedback. |

---

## Multi-Dimensional Evaluation Vector

We evaluate performance across the full vector $R = (A, C, E, F, V, T)$:
- **$A$ (Accuracy)**: Final task correctness.
- **$C$ (Consistency)**: $\text{pass}^k$ (all $k$ runs succeed), capturing deterministic composability.
- **$\text{pass}@k$**: Searchable capability across $k$ attempts.
- **$E$ (Error Propagation)**: $EPC_k = P(N_{i+k} \text{ wrong} \mid N_i \text{ wrong})$ under deliberate error injection.
- **$F$ (False Promotion Rate)**: Rate at which unjustified claims cross an epistemic boundary.
- **$V$ (Abstention Rate)**: Safe refusal when inputs violate contracts (`insufficient`).
- **$T$ (Token Cost)**: Controlled token expenditure per episode.
- **$\beta$ (Horizon Decay Rate)**: Fitted exponential decay $\log P(\text{success}) = \alpha - \beta d$.

---

## Quickstart

### 1. Installation

Using `uv` exclusively:

```bash
git clone https://github.com/nshkrdotcom/esc.git
cd esc
uv sync
```

### 2. Run Test Suite

```bash
uv run pytest
```

### 3. Inspect EpiDAG Benchmark Tasks

```bash
# Inspect hidden synthetic DAG at depth 4
uv run esc bench --depth 4

# Inspect hidden synthetic DAG at depth 8
uv run esc bench --depth 8
```

### 4. Run Experiment 1 (Offline Mock Simulation)

```bash
uv run esc run --depths 2,4,8,16 --tasks-per-depth 4 --repetitions 3 --mock
```

### 5. Run with Local Qwen3-14B or API

```bash
# Using local Ollama with Qwen3-14B:
uv run esc run --depths 2,4,8 --tasks-per-depth 3 --repetitions 3 --no-mock --model ollama_chat/hf.co/Qwen/Qwen3-14B-GGUF:Q4_K_M

# Or test GEPA reflective failure metric:
uv run esc gepa-demo
```

---

## Project Structure

```
esc/
├── README.md
├── pyproject.toml
├── LICENSE
├── docs/
│   ├── ARCHITECTURE.md          # State kernel, context compiler, and witness mechanics
│   ├── EXPERIMENT_SPEC.md       # Full specification of Experiment 1 & Experiment 2
│   └── EPIDAG_BENCHMARK.md      # Synthetic DAG generator and error injection protocol
├── src/esc/
│   ├── core/
│   │   ├── types.py             # EpistemicLevel, Fact, Evidence, StepSpec, StepResult
│   │   ├── kernel.py            # StateKernel context compiler and commit logic
│   │   ├── witness.py           # Type I (deterministic), Type II (grounded), Type III
│   │   └── signatures.py        # ResolveStep and ContinuousSolve DSPy signatures
│   ├── workers/
│   │   ├── epistemic.py         # EpistemicWorker (RLM disposable cognitive process)
│   │   ├── continuous.py        # ContinuousWorker (monolithic context)
│   │   └── mock.py              # Stochastic simulation workers for offline validation
│   ├── benchmark/
│   │   ├── corpus.py            # Synthetic document corpus and context projection
│   │   ├── tasks.py             # EpiDAGTask and EpiDAGNode schemas
│   │   └── generator.py         # Hidden DAG generator (d in {2, 4, 8, 16})
│   ├── systems/
│   │   ├── base.py              # BaseSystem and SystemResult
│   │   ├── system_a.py          # Condition A: Continuous
│   │   ├── system_b.py          # Condition B: Search-Heavy (Best-of-N)
│   │   ├── system_c.py          # Condition C: Isolated (EpiDSPy)
│   │   └── system_d.py          # Condition D: Isolated + GEPA
│   ├── eval/
│   │   ├── metrics.py           # EvalVector, pass@k, pass^k, EPC_k
│   │   ├── decay.py             # Horizon decay regression and H1 hypothesis test
│   │   └── gepa_metric.py       # GEPA natural-language feedback metric
│   ├── runner.py                # Coordinated experiment coordinator
│   └── cli.py                   # Typer CLI (esc run, bench, gepa-demo)
└── tests/
    ├── test_types_and_kernel.py
    ├── test_witnesses.py
    ├── test_epidag_generator.py
    ├── test_systems.py
    └── test_eval_metrics.py
```

---

## License

MIT License © 2026 nshkrdotcom.
