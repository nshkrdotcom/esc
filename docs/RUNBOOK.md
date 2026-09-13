# ESC pilot runbook

Read [the pre-experiment review](PREFLIGHT_REVIEW.md) before running. The current implementation supports a pipeline pilot across A/B/C. Matched-compute inference, live EPCₖ comparisons, GEPA compilation, and the full ablation study remain unfinished.

## Offline preparation

Use the locked environment and test without invoking a model:

```bash
uv sync --locked
uv run pytest
uv run esc --help
uv run esc bench --depth 4

# Optional real-interpreter test with scripted responses (no model inference)
ESC_TEST_RLM=1 uv run pytest tests/test_rlm_integration.py
```

DSPy's RLM requires a working Deno/Pyodide interpreter. `deno --version` should resolve a Deno 2.x executable; if a compatible system runtime is unavailable, install the DSPy Deno extra using `uv add 'dspy[deno]==3.3.1'`. RLM initialization or runtime failure now stops the run instead of silently switching architectures. The interpreter may need to populate its runtime cache on first use.

Check the local server/model without sending an inference request:

```bash
ollama list
nvidia-smi
```

The default model name is `ollama_chat/qwen3:14b`. If only the Hugging Face name is installed, pass `--model ollama_chat/hf.co/Qwen/Qwen3-14B-GGUF:Q4_K_M` instead. Model weights, KV cache, runtime buffers, and desktop GPU usage share VRAM. The Python REPL runs outside GPU memory; no fixed amount of free VRAM is guaranteed.

## Future live pilot

This command is documented for the user to run after review; it was not executed during the review:

```bash
uv run esc run \
  --depths 2,4,8 \
  --tasks-per-depth 2 \
  --repetitions 2 \
  --no-mock \
  --model ollama_chat/qwen3:14b \
  --temperature 0.6 \
  --max-tokens 2048 \
  --num-ctx 8192 \
  --seed 42 \
  --output-dir outputs/pilot_qwen3_001
```

This requests 36 episodes: 3 task sizes × 2 tasks × 2 repetitions × 3 conditions. B performs three continuous rollouts within each of its episodes. Output token limits apply per LM call; RLM can make multiple root and recursive calls. Total episode tokens are measured, not matched or capped. The local default disables Qwen thinking; this is an explicit pilot configuration, not a model-quality comparison. Reducing context may truncate useful information; increasing it needs a fresh VRAM check.

A uses up to 16 REPL iterations and 24 recursive calls; C uses up to 8 iterations and 12 recursive calls per step. These counts do not include a possible final extraction call and do not ensure equal compute. Sampling caches are disabled so repeated calls reach the backend.

For an offline harness exercise, replace `--no-mock` with `--mock` and select a different output directory. Mock workers use hidden labels and scripted failure rules; their accuracy is not experimental evidence.

## Artifacts and failures

Each output directory belongs to one run. Existing run artifacts cause an error rather than being overwritten.

- `manifest.json`: model, safe sampling settings, seed, versions, configuration and limitations.
- `tasks.json`: reproducible corpus, public task instructions and evaluator labels. Only public inputs reach live workers.
- `runs.jsonl`: one flushed record per completed episode, repetition ID, measured usage, and available audit/trajectory data. B stores its constituent rollout records.
- `experiment_1_summary.json`: overall and per-depth vectors and descriptive fits, written on successful completion.
- `failure.json`: task, condition, repetition, and exception if an episode fails. The exception is re-raised and the run stops.

There is no automatic resume. Earlier completed episodes remain readable after failure; use a new directory for another run. No summary means the run did not finish. Unmeasured/inapplicable metrics display N/A. H₁ is explicitly untested.

`--depths` retains the original CLI name but selects nominal task sizes; measured dependency depths are 2, 4, 7, and 15. See [EpiDAG limitations](EPIDAG_BENCHMARK.md).

## Deferred stages

`uv run esc gepa-demo` only demonstrates textual metric feedback. It does not optimize a worker. `--reflection-model`, live Condition D, and `llm_verifier` are rejected until implemented. Do not launch a large study using the old runbook's cost estimates or treat the pilot's slopes as a hypothesis test.
