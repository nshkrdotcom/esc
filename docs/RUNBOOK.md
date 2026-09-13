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

Create the ESC alias from the installed Hugging Face weights:

```bash
ollama create esc-qwen3:14b-nothink -f models/Modelfile.qwen3-nothink
```

The default model is `ollama_chat/esc-qwen3:14b-nothink`. It shares the existing 9GB weight blob. The original imported `qwen3:14b` template generated thinking even when the API requested `think=false`; its output could exhaust the limit before emitting an answer. The ESC template explicitly closes the thinking block before generation. It retains the prior-message thinking field so Ollama 0.33's capability selection uses this template rather than the embedded GGUF template. Keep this model alias separate from the original.

Model weights, KV cache, runtime buffers, and desktop GPU usage share VRAM. The Python REPL runs outside GPU memory; no fixed amount of free VRAM is guaranteed.

## Future live pilot

Use a fresh directory for each run. The original `pilot_qwen3_001` was interrupted and should remain an artifact of the old configuration:

```bash
uv run esc run \
  --depths 2,4,8 \
  --tasks-per-depth 2 \
  --repetitions 2 \
  --no-mock \
  --model ollama_chat/esc-qwen3:14b-nothink \
  --temperature 0.6 \
  --max-tokens 1024 \
  --num-ctx 8192 \
  --max-iters 4 \
  --max-subcalls 4 \
  --request-timeout 120 \
  --seed 42 \
  --output-dir outputs/pilot_qwen3_002
```

This requests 36 episodes: 3 task sizes × 2 tasks × 2 repetitions × 3 conditions. B performs three continuous rollouts within each of its episodes. Output token limits apply per LM call; RLM can make multiple root and recursive calls. Total episode tokens are measured, not matched or capped. The local default disables Qwen thinking; this is an explicit pilot configuration, not a model-quality comparison. Reducing context may truncate useful information; increasing it needs a fresh VRAM check.

A and each B rollout now use at most 4 REPL iterations and 4 recursive calls. C uses those limits per step. One additional extraction call is possible per invocation. `--max-iters`, `--max-subcalls`, `--max-output-chars` (default 4000), and `--n-samples` (default 3) are recorded in the manifest and propagated to every worker. These limits do not ensure equal episode compute. The backend request timeout is 120 seconds; it is not an episode deadline. Automatic adapter retries in JSON format are disabled, and provider retries remain disabled.

For an offline harness exercise, replace `--no-mock` with `--mock` and select a different output directory. Mock workers use hidden labels and scripted failure rules; their accuracy is not experimental evidence.

## Artifacts and failures

Each output directory belongs to one run. Existing run artifacts cause an error rather than being overwritten.

- `manifest.json`: model, safe sampling settings, seed, versions, configuration and limitations.
- `tasks.json`: reproducible corpus, public task instructions and evaluator labels. Only public inputs reach live workers.
- `runs.jsonl`: one flushed record per completed episode, repetition ID, measured usage, and available audit/trajectory data. B stores its constituent rollout records.
- `events.jsonl`: flushed episode starts/ends, LM starts/ends with raw outputs, usage and finish reasons, and interpreter outputs. It retains diagnostics for a failed or interrupted episode; `finish_reasons: ["length"]` identifies output truncation.
- `experiment_1_summary.json`: overall and per-depth vectors and descriptive fits, written on successful completion.
- `failure.json`: task, condition, repetition, and exception if an episode fails or is interrupted with Ctrl-C. The exception is re-raised and the run stops.

There is no automatic resume. Earlier completed episodes remain readable after failure; use a new directory for another run. No summary means the run did not finish. Unmeasured/inapplicable metrics display N/A. H₁ is explicitly untested.

`--depths` retains the original CLI name but selects nominal task sizes; measured dependency depths are 2, 4, 7, and 15. See [EpiDAG limitations](EPIDAG_BENCHMARK.md).

## Deferred stages

`uv run esc gepa-demo` only demonstrates textual metric feedback. It does not optimize a worker. `--reflection-model`, live Condition D, and `llm_verifier` are rejected until implemented. Do not launch a large study using the old runbook's cost estimates or treat the pilot's slopes as a hypothesis test.
