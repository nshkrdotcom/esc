# Local runtime configuration fix

## Cause and verification

The original imported Qwen3-14B template left the generation prefix inside an open `<think>` block. The CLI's `reasoning_effort="none"` correctly became Ollama's `think=false`, but the prompt still triggered reasoning. A direct request to return only `4` for `2+2` consumed all 128 allowed tokens without answering.

The corrected model alias `esc-qwen3:14b-nothink` reuses the existing Hugging Face Q4_K_M weights and supplies a generation prefix ending with a closed `<think>...</think>` block. Ollama 0.33.1 also selects between its Go template and the GGUF's embedded Jinja template based on capabilities. Retaining the prior assistant `.Thinking` branch is necessary for it to select our corrected template. Removing that branch caused Ollama to prefer the embedded template and reproduce the problem.

After fixing both behaviors, the same request returned `4` with 2 completion tokens in 1.3 seconds and `finish_reason="stop"`.

Relevant primary references: [Qwen3-14B non-thinking mode](https://huggingface.co/Qwen/Qwen3-14B), [Ollama thinking requests](https://docs.ollama.com/capabilities/thinking), and [Ollama 0.33.1 template capability selection](https://github.com/ollama/ollama/blob/v0.33.1/server/images.go).

## Reproduction

```bash
ollama create esc-qwen3:14b-nothink -f models/Modelfile.qwen3-nothink
ollama show esc-qwen3:14b-nothink --template
```

The effective template should be the Go template from the Modelfile, with an explicitly closed thinking block at the generation prefix. If a server update changes template selection, verify this again before interpreting model results. The original model aliases are preserved.

The pilot now defaults to 1,024 generated tokens per call, 8,192 context tokens, 4 RLM iterations, and 4 recursive calls per invocation. One final extraction call remains possible. All limits are configurable and persisted. Provider requests time out after 120 seconds; automatic provider and adapter fallback retries are disabled. The context window was retained because shrinking it could remove useful task state.

## Completed validation

`outputs/config_validation_001` contains a completed local A/B/C run on one two-node task, one repetition, with three rollouts inside B:

| Condition | Total provider tokens | Wall time | Correct |
|---|---:|---:|---|
| A | 3,647 | 13.9 s | yes |
| B (three rollouts) | 17,803 | 53.5 s | yes |
| C | 8,625 | 23.2 s | yes |

All 16 LM calls ended with `stop`; none were truncated or raised a backend error. This validates the runtime fix on a small task, not accuracy or scalability across the full benchmark.

`events.jsonl` now preserves LM outputs, finish reasons and interpreter outputs during an episode. Ctrl-C writes an interruption record to `failure.json`; previously, interrupting a long episode left only its last completed predecessor. Completed records remain in `runs.jsonl`.

Verification: `ESC_TEST_RLM=1 uv run pytest -q` passed all 69 tests, and `git diff --check` passed. No code was committed during this configuration fix.
