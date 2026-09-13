"""Provider-reported token usage, including RLM root and recursive calls."""
from dspy import Module
from dspy.utils.usage_tracker import track_usage
from esc.budget import current_ledger


def invoke_with_usage(worker, **inputs):
    with track_usage() as tracker:
        try:
            prediction = worker(**inputs) if isinstance(worker, Module) else worker.forward(**inputs)
        finally:
            ledger = current_ledger()
            if ledger:
                ledger.check()
    usage = tracker.get_total_tokens()
    if not usage or any(
        "prompt_tokens" not in item or "completion_tokens" not in item
        for item in usage.values()
    ):
        raise RuntimeError("LM did not report prompt/completion usage; refusing fabricated token counts")
    prompt = sum(item["prompt_tokens"] for item in usage.values())
    completion = sum(item["completion_tokens"] for item in usage.values())
    return prediction, {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "lm_calls": sum(len(entries) for entries in tracker.usage_data.values()),
        "by_model": usage,
    }
