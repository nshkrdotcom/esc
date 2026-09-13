"""Explicit, recorded execution limits for the local pipeline pilot."""
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL = "ollama_chat/esc-qwen3:14b-nothink"


class RLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iters: int = Field(default=4, ge=1)
    max_llm_calls: int = Field(default=4, ge=0)
    max_output_chars: int = Field(default=4000, ge=1)


def task_lm(model=DEFAULT_MODEL, *, max_tokens=1024, num_ctx=8192,
            temperature=0.6, request_timeout=120.0):
    """Use the same uncached settings for root and recursive LM requests."""
    from esc.budget import BudgetedLM

    if max_tokens < 1 or request_timeout <= 0 or temperature < 0:
        raise ValueError("Invalid LM token limit, temperature, or timeout")
    options = {}
    if model.startswith("ollama_chat/"):
        if num_ctx <= max_tokens:
            raise ValueError("num_ctx must exceed max_tokens to leave room for input")
        options = {"api_base": "http://localhost:11434", "num_ctx": num_ctx,
                   "reasoning_effort": "none", "top_p": 0.8, "top_k": 20}
    return BudgetedLM(model, cache=False, temperature=temperature, max_tokens=max_tokens,
                   num_retries=0, timeout=request_timeout, **options)
