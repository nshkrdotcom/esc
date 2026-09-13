"""Regression checks for the live pilot's execution limits and diagnostics."""
import json
from pathlib import Path
from unittest.mock import Mock

import dspy
import pytest
from typer.testing import CliRunner

from esc import cli
from esc.config import DEFAULT_MODEL, RLMConfig, task_lm
from esc.runner import run_experiment_1
from esc.systems.system_a import SystemAContinuous
from esc.systems.system_b import SystemBSearchHeavy
from esc.systems.system_c import SystemCIsolated
from esc.telemetry import RunTelemetry


def test_limits_reach_every_root_and_search_rollout(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(dspy, "RLM", constructor)
    limits = RLMConfig(max_iters=2, max_llm_calls=0, max_output_chars=789)
    lm = task_lm()
    for system in (SystemAContinuous, SystemBSearchHeavy, SystemCIsolated):
        system(sub_lm=lm, rlm_config=limits)
    assert constructor.call_count == 3
    for call in constructor.call_args_list:
        assert call.kwargs == {"sub_lm": lm, **limits.model_dump()}


def test_request_options_are_explicit_for_recursive_lm_too():
    lm = task_lm(max_tokens=777, num_ctx=8192, request_timeout=33)
    assert lm.model == DEFAULT_MODEL
    assert not lm.cache and lm.num_retries == 0
    assert lm.kwargs["max_tokens"] == 777
    assert lm.kwargs["timeout"] == 33
    assert lm.kwargs["reasoning_effort"] == "none"
    assert lm.kwargs["num_ctx"] == 8192


def test_litellm_translates_nonthinking_request():
    from litellm.llms.ollama.chat.transformation import OllamaChatConfig
    config = OllamaChatConfig()
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "none", "max_tokens": 777},
        optional_params={}, model="esc-qwen3:14b-nothink", drop_params=False,
    )
    request = config.transform_request(
        model="esc-qwen3:14b-nothink", messages=[{"role": "user", "content": "test"}],
        optional_params=params, litellm_params={}, headers={},
    )
    assert request["think"] is False
    assert request["options"]["num_predict"] == 777


def test_nonthinking_template_closes_generation_prefix():
    template = (Path(__file__).parents[1] / "models/Modelfile.qwen3-nothink").read_text()
    assert '<|im_start|>assistant\n<think>\n</think>\n\n' in template
    # Needed for Ollama 0.33's capability selection to retain the Go template.
    assert '<think>{{ .Thinking }}</think>' in template


def test_cli_propagates_overrides(monkeypatch):
    runner = Mock(side_effect=RuntimeError("stop before execution"))
    monkeypatch.setattr(cli, "run_experiment_1", runner)
    CliRunner().invoke(cli.app, ["run", "--max-iters", "2", "--max-subcalls", "0",
                                "--max-output-chars", "1234", "--n-samples", "2"])
    assert runner.call_args.kwargs["rlm_config"] == RLMConfig(max_iters=2, max_llm_calls=0, max_output_chars=1234)
    assert runner.call_args.kwargs["n_samples"] == 2


def test_interrupt_records_failure_and_preserves_completed_work(tmp_path, monkeypatch):
    monkeypatch.setattr(SystemBSearchHeavy, "run", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        run_experiment_1(depths=[2], tasks_per_depth=1, repetitions=1, output_dir=tmp_path,
                         rlm_config=RLMConfig(max_iters=2), n_samples=2)
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path / "failure.json").read_text())["error_type"] == "KeyboardInterrupt"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "episode_failed"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["rlm"]["max_iters"] == 2
    assert manifest["n_samples"] == 2


def test_telemetry_persists_truncated_response(tmp_path):
    from types import SimpleNamespace
    callback = RunTelemetry(tmp_path / "events.jsonl")
    lm = SimpleNamespace(model="fixture", history=[])
    callback.on_lm_start("test", lm, {})
    lm.history.append({"usage": {"completion_tokens": 128},
                       "response": SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")])})
    callback.on_lm_end("test", ["partial output"])
    events = [json.loads(line) for line in callback.path.read_text().splitlines()]
    assert events[-1]["finish_reasons"] == ["length"]
    assert events[-1]["outputs"] == ["partial output"]


def test_bad_context_budget_is_rejected_before_inference():
    with pytest.raises(ValueError, match="num_ctx"):
        task_lm(max_tokens=8192, num_ctx=8192)
